import argparse
import os

from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment, get_mean_lr, str2bool, compute_weights
from util.cluster_and_log_utils import log_accs_from_preds, cluster_acc, add_to_label_same_w
from util.memory_queue_utils import MemoryQueue
from config import exp_root, dino_pretrain_path
from model import ReLKDHead, info_nce_logits, coarse_info_nce_logits, get_coarse_sup_logits_mean_labels, get_coarse_sup_logits_random_labels, get_coarse_sup_logits_mq_labels, SupConLoss, CoarseSupConLoss, DistillLoss, TCALoss, PrototypesLoss, ContrastiveLearningViewGenerator, get_params_groups

from vit_model import vision_transformer as vits

def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
    start_epoch = 0

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['model'])
        optimizer.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['optimizer'])
        exp_lr_scheduler.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['scheduler'])
        start_epoch = torch.load(args.warmup_model_dir, map_location='cpu')['epoch']


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    
    coarse_weight_schedule = compute_weights(
        t_values=np.linspace(0, args.epochs, args.epochs),
        T_start=args.warmup_coarse_weight_start_epoch,
        T_end=args.warmup_coarse_weight_end_epoch,
        lambda_final=args.coarse_weight
    )
    distill_weight_schedule = compute_weights(
        t_values=np.linspace(0, args.epochs, args.epochs),
        T_start=args.warmup_distill_weight_start_epoch,
        T_end=args.warmup_distill_weight_end_epoch,
        lambda_final=args.distill_weight
    )

    if args.use_memory_queue:
        memory_queue = MemoryQueue(max_size=args.mq_maxsize, fine_label_num=args.mlp_out_dim, coarse_label_num=args.coarse_out_dim)

    label_same_fine2coarse_w = np.zeros((args.mlp_out_dim, args.coarse_out_dim), dtype=int)
    label_same_coarse2coarse_w = np.zeros((args.coarse_out_dim, args.coarse_out_dim), dtype=int)
    for epoch in range(start_epoch, args.epochs):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            if args.use_coarse_label:
                images, class_labels, coarse_labels, uq_idxs, mask_lab = batch
                coarse_labels = coarse_labels.cuda(non_blocking=True)
            else:
                images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_rep, student_pred, student_proj, student_out, student_coarse_out, student_coarse_from_fine_out = student(images)
                teacher_rep = student_rep.detach()
                teacher_out = student_out.detach()
                teacher_coarse_out = student_coarse_out.detach()

                coarse_prototypes = student.projector.get_coarse_prototypes()
                fine_prototypes = student.projector.get_fine_prototypes()
                # coarse_from_fine_prototyps = student.projector.get_coarse_from_fine_prototypes()

                # prototypes loss
                fine_prototypes_loss = PrototypesLoss()(fine_prototypes)
                coarse_prototypes_loss = PrototypesLoss()(coarse_prototypes)

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                if args.use_coarse_label: 
                    coarse_sup_labels = torch.cat([coarse_labels[mask_lab] for _ in range(2)], dim=0)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # coarse clustering, sup,
                coarse_sup_logits = torch.cat([f[mask_lab] for f in (student_coarse_out).chunk(2)], dim=0)
                coarse_from_fine_sup_logits = torch.cat([f[mask_lab] for f in (student_coarse_from_fine_out).chunk(2)], dim=0)
                teacher_sup_logits = torch.cat([f[mask_lab] for f in teacher_out.chunk(2)], dim=0)
                coarse_teacher_sup_logits = torch.cat([f[mask_lab] for f in teacher_coarse_out.chunk(2)], dim=0)

                if args.use_coarse_label:
                    coarse_gt_cls_loss = nn.CrossEntropyLoss()(coarse_sup_logits / 0.1, coarse_sup_labels)

                if args.use_memory_queue:
                    # use memory queue
                    coarse_sup_cls_loss = torch.tensor(0., requires_grad=True)
                    if epoch >= args.mq_start_add_epoch:
                        memory_queue.add(teacher_sup_logits, coarse_teacher_sup_logits)
                    if epoch >= args.mq_start_query_epoch:
                        if args.mq_query_mode == 'soft':
                            mq_labels = memory_queue.soft_query()
                            coarse_sup_logits_labels = get_coarse_sup_logits_mq_labels(fine_labels=sup_labels, mq_labels=mq_labels)
                            coarse_sup_cls_loss = cluster_criterion(coarse_sup_logits, coarse_sup_logits_labels, epoch)
                        elif args.mq_query_mode == 'hard':
                            mq_labels = memory_queue.hard_query()
                            coarse_sup_logits_labels = get_coarse_sup_logits_mq_labels(fine_labels=sup_labels, mq_labels=mq_labels)
                            coarse_sup_cls_loss = nn.CrossEntropyLoss()(coarse_sup_logits, coarse_sup_logits_labels)
                        else:
                            raise ValueError(f'The options of args.mq_query_mode is [`soft`, `hard`], but find {args.mq_query_mode}')
                else:
                    # NOTE: mean
                    coarse_sup_logits_labels = get_coarse_sup_logits_mean_labels(teacher_coarse_logits=coarse_teacher_sup_logits, fine_labels=sup_labels, fine_out_dim=args.mlp_out_dim)

                    # NOTE: soft loss
                    coarse_sup_cls_loss = cluster_criterion(coarse_sup_logits, coarse_sup_logits_labels, epoch)
                    # NOTE: hard loss
                    # _, coarse_sup_logits_labels = coarse_sup_logits_labels.max(1)
                    # coarse_sup_cls_loss = nn.CrossEntropyLoss()(coarse_sup_logits, coarse_sup_logits_labels)
                ## tca
                coarse_cls_loss = TCALoss()(coarse_logits=coarse_sup_logits, fine_labels=sup_labels, coarse_prototypes=coarse_prototypes, fine_prototypes=fine_prototypes)

                # coarse clustering, unsup
                coarse_cluster_loss = cluster_criterion(student_coarse_out, teacher_coarse_out, epoch)
                coarse_avg_probs = (student_coarse_out / 0.1).softmax(dim=1).mean(dim=0)
                coarse_me_max_loss = - torch.sum(torch.log((coarse_avg_probs + 1e-7)**(-coarse_avg_probs))) + math.log(float(len(coarse_avg_probs)))
                coarse_cluster_loss += args.memax_weight * coarse_me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # coarse represent learning, unsup
                #TODO: coarse_confidence 
                coarse_contrastive_logits, coarse_contrastive_labels, coarse_confidence = coarse_info_nce_logits(features=student_proj, prototypes=coarse_prototypes, coarse_logits=student_coarse_out, confidence_t=args.confidence_t)
                coarse_contrastive_loss_list = torch.nn.CrossEntropyLoss(reduction='none')(coarse_contrastive_logits, coarse_contrastive_labels)
                coarse_contrastive_loss = torch.sum(coarse_contrastive_loss_list * coarse_confidence) / len(coarse_contrastive_loss_list)
                # coarse_contrastive_loss = torch.sum(coarse_contrastive_loss_list) / len(coarse_contrastive_loss_list)

                # representation learning, sup                
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                # coarse representation learning, sup 
                # use simsiam-like sim function
                student_pred = torch.cat([f[mask_lab].unsqueeze(1) for f in student_pred.chunk(2)], dim=1)
                teacher_rep = torch.cat([f[mask_lab].unsqueeze(1) for f in teacher_rep.chunk(2)], dim=1)
                sup_con_labels = class_labels[mask_lab]
                coarse_sup_con_loss = CoarseSupConLoss()(teacher_rep, student_pred, labels=sup_con_labels)

                # distill loss, unsup
                distill_loss = cluster_criterion(student_coarse_from_fine_out, teacher_coarse_out, epoch)
                distill_avg_probs = (student_coarse_from_fine_out / 0.1).softmax(dim=1).mean(dim=0)
                distill_me_max_loss = - torch.sum(torch.log(distill_avg_probs ** (-distill_avg_probs))) + math.log(float(len(distill_avg_probs)))
                distill_loss += args.memax_weight * distill_me_max_loss

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'coarse_cls_loss: {coarse_cls_loss.item():.4f} '
                pstr += f'coarse_sup_cls_loss: {coarse_sup_cls_loss.item():.4f} '
                if args.use_coarse_label:
                    pstr += f'coarse_gt_cls_loss: {coarse_gt_cls_loss.item():.4f} '
                pstr += f'coarse_cluster_loss: {coarse_cluster_loss.item():.4f} '
                pstr += f'coarse_sup_con_loss: {coarse_sup_con_loss.item():.4f} '
                pstr += f'coarse_contrastive_loss: {coarse_contrastive_loss.item():.4f} '
                pstr += f'fine_prototypes_loss: {fine_prototypes_loss.item():.4f} '
                pstr += f'coarse_prototypes_loss: {coarse_prototypes_loss.item():.4f} '
                pstr += f'distill_loss: {distill_loss.item():.4f} '

                fine_loss = 0.
                if args.use_prototypes_loss:
                    fine_loss = args.sup_weight * (cls_loss + sup_con_loss) + (1 - args.sup_weight) * (cluster_loss + contrastive_loss + fine_prototypes_loss)
                else:
                    fine_loss = args.sup_weight * (cls_loss + sup_con_loss) + (1 - args.sup_weight) * (cluster_loss + contrastive_loss)

                coarse_loss = 0.
                coarse_sup_loss = coarse_sup_con_loss + coarse_sup_cls_loss
                coarse_unsup_loss = coarse_cluster_loss + coarse_contrastive_loss
                if args.use_prototypes_loss:
                    coarse_unsup_loss += coarse_prototypes_loss
                if args.use_coarse_label and args.use_gt_coarse_label:
                    coarse_unsup_loss += coarse_gt_cls_loss

                coarse_loss = args.sup_weight * coarse_sup_loss + (1 - args.sup_weight) * coarse_unsup_loss

                loss = 0.
                if args.fine_weight == -1:
                    loss = (1.0 - coarse_weight_schedule[epoch]) * fine_loss + coarse_weight_schedule[epoch] * coarse_loss + distill_weight_schedule[epoch] * distill_loss
                else:
                    loss = args.fine_weight * fine_loss + coarse_weight_schedule[epoch] * coarse_loss + distill_weight_schedule[epoch] * distill_loss
                
            # Train acc
            _, sup_pred = sup_logits.max(1)
            sup_acc = (sup_pred == sup_labels).float().mean().item()

            if args.use_coarse_label:
                _, sup_coarse_pred = coarse_sup_logits.max(1)
                _, sup_coarse_from_fine_pred = coarse_from_fine_sup_logits.max(1)
                add_to_label_same_w(label_same_fine2coarse_w, label_same_coarse2coarse_w, sup_labels.cpu().numpy(), sup_coarse_pred.cpu().numpy(), args.dataset_name)

            loss_record.update(loss.item(), class_labels.size(0))

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        # Step schedule
        exp_lr_scheduler.step()

        if epoch + 1 == args.epochs:
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test, ind_test, w_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
            save_dict = {
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': exp_lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'ind_test': ind_test,
                'w_test': w_test,
                'label_same_fine2coarse_w': label_same_fine2coarse_w,
                'label_same_coarse2coarse_w': label_same_coarse2coarse_w
            }
            save_path = os.path.join(args.model_dir, f'model_{epoch + 1}.pt')
            torch.save(save_dict, save_path)
            args.logger.info("model saved to {}.".format(save_path))



def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    if args.use_coarse_label:
        coarse_preds, coarse_from_fine_preds, coarse_targets = [], [], []
    mask = np.array([])
    for batch_idx, data in enumerate(tqdm(test_loader)):
        if args.use_coarse_label:
            (images, label, coarse_label, _) = data
        else:
            (images, label, _) = data
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, _, logits, coarse_logits, coarse_from_fine_logits = model(images)
            # _, logits, coarse_logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))
            if args.use_coarse_label:
                coarse_preds.append(coarse_logits.argmax(1).cpu().numpy())
                coarse_from_fine_preds.append(coarse_from_fine_logits.argmax(1).cpu().numpy())
                coarse_targets.append(coarse_label.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc, ind, w = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    return all_acc, old_acc, new_acc, ind, w

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_arch', type=str, default='vit')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2', 'v2b'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='ReLKD', help='model name')

    # SimGCD原参数，无需改动
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet, scars, fgvc_aricraft')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', type=str2bool, default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--setting', type=str, default='default', help='dataset setting')
    # parser.add_argument('--save_freq', type=int, default=50, help='save frequency of model when training')

    # 粗粒度层级参数
    parser.add_argument('--coarse_label_num', type=int, default=-1) # 一般设置为-1即可，会根据数据集自动调整
    parser.add_argument('--use_coarse_label', type=str2bool, default=False) # cifar和imagenet数据集需要设置为True
    parser.add_argument('--sup_coarse_con_weight', type=float, default=0.5)

    # ema 参数，不使用ema不用更改
    parser.add_argument('--use_ema', type=str2bool, default=False)
    parser.add_argument('--momentum_ema', type=float, default=0.999)
    parser.add_argument('--interval_ema', type=int, default=1, help='ema update interval')

    # 内存队列参数
    parser.add_argument('--use_memory_queue', type=str2bool, default=False)
    parser.add_argument('--mq_start_add_epoch', type=int, default=30, help='start epoch of adding data to memory queue')
    parser.add_argument('--mq_start_query_epoch', type=int, default=40, help='start epoch of quering data from memory queue')
    parser.add_argument('--mq_query_mode', type=str, default='soft', help='options: soft, hard')
    parser.add_argument('--mq_maxsize', type=int, default=1024, help='max size of memory queue')

    parser.add_argument('--use_prototypes_attention', type=str2bool, default=False)

    # 损失权重调整
    parser.add_argument('--fine_weight', type=float, default=1.0, help='Weight of fine-grained loss')
    parser.add_argument('--warmup_coarse_weight', type=float, default=0.0, help='Initial value for coarse_weight')
    parser.add_argument('--warmup_coarse_weight_start_epoch', type=int, default=30, help='start epoch of coarse_weight warmup')
    parser.add_argument('--warmup_coarse_weight_end_epoch', type=int, default=60, help='end epoch of coarse_weight warmup')
    parser.add_argument('--coarse_weight', type=float, default=0.5, help='final coarse_weight')
    parser.add_argument('--warmup_distill_weight_start_epoch', type=int, default=120, help='start epoch of distill_weight warmup')
    parser.add_argument('--warmup_distill_weight_end_epoch', type=int, default=150, help='end epoch of distill_weight warmup')
    parser.add_argument('--distill_weight', type=float, default=1.0, help='final distill_weight')
    parser.add_argument('--confidence_t', type=float, default=0.1, help='confidence temperature for coarse contrastive loss')
    parser.add_argument('--use_prototypes_loss', type=str2bool, default=True, help='use prototypes loss or not')
    parser.add_argument('--use_gt_coarse_label', type=str2bool, default=False, help='use ground truth coarse label or not')

    # 测试参数
    parser.add_argument('--do_test', type=str2bool, default=False)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'{args.model_name}'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    # TODO: set coarse label num
    if args.coarse_label_num == -1:
        if args.dataset_name == 'cifar100' or args.dataset_name == 'cifar100small':
            args.coarse_out_dim = 20
        elif args.dataset_name == 'cifar10':
            args.coarse_out_dim = 2
        elif args.dataset_name == 'aircraft':
            args.coarse_out_dim = 20
        elif args.dataset_name == 'scars':
            args.coarse_out_dim = 30
        elif args.dataset_name == 'imagenet':
            args.coarse_out_dim = 10
        elif args.dataset_name == 'imagenet_200':
            args.coarse_out_dim = 20
    else:
        args.coarse_out_dim = args.coarse_label_num
        args.use_coarse_label = False


    if args.model_arch == 'vit':
        args.feat_dim = 768
        args.image_size = 224
        args.num_mlp_layers = 3
        backbone = vits.__dict__['vit_base']()
        pretrain_path = dino_pretrain_path
        state_dict = torch.load(pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in backbone.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    else:
        raise NotImplementedError

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    args.double_transform = False
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = ReLKDHead(in_dim=args.feat_dim, out_dim_fine=args.mlp_out_dim, out_dim_coarse=args.coarse_out_dim, mlp_nlayers=args.num_mlp_layers, double_transform=args.double_transform)
    model = nn.Sequential(OrderedDict([
        ('backbone', backbone),
        ('projector', projector)
    ])).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    # test(model, test_loader_labelled, epoch=None, save_name='Test ACC', args=args)
    train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    if args.do_test:
        if args.warmup_model_dir is None:
            raise ValueError('args.warmup_model_dir is None')
        _, __, ___, ____, ind, w = test(model, test_loader_unlabelled, epoch=None, save_name='Train ACC Unlabelled', args=args)
        _, __, ___, ____, ind_test, w_test = test(model, test_loader_labelled, epoch=None, save_name='Test ACC', args=args)
        save_dict = {
            'ind': ind,
            'w': w,
            'ind_test': ind_test,
            'w_test': w_test
        }
        save_path = os.path.join(args.model_dir, f'ind_w.pt')
        torch.save(save_dict, save_path)
        args.logger.info("ind and w saved to {}.".format(save_path))
