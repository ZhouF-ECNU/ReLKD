# ReLKD

Implementation of ["ReLKD: Inter-Class Relation Learning with Knowledge Distillation for Generalized Category Discovery"]. (Accepted by ECAI 2025)

## Paper abstract
Generalized Category Discovery (GCD) faces the challenge of categorizing unlabeled data containing both known and novel classes, given only labels for known classes. Previous studies often treat each class independently, neglecting the inherent interclass relations. Obtaining such inter-class relations directly presents a significant challenge in real-world scenarios. To address this issue, we propose ReLKD, an end-to-end framework that effectively exploits implicit inter-class relations and leverages this knowledge to enhance the classification of novel classes. ReLKD comprises three key modules: a target-grained module for learning discriminative representations, a coarse-grained module for capturing hierarchical class relations, and a distillation module for transferring knowledge from the coarse-grained module to refine the target-grained module’s representation learning. Extensive experiments on four datasets demonstrate the effectiveness of ReLKD, particularly in scenarios with limited labeled data.

## Running

### Dependencies

Python version 3.9

```
pip install -r requirements.txt
```

### Datasets
We evaluate the effectiveness of ReLKD on four widely used benchmark datasets.
* CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
* ImageNet-100: https://www.image-net.org
* Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
* Scars: https://github.com/cyizhuo/Stanford_Cars_dataset


### Config

Set paths to datasets and desired log directories in ```config.py```


### Scripts
**Train and eval**:
```
bash scripts/run_${DATASET_NAME}.sh
```


## Citation
>Zhou F., Chen Z., Pavlovski M., Zhang Y., "ReLKD: Inter-Class Relation Learning with Knowledge Distillation for Generalized Category Discovery", Proc. 28th European Conference on Artificial Intelligence (ECAI'25), 2025.



