# IDPOC: Class Incremental Learning Based on Identically Distributed Parallel One-Class Classifiers

## Paper

Official repository of Class Incremental Learning Based on Identically Distributed Parallel One-Class Classifiers

Published at **NeuroComputing**

![geometry](img/README/geometry.png)

## Setup

-   Use `./utils/main.py` to run experiments.
-   Some training result can be found in folder `./result`.

## Datasets

**Class-IL / Task-IL settings**

-   Sequential MNIST
-   Sequential CIFAR-10
-   Sequential CIFAR-100
-   Sequential Tiny ImageNet

## Performance

|         | MNIST     | CIFAR10 | CIFAR100  | Tiny-ImageNet |
| ------- | --------- | ------- | --------- | ------------- |
| DER++   | 85.61     | 64.88   | 24.75     | 10.96         |
| LWF     | 21.62     | 19.59   | 9.20      | 9.36          |
| OWM     | 96.30     | 52.83   | 27.63     | 15.30         |
| ILCOC   | 86.05     | 38.40   | 24.39     | 16.97         |
| DisCOIL | **96.69** | 44.54   | 27.50     | 19.75         |
| IDPOC   | 87.51     | 55.50   | **29.08** | **22.55**     |

## Visualizaiton of score distributions

![score](img/README/score.png)

## Requirement

+ numpy==1.16.4
+ Pillow==6.1.0
+ torch==1.3.1
+ torchvision==0.4.2

## Related repository

https://github.com/aimagelab/mammoth

## Citation

@article{sun2023class,

  title={Class Incremental Learning based on Identically Distributed Parallel One-Class Classifiers},
  
  author={Sun, Wenju and Li, Qingyong and Zhang, Jing and Wang, Wen and Geng, YangLi-ao},
  
  journal={Neurocomputing},
  
  volume={556},
  
  pages={126579},
  
  year={2023},
  
  publisher={Elsevier}
  
}
