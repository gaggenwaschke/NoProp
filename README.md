# NoProp: Community Implementation
Hyungon Ryu | Sr. Solution Architect | NVIDIA AI Technology Center Korea

It's an unofficial community-driven implementation of the NoProp method described in Li et al., "NoProp: Training Neural Networks without Back-propagation or Forward-propagation" [arXiv:2503.24322v1](https://arxiv.org/html/2503.24322v1)

## Overview

NoProp is a novel approach for training neural networks without relying on standard back-propagation or forward-propagation steps. This repository provides:
 - Discrete-Time (DT) and Continuous-Time (CT) implementations.
 - Support for benchmark image classification tasks (MNIST, CIFAR-10, CIFAR-100).
 - Scripts for training, evaluation, and visualization of results.

![Architecture](https://arxiv.org/html/2503.24322v1/extracted/6324620/plots/Noprop_clear.png)

- Figure 1:Architecture of NoProp. $z_0$ represents Gaussian noise, while $z_1,…,z_T$ are successive transformations of $z_0$ through the learned dynamics $u_1,…,u_T$, with each layer conditioned on the image $x$, ultimately producing the class prediction $\hat{y}$.


for more detail, check the original paper [arXiv:2503.24322v1](https://arxiv.org/html/2503.24322v1)

## implementation  
- Modular Design: Duplicate paper and easily extend and investigate NoProp with different model architectures and datasets.
- Modular Backbone Design: Easily configure ResNet-18, ResNet-50, or ResNet-152 backbones.
- CLS Headers with time and Noise : Embed noise (Zt) and time-step (T), then fuse with feature header for classification.
- continous train scheme : follow paper's train scheme with random T for continous train. 
- Scheduler Options: Support both Euler and Heun integration schemes for diffusion timesteps.
- Evaluation Hooks:
  - Automatic Heun integration with T=40 evaluation at the end of every epoch.
  - Post-training evaluation across customizable T values (e.g., [2,5,10,20,40,60,100]).
  - Benchmarks: Pre-configured for MNIST. you can easily evaluate for CIFAR-10, and CIFAR-100.
## Usage 

#### Quick Start 

copy the code
```
git clone https://github.com/yhgon/NoProp.git
```

run with default option for mnist dataset
```
python NoProp/src/nopropct_mnist.py
```

run with configure dataset and backbone model 
```
python NoProp/src/noprop_simple.py --dataset cifar10 --backbone resnet50
```

run with configure dataset and backbone model  default epoch is `400`
```
python NoProp/src/noprop_simple.py --dataset cifar100 --backbone resnet152
```

## log 
you can compare the log and figure in paper
[log of train/eval for mnist](logs/log_mnist.md)
![log_mnist](https://arxiv.org/html/2503.24322v1/extracted/6324620/plots/continuous_MNIST.png)

[log of train/eval for mnist](logs/log_cifar10.md) 
 ![log_cifar10](https://arxiv.org/html/2503.24322v1/extracted/6324620/plots/continuous_CIFAR-10.png)

[log of train/eval for mnist](logs/log_cifar100.md) 
![log_cifar100](https://arxiv.org/html/2503.24322v1/extracted/6324620/plots/continuous_CIFAR-100.png)

- Figure 3:Test accuracies (%) plotted against cumulative training time (in seconds) for models using one-hot label embedding in the continuous-time setting. All models within each plot were trained on the same type of GPU to ensure a fair comparison. NoProp-CT achieves strong performance in terms of both accuracy and speed compared to adjoint sensitivity. For CIFAR-100, NoProp-FM does not learn effectively with one-hot label embedding.



## Citation 
```
@misc{li2025noprop,
  title={NoProp: Training Neural Networks without Back-propagation or Forward-propagation},
  author={Li, Qinyu and Teh, Yee Whye and Pascanu, Razvan},
  year={2025},
  eprint={2503.24322v1},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

```
@misc{ryu2025nopropcode,
  title={NoProp: Community Implementation Code},
  author={Ryu, Hyungon},
  year={2025},
  howpublished={\url{https://github.com/yhgon/NoProp}} 
}
```


