# NoProp
Hyungon Ryu | NVAITC Korea

implement of NoProp-CT [arXiv:2503.24322v1](https://arxiv.org/html/2503.24322v1)

![Architecture](https://arxiv.org/html/2503.24322v1/extracted/6324620/plots/Noprop_clear.png)
Figure 1:Architecture of NoProp. $z_0$ represents Gaussian noise, while $z_1,…,z_T$ are successive transformations of $z_0$ through the learned dynamics $u_1,…,u_T$, with each layer conditioned on the image $x$, ultimately producing the class prediction $\hat{y}$.

![log](https://arxiv.org/html/2503.24322v1/extracted/6324620/plots/continuous_CIFAR-100.png)
Figure 3:Test accuracies (%) plotted against cumulative training time (in seconds) for models using one-hot label embedding in the continuous-time setting. All models within each plot were trained on the same type of GPU to ensure a fair comparison. NoProp-CT achieves strong performance in terms of both accuracy and speed compared to adjoint sensitivity. For CIFAR-100, NoProp-FM does not learn effectively with one-hot label embedding.


## implementation  
- modular design for backbone 
- configure backbone network with ResNet-18,50,152
- add Zt, T and fused header for cls
- scheduler with euler and heun
- evaluate accuracy with heun T=40 for every epoch
- evaluate various T after finish train

## log 
[train/eval for mnist](log01.md)
 
