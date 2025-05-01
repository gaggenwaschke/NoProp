# NoProp
implement of NoProp-CT [arXiv:2503.24322v1](https://arxiv.org/html/2503.24322v1)

## modification 

- configure backbone network with ResNet-18,50,152
- add Zt, T and fused header 
- scheduler with euler and heun
- evaluate accuracy with heun T=40 for every epoch
- evaluate various T after finish train

