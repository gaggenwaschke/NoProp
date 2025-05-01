# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import math
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ----------------------------------------------------------------------------
# Sinusoidal embedding for scalar t ∈ [0,1]
# ----------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
    )
    args = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

# ----------------------------------------------------------------------------
# ResNet backbone selector
# ----------------------------------------------------------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, name: str, embed_dim: int = 256):
        super().__init__()
        resnets = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'resnet152': models.resnet152,
        }
        assert name in resnets, f"Unsupported backbone '{name}'"
        resnet = resnets[name](weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # drop fc
        feat_dim = resnet.fc.in_features
        self.proj = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = self.features(x).view(B, -1)
        return self.proj(out)

# ----------------------------------------------------------------------------
# z_t encoder
# ----------------------------------------------------------------------------
class ZEncoder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU()
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# ----------------------------------------------------------------------------
# time embedding encoder
# ----------------------------------------------------------------------------
class TEncoder(nn.Module):
    def __init__(self, time_emb_dim: int = 64, embed_dim: int = 256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, embed_dim),
            nn.ReLU()
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_embedding(t, self.time_emb_dim)
        return self.net(te)

# ----------------------------------------------------------------------------
# fuse head to combine image, z, and t features
# ----------------------------------------------------------------------------
class FuseHead(nn.Module):
    def __init__(self, embed_dim: int = 256, mid_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, mid_dim),
            nn.BatchNorm1d(mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, num_classes)
        )
    def forward(self, fx: torch.Tensor, fz: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([fx, fz, ft], dim=1)
        return self.net(cat)
    @property
    def out_features(self):
        return self.net[-1].out_features

# ----------------------------------------------------------------------------
# noise schedule module for learnable gamma and alpha_bar
# ----------------------------------------------------------------------------
class NoiseSchedule(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gamma_tilde = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, 1), nn.Softplus()
        )
        self.gamma0 = nn.Parameter(torch.tensor(-5.0))
        self.gamma1 = nn.Parameter(torch.tensor(5.0))

    def _gamma_bar(self, t: torch.Tensor) -> torch.Tensor:
        g0 = self.gamma_tilde(torch.zeros_like(t))
        g1 = self.gamma_tilde(torch.ones_like(t))
        return ((self.gamma_tilde(t) - g0) / (g1 - g0 + 1e-8)).clamp(0,1)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        γt = self.gamma0 + (self.gamma1 - self.gamma0) * (1 - self._gamma_bar(t))
        return torch.sigmoid(-γt).clamp(1e-5, 1-1e-5)

# ----------------------------------------------------------------------------
# NoPropCT model integrating all components
# ----------------------------------------------------------------------------
class NoPropCT(nn.Module):
    def __init__(self,
                 backbone: str = 'resnet18',
                 num_classes: int = 10,
                 time_emb_dim: int = 64,
                 embed_dim: int = 256):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, embed_dim)
        self.z_enc = ZEncoder(num_classes, embed_dim)
        self.t_enc = TEncoder(time_emb_dim, embed_dim)
        self.fuse = FuseHead(embed_dim, mid_dim=128, num_classes=num_classes)
        self.noise = NoiseSchedule(hidden_dim=64)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise.alpha_bar(t)

    def forward_u(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        fx = self.backbone(x)
        fz = self.z_enc(z_t)
        ft = self.t_enc(t)
        return self.fuse(fx, fz, ft)

# ----------------------------------------------------------------------------
# single training step
# ----------------------------------------------------------------------------
def train_step(model, x, y, optimizer, device, η: float = 1.0) -> float:
    B = x.size(0)
    m = model.fuse.out_features
    # one-hot
    u_y = torch.eye(m, device=device)[y]
    # sample t
    t = torch.rand(B,1,device=device,requires_grad=True)
    αb = model.alpha_bar(t)
    snr = αb / (1 - αb)
    snr_p = torch.autograd.grad(snr.sum(), t, create_graph=True)[0]
    # noisy z_t
    eps = torch.randn_like(u_y)
    zt = αb * u_y + torch.sqrt(1 - αb) * eps
    # prediction
    logits = model.forward_u(x, zt, t)
    p = F.softmax(logits, dim=1)
    mse = F.mse_loss(p, u_y, reduction='none').sum(dim=1, keepdim=True)
    loss_sdm = 0.5 * η * (snr_p * mse).mean()
    # KL term
    loss_kl = 0.5 * (u_y.pow(2).sum(dim=1)).mean()
    # cross‐entropy at t=1
    t1 = torch.ones_like(t)
    αb1 = model.alpha_bar(t1)
    z1 = αb1 * u_y + torch.sqrt(1 - αb1) * torch.randn_like(u_y)
    loss_ce = F.cross_entropy(model.forward_u(x, z1, t1), y)
    loss = loss_ce + loss_kl + loss_sdm

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------------------------------------------------------
# inference routines
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_noprop_ct_inference(model: nn.Module, x: torch.Tensor, T_steps: int = 1000) -> torch.Tensor:
    model.eval()
    B = x.size(0)
    m = model.fuse.out_features
    dt = 1.0 / T_steps
    z = torch.randn(B, m, device=x.device)
    for i in range(T_steps):
        t = torch.full((B,1), float(i) / T_steps, device=x.device)
        αb = model.alpha_bar(t)
        p = F.softmax(model.forward_u(x, z, t), dim=1)
        z = z + dt * (p - z) / (1 - αb)
    return z.argmax(dim=1)
    
# ----------------------------------------------------------------------------
# inference routines with heun
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_noprop_ct_inference_heun(model: nn.Module, x: torch.Tensor, T_steps: int = 40) -> torch.Tensor:
    model.eval()
    B = x.size(0)
    m = model.fuse.out_features
    dt = 1.0 / T_steps
    z = torch.randn(B, m, device=x.device)
    for i in range(T_steps):
        t_n = torch.full((B,1), float(i) / T_steps, device=x.device)
        t_np1 = torch.full((B,1), float(i+1) / T_steps, device=x.device)
        αn = model.alpha_bar(t_n)
        p_n = F.softmax(model.forward_u(x, z, t_n), dim=1)
        f_n = (p_n - z) / (1 - αn)
        z_mid = z + dt * f_n
        αm = model.alpha_bar(t_np1)
        p_mid = F.softmax(model.forward_u(x, z_mid, t_np1), dim=1)
        f_mid = (p_mid - z_mid) / (1 - αm)
        z = z + 0.5 * dt * (f_n + f_mid)
    return z.argmax(dim=1)

# ----------------------------------------------------------------------------
# train & eval loop per backbone + dataset
# ----------------------------------------------------------------------------
def train_and_eval(backbone: str, dataset: str, data_root: str):
    # dataset‐specific setup
    if dataset == 'mnist':
        ds_train = torchvision.datasets.MNIST(data_root, train=True,  download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Lambda(lambda x: x.repeat(3,1,1)),
                         transforms.Normalize((0.1307,)*3, (0.3081,)*3),
                     ]))
        ds_test  = torchvision.datasets.MNIST(data_root, train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Lambda(lambda x: x.repeat(3,1,1)),
                         transforms.Normalize((0.1307,)*3, (0.3081,)*3),
                     ]))
        num_classes = 10
        input_size = 28
    elif dataset == 'cifar10':
        mean, std = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
        ds_train = torchvision.datasets.CIFAR10(data_root, train=True,  download=True,
                     transform=transforms.Compose([
                         transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std),
                     ]))
        ds_test  = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std),
                     ]))
        num_classes = 10
        input_size = 32
    elif dataset == 'cifar100':
        mean, std = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)
        ds_train = torchvision.datasets.CIFAR100(data_root, train=True,  download=True,
                     transform=transforms.Compose([
                         transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std),
                     ]))
        ds_test  = torchvision.datasets.CIFAR100(data_root, train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std),
                     ]))
        num_classes = 100
        input_size = 32
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- {backbone} on {dataset.upper()} ({num_classes} classes) using {device} ---")

    tr_loader = DataLoader(ds_train, batch_size=2048, shuffle=True,  num_workers=8, drop_last=True)
    te_loader = DataLoader(ds_test,  batch_size=2048, shuffle=False, num_workers=8)

    model = NoPropCT(backbone, num_classes=num_classes, time_emb_dim=64, embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    for ep in range(1, 2001):
        t0, total_loss = time.time(), 0.0
        model.train()
        for x,y in tr_loader:
            x,y = x.to(device), y.to(device)
            total_loss += train_step(model, x, y, optimizer, device) * x.size(0)
        avg_loss = total_loss / len(ds_train)
        print(f"Epoch {ep:02d} loss {avg_loss:.4f} | train {time.time()-t0:.1f}s", end='')

        if ep % 10 == 0:
            # quick eval
            model.eval()
            corr = tot = 0
            eval_t0 = time.time()
            for x,y in te_loader:
                x,y = x.to(device), y.to(device)
                preds = run_noprop_ct_inference_heun(model, x, T_steps=40)
                corr += (preds == y).sum().item()
                tot  += y.size(0)
            print(f" | Acc {100*corr/tot:.2f}% | infer {time.time()-eval_t0:.1f}s", end='')
        print()

    # final multi-T Heun
    print("\nFinal Heun multi-T eval:")
    for T in [2,5,10,20,30,40,50,60,70,80,90,100,200]:
        ti = time.time()
        corr = tot = 0
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            preds = run_noprop_ct_inference_heun(model, x, T)
            corr += (preds == y).sum().item(); tot += y.size(0)
        print(f"Heun T={T:3d} acc {corr/tot:.4%} | infer {time.time()-ti:.1f}s")

    # final multi-T Euler
    print("\nFinal Euler multi-T eval:")
    for T in [2,5,10,20,30,40,50,60,70,80,90,100,200]:
        ti = time.time()
        corr = tot = 0
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            preds = run_noprop_ct_inference(model, x, T)
            corr += (preds == y).sum().item(); tot += y.size(0)
        print(f"Euler T={T:3d} acc {corr/tot:.4%} | infer {time.time()-ti:.1f}s")


    # cleanup
    del model, optimizer, ds_train, ds_test, tr_loader, te_loader
    torch.cuda.empty_cache(); gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist','cifar10','cifar100'], required=True)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--backbones', nargs='+', default=['resnet18','resnet50','resnet152'])
    args = parser.parse_args()

    for backbone in args.backbones:
        train_and_eval(backbone, args.dataset, args.data_root)
