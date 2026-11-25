"""Domain generalization via MixStyle (Zhou et al., ICLR 2021).

Idea: during training, mix the per-sample feature statistics (channel mean/std,
i.e. "style") between shuffled samples in the batch. This synthesises novel
styles the model never saw, pushing it to rely on content (the disease) rather
than domain-specific style (colour/lighting/camera) — exactly the cross-dataset
shift this project targets. MixStyle is a no-op at eval time.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        lmda = self.beta.sample((b, 1, 1, 1)).to(x.device)
        perm = torch.randperm(b, device=x.device)
        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)
        return x_norm * sig_mix + mu_mix


class MixStyleNet(nn.Module):
    """Wraps a torchvision backbone (MobileNetV3 / EfficientNet) and applies
    MixStyle after selected early feature blocks."""

    def __init__(self, backbone: nn.Module, insert_after=(1, 2, 3),
                 p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.mixstyle = MixStyle(p, alpha)
        self.insert_after = set(insert_after)

    def forward(self, x):
        for i, block in enumerate(self.features):
            x = block(x)
            if self.training and i in self.insert_after:
                x = self.mixstyle(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def to_mixstyle(backbone: nn.Module, insert_after=(1, 2, 3),
                p: float = 0.5, alpha: float = 0.1) -> nn.Module:
    return MixStyleNet(backbone, insert_after=insert_after, p=p, alpha=alpha)


# ----------------------------------------------------------------------------
# RandConv (Xu et al., ICLR 2021): a second, independent DG method.
# Applies a convolution with FRESH RANDOM weights to the input each step,
# randomising local texture/colour while preserving global shape, so the model
# is pushed to rely on shape/content rather than dataset-specific appearance.
# No-op at inference.
# ----------------------------------------------------------------------------
class RandConv(nn.Module):
    def __init__(self, kernel_sizes=(1, 3, 5, 7), p: float = 0.5, mix: bool = True):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.p = p
        self.mix = mix

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        k = random.choice(self.kernel_sizes)
        # fresh random (non-learnable) weights, 3->3 channels, per-filter L2-normed
        w = torch.randn(3, 3, k, k, device=x.device)
        w = w / (w.reshape(3, -1).norm(dim=1).reshape(3, 1, 1, 1) + 1e-6)
        out = F.conv2d(x, w, padding=k // 2)
        if self.mix:
            a = torch.rand(1, device=x.device).item()
            out = a * x + (1 - a) * out
        return out


class RandConvNet(nn.Module):
    def __init__(self, backbone: nn.Module, p: float = 0.5):
        super().__init__()
        self.randconv = RandConv(p=p)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(self.randconv(x))


def to_randconv(backbone: nn.Module, p: float = 0.5) -> nn.Module:
    return RandConvNet(backbone, p=p)


# ----------------------------------------------------------------------------
# AdaBN — test-time BatchNorm adaptation (Li et al., 2018).
# The proposed FIX: prediction collapse on an unseen dataset is largely caused
# by BatchNorm running statistics (mean/var) being tuned to the SOURCE domain.
# AdaBN recomputes those statistics on the (unlabelled) TARGET data at test time
# so normalisation matches the target distribution. No labels, no re-training.
# ----------------------------------------------------------------------------
@torch.no_grad()
def adabn_adapt(model: nn.Module, loader, device) -> None:
    """In-place: reset BN running stats and re-estimate them on the target."""
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None      # cumulative moving average over all target batches
        m.train()              # use/update batch statistics during the passes
    for x, *_ in loader:
        model(x.to(device))
    model.eval()               # now evaluate using the target-adapted statistics


# ----------------------------------------------------------------------------
# Partial AdaBN (Schneider et al., 2020 style): interpolate SOURCE and TARGET
# BatchNorm statistics by a coefficient lam in [0,1].  lam=1 -> full AdaBN,
# lam=0 -> untouched source model.  A soft alternative to the hard reset.
# ----------------------------------------------------------------------------
@torch.no_grad()
def partial_adabn_adapt(model: nn.Module, loader, device, lam: float) -> None:
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    src = [(m.running_mean.detach().clone(), m.running_var.detach().clone())
           for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    for x, *_ in loader:
        model(x.to(device))
    for m, (sm, sv) in zip(bns, src):          # running_* now hold target stats
        m.running_mean.copy_((1 - lam) * sm + lam * m.running_mean)
        m.running_var.copy_((1 - lam) * sv + lam * m.running_var)
    model.eval()


# ----------------------------------------------------------------------------
# Tent (Wang et al., ICLR 2021): fully test-time adaptation.  Uses target batch
# statistics AND updates the BatchNorm affine params (gamma, beta) by minimising
# the entropy of predictions.  No labels.  (One pass over the target here.)
# ----------------------------------------------------------------------------
def tent_adapt(model: nn.Module, loader, device, lr: float = 1e-3) -> None:
    import torch.optim as optim
    model.eval()
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train(); m.requires_grad_(True)               # batch stats + trainable affine
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    opt = optim.Adam(params, lr=lr)
    for x, *_ in loader:
        out = model(x.to(device))
        p = out.softmax(1)
        ent = -(p * (p + 1e-9).log()).sum(1).mean()          # entropy minimisation
        opt.zero_grad(); ent.backward(); opt.step()
    model.eval()


def _bn_affine_params(model):
    """Configure BN: train mode (batch stats) + trainable affine only. Returns params."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train(); m.requires_grad_(True)
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    return params


# ----------------------------------------------------------------------------
# EATA (Niu et al., ICML 2022): reliable + weighted entropy minimisation.  Only
# LOW-entropy (reliable) samples are used, each weighted by exp(E0 - H); BN affine
# is adapted.  (We omit EATA's Fisher source-anti-forgetting term, which regularises
# in-distribution performance; our evaluation is target-only, so it does not affect
# the reported metric.)  Reliable threshold E0 = 0.4 * ln(C), as in the paper.
# ----------------------------------------------------------------------------
def eata_adapt(model: nn.Module, loader, device, lr: float = 1e-3) -> None:
    import math, torch.optim as optim
    model.eval()
    params = _bn_affine_params(model)
    e0 = 0.4 * math.log(config.NUM_CLASSES)
    opt = optim.Adam(params, lr=lr)
    for x, *_ in loader:
        out = model(x.to(device))
        p = out.softmax(1)
        ent = -(p * (p + 1e-9).log()).sum(1)
        mask = ent < e0                                       # reliable-sample selection
        if mask.sum() == 0:
            continue
        w = torch.exp((e0 - ent[mask]).detach())              # entropy weighting
        loss = (ent[mask] * w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


# ----------------------------------------------------------------------------
# SAR (Niu et al., ICLR 2023): sharpness-aware + reliable entropy minimisation.
# Reliable (low-entropy) samples only; a SAM ascent step perturbs the BN affine to
# the local-max direction (rho-ball) before the descent step, seeking a flat, stable
# minimum.  BN affine adapted; import config for class count.
# ----------------------------------------------------------------------------
import config  # noqa: E402


def sar_adapt(model: nn.Module, loader, device, lr: float = 1e-3, rho: float = 0.05) -> None:
    import math, torch.optim as optim
    model.eval()
    params = _bn_affine_params(model)
    e0 = 0.4 * math.log(config.NUM_CLASSES)
    opt = optim.SGD(params, lr=lr, momentum=0.9)

    def reliable_entropy():
        out = model(x.to(device)); p = out.softmax(1)
        ent = -(p * (p + 1e-9).log()).sum(1)
        m = ent < e0
        return ent[m].mean() if m.sum() > 0 else ent.mean()

    for x, *_ in loader:
        loss = reliable_entropy()
        opt.zero_grad(); loss.backward()
        with torch.no_grad():                                 # SAM ascent perturbation
            gnorm = torch.norm(torch.stack([q.grad.norm() for q in params if q.grad is not None]))
            scale = rho / (gnorm + 1e-12)
            eps = [(q.grad * scale) if q.grad is not None else None for q in params]
            for q, e in zip(params, eps):
                if e is not None:
                    q.add_(e)
        loss2 = reliable_entropy()                            # gradient at perturbed weights
        opt.zero_grad(); loss2.backward()
        with torch.no_grad():                                 # restore, then descent
            for q, e in zip(params, eps):
                if e is not None:
                    q.sub_(e)
        opt.step()
    model.eval()
