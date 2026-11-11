"""Ban KHA VI (torch) cua the K de CHIEU twin ve mat K = const (Gauss-Newton/Adam tren u, muc 5.3.3a cua de xuat).
Cac muc dem/nguong (HCR, SR, VCR, NCV, NCA) dung xap xi sigmoid/tanh de co gradient; SAU DO do sai so bang K THAT (compute_K).
Thu tu cot KHOP knowledge.ALL_NAMES (kiem bang tests/test_softk)."""
from __future__ import annotations
import math
import torch
from .knowledge import ALL_NAMES


def decode_t(p1, beta1, nu, om):
    beta = torch.cat([beta1[:, None], beta1[:, None] + torch.cumsum(om, 1)], 1)
    s = torch.exp(nu)
    d = torch.stack([s * torch.cos(beta), s * torch.sin(beta)], -1)
    return torch.cat([p1[:, None], p1[:, None] + torch.cumsum(d, 1)], 1)


def _slope(v):
    n = v.shape[1]
    t = torch.arange(n, dtype=v.dtype, device=v.device) - (n - 1) / 2.0
    return (v * t).sum(1) / (t ** 2).sum()


def _ac1(v):
    a = v[:, :-1] - v[:, :-1].mean(1, keepdim=True); b = v[:, 1:] - v[:, 1:].mean(1, keepdim=True)
    return (a * b).sum(1) / (torch.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-9)


def soft_K(p1, beta1, nu, om, tau=0.05, soft=True):
    """Tat ca tensor float64/float32 batched. soft=False: dung nguong cung (de kiem so khop voi compute_K)."""
    X = decode_t(p1, beta1, nu, om)
    s = torch.exp(nu); path = s.sum(1)
    net_v = X[:, -1] - X[:, 0]; net = torch.sqrt((net_v ** 2).sum(1) + 1e-12)
    ext = X.amax(1) - X.amin(1)
    acc = s[:, 1:] - s[:, :-1]
    c = torch.cos(om).mean(1); p = s.mean(1); bcv = s.std(1, unbiased=False) / (p + 1e-9)
    sin_denom = (1 + c) / (1 - c + 1e-6) + bcv ** 2
    sinu = 2.0 * (p * sin_denom) ** -0.5
    if soft:
        sg = lambda x: torch.sigmoid(x / tau)
        hcr = sg(om.abs() - math.radians(19)).sum(1) / (path + 1e-9)
        sr = sg(0.25 - s).sum(1) / (path + 1e-9)
        vcr = sg(acc.abs() / (s[:, :-1] + 1e-9) - 0.26).sum(1) / (path + 1e-9)
        sa = torch.tanh(acc / 0.05); ncv = (0.5 * (1 - sa[:, 1:] * sa[:, :-1])).sum(1)
        acc2 = acc[:, 1:] - acc[:, :-1]; sa2 = torch.tanh(acc2 / 0.05); nca = (0.5 * (1 - sa2[:, 1:] * sa2[:, :-1])).sum(1)
    else:
        hcr = (om.abs() > math.radians(19)).double().sum(1) / (path + 1e-9)
        sr = (s < 0.25).double().sum(1) / (path + 1e-9)
        vcr = ((acc.abs() / (s[:, :-1] + 1e-9)) > 0.26).double().sum(1) / (path + 1e-9)
        sa = torch.sign(acc); ncv = (sa[:, 1:] * sa[:, :-1] < 0).double().sum(1)
        acc2 = acc[:, 1:] - acc[:, :-1]; sa2 = torch.sign(acc2); nca = (sa2[:, 1:] * sa2[:, :-1] < 0).double().sum(1)
    acc2 = acc[:, 1:] - acc[:, :-1]
    srt = torch.topk(s, 3, dim=1).values.mean(1)
    top3_acc = torch.topk(acc.abs(), 3, dim=1).values.mean(1)
    cols = [s.mean(1), s.std(1, unbiased=False), path, net, ext[:, 0], ext[:, 1], net_v[:, 0] / (net + 1e-9), net_v[:, 1] / (net + 1e-9),
            om.abs().mean(1), om.std(1, unbiased=False), srt, top3_acc,
            net / (path + 1e-9), sinu, hcr, sr, vcr, _ac1(om), _ac1(nu),
            nu.mean(1), _slope(nu), torch.cos(beta1), torch.sin(beta1), om.mean(1), _slope(om),
            ncv, nca, torch.sqrt((acc2 ** 2).mean(1))]
    K = torch.stack(cols, 1)
    assert K.shape[1] == len(ALL_NAMES)
    return K
