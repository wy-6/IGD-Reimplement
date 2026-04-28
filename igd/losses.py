from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambd)


@dataclass
class AMSoftmaxConfig:
    s: float = 1.0
    m: float = 0.3


class AMSoftmaxHead(nn.Module):
    """
    Additive Margin Softmax（AM-Softmax）。
    logits = s * (cos(theta) - m on target class)
    """

    def __init__(self, in_dim: int, num_classes: int, cfg: AMSoftmaxConfig):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.s = float(cfg.s)
        self.m = float(cfg.m)
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        cosine = F.linear(x_norm, w_norm)  # [B, C]
        if labels is None:
            return self.s * cosine
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        adjusted = cosine - one_hot * self.m
        return self.s * adjusted


def am_softmax_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)

