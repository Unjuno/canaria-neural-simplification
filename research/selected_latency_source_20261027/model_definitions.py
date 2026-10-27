"""Exact upstream ResBlock/TinyRes only; no training or dataset entry point."""
import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, d)

    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        z = self.fc2(z)
        return x + z

class TinyRes(nn.Module):
    def __init__(self, d, w, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = nn.Linear(d, w, bias=False)
        self.w2 = nn.Linear(w, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))
