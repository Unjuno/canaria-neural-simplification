"""Exact allowlisted definitions from the historical Digits source.

This file contains no dataset loader, historical full-grid runner or test evaluation.
See SOURCE_MAP.json for upstream identity and exact definition hashes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Net(nn.Module):
    def __init__(self, seed, d=64, depth=4):
        super().__init__()
        torch.manual_seed(seed)
        self.stem = nn.Linear(64, d)
        self.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
        self.head = nn.Linear(d, 10)

    def forward(self, x, return_acts=False):
        h = F.gelu(self.stem(x))
        acts = [h]
        for block in self.blocks:
            h = block(h)
            acts.append(h)
        logits = self.head(h)
        return (logits, acts) if return_acts else logits


class TinyRes(nn.Module):
    def __init__(self, d, w, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = nn.Linear(d, w, bias=False)
        self.w2 = nn.Linear(w, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


def train_model(seed, Xt, yt, epochs=60):
    model = Net(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 999)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), 64):
            ix = perm[i : i + 64]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
    return model
