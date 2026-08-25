from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


torch.set_num_threads(1)


class ResBlock(nn.Module):
    def __init__(self, d: int = 64):
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
    def __init__(self, seed: int, d: int = 64, depth: int = 4):
        super().__init__()
        torch.manual_seed(seed)
        self.stem = nn.Linear(64, d)
        self.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
        self.head = nn.Linear(d, 10)

    def forward(self, x, return_acts: bool = False):
        h = F.gelu(self.stem(x))
        acts_out = [h]
        for block in self.blocks:
            h = block(h)
            acts_out.append(h)
        logits = self.head(h)
        return (logits, acts_out) if return_acts else logits


class TinyRes(nn.Module):
    def __init__(self, d: int, w: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = nn.Linear(d, w, bias=False)
        self.w2 = nn.Linear(w, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


class PairReplacedNet(nn.Module):
    def __init__(self, teacher, k: int, r1=None, r2=None, comp=None):
        super().__init__()
        self.stem = copy.deepcopy(teacher.stem)
        self.blocks = copy.deepcopy(teacher.blocks)
        self.head = copy.deepcopy(teacher.head)
        self.k, self.r1, self.r2, self.comp = k, r1, r2, comp

    def forward(self, x):
        h = F.gelu(self.stem(x))
        i = 0
        while i < len(self.blocks):
            if i == self.k:
                h = self.comp(h) if self.comp is not None else self.r2(self.r1(h))
                i += 2
            else:
                h = self.blocks[i](h)
                i += 1
        return self.head(h)


def datasets():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=1234, stratify=y)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=5678, stratify=y[train_idx]
    )
    return (
        torch.tensor(X[train_idx]), torch.tensor(y[train_idx], dtype=torch.long),
        torch.tensor(X[val_idx]), torch.tensor(y[val_idx], dtype=torch.long),
        torch.tensor(X[test_idx]), torch.tensor(y[test_idx], dtype=torch.long),
    )


def train_model(seed, X, y, epochs: int = 60):
    model = Net(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 999)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 64):
            ix = perm[i:i + 64]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[ix]), y[ix])
            loss.backward()
            opt.step()
    return model


def fit_map(module, X, Y, updates: int, seed: int):
    opt = torch.optim.AdamW(module.parameters(), lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(X)
    for _ in range(updates):
        ix = torch.randint(0, n, (128,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(module(X[ix]), Y[ix])
        loss.backward()
        opt.step()
    return module


def accuracy(model, X, y):
    with torch.no_grad():
        return float((model(X).argmax(-1) == y).float().mean())


def acts(model, X):
    with torch.no_grad():
        return model(X, return_acts=True)[1]


def quantize_per_matrix(module, bits: int):
    q = copy.deepcopy(module)
    if bits == 32:
        return q, 0
    qmax = (1 << (bits - 1)) - 1
    scales = 0
    with torch.no_grad():
        for layer in q.modules():
            if isinstance(layer, nn.Linear):
                w = layer.weight
                mx = float(w.abs().max())
                scale = mx / qmax if mx > 0 else 1.0
                layer.weight.copy_(torch.clamp(torch.round(w / scale), -qmax, qmax) * scale)
                scales += 1
    return q, scales


def quantize_rowwise(module, bits: int):
    q = copy.deepcopy(module)
    qmax = (1 << (bits - 1)) - 1
    scales = 0
    with torch.no_grad():
        for layer in q.modules():
            if isinstance(layer, nn.Linear):
                w = layer.weight
                mx = w.abs().amax(dim=1, keepdim=True)
                scale = torch.where(mx > 0, mx / qmax, torch.ones_like(mx))
                layer.weight.copy_(torch.clamp(torch.round(w / scale), -qmax, qmax) * scale)
                scales += w.shape[0]
    return q, scales


def code_bits(nparams: int, bits: int, nscales: int):
    if bits == 32:
        return nparams * 32
    return nparams * bits + nscales * 16
