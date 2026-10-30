"""Exact architecture/initialization copied from preflight commit 9154339."""
import torch
from torch import nn
from torch.nn import functional as F
class Block(nn.Module):
    def __init__(self,ci,co,stride):
        super().__init__(); self.stride=stride; self.co=co
        self.c1=nn.Conv2d(ci,co,3,stride,1,bias=False); self.b1=nn.BatchNorm2d(co)
        self.c2=nn.Conv2d(co,co,3,1,1,bias=False); self.b2=nn.BatchNorm2d(co)
    def forward(self,x):
        y=F.relu(self.b1(self.c1(x))); y=self.b2(self.c2(y))
        z=x[:,:,::self.stride,::self.stride]
        if z.shape[1]!=self.co:
            pad=(self.co-z.shape[1])//2; z=F.pad(z,(0,0,0,0,pad,pad))
        return F.relu(y+z)
class ResNet20(nn.Module):
    def __init__(self):
        super().__init__(); self.stem=nn.Conv2d(3,16,3,1,1,bias=False); self.bn=nn.BatchNorm2d(16)
        blocks=[]; ci=16
        for co in (16,32,64):
            for b in range(3):
                blocks.append(Block(ci,co,2 if b==0 and ci!=co else 1));ci=co
        self.blocks=nn.Sequential(*blocks); self.fc=nn.Linear(64,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x): return self.fc(self.blocks(F.relu(self.bn(self.stem(x)))).mean((2,3)))
