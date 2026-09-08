import argparse, json, math, os, random, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Small deterministic ViT generalization experiment for Canaria.
# Historical-result compatible philosophy: baseline-only eligibility, matched continuation control,
# replacement fitted without labels, candidate core frozen during repair.

torch.set_num_threads(1)
DEVICE = torch.device('cpu')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def data_loaders(batch=128):
    d = load_digits()
    X = (d.images.astype(np.float32) / 16.0)[:,None,:,:]
    y = d.target.astype(np.int64)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, random_state=12345, stratify=y)
    Xtr=torch.from_numpy(X[tr]); ytr=torch.from_numpy(y[tr])
    Xte=torch.from_numpy(X[te]); yte=torch.from_numpy(y[te])
    train_ds=torch.utils.data.TensorDataset(Xtr,ytr)
    test_ds=torch.utils.data.TensorDataset(Xte,yte)
    return train_ds, test_ds

class Block(nn.Module):
    def __init__(self,d=32,heads=4,mlp=64):
        super().__init__()
        self.n1=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.0)
        self.n2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,mlp),nn.GELU(),nn.Linear(mlp,d))
    def forward(self,x):
        z=self.n1(x); a,_=self.attn(z,z,z,need_weights=False); x=x+a
        x=x+self.mlp(self.n2(x)); return x

class SmallViT(nn.Module):
    def __init__(self,depth=4,d=32,heads=4,mlp=64,compiler=None):
        super().__init__(); self.d=d
        self.patch=nn.Conv2d(1,d,2,2)
        self.cls=nn.Parameter(torch.zeros(1,1,d)); nn.init.normal_(self.cls,std=0.02)
        self.pos=nn.Parameter(torch.zeros(1,17,d)); nn.init.normal_(self.pos,std=0.02)
        self.blocks=nn.ModuleList([Block(d,heads,mlp) for _ in range(depth)])
        self.compiler=compiler
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,10)
    def embed(self,x):
        x=self.patch(x).flatten(2).transpose(1,2)
        c=self.cls.expand(x.size(0),-1,-1)
        return torch.cat([c,x],1)+self.pos
    def core(self,h):
        if self.compiler is not None: return self.compiler(h)
        for b in self.blocks: h=b(h)
        return h
    def forward(self,x):
        h=self.core(self.embed(x)); return self.head(self.norm(h[:,0]))

@torch.no_grad()
def acc(model, ds, batch=256):
    model.eval(); n=0;c=0
    for x,y in torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False):
        p=model(x).argmax(1); c += int((p==y).sum()); n += y.numel()
    return c/n

def train_cls(model, ds, epochs, lr=2e-3, wd=1e-4, trainable=None, seed=0):
    if trainable is not None:
        for p in model.parameters(): p.requires_grad=False
        for obj in trainable:
            if isinstance(obj, nn.Parameter): obj.requires_grad=True
            else:
                for p in obj.parameters(): p.requires_grad=True
    ps=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(ps,lr=lr,weight_decay=wd)
    lossf=nn.CrossEntropyLoss()
    g=torch.Generator().manual_seed(seed)
    dl=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True,generator=g)
    model.train()
    for _ in range(epochs):
        for x,y in dl:
            opt.zero_grad(); loss=lossf(model(x),y); loss.backward(); opt.step()
    return model

@torch.no_grad()
def collect_core_io(model, ds, ncal=512):
    model.eval(); xs=[];ys=[]; n=0
    dl=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False)
    for x,_ in dl:
        h0=model.embed(x); h1=h0
        for b in model.blocks: h1=b(h1)
        take=min(x.size(0),ncal-n)
        xs.append(h0[:take].detach()); ys.append(h1[:take].detach()); n+=take
        if n>=ncal: break
    return torch.cat(xs,0),torch.cat(ys,0)

class Compiler(nn.Module):
    def __init__(self,d=32,heads=4,mlp=32,depth=1):
        super().__init__(); self.blocks=nn.ModuleList([Block(d,heads,mlp) for _ in range(depth)])
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return x

def fit_compiler(teacher, train_ds, mlp_dim=32, depth=1, epochs=60, seed=0):
    set_seed(seed+700000)
    comp=Compiler(d=teacher.d,heads=4,mlp=mlp_dim,depth=depth)
    X,Y=collect_core_io(teacher,train_ds,ncal=512)
    ds=torch.utils.data.TensorDataset(X,Y)
    g=torch.Generator().manual_seed(seed+700001)
    dl=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,generator=g)
    opt=torch.optim.AdamW(comp.parameters(),lr=3e-3,weight_decay=1e-5)
    for _ in range(epochs):
        comp.train()
        for a,b in dl:
            opt.zero_grad(); pred=comp(a); loss=((pred-b)**2).mean(); loss.backward(); opt.step()
    return comp

def clone_baseline_with_compiler(base, comp):
    import copy
    m=copy.deepcopy(base)
    m.compiler=copy.deepcopy(comp)
    m.blocks=nn.ModuleList([])
    return m

def nparams(m): return sum(p.numel() for p in m.parameters())

def run(seed, outdir, base_epochs=40, compiler_mlp=32, compiler_depth=1, compiler_epochs=60):
    import copy, time
    set_seed(seed)
    tr,te=data_loaders()
    base=SmallViT(depth=4,d=32,heads=4,mlp=64)
    train_cls(base,tr,base_epochs,lr=2e-3,wd=1e-4,seed=seed)
    base_acc=acc(base,te)
    rec={'seed':seed,'baseline_acc':base_acc,'baseline_params':nparams(base)}
    if base_acc < 0.95:
        rec['eligible']=False
        Path(outdir).mkdir(parents=True,exist_ok=True)
        (Path(outdir)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2))
        return rec
    rec['eligible']=True
    comp=fit_compiler(base,tr,mlp_dim=compiler_mlp,depth=compiler_depth,epochs=compiler_epochs,seed=seed)
    cand=clone_baseline_with_compiler(base,comp)
    rec['compiled_params']=nparams(cand)
    rec['param_reduction']=1-rec['compiled_params']/rec['baseline_params']
    rec['tau0_acc']=acc(cand,te); rec['tau0_utility']=rec['tau0_acc']/base_acc
    # matched controls from same baseline, continued all params; compiled repairs shell only.
    for tau in (2,8):
        ctrl=copy.deepcopy(base)
        train_cls(ctrl,tr,tau,lr=8e-4,wd=1e-4,seed=seed+10000+tau)
        ctrl_acc=acc(ctrl,te)
        cm=copy.deepcopy(cand)
        shell=[cm.patch,cm.norm,cm.head,cm.cls,cm.pos]
        train_cls(cm,tr,tau,lr=8e-4,wd=1e-4,trainable=shell,seed=seed+20000+tau)
        ca=acc(cm,te)
        rec[f'tau{tau}_control_acc']=ctrl_acc
        rec[f'tau{tau}_compiled_acc']=ca
        rec[f'tau{tau}_utility']=ca/ctrl_acc
    Path(outdir).mkdir(parents=True,exist_ok=True)
    (Path(outdir)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2))
    return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--base-epochs',type=int,default=40); ap.add_argument('--compiler-mlp',type=int,default=32); ap.add_argument('--compiler-depth',type=int,default=1); ap.add_argument('--compiler-epochs',type=int,default=60)
    a=ap.parse_args(); print(json.dumps(run(a.seed,a.out,a.base_epochs,a.compiler_mlp,a.compiler_depth,a.compiler_epochs),indent=2))
