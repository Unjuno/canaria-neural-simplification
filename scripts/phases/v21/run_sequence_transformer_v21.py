import argparse, json, random, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
SEQ_LEN=16; VOCAB=32; NCLASS=4

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_split(n, seed):
    assert n % 4 == 0
    rng=np.random.default_rng(seed)
    X=np.empty((n,SEQ_LEN),dtype=np.int64); y=np.empty(n,dtype=np.int64); k=0
    for cls in range(4):
        b12=(cls>>1)&1; b34=cls&1
        for _ in range(n//4):
            pos=rng.choice(SEQ_LEN,size=4,replace=False); p1,p2,p3,p4=pos
            if bool(p1<p2) != bool(b12): p1,p2=p2,p1
            if bool(p3<p4) != bool(b34): p3,p4=p4,p3
            s=rng.integers(5,VOCAB,size=SEQ_LEN,dtype=np.int64); s[p1]=1; s[p2]=2; s[p3]=3; s[p4]=4
            X[k]=s; y[k]=cls; k+=1
    perm=rng.permutation(n); return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])

def datasets():
    Xtr,ytr=make_split(2400,20260824); Xte,yte=make_split(800,20260825)
    return torch.utils.data.TensorDataset(Xtr,ytr), torch.utils.data.TensorDataset(Xte,yte)

class Block(nn.Module):
    def __init__(self,d=24,heads=4,mlp=48):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.0)
        self.n2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,mlp),nn.GELU(),nn.Linear(mlp,d))
    def forward(self,x):
        z=self.n1(x); a,_=self.attn(z,z,z,need_weights=False); x=x+a; return x+self.mlp(self.n2(x))

class SeqTransformer(nn.Module):
    def __init__(self,depth=4,d=24,heads=4,mlp=48,compiler=None):
        super().__init__(); self.d=d; self.tok=nn.Embedding(VOCAB,d)
        self.cls=nn.Parameter(torch.zeros(1,1,d)); nn.init.normal_(self.cls,std=.02)
        self.pos=nn.Parameter(torch.zeros(1,SEQ_LEN+1,d)); nn.init.normal_(self.pos,std=.02)
        self.blocks=nn.ModuleList([Block(d,heads,mlp) for _ in range(depth)]); self.compiler=compiler
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,NCLASS)
    def embed(self,t):
        h=self.tok(t); c=self.cls.expand(t.size(0),-1,-1); return torch.cat([c,h],1)+self.pos
    def core(self,h):
        if self.compiler is not None: return self.compiler(h)
        for b in self.blocks: h=b(h)
        return h
    def forward(self,t):
        h=self.core(self.embed(t)); return self.head(self.norm(h[:,0]))

class Compiler(nn.Module):
    def __init__(self,d=24,heads=4,mlp=24,depth=2):
        super().__init__(); self.blocks=nn.ModuleList([Block(d,heads,mlp) for _ in range(depth)])
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return x

@torch.no_grad()
def acc(model,ds,batch=512):
    model.eval(); c=n=0
    for x,y in torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False):
        p=model(x).argmax(1); c+=int((p==y).sum()); n+=y.numel()
    return c/n

def train_cls(model,ds,epochs,lr=2e-3,wd=1e-4,trainable=None,seed=0):
    if trainable is not None:
        for p in model.parameters(): p.requires_grad=False
        for obj in trainable:
            if isinstance(obj,nn.Parameter): obj.requires_grad=True
            else:
                for p in obj.parameters(): p.requires_grad=True
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=wd)
    lossf=nn.CrossEntropyLoss(); g=torch.Generator().manual_seed(seed)
    dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True,generator=g)
    for _ in range(epochs):
        model.train()
        for x,y in dl:
            opt.zero_grad(); loss=lossf(model(x),y); loss.backward(); opt.step()
    return model

@torch.no_grad()
def collect_core_io(model,ds,ncal=512):
    model.eval(); xs=[];ys=[];n=0
    for t,_ in torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False):
        h0=model.embed(t); h1=h0
        for b in model.blocks: h1=b(h1)
        take=min(t.size(0),ncal-n); xs.append(h0[:take]); ys.append(h1[:take]); n+=take
        if n>=ncal: break
    return torch.cat(xs,0),torch.cat(ys,0)

def fit_compiler(teacher,tr,epochs=60,seed=0):
    set_seed(seed+700000); comp=Compiler(); X,Y=collect_core_io(teacher,tr,512)
    ds=torch.utils.data.TensorDataset(X,Y); g=torch.Generator().manual_seed(seed+700001)
    dl=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,generator=g); opt=torch.optim.AdamW(comp.parameters(),lr=3e-3,weight_decay=1e-5)
    for _ in range(epochs):
        comp.train()
        for a,b in dl:
            opt.zero_grad(); loss=((comp(a)-b)**2).mean(); loss.backward(); opt.step()
    return comp

def with_compiler(base,comp):
    m=copy.deepcopy(base); m.compiler=copy.deepcopy(comp); m.blocks=nn.ModuleList([]); return m

def nparams(m): return sum(p.numel() for p in m.parameters())

def run(seed,outdir,base_epochs=18,compiler_epochs=60):
    set_seed(seed); tr,te=datasets(); base=SeqTransformer(); train_cls(base,tr,base_epochs,seed=seed); ba=acc(base,te)
    rec={'seed':seed,'baseline_acc':ba,'baseline_params':nparams(base)}
    if ba<.95:
        rec['eligible']=False
    else:
        rec['eligible']=True; comp=fit_compiler(base,tr,compiler_epochs,seed); cand=with_compiler(base,comp)
        rec['compiled_params']=nparams(cand); rec['param_reduction']=1-rec['compiled_params']/rec['baseline_params']
        rec['tau0_acc']=acc(cand,te); rec['tau0_utility']=rec['tau0_acc']/ba
        for tau in (2,8):
            ctrl=copy.deepcopy(base); train_cls(ctrl,tr,tau,lr=8e-4,seed=seed+10000+tau); ca0=acc(ctrl,te)
            cm=copy.deepcopy(cand); train_cls(cm,tr,tau,lr=8e-4,trainable=[cm.tok,cm.cls,cm.pos,cm.norm,cm.head],seed=seed+20000+tau); ca=acc(cm,te)
            rec[f'tau{tau}_control_acc']=ca0; rec[f'tau{tau}_compiled_acc']=ca; rec[f'tau{tau}_utility']=ca/ca0
    Path(outdir).mkdir(parents=True,exist_ok=True); (Path(outdir)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2)); return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); print(json.dumps(run(a.seed,a.out),indent=2))
