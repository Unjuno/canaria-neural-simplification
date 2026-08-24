import argparse, copy, json, math, random, re, zlib, struct
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import sklearn

torch.set_num_threads(1)
DEVICE=torch.device('cpu')
SEQ_LEN=48; D=24; HEADS=4; PROMPT_LEN=12; ROLLOUT=24
ALPHABET="abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'-()/\n"
ITOS=list(ALPHABET)+['?']
STOI={c:i for i,c in enumerate(ITOS)}
VOCAB=len(ITOS)

TRAIN_DOCS=['breast_cancer.rst','california_housing.rst','covtype.rst','diabetes.rst','digits.rst','iris.rst','kddcup99.rst','lfw.rst','linnerud.rst']
VAL_DOCS=['olivetti_faces.rst','rcv1.rst']
TEST_DOCS=['species_distributions.rst','twenty_newsgroups.rst','wine_data.rst']

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def descr_root():
    return Path(sklearn.__file__).resolve().parent/'datasets'/'descr'

def normalize_text(s):
    s=s.lower().replace('\r','\n')
    s=re.sub(r'https?://\S+',' ',s)
    s=re.sub(r'[`*_={}<>|]+',' ',s)
    s=re.sub(r'[^a-z0-9 .,;:!?\'\-()/\n]+',' ',s)
    s=re.sub(r'[ \t]+',' ',s)
    s=re.sub(r'\n{3,}','\n\n',s)
    return s.strip()+"\n"

def load_docs(names):
    root=descr_root(); return [normalize_text((root/n).read_text(errors='ignore')) for n in names]

def encode(s):
    unk=VOCAB-1
    return torch.tensor([STOI.get(c,unk) for c in s],dtype=torch.long)

def make_windows(names, n, seed):
    rng=np.random.default_rng(seed); docs=[encode(x) for x in load_docs(names)]
    valid=[d for d in docs if len(d)>SEQ_LEN+1]
    X=torch.empty((n,SEQ_LEN+1),dtype=torch.long)
    weights=np.array([max(1,len(d)-SEQ_LEN-1) for d in valid],dtype=float); weights/=weights.sum()
    for i in range(n):
        j=int(rng.choice(len(valid),p=weights)); d=valid[j]; start=int(rng.integers(0,len(d)-SEQ_LEN-1)); X[i]=d[start:start+SEQ_LEN+1]
    return torch.utils.data.TensorDataset(X)

def datasets():
    tr=make_windows(TRAIN_DOCS,1024,20260824)
    va=make_windows(VAL_DOCS,256,20260825)
    te=make_windows(TEST_DOCS,256,20260826)
    return tr,va,te

class CausalBlock(nn.Module):
    def __init__(self,d=24,heads=4,mlp=48):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.0)
        self.n2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,mlp),nn.GELU(),nn.Linear(mlp,d))
    def forward(self,x):
        T=x.size(1); mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),diagonal=1)
        z=self.n1(x); a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False); x=x+a; return x+self.mlp(self.n2(x))

class DecoderLM(nn.Module):
    def __init__(self,depth=4,d=24,heads=4,mlp=48,compiler=None):
        super().__init__(); self.tok=nn.Embedding(VOCAB,d); self.pos=nn.Parameter(torch.zeros(1,SEQ_LEN,d)); nn.init.normal_(self.pos,std=.02)
        self.blocks=nn.ModuleList([CausalBlock(d,heads,mlp) for _ in range(depth)]); self.compiler=compiler
        self.norm=nn.LayerNorm(d); self.lm_head=nn.Linear(d,VOCAB)
    def embed(self,t): return self.tok(t)+self.pos[:,:t.size(1)]
    def core(self,h):
        if self.compiler is not None: return self.compiler(h)
        for b in self.blocks: h=b(h)
        return h
    def forward(self,t): return self.lm_head(self.norm(self.core(self.embed(t))))

class Compiler(nn.Module):
    def __init__(self,d=24,heads=4,mlp=24,depth=2):
        super().__init__(); self.blocks=nn.ModuleList([CausalBlock(d,heads,mlp) for _ in range(depth)])
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return x

def train_lm(model,ds,epochs,lr=2e-3,wd=1e-4,trainable=None,seed=0):
    if trainable is not None:
        for p in model.parameters(): p.requires_grad=False
        for obj in trainable:
            if isinstance(obj,nn.Parameter): obj.requires_grad=True
            else:
                for p in obj.parameters(): p.requires_grad=True
    ps=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(ps,lr=lr,weight_decay=wd); lossf=nn.CrossEntropyLoss(); g=torch.Generator().manual_seed(seed)
    dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True,generator=g)
    for _ in range(epochs):
        model.train()
        for (t,) in dl:
            inp=t[:,:-1]; y=t[:,1:]; opt.zero_grad(); lg=model(inp); loss=lossf(lg.reshape(-1,VOCAB),y.reshape(-1)); loss.backward(); opt.step()
    return model

@torch.no_grad()
def tf_metrics(model,ds,batch=128):
    model.eval(); lossf=nn.CrossEntropyLoss(reduction='sum'); total_loss=0.; total=0; correct=0
    for (t,) in torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False):
        inp=t[:,:-1]; y=t[:,1:]; lg=model(inp); total_loss+=float(lossf(lg.reshape(-1,VOCAB),y.reshape(-1))); total+=y.numel(); correct+=int((lg.argmax(-1)==y).sum())
    nll=total_loss/total; return {'nll':nll,'ppl':math.exp(nll),'token_acc':correct/total}

@torch.no_grad()
def rollout_sequences(model,ds,max_sequences=32):
    model.eval(); outs=[]; n=0
    for (t,) in torch.utils.data.DataLoader(ds,batch_size=32,shuffle=False):
        if n>=max_sequences: break
        t=t[:max_sequences-n]; cur=t[:,:PROMPT_LEN].clone()
        for _ in range(ROLLOUT):
            nxt=model(cur)[:,-1,:].argmax(-1,keepdim=True); cur=torch.cat([cur,nxt],1)
        outs.append(cur[:,PROMPT_LEN:].cpu()); n+=t.size(0)
    return torch.cat(outs,0)

@torch.no_grad()
def generation_agreement(a,b,ds,max_sequences=32):
    A=rollout_sequences(a,ds,max_sequences); B=rollout_sequences(b,ds,max_sequences); eq=A.eq(B)
    first=[]
    for row in eq:
        bad=(~row).nonzero(as_tuple=False); first.append(ROLLOUT if bad.numel()==0 else int(bad[0,0]))
    return {'token_agreement':float(eq.float().mean()),'exact_agreement':float(eq.all(1).float().mean()),'first_divergence_mean':float(np.mean(first))}

@torch.no_grad()
def collect_core_io(model,ds,ncal=512):
    xs=[]; ys=[]; n=0; model.eval()
    for (t,) in torch.utils.data.DataLoader(ds,batch_size=64,shuffle=False):
        inp=t[:,:-1]; h0=model.embed(inp); h=h0
        for b in model.blocks: h=b(h)
        take=min(inp.size(0),ncal-n); xs.append(h0[:take]); ys.append(h[:take]); n+=take
        if n>=ncal: break
    return torch.cat(xs),torch.cat(ys)

def fit_compiler(teacher,tr,epochs=20,seed=0):
    set_seed(seed+700000); comp=Compiler(); X,Y=collect_core_io(teacher,tr,512); ds=torch.utils.data.TensorDataset(X,Y); g=torch.Generator().manual_seed(seed+700001)
    dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True,generator=g); opt=torch.optim.AdamW(comp.parameters(),lr=3e-3,weight_decay=1e-5)
    for _ in range(epochs):
        comp.train()
        for a,b in dl:
            opt.zero_grad(); loss=((comp(a)-b)**2).mean(); loss.backward(); opt.step()
    return comp

def with_compiler(base,comp):
    m=copy.deepcopy(base); m.compiler=copy.deepcopy(comp); m.blocks=nn.ModuleList([]); return m

def nparams(m): return sum(p.numel() for p in m.parameters())

def quantize_state(m):
    q=copy.deepcopy(m)
    with torch.no_grad():
        for p in q.parameters():
            if not p.is_floating_point(): continue
            mx=float(p.abs().max()); s=max(mx/127.0,1e-12); z=torch.clamp(torch.round(p/s),-127,127); p.copy_(z*s)
    return q

def state_stream(m):
    out=bytearray(b'CQ8R')
    for name,t in m.state_dict().items():
        arr=t.detach().cpu().numpy().astype(np.float32); mx=float(np.max(np.abs(arr))) if arr.size else 0.; s=max(mx/127.0,1e-12); q=np.clip(np.round(arr/s),-127,127).astype(np.int8)
        nb=name.encode(); out+=struct.pack('<H',len(nb))+nb+struct.pack('<B',arr.ndim)+struct.pack('<'+'I'*arr.ndim,*arr.shape)+struct.pack('<fI',s,q.size)+q.tobytes()
    return bytes(out)
