import argparse, copy, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
DEVICE=torch.device('cpu')
SEQ_LEN=20; VOCAB=16; D=24; HEADS=4; PROMPT_LEN=4

# token layout: 0=BOS, 1..3=MODE, 4..7=KEY, 8..15=DATA

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_split(n, seed):
    rng=np.random.default_rng(seed)
    X=np.empty((n,SEQ_LEN),dtype=np.int64)
    for i in range(n):
        mode=int(rng.integers(1,4)); key=int(rng.integers(0,4)); state=int(rng.integers(0,8))
        s=np.empty(SEQ_LEN,dtype=np.int64); s[0]=0; s[1]=mode; s[2]=4+key; s[3]=8+state
        for t in range(4,SEQ_LEN):
            step=mode+key+(1 if (t%2)==0 else 3); state=(state+step)%8; s[t]=8+state
        X[i]=s
    return torch.from_numpy(X)

def datasets():
    tr=make_split(3072,20260824); va=make_split(768,20260825); te=make_split(768,20260826)
    return torch.utils.data.TensorDataset(tr), torch.utils.data.TensorDataset(va), torch.utils.data.TensorDataset(te)

class CausalBlock(nn.Module):
    def __init__(self,d=24,heads=4,mlp=48):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.0)
        self.n2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,mlp),nn.GELU(),nn.Linear(mlp,d))
    def forward(self,x):
        T=x.size(1); mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),diagonal=1)
        z=self.n1(x); a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False); x=x+a
        return x+self.mlp(self.n2(x))

class DecoderLM(nn.Module):
    def __init__(self,depth=4,d=24,heads=4,mlp=48,compiler=None):
        super().__init__(); self.d=d
        self.tok=nn.Embedding(VOCAB,d); self.pos=nn.Parameter(torch.zeros(1,SEQ_LEN,d)); nn.init.normal_(self.pos,std=.02)
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

def continuation_logits_targets(model,t):
    logits=model(t); return logits[:,PROMPT_LEN-1:-1,:], t[:,PROMPT_LEN:]

def train_lm(model,ds,epochs,lr=2e-3,wd=1e-4,trainable=None,seed=0):
    if trainable is not None:
        for p in model.parameters(): p.requires_grad=False
        for obj in trainable:
            if isinstance(obj,nn.Parameter): obj.requires_grad=True
            else:
                for p in obj.parameters(): p.requires_grad=True
    ps=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(ps,lr=lr,weight_decay=wd); lossf=nn.CrossEntropyLoss(); g=torch.Generator().manual_seed(seed)
    dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True,generator=g)
    for _ in range(epochs):
        model.train()
        for (t,) in dl:
            opt.zero_grad(); lg,y=continuation_logits_targets(model,t); loss=lossf(lg.reshape(-1,VOCAB),y.reshape(-1)); loss.backward(); opt.step()
    return model

@torch.no_grad()
def tf_metrics(model,ds,batch=256):
    model.eval(); total_loss=0.0; total_tok=0; correct=0; lossf=nn.CrossEntropyLoss(reduction='sum')
    for (t,) in torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False):
        lg,y=continuation_logits_targets(model,t); total_loss+=float(lossf(lg.reshape(-1,VOCAB),y.reshape(-1))); total_tok+=y.numel(); correct+=int((lg.argmax(-1)==y).sum())
    nll=total_loss/total_tok; return {'nll':nll,'ppl':math.exp(nll),'token_acc':correct/total_tok}

@torch.no_grad()
def generation_metrics(model,ds,max_sequences=256):
    model.eval(); tok_correct=0; tok_total=0; seq_correct=0; nseq=0; first_errors=[]
    for (full,) in torch.utils.data.DataLoader(ds,batch_size=64,shuffle=False):
        full=full[:max(0,max_sequences-nseq)]
        if full.numel()==0: break
        cur=full[:,:PROMPT_LEN].clone()
        for _ in range(PROMPT_LEN,SEQ_LEN):
            nxt=model(cur)[:,-1,:].argmax(-1,keepdim=True); cur=torch.cat([cur,nxt],dim=1)
        pred=cur[:,PROMPT_LEN:]; gold=full[:,PROMPT_LEN:]; eq=(pred==gold)
        tok_correct+=int(eq.sum()); tok_total+=eq.numel(); seq_correct+=int(eq.all(dim=1).sum())
        for row in eq:
            bad=(~row).nonzero(as_tuple=False); first_errors.append(SEQ_LEN-PROMPT_LEN if bad.numel()==0 else int(bad[0,0]))
        nseq+=full.size(0)
        if nseq>=max_sequences: break
    return {'gen_token_acc':tok_correct/tok_total,'gen_exact_seq':seq_correct/nseq,'gen_first_error_mean':float(np.mean(first_errors))}

@torch.no_grad()
def collect_core_io(model,ds,ncal=512):
    model.eval(); xs=[]; ys=[]; n=0
    for (t,) in torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False):
        h0=model.embed(t); h1=h0
        for b in model.blocks: h1=b(h1)
        take=min(t.size(0),ncal-n); xs.append(h0[:take]); ys.append(h1[:take]); n+=take
        if n>=ncal: break
    return torch.cat(xs,0),torch.cat(ys,0)

def fit_compiler(teacher,tr,epochs=50,seed=0):
    set_seed(seed+700000); comp=Compiler(depth=2,mlp=24); X,Y=collect_core_io(teacher,tr,512)
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

def run(seed,outdir,base_epochs=14,compiler_epochs=50):
    set_seed(seed); tr,va,te=datasets(); base=DecoderLM(depth=4,mlp=48)
    train_lm(base,tr,base_epochs,lr=2e-3,wd=1e-4,seed=seed)
    vm=tf_metrics(base,va); tm=tf_metrics(base,te); gm=generation_metrics(base,te)
    rec={'seed':seed,'baseline_val':vm,'baseline_test':tm,'baseline_generation':gm,'baseline_params':nparams(base)}
    if vm['token_acc']<.95:
        rec['eligible']=False; Path(outdir).mkdir(parents=True,exist_ok=True); (Path(outdir)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2)); return rec
    rec['eligible']=True; comp=fit_compiler(base,tr,epochs=compiler_epochs,seed=seed); cand=with_compiler(base,comp)
    rec['compiled_params']=nparams(cand); rec['param_reduction']=1-rec['compiled_params']/rec['baseline_params']
    c0=tf_metrics(cand,te); g0=generation_metrics(cand,te); rec['tau0']=c0; rec['tau0_generation']=g0; rec['tau0_utility']=tm['ppl']/c0['ppl']
    for tau in (2,8):
        ctrl=copy.deepcopy(base); train_lm(ctrl,tr,tau,lr=8e-4,wd=1e-4,seed=seed+10000+tau); ct=tf_metrics(ctrl,te); cg=generation_metrics(ctrl,te)
        cm=copy.deepcopy(cand); shell=[cm.tok,cm.pos,cm.norm,cm.lm_head]
        train_lm(cm,tr,tau,lr=8e-4,wd=1e-4,trainable=shell,seed=seed+20000+tau); mt=tf_metrics(cm,te); mg=generation_metrics(cm,te)
        rec[f'tau{tau}_control']=ct; rec[f'tau{tau}_control_generation']=cg; rec[f'tau{tau}_compiled']=mt; rec[f'tau{tau}_compiled_generation']=mg; rec[f'tau{tau}_utility']=ct['ppl']/mt['ppl']
    Path(outdir).mkdir(parents=True,exist_ok=True); (Path(outdir)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2)); return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--base-epochs',type=int,default=14); ap.add_argument('--compiler-epochs',type=int,default=50)
    a=ap.parse_args(); print(json.dumps(run(a.seed,a.out,a.base_epochs,a.compiler_epochs),indent=2))