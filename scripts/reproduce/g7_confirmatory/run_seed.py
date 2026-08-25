from __future__ import annotations
import argparse, copy, json, math, random, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import sklearn

torch.set_num_threads(1)
SEQ_LEN=48
ALPHABET="abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'-()/\n"
ITOS=list(ALPHABET)+['?']; STOI={c:i for i,c in enumerate(ITOS)}; VOCAB=len(ITOS)
TRAIN_DOCS=['breast_cancer.rst','california_housing.rst','covtype.rst','diabetes.rst','digits.rst','iris.rst','kddcup99.rst','lfw.rst','linnerud.rst']
VAL_DOCS=['olivetti_faces.rst','rcv1.rst']
TEST_DOCS=['species_distributions.rst','twenty_newsgroups.rst','wine_data.rst']

def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def descr_root(): return Path(sklearn.__file__).resolve().parent/'datasets'/'descr'
def normalize_text(s):
    s=s.lower().replace('\r','\n'); s=re.sub(r'https?://\S+',' ',s); s=re.sub(r'[`*_={}<>|]+',' ',s); s=re.sub(r"[^a-z0-9 .,;:!?'\-()/\n]+",' ',s); s=re.sub(r'[ \t]+',' ',s); s=re.sub(r'\n{3,}','\n\n',s); return s.strip()+"\n"
def encode(s): return torch.tensor([STOI.get(c,VOCAB-1) for c in s],dtype=torch.long)
def make_windows(names,n,seed):
    rng=np.random.default_rng(seed); root=descr_root(); docs=[encode(normalize_text((root/x).read_text(errors='ignore'))) for x in names]; valid=[d for d in docs if len(d)>SEQ_LEN+1]
    X=torch.empty((n,SEQ_LEN+1),dtype=torch.long); w=np.array([max(1,len(d)-SEQ_LEN-1) for d in valid],dtype=float); w/=w.sum()
    for i in range(n):
        j=int(rng.choice(len(valid),p=w)); d=valid[j]; start=int(rng.integers(0,len(d)-SEQ_LEN-1)); X[i]=d[start:start+SEQ_LEN+1]
    return torch.utils.data.TensorDataset(X)
def datasets(): return make_windows(TRAIN_DOCS,1024,20260824),make_windows(VAL_DOCS,256,20260825),make_windows(TEST_DOCS,256,20260826)

class CausalBlock(nn.Module):
    def __init__(self,d=24,heads=4,mlp=48):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.0); self.n2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,mlp),nn.GELU(),nn.Linear(mlp,d))
    def forward(self,x):
        T=x.size(1); mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),diagonal=1); z=self.n1(x); a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False); x=x+a; return x+self.mlp(self.n2(x))
class DecoderLM(nn.Module):
    def __init__(self,depth=4,d=24,heads=4,mlp=48):
        super().__init__(); self.tok=nn.Embedding(VOCAB,d); self.pos=nn.Parameter(torch.zeros(1,SEQ_LEN,d)); nn.init.normal_(self.pos,std=.02); self.blocks=nn.ModuleList([CausalBlock(d,heads,mlp) for _ in range(depth)]); self.compiler=None; self.norm=nn.LayerNorm(d); self.lm_head=nn.Linear(d,VOCAB)
    def embed(self,t): return self.tok(t)+self.pos[:,:t.size(1)]
    def core(self,h):
        if self.compiler is not None: return self.compiler(h)
        for block in self.blocks: h=block(h)
        return h
    def forward(self,t): return self.lm_head(self.norm(self.core(self.embed(t))))
class GenericCompiler(nn.Module):
    def __init__(self,d=24,heads=4,mlp=24,depth=2): super().__init__(); self.blocks=nn.ModuleList([CausalBlock(d,heads,mlp) for _ in range(depth)])
    def forward(self,x):
        for block in self.blocks: x=block(x)
        return x

@torch.no_grad()
def tf_metrics(model,ds,batch=128):
    model.eval(); lossf=nn.CrossEntropyLoss(reduction='sum'); total_loss=0.; total=0; correct=0
    for (t,) in torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False):
        inp=t[:,:-1]; y=t[:,1:]; lg=model(inp); total_loss+=float(lossf(lg.reshape(-1,VOCAB),y.reshape(-1))); total+=y.numel(); correct+=int((lg.argmax(-1)==y).sum())
    nll=total_loss/total; return {'nll':nll,'ppl':math.exp(nll),'token_acc':correct/total}
def nparams(m): return sum(p.numel() for p in m.parameters())
@torch.no_grad()
def collect_io(model,ds,ncal=512):
    xs=[]; ys=[]; n=0; model.eval()
    for (t,) in torch.utils.data.DataLoader(ds,batch_size=64,shuffle=False):
        inp=t[:,:-1]; x=model.embed(inp); y=model.core(x); take=min(inp.size(0),ncal-n); xs.append(x[:take].cpu()); ys.append(y[:take].cpu()); n+=take
        if n>=ncal: break
    return torch.cat(xs),torch.cat(ys)
def fit_replacement(model,tr,depth,mlp,epochs,seed):
    set_seed(seed); comp=GenericCompiler(depth=depth,mlp=mlp); X,Y=collect_io(model,tr,512); ds=torch.utils.data.TensorDataset(X,Y); opt=torch.optim.AdamW(comp.parameters(),lr=3e-3,weight_decay=1e-5); losses=[]; updates=0
    for ep in range(epochs):
        dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed+ep)); s=n=0
        for a,y in dl:
            opt.zero_grad(); pred=comp(a); loss=((pred-y)**2).mean(); loss.backward(); opt.step(); s+=loss.detach().item()*a.size(0); n+=a.size(0); updates+=1
        losses.append(s/n)
    return comp,{'epochs':epochs,'updates':updates,'mse_first':losses[0],'mse_last':losses[-1]}
def task_optimizer(model,lr): return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
def named_opt_state(model,opt):
    byid={id(p):name for name,p in model.named_parameters()}; out={}
    for p,st in opt.state.items():
        name=byid.get(id(p))
        if name is not None: out[name]={k:(v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in st.items()}
    return out
def restore_named_state(model,opt,state,allow_prefixes=('tok.','pos','norm.','lm_head.')):
    for name,p in model.named_parameters():
        if name in state and any(name.startswith(pre) for pre in allow_prefixes): opt.state[p]={k:(v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in state[name].items()}
def set_lr(opt,lr):
    for group in opt.param_groups: group['lr']=lr
def epoch_lr(epoch_idx): return 2e-3 if epoch_idx<8 else 7e-4
def train_one_epoch(model,tr,opt,seed_epoch):
    model.train(); lossf=nn.CrossEntropyLoss(); dl=torch.utils.data.DataLoader(tr,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed_epoch)); s=0.; n=0; correct=0
    for (t,) in dl:
        inp=t[:,:-1]; y=t[:,1:]; opt.zero_grad(); lg=model(inp); loss=lossf(lg.reshape(-1,VOCAB),y.reshape(-1)); loss.backward(); opt.step(); s+=loss.detach().item()*y.numel(); n+=y.numel(); correct+=int((lg.argmax(-1)==y).sum())
    return {'nll':s/n,'ppl':math.exp(s/n),'token_acc':correct/n}
def new_model(depth,mlp,seed): set_seed(seed); return DecoderLM(depth=depth,d=24,heads=4,mlp=mlp)
def replace_inplace(model,comp): model.blocks=nn.ModuleList([]); model.compiler=comp; return model
def shape(m):
    blocks=m.compiler.blocks if m.compiler is not None else m.blocks; return {'depth':len(blocks),'mlp':blocks[0].mlp[0].out_features,'kind':'compiler' if m.compiler is not None else 'blocks'}
def eval_model(m,va,te): return {'val':tf_metrics(m,va),'test':tf_metrics(m,te),'params':nparams(m),'shape':shape(m)}
def continue_from(model,opt_state,tr,va,start_epoch,end_epoch,seed):
    opt=task_optimizer(model,epoch_lr(start_epoch)); restore_named_state(model,opt,opt_state); curve=[]
    for ep in range(start_epoch,end_epoch): set_lr(opt,epoch_lr(ep)); train=train_one_epoch(model,tr,opt,seed+50000+ep); curve.append({'epoch':ep+1,'train':train,'val':tf_metrics(model,va)})
    return model,opt,curve

def run(seed):
    tr,va,te=datasets(); m=new_model(4,48,seed); opt=task_optimizer(m,epoch_lr(0)); snaps={}; curve=[]
    for ep in range(12):
        set_lr(opt,epoch_lr(ep)); trm=train_one_epoch(m,tr,opt,seed+50000+ep); curve.append({'epoch':ep+1,'train':trm,'val':tf_metrics(m,va)})
        if ep+1 in (4,8,12): snaps[ep+1]=(copy.deepcopy(m),named_opt_state(m,opt))
    rec={'seed':seed,'large_reference':{'final':eval_model(snaps[12][0],va,te)},'task_updates':48}
    sm=new_model(2,24,seed); so=task_optimizer(sm,epoch_lr(0))
    for ep in range(12): set_lr(so,epoch_lr(ep)); train_one_epoch(sm,tr,so,seed+50000+ep)
    rec['small_from_start']={'final':eval_model(sm,va,te)}
    x=copy.deepcopy(snaps[12][0]); c,f=fit_replacement(x,tr,2,24,40,seed+120000); replace_inplace(x,c); rec['terminal_posthoc']={'after_consolidation':eval_model(x,va,te),'final':eval_model(x,va,te),'fit':f}
    x=copy.deepcopy(snaps[8][0]); st=snaps[8][1]; c,f=fit_replacement(x,tr,2,24,40,seed+130000); replace_inplace(x,c); t0=eval_model(x,va,te); x,o,cv=continue_from(x,st,tr,va,8,12,seed); rec['late_one_shot']={'after_consolidation':t0,'recovery_curve':cv,'final':eval_model(x,va,te),'fit':f}
    x=copy.deepcopy(snaps[4][0]); st=snaps[4][1]; c,f=fit_replacement(x,tr,2,24,40,seed+140000); replace_inplace(x,c); t0=eval_model(x,va,te); x,o,cv=continue_from(x,st,tr,va,4,12,seed); rec['early_one_shot']={'after_consolidation':t0,'recovery_curve':cv,'final':eval_model(x,va,te),'fit':f}
    x=copy.deepcopy(snaps[4][0]); st=snaps[4][1]; c1,f1=fit_replacement(x,tr,3,36,15,seed+150000); replace_inplace(x,c1); t01=eval_model(x,va,te); x,o,cv1=continue_from(x,st,tr,va,4,8,seed); pre2=eval_model(x,va,te); st2=named_opt_state(x,o); c2,f2=fit_replacement(x,tr,2,24,14,seed+160000); replace_inplace(x,c2); t02=eval_model(x,va,te); x,o,cv2=continue_from(x,st2,tr,va,8,12,seed)
    rec['progressive_compute_matched']={'after_first':t01,'first_recovery_curve':cv1,'before_second':pre2,'after_second':t02,'second_recovery_curve':cv2,'final':eval_model(x,va,te),'fit1':f1,'fit2':f2}
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=4300); ap.add_argument('--out',type=Path,default=Path('g7_reproduction.json')); a=ap.parse_args(); result=run(a.seed); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)); print(json.dumps({k:result[k]['final']['test']['ppl'] for k in ['large_reference','small_from_start','terminal_posthoc','late_one_shot','early_one_shot','progressive_compute_matched']},indent=2))
if __name__=='__main__': main()
