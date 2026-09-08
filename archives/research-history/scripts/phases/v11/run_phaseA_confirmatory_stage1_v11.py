import os, json, math, random, argparse, hashlib, importlib.util, time
from copy import deepcopy
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
OUT=os.path.join(ROOT,'raw_experiments','canary_phaseA_confirmatory_stage1_v11'); os.makedirs(OUT,exist_ok=True)
torch.set_num_threads(4)
# Existing frozen training/model helpers
spec=importlib.util.spec_from_file_location('r',os.path.join(ROOT,'scripts','run_canary_global_relocation_pilot_v9.py')); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
TAUS=[0,1,2,4,8]; FLOOR=.95; CALN=192
GRAMMAR=['original','channel_affine','conv1','conv3','conv5','lowrank3_r2','lowrank3_r4','dwpw3']

class ChannelAffine(nn.Module):
    def __init__(self,a,b): super().__init__(); self.register_buffer('a',a.view(1,-1,1,1)); self.register_buffer('b',b.view(1,-1,1,1))
    def forward(self,x): return self.a*x+self.b
class LowRankCore(nn.Module):
    def __init__(self,first,second): super().__init__(); self.first=first; self.second=second
    def forward(self,x): return self.second(self.first(x))
def fit_channel_affine(Z,Y):
    aa=[]; bb=[]
    for c in range(Z.shape[1]):
        x=Z[:,c].reshape(-1); y=Y[:,c].reshape(-1); xm=x.mean(); ym=y.mean(); var=((x-xm)**2).mean()+1e-8
        a=((x-xm)*(y-ym)).mean()/var; b=ym-a*xm; aa.append(a); bb.append(b)
    return ChannelAffine(torch.stack(aa),torch.stack(bb))
def lowrank_from_linear(core,rank):
    c=core.conv; W=c.weight.detach().clone(); b=c.bias.detach().clone(); o,i,k,_=W.shape; M=W.reshape(o,-1)
    U,S,Vh=torch.linalg.svd(M,full_matrices=False); rr=min(rank,len(S)); A=(torch.diag(torch.sqrt(S[:rr]))@Vh[:rr]).reshape(rr,i,k,k); B=(U[:,:rr]@torch.diag(torch.sqrt(S[:rr]))).reshape(o,rr,1,1)
    first=nn.Conv2d(i,rr,k,padding=k//2,bias=False); second=nn.Conv2d(rr,o,1,bias=True)
    with torch.no_grad(): first.weight.copy_(A); second.weight.copy_(B); second.bias.copy_(b)
    return LowRankCore(first,second)

class DWPW(nn.Module):
    def __init__(self,dw,pw): super().__init__(); self.dw=dw; self.pw=pw
    def forward(self,x): return self.pw(self.dw(x))
def dwpw_from_linear(core):
    c=core.conv; W=c.weight.detach().clone(); b=c.bias.detach().clone(); o,i,k,_=W.shape; D=torch.zeros(i,k,k); P=torch.zeros(o,i)
    for ci in range(i):
        M=W[:,ci].reshape(o,-1); U,S,Vh=torch.linalg.svd(M,full_matrices=False); p=U[:,0]*torch.sqrt(S[0]); d=Vh[0]*torch.sqrt(S[0]); P[:,ci]=p; D[ci]=d.reshape(k,k)
    dw=nn.Conv2d(i,i,k,padding=k//2,groups=i,bias=False); pw=nn.Conv2d(i,o,1,bias=True)
    with torch.no_grad(): dw.weight.copy_(D[:,None]); pw.weight.copy_(P[:,:,None,None]); pw.bias.copy_(b)
    return DWPW(dw,pw)
def nstate(m): return sum(x.numel() for x in list(m.parameters())+list(m.buffers()) if x.is_floating_point())
def replace_span(base,start,end,cand):
    m=deepcopy(base); cc=deepcopy(cand)
    for p in cc.parameters(): p.requires_grad_(False)
    m.blocks[start]=cc
    for j in range(start+1,end+1): m.blocks[j]=r.IdentityBlock()
    return m

def fit_conv_fast(Z,Y,k,ridge=1e-5):
    n,ch,h,w=Z.shape; oc=Y.shape[1]
    U=F.unfold(Z,kernel_size=k,padding=k//2).transpose(1,2).reshape(-1,ch*k*k)
    A=torch.cat([U,torch.ones(U.shape[0],1,dtype=U.dtype)],1); B=Y.permute(0,2,3,1).reshape(-1,oc)
    ATA=A.T@A; ATB=A.T@B; I=torch.eye(ATA.shape[0],dtype=ATA.dtype); I[-1,-1]=0.0
    sol=torch.linalg.solve(ATA+ridge*I,ATB)
    c=nn.Conv2d(ch,oc,k,padding=k//2)
    with torch.no_grad(): c.weight.copy_(sol[:-1].T.reshape(oc,ch,k,k)); c.bias.copy_(sol[-1])
    return r.LinearCore(c)

def build_candidates(Z,Y):
    c1=fit_conv_fast(Z,Y,1); c3=fit_conv_fast(Z,Y,3); c5=fit_conv_fast(Z,Y,5)
    return [('channel_affine',fit_channel_affine(Z,Y)),('conv1',c1),('conv3',c3),('conv5',c5),('lowrank3_r2',lowrank_from_linear(c3,2)),('lowrank3_r4',lowrank_from_linear(c3,4)),('dwpw3',dwpw_from_linear(c3))]
def relmse(a,b): return float(((a-b).pow(2).mean()/(b.pow(2).mean()+1e-12)).item())
def set_shell_trainable(m,start):
    for p in m.parameters(): p.requires_grad_(True)
    for p in m.blocks[start].parameters(): p.requires_grad_(False)
def repair_dataset(data,seed):
    Xtr,ytr,_,_=data; Xa=r.make_aug(Xtr,seed=seed+5000,noise=.08); return torch.cat([Xtr,Xa]),torch.cat([ytr,ytr])
def train_one_epoch(m,opt,X,y,seed,epoch):
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,y),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed*10000+epoch))
    m.train()
    for xb,yb in dl: opt.zero_grad(set_to_none=True); F.cross_entropy(m(xb),yb).backward(); opt.step()
def region_tensors(m,start,end):
    d={'pre_shell':[],'compiled_span':[],'post_shell':[]}
    for name,t in list(m.named_parameters())+list(m.named_buffers()):
        if not t.is_floating_point(): continue
        if name.startswith('stem.') or name.startswith('b_in.'): reg='pre_shell'
        elif name.startswith('blocks.'):
            bi=int(name.split('.')[1]); reg='pre_shell' if bi<start else ('compiled_span' if bi<=end else 'post_shell')
        else: reg='post_shell'
        d[reg].append(t)
    return d
def ledger(m,start,end,full=True):
    out={}; regs=region_tensors(m,start,end)
    for reg,ts in regs.items():
        out[reg+'_fixed32_bits']=sum(t.numel()*32 for t in ts)
        if full:
            out[reg+'_q8_entropy_bits']=sum(r.entropy_bits_for_tensor(t)[0] for t in ts)
            out[reg+'_rank99_bits']=sum(r.rank99_bits_for_tensor(t)[0] for t in ts)
    out['total_fixed32_bits']=sum(out[z+'_fixed32_bits'] for z in regs)
    if full:
        out['total_q8_entropy_bits']=sum(out[z+'_q8_entropy_bits'] for z in regs); out['total_rank99_bits']=sum(out[z+'_rank99_bits'] for z in regs)
    return out
def evaluate(m,start,end,cname,C,tau,control_clean,control_aug,Xv,yv,Xa,fit):
    clean=r.acc(m,Xv,yv); aug=r.acc(m,Xa,yv); U=aug/(control_aug+1e-12); viable=U>=FLOOR-1e-12
    return dict(candidate=cname,C=int(C),tau=tau,clean=clean,aug=aug,utility=U,viable=viable,fit_rel_mse=fit,control_clean=control_clean,control_aug=control_aug,total_fixed32_bits=int(nstate(m)*32))

def run_seed(seed):
    sd=os.path.join(OUT,f'seed_{seed}'); os.makedirs(sd,exist_ok=True); t0=time.time(); print('BASE',seed,flush=True)
    base,data=r.train_base(seed); Xtr,ytr,Xv,yv=data; Xa=r.make_aug(Xv,123); bc=r.acc(base,Xv,yv); ba=r.acc(base,Xa,yv)
    if bc<.95: raise RuntimeError(f'baseline ineligible {seed}: {bc}')
    torch.save(base.state_dict(),os.path.join(sd,'baseline_state.pt'))
    RX,RY=repair_dataset(data,seed)
    ctrls={0:deepcopy(base)}; cm={0:(bc,ba)}; c=deepcopy(base); opt=torch.optim.AdamW(c.parameters(),lr=7e-4,weight_decay=1e-4)
    for ep in range(1,9):
        train_one_epoch(c,opt,RX,RY,seed,ep)
        if ep in TAUS: ctrls[ep]=deepcopy(c); cm[ep]=(r.acc(c,Xv,yv),r.acc(c,Xa,yv))
    pd.DataFrame([dict(seed=seed,tau=t,control_clean=cm[t][0],control_aug=cm[t][1]) for t in TAUS]).to_csv(os.path.join(sd,'matched_controls.csv'),index=False)
    rows=[]; mins=[]
    for start in range(r.N_BLOCKS):
      for end in range(start,r.N_BLOCKS):
        with torch.no_grad(): Z=r.span_input(base,Xtr[:CALN],start); Y=r.span_output(base,Z,start,end); Zv=r.span_input(base,Xv,start); Yv=r.span_output(base,Zv,start,end)
        orig=sum(nstate(base.blocks[j]) for j in range(start,end+1)); local=[]
        for tau in TAUS:
            rr=evaluate(ctrls[tau],start,end,'original',orig,tau,*cm[tau],Xv,yv,Xa,0.0); rr.update(seed=seed,start=start,end=end,width=end-start+1); rows.append(rr); local.append(rr)
        cand_list=build_candidates(Z,Y); order={n:i for i,n in enumerate(GRAMMAR)}
        cand_list=sorted(cand_list,key=lambda z:(nstate(z[1]),order[z[0]]))
        unresolved=set(TAUS)
        for cname,cand in cand_list:
            C=nstate(cand)
            if C>=orig: continue
            if not unresolved: break
            with torch.no_grad(): fit=relmse(cand(Zv),Yv)
            m=replace_span(base,start,end,cand); set_shell_trainable(m,start); o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=7e-4,weight_decay=1e-4)
            prev=0; max_tau=max(unresolved)
            for tau in TAUS:
                if tau>max_tau: break
                for ep in range(prev+1,tau+1): train_one_epoch(m,o,RX,RY,seed,ep)
                if tau in unresolved:
                    rr=evaluate(m,start,end,cname,C,tau,*cm[tau],Xv,yv,Xa,fit); rr.update(seed=seed,start=start,end=end,width=end-start+1); rows.append(rr); local.append(rr)
                    if rr['viable']: unresolved.remove(tau)
                prev=tau
                if not unresolved: break
        L=pd.DataFrame(local)
        for tau in TAUS:
            g=L[(L.tau==tau)&L.viable].copy(); g['go']=g.candidate.map(order); g=g.sort_values(['C','go','utility'],ascending=[True,True,False]); b=g.iloc[0]; orow=L[(L.tau==tau)&(L.candidate=='original')].iloc[0]
            mins.append(dict(seed=seed,start=start,end=end,width=end-start+1,tau=tau,original_C=orig,C=int(b.C),complexity_ratio=float(b.C/orig),selected_candidate=b.candidate,selected_utility=float(b.utility),selected_aug=float(b.aug),selected_clean=float(b.clean),selected_fit=float(b.fit_rel_mse),selected_global_fixed32_gain=float(1-b.total_fixed32_bits/orow.total_fixed32_bits)))
        pd.DataFrame(rows).to_csv(os.path.join(sd,'candidate_evaluations_partial.csv'),index=False); pd.DataFrame(mins).to_csv(os.path.join(sd,'span_minima_partial.csv'),index=False)
        print('SPAN',seed,start,end,'elapsed',round(time.time()-t0,1),flush=True)
    R=pd.DataFrame(rows); S=pd.DataFrame(mins); R.to_csv(os.path.join(sd,'candidate_evaluations.csv'),index=False); S.to_csv(os.path.join(sd,'span_minima.csv'),index=False)
    ev=[]
    for tau in TAUS:
        s=S[S.tau==tau]; ix={(int(x.start),int(x.end)):x for _,x in s.iterrows()}
        for (a,b),p in ix.items():
            if b<=a: continue
            for sp in range(a,b):
                L=ix[(a,sp)]; Rr=ix[(sp+1,b)]; G=1-p.C/(L.C+Rr.C)
                ev.append(dict(event_id=f's{seed}_t{tau}_p{a}_{b}_sp{sp}',seed=seed,tau=tau,parent_start=a,parent_end=b,parent_width=b-a+1,split=sp,left_C=int(L.C),right_C=int(Rr.C),parent_C=int(p.C),simplification_gain=float(G),G_positive=bool(G>0),strong_G=bool(G>=.25 and p.selected_utility>=FLOOR),parent_candidate=p.selected_candidate,parent_utility=float(p.selected_utility),parent_global_fixed32_gain=float(p.selected_global_fixed32_gain)))
    E=pd.DataFrame(ev).sort_values('event_id'); E.to_csv(os.path.join(sd,'composition_events.csv'),index=False,float_format='%.12g')
    json.dump({'stage':'Stage1 only — no Canary measured','seed':seed,'baseline_clean':bc,'baseline_aug':ba,'architecture':'residual8 scale=.5 ch=8','taus':TAUS,'utility_floor':FLOOR,'grammar':GRAMMAR,'calibration_n':CALN,'elapsed_s':time.time()-t0},open(os.path.join(sd,'run_metadata.json'),'w'),indent=2)
    print('DONE',seed,'events',len(E),'elapsed',round(time.time()-t0,1),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',required=True,type=int); run_seed(ap.parse_args().seed)
