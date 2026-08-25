import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class Block(nn.Module):
    def __init__(self,d=32,heads=4,mlp=64):
        super().__init__(); self.d=d; self.mlpw=mlp
        self.n1=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.0)
        self.n2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,mlp),nn.GELU(),nn.Linear(mlp,d))
    def forward(self,x):
        z=self.n1(x); a,_=self.attn(z,z,z,need_weights=False); x=x+a
        return x+self.mlp(self.n2(x))

class SmallViT(nn.Module):
    def __init__(self,depth=4,d=32,heads=4,mlp=64):
        super().__init__(); self.d=d
        self.patch=nn.Conv2d(1,d,2,2)
        self.cls=nn.Parameter(torch.zeros(1,1,d)); nn.init.normal_(self.cls,std=.02)
        self.pos=nn.Parameter(torch.zeros(1,17,d)); nn.init.normal_(self.pos,std=.02)
        self.blocks=nn.ModuleList([Block(d,heads,mlp) for _ in range(depth)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,10)
    def embed(self,x):
        x=self.patch(x).flatten(2).transpose(1,2)
        c=self.cls.expand(x.size(0),-1,-1)
        return torch.cat([c,x],1)+self.pos
    def forward(self,x):
        h=self.embed(x)
        for b in self.blocks: h=b(h)
        return self.head(self.norm(h[:,0]))

class Identity(nn.Module):
    def forward(self,x): return x

def data_split():
    d=load_digits(); X=(d.images.astype(np.float32)/16.0)[:,None,:,:]; y=d.target.astype(np.int64)
    idx=np.arange(len(y))
    tr,temp=train_test_split(idx,test_size=.30,random_state=24680,stratify=y)
    va,te=train_test_split(temp,test_size=.50,random_state=13579,stratify=y[temp])
    def ds(ii): return torch.utils.data.TensorDataset(torch.from_numpy(X[ii]),torch.from_numpy(y[ii]))
    return ds(tr),ds(va),ds(te)

def train_cls(m,ds,epochs=45,seed=0):
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
    lossf=nn.CrossEntropyLoss()
    for ep in range(epochs):
        g=torch.Generator().manual_seed(seed+ep)
        dl=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True,generator=g)
        m.train()
        for x,y in dl:
            opt.zero_grad(); loss=lossf(m(x),y); loss.backward(); opt.step()
    return m

@torch.no_grad()
def accuracy(m,ds):
    m.eval(); c=n=0
    for x,y in torch.utils.data.DataLoader(ds,batch_size=256,shuffle=False):
        c += int((m(x).argmax(1)==y).sum()); n += len(y)
    return c/n

@torch.no_grad()
def collect_span(m,ds,start=1,nfit=512,nhold=256):
    m.eval(); ins=[]; mids=[]; outs=[]
    for x,_ in torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False):
        h=m.embed(x)
        for j in range(start): h=m.blocks[j](h)
        hi=h; hm=m.blocks[start](hi); ho=m.blocks[start+1](hm)
        ins.append(hi); mids.append(hm); outs.append(ho)
    I=torch.cat(ins); M=torch.cat(mids); O=torch.cat(outs)
    need=nfit+nhold
    if len(I)<need: raise RuntimeError('not enough train examples')
    return (I[:nfit],M[:nfit],O[:nfit]),(I[nfit:need],M[nfit:need],O[nfit:need])

def fit_map(X,Y,width,epochs,seed):
    set_seed(seed)
    c=Block(d=32,heads=4,mlp=width)
    opt=torch.optim.AdamW(c.parameters(),lr=3e-3,weight_decay=1e-5)
    ds=torch.utils.data.TensorDataset(X,Y); updates=0
    for ep in range(epochs):
        g=torch.Generator().manual_seed(seed+1000+ep)
        dl=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,generator=g)
        c.train()
        for a,b in dl:
            opt.zero_grad(); p=c(a); loss=((p-b)**2).mean(); loss.backward(); opt.step(); updates+=1
    return c,updates

def count_params(m): return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def nmse(pred,target):
    mse=float(((pred-target)**2).mean())
    den=float(((target-target.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
    return mse/den,mse,den

def eval_candidate(base,mods,start,val,test):
    m=copy.deepcopy(base)
    m.blocks[start]=copy.deepcopy(mods[0]); m.blocks[start+1]=copy.deepcopy(mods[1])
    return accuracy(m,val),accuracy(m,test)

def run(seed,out,epochs=45,fit_epochs=40,widths=(8,16,32,64),start=1):
    t=time.time(); set_seed(seed)
    tr,va,te=data_split(); base=SmallViT(); train_cls(base,tr,epochs=epochs,seed=seed+50000)
    ba_v=accuracy(base,va); ba_t=accuracy(base,te)
    rec={'seed':seed,'baseline_val_acc':ba_v,'baseline_test_acc':ba_t,'baseline_params':count_params(base),'eligible':ba_v>=.95,'span':[start,start+1],'widths':list(widths),'fit_epochs':fit_epochs}
    Path(out).mkdir(parents=True,exist_ok=True)
    if not rec['eligible']:
        (Path(out)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2)); return rec
    fit,hold=collect_span(base,tr,start=start); X,M,Y=fit; Xh,Mh,Yh=hold
    comp0={}; comp1={}; updates={}
    for w in widths:
        c0,u0=fit_map(X,M,w,fit_epochs,seed+100000+w); c1,u1=fit_map(M,Y,w,fit_epochs,seed+200000+w)
        comp0[w]=c0; comp1[w]=c1; updates[f'c0_{w}']=u0; updates[f'c1_{w}']=u1
    composed={}
    for w in widths:
        c,u=fit_map(X,Y,w,fit_epochs,seed+300000+w); composed[w]=c; updates[f'whole_{w}']=u
    opts=[('id',Identity(),0)]+[(str(w),comp0[w],count_params(comp0[w])) for w in widths]
    opts1=[('id',Identity(),0)]+[(str(w),comp1[w],count_params(comp1[w])) for w in widths]
    component=[]
    for n0,m0,p0 in opts:
        for n1,m1,p1 in opts1:
            with torch.no_grad(): pred=m1(m0(Xh))
            e,raw,_=nmse(pred,Yh); vaa,tea=eval_candidate(base,[m0,m1],start,va,te)
            component.append({'choice':[n0,n1],'replacement_params':p0+p1,'hold_nmse':e,'hold_mse':raw,'val_acc':vaa,'test_acc':tea,'val_utility':vaa/ba_v,'test_utility':tea/ba_t,'compiler_updates':(0 if n0=='id' else updates[f'c0_{n0}'])+(0 if n1=='id' else updates[f'c1_{n1}'])})
    whole=[]
    for name,m,p in [('id',Identity(),0)]+[(str(w),composed[w],count_params(composed[w])) for w in widths]:
        with torch.no_grad(): pred=m(Xh)
        e,raw,_=nmse(pred,Yh); vaa,tea=eval_candidate(base,[m,Identity()],start,va,te)
        whole.append({'choice':name,'replacement_params':p,'hold_nmse':e,'hold_mse':raw,'val_acc':vaa,'test_acc':tea,'val_utility':vaa/ba_v,'test_utility':tea/ba_t,'compiler_updates':0 if name=='id' else updates[f'whole_{name}']})
    rec['componentwise']=component; rec['composed']=whole; rec['runtime_sec']=time.time()-t
    (Path(out)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2)); return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=45); ap.add_argument('--fit-epochs',type=int,default=40)
    a=ap.parse_args(); r=run(a.seed,a.out,a.epochs,a.fit_epochs)
    print(json.dumps({'seed':a.seed,'eligible':r['eligible'],'base_val':r['baseline_val_acc'],'base_test':r['baseline_test_acc'],'runtime_sec':r.get('runtime_sec')},indent=2))
