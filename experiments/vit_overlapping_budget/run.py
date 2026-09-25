from __future__ import annotations
import os
for k,v in {"ATEN_CPU_CAPABILITY":"avx2","MKL_CBWR":"COMPATIBLE","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}.items(): os.environ[k]=v
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np, torch
from torch import nn
import importlib.util
BASE=Path(__file__).resolve().parents[2]/"scripts/replication/vit_compositional.py"
spec=importlib.util.spec_from_file_location("vit",BASE); vit=importlib.util.module_from_spec(spec); spec.loader.exec_module(vit)
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled=False
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def orders(seed,n,epochs): return [torch.randperm(n,generator=torch.Generator().manual_seed(seed+400000+e)) for e in range(epochs)]
def fit(model,X,Y,ords):
    m=copy.deepcopy(model).train(); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=1e-5,foreach=False)
    for o in ords:
      for ix in o.split(64):
        opt.zero_grad(set_to_none=True); loss=(m(X[ix])-Y[ix]).square().mean(); loss.backward(); opt.step()
    return m.eval()
@torch.no_grad()
def valacc(base,repl,pair,va):
    m=copy.deepcopy(base); m.blocks[1]=copy.deepcopy(repl[0] if pair else repl); m.blocks[2]=copy.deepcopy(repl[1] if pair else vit.Identity()); return vit.accuracy(m,va)
def run(seed,out):
    P=json.loads(Path(__file__).with_name("PROTOCOL.json").read_text())
    if seed not in P["seeds"]: raise ValueError("seed outside locked plan")
    out=Path(out); out.mkdir(parents=True,exist_ok=False); started=time.time()
    set_seed(seed); tr,va,te=vit.data_split(); del te; base=vit.SmallViT(); vit.train_cls(base,tr,epochs=45,seed=seed+50000); bva=vit.accuracy(base,va)
    (X,M,Y),(Xh,Mh,Yh)=vit.collect_span(base,tr,start=1,nfit=512,nhold=256); ords=orders(seed,len(X),40)
    set_seed(seed+100008); pair=nn.Sequential(vit.Block(mlp=8),vit.Block(mlp=8)); pair=fit(pair,X,Y,ords)
    set_seed(seed+300084); direct=vit.Block(mlp=84); direct=fit(direct,X,Y,ords)
    pn,dn=vit.count_params(pair),vit.count_params(direct); rel=abs(dn-pn)/pn
    if pn!=9808 or dn!=9844 or rel>P["budget_relative_difference_max"]: raise RuntimeError("budget match invariant")
    with torch.no_grad(): pe=vit.nmse(pair(Xh),Yh)[0]; de=vit.nmse(direct(Xh),Yh)[0]
    r={"seed":seed,"eligible":bva>=.95,"teacher_val_acc":bva,
       "component_pair8":{"parameters":pn,"hold_nmse":pe,"val_acc":valacc(base,pair,True,va)},
       "direct_single84":{"parameters":dn,"hold_nmse":de,"val_acc":valacc(base,direct,False,va)},
       "budget_relative_difference":rel,"test_used":False,"elapsed_not_benchmark":time.time()-started}
    (out/"RESULT.json").write_text(json.dumps(r,indent=2)+"\n"); return r
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);p.add_argument("--out",required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),indent=2))
