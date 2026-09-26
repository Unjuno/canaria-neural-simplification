from __future__ import annotations
import os
for k,v in {"ATEN_CPU_CAPABILITY":"avx2","MKL_CBWR":"COMPATIBLE","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}.items(): os.environ[k]=v
import argparse,copy,json,random,time,importlib.util
from pathlib import Path
import numpy as np, torch
from torch import nn
BASE=Path(__file__).resolve().parents[2]/"scripts/replication/vit_compositional.py"
sp=importlib.util.spec_from_file_location("vit",BASE); vit=importlib.util.module_from_spec(sp); sp.loader.exec_module(vit)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
def ss(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def ords(s,n): return [torch.randperm(n,generator=torch.Generator().manual_seed(s+400000+e)) for e in range(40)]
def fit(m,X,Y,O):
 m=copy.deepcopy(m).train();o=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=1e-5,foreach=False)
 for q in O:
  for ix in q.split(64): o.zero_grad(set_to_none=True);loss=(m(X[ix])-Y[ix]).square().mean();loss.backward();o.step()
 return m.eval()
@torch.no_grad()
def va(base,r,pair,ds):
 m=copy.deepcopy(base);m.blocks[1]=copy.deepcopy(r[0] if pair else r);m.blocks[2]=copy.deepcopy(r[1] if pair else vit.Identity());return vit.accuracy(m,ds)
def run(seed,out):
 P=json.loads(Path(__file__).with_name("PROTOCOL.json").read_text())
 if seed not in P["seeds"]: raise ValueError("seed")
 out=Path(out);out.mkdir(parents=True,exist_ok=False);t=time.time();ss(seed);tr,vds,te=vit.data_split();del te;base=vit.SmallViT();vit.train_cls(base,tr,epochs=45,seed=seed+50000);bva=vit.accuracy(base,vds)
 (X,M,Y),(Xh,Mh,Yh)=vit.collect_span(base,tr,start=1,nfit=512,nhold=256);O=ords(seed,len(X));rows=[]
 for z in P["pairs"]:
  pw,sw=z["pair_width"],z["single_width"];ss(seed+100000+pw);pair=nn.Sequential(vit.Block(mlp=pw),vit.Block(mlp=pw));pair=fit(pair,X,Y,O);ss(seed+300000+sw);single=vit.Block(mlp=sw);single=fit(single,X,Y,O)
  pp,sp=vit.count_params(pair),vit.count_params(single);mis=(sp-pp)/pp
  if abs(mis)>P["max_absolute_budget_mismatch_fraction"]: raise RuntimeError("budget mismatch")
  with torch.no_grad(): pe=vit.nmse(pair(Xh),Yh)[0];se=vit.nmse(single(Xh),Yh)[0]
  rows.append({"pair_width":pw,"single_width":sw,"pair_parameters":pp,"single_parameters":sp,"budget_mismatch_fraction":mis,"pair_nmse":pe,"single_nmse":se,"pair_val_acc":va(base,pair,True,vds),"single_val_acc":va(base,single,False,vds)})
 r={"seed":seed,"eligible":bva>=.95,"teacher_val_acc":bva,"comparisons":rows,"test_used":False,"elapsed_not_benchmark":time.time()-t};(out/"RESULT.json").write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);p.add_argument("--out",required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),indent=2))
