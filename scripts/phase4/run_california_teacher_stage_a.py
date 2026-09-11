#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,platform
from pathlib import Path

os.environ.setdefault('ATEN_CPU_CAPABILITY','avx2')
os.environ.setdefault('MKL_CBWR','COMPATIBLE')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_DYNAMIC','FALSE')
os.environ.setdefault('OMP_DYNAMIC','FALSE')

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'results/phase4/california_regression/STAGE_A_PROTOCOL.json'
BASE=ROOT/'scripts/phase3/regression_external_validity/run_seed.py'

def sha256_bytes(x):return hashlib.sha256(x).hexdigest()
def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_base():
    s=importlib.util.spec_from_file_location('phase3base',BASE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def install_sqrt64():
    original_step=torch.optim.AdamW.step;original_sqrt=torch.Tensor.sqrt;counts={'adamw_steps':0,'float32_sqrt_calls':0}
    def stable_sqrt(t):
        if t.device.type=='cpu' and t.dtype==torch.float32:
            counts['float32_sqrt_calls']+=1;return original_sqrt(t.to(torch.float64)).to(torch.float32)
        return original_sqrt(t)
    def step(self,*a,**kw):
        if torch.Tensor.sqrt is not original_sqrt:raise RuntimeError('concurrent sqrt override')
        counts['adamw_steps']+=1;torch.Tensor.sqrt=stable_sqrt
        try:return original_step(self,*a,**kw)
        finally:torch.Tensor.sqrt=original_sqrt
    torch.optim.AdamW.step=step
    return counts,lambda:(setattr(torch.optim.AdamW,'step',original_step),setattr(torch.Tensor,'sqrt',original_sqrt))

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise RuntimeError('ATen AVX2 profile inactive')

def train(base,seed,Xt,yt,recipe):
    model=base.Net(seed,input_dim=Xt.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=float(recipe['lr']),weight_decay=float(recipe['weight_decay']))
    gen=torch.Generator().manual_seed(seed+999);batch=int(recipe['batch_size'])
    for _ in range(int(recipe['epochs'])):
        perm=torch.randperm(len(Xt),generator=gen)
        for i in range(0,len(Xt),batch):
            ix=perm[i:i+batch];opt.zero_grad();loss=F.mse_loss(model(Xt[ix]),yt[ix]);loss.backward();opt.step()
    return model

def metrics(model,X,y):
    with torch.no_grad():
        p=model(X);mse=float(F.mse_loss(p,y));sse=float(((p-y)**2).sum());sst=float(((y-y.mean())**2).sum())+1e-12
        return mse,1.0-sse/sst

def main(out):
    protocol=json.loads(PROTOCOL.read_text());assert protocol['status']=='LOCKED_BEFORE_STAGE_A_OUTCOMES';configure();base=load_base();counts,restore=install_sqrt64()
    X,y=fetch_california_housing(return_X_y=True);X=np.asarray(X,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
    if X.shape!=(20640,8) or y.shape!=(20640,):raise RuntimeError(f'unexpected dataset shape {X.shape} {y.shape}')
    all_idx=np.arange(len(y));train_pool,test_idx=train_test_split(all_idx,test_size=.20,random_state=3119757572);train_idx,val_idx=train_test_split(train_pool,test_size=.20,random_state=2895583027)
    xm=X[train_idx].mean(0,keepdims=True);xs=X[train_idx].std(0,keepdims=True);xs[xs<1e-8]=1.;ym=float(y[train_idx].mean());ys=float(y[train_idx].std());ys=1. if ys<1e-8 else ys
    Xt=torch.tensor((X[train_idx]-xm)/xs,dtype=torch.float32);yt=torch.tensor((y[train_idx]-ym)/ys,dtype=torch.float32)
    Xv=torch.tensor((X[val_idx]-xm)/xs,dtype=torch.float32);yv=torch.tensor((y[val_idx]-ym)/ys,dtype=torch.float32)
    rows=[]
    try:
        for ri,recipe in enumerate(protocol['teacher_recipes']):
            for seed in protocol['teacher_initialization_seeds']:
                m=train(base,seed,Xt,yt,recipe);tm,tr=metrics(m,Xt,yt);vm,vr=metrics(m,Xv,yv)
                rows.append({'recipe_index':ri,'recipe_id':recipe['id'],'seed':seed,'train_mse':tm,'train_r2':tr,'validation_mse':vm,'validation_r2':vr})
    finally:restore()
    if len(rows)!=21 or counts['float32_sqrt_calls']<=0:raise RuntimeError('incomplete calibration/profile')
    ag=[]
    for ri,recipe in enumerate(protocol['teacher_recipes']):
        z=[r for r in rows if r['recipe_id']==recipe['id']];v=np.asarray([r['validation_r2'] for r in z],dtype=np.float64)
        ag.append({'recipe_index':ri,'recipe':recipe,'mean_validation_r2':float(v.mean()),'std_validation_r2':float(v.std(ddof=0)),'min_validation_r2':float(v.min()),'max_validation_r2':float(v.max()),
                   'eligible':bool(v.mean()>=.65 and v.min()>=.60)})
    eligible=[x for x in ag if x['eligible']];eligible.sort(key=lambda x:(-x['min_validation_r2'],-x['mean_validation_r2'],x['std_validation_r2'],x['recipe_index']));winner=eligible[0] if eligible else None
    result={'experiment':'PHASE4_CALIFORNIA_TEACHER_STAGE_A','evidence_class':'PROSPECTIVE_EXPLORATORY_TEACHER_CALIBRATION','stage_a_status':'PASS_SELECT_RECIPE' if winner else 'STOP_NO_COMPETENT_STABLE_TEACHER',
            'test_evaluated':False,'replacement_fitting_performed':False,'dataset':{'shape':list(X.shape),'target_shape':list(y.shape),'X_float32_sha256':sha256_bytes(X.tobytes()),'y_float32_sha256':sha256_bytes(y.tobytes()),'sklearn':sklearn.__version__},
            'split':{'n_train':len(train_idx),'n_val':len(val_idx),'n_test_held_out':len(test_idx),'train_indices_sha256':sha256_bytes(np.asarray(sorted(train_idx),dtype=np.int64).tobytes()),'validation_indices_sha256':sha256_bytes(np.asarray(sorted(val_idx),dtype=np.int64).tobytes()),'test_indices_sha256':sha256_bytes(np.asarray(sorted(test_idx),dtype=np.int64).tobytes())},
            'protocol_sha256':sha256(PROTOCOL),'base_runner_git_blob':'8dd3ddefb59ba250f5deca2d98d04fbff0055748','rows':rows,'recipe_aggregates':ag,'chosen_recipe':None if winner is None else winner['recipe'],
            'chosen_recipe_mean_validation_r2':None if winner is None else winner['mean_validation_r2'],'chosen_recipe_min_validation_r2':None if winner is None else winner['min_validation_r2'],
            'numeric_profile':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'effective_cpu_capability':torch.backends.cpu.get_cpu_capability(),'MKL_CBWR':os.environ.get('MKL_CBWR'),'threads':torch.get_num_threads(),'mkldnn':torch.backends.mkldnn.enabled,'deterministic':torch.are_deterministic_algorithms_enabled(),'sqrt_intervention':counts}}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'stage_a_status':result['stage_a_status'],'chosen_recipe':result['chosen_recipe'],'mean_validation_r2':result['chosen_recipe_mean_validation_r2'],'min_validation_r2':result['chosen_recipe_min_validation_r2'],'dataset_X_sha256':result['dataset']['X_float32_sha256']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
