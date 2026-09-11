#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, sys
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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'results/phase3c/nested_cv_teacher/PROTOCOL.json'
AMENDMENT=ROOT/'results/phase3c/nested_cv_teacher/PRE_OUTCOME_AMENDMENT.json'
STAGEA=ROOT/'scripts/phase3b/stronger_teacher_regression/explore_teacher.py'
BASE=ROOT/'scripts/phase3/regression_external_validity/run_seed.py'

def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git_blob_bytes(path):
    raw=Path(path).read_bytes();return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def install_scoped_sqrt64():
    original_step=torch.optim.AdamW.step;original_sqrt=torch.Tensor.sqrt;counts={'float32_sqrt_calls':0,'adamw_steps':0}
    def stable_sqrt(t):
        if t.device.type=='cpu' and t.dtype==torch.float32:
            counts['float32_sqrt_calls']+=1
            return original_sqrt(t.to(torch.float64)).to(torch.float32)
        return original_sqrt(t)
    def step(self,*a,**kw):
        if torch.Tensor.sqrt is not original_sqrt:raise RuntimeError('unexpected concurrent sqrt override')
        counts['adamw_steps']+=1;torch.Tensor.sqrt=stable_sqrt
        try:return original_step(self,*a,**kw)
        finally:torch.Tensor.sqrt=original_sqrt
    torch.optim.AdamW.step=step
    def restore():torch.optim.AdamW.step=original_step;torch.Tensor.sqrt=original_sqrt
    return counts,restore

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise RuntimeError('ATen AVX2 not active')

def normalize_fold(X,y,tr,va):
    xm=X[tr].mean(axis=0,keepdims=True);xs=X[tr].std(axis=0,keepdims=True);xs[xs<1e-8]=1.0
    ym=float(y[tr].mean());ys=float(y[tr].std());ys=1.0 if ys<1e-8 else ys
    return (torch.tensor((X[tr]-xm)/xs,dtype=torch.float32),torch.tensor((y[tr]-ym)/ys,dtype=torch.float32),
            torch.tensor((X[va]-xm)/xs,dtype=torch.float32),torch.tensor((y[va]-ym)/ys,dtype=torch.float32))

def train(base,seed,Xt,yt,input_dim,recipe):
    model=base.Net(seed,input_dim=input_dim);opt=torch.optim.AdamW(model.parameters(),lr=float(recipe['lr']),weight_decay=float(recipe['weight_decay']))
    gen=torch.Generator().manual_seed(seed+999);batch=len(Xt) if recipe['batch_size']=='full_train_fold' else int(recipe['batch_size'])
    for _ in range(int(recipe['epochs'])):
        perm=torch.randperm(len(Xt),generator=gen)
        for i in range(0,len(Xt),batch):
            ix=perm[i:i+batch];opt.zero_grad();loss=F.mse_loss(model(Xt[ix]),yt[ix]);loss.backward();opt.step()
    return model

def r2(model,X,y):
    with torch.no_grad():
        pred=model(X);sse=float(((pred-y)**2).sum());sst=float(((y-y.mean())**2).sum())+1e-12;return 1.0-sse/sst

def choose(aggregates):
    baseline=next(x for x in aggregates if x['recipe']['id']=='baseline_60');bm=baseline['mean_validation_r2']
    for x in aggregates:
        x['improvement_over_baseline_mean_r2']=float(x['mean_validation_r2']-bm)
        x['eligible']=bool(x['mean_validation_r2']>=0.35 and x['improvement_over_baseline_mean_r2']>=0.10)
    eligible=[x for x in aggregates if x['eligible']]
    eligible.sort(key=lambda x:(-x['min_fold_mean_validation_r2'],-x['mean_validation_r2'],x['std_validation_r2'],x['recipe_index']))
    return baseline, eligible[0] if eligible else None

def aggregate(rows,recipes,include_folds=None):
    rr=rows if include_folds is None else [r for r in rows if r['fold'] in include_folds];out=[]
    for i,recipe in enumerate(recipes):
        z=[r for r in rr if r['recipe_id']==recipe['id']];vals=np.asarray([r['validation_r2'] for r in z],dtype=np.float64)
        fold_means=[]
        for f in sorted(set(r['fold'] for r in z)):
            fv=[r['validation_r2'] for r in z if r['fold']==f];fold_means.append(float(np.mean(fv)))
        out.append({'recipe_index':i,'recipe':recipe,'n_values':len(z),'mean_validation_r2':float(vals.mean()),'std_validation_r2':float(vals.std(ddof=0)),
                    'min_validation_r2':float(vals.min()),'max_validation_r2':float(vals.max()),'fold_mean_validation_r2':fold_means,
                    'min_fold_mean_validation_r2':float(min(fold_means))})
    return out

def main(out):
    p=json.loads(PROTOCOL.read_text());a=json.loads(AMENDMENT.read_text());assert p['status']=='LOCKED_BEFORE_PHASE3C_OUTCOMES';assert a['status']=='PRE_OUTCOME_PROTOCOL_CORRECTION'
    assert a['outcomes_generated_before_this_amendment']==0 and git_blob_bytes(STAGEA)=='b6518c84b0057d2da77a02432024c58c07815587'
    recipes=a['effective_teacher_recipes'];configure();base=load_module('phase3base',BASE);counts,restore=install_scoped_sqrt64()
    X,y=load_diabetes(return_X_y=True);X=X.astype(np.float32);y=y.astype(np.float32);idx=np.arange(len(y))
    outer_train,outer_test=train_test_split(idx,test_size=.25,random_state=3240137098)
    kf=KFold(n_splits=5,shuffle=True,random_state=3502951522);rows=[]
    try:
        for fold,(tri,vai) in enumerate(kf.split(outer_train)):
            tr=outer_train[tri];va=outer_train[vai];Xt,yt,Xv,yv=normalize_fold(X,y,tr,va)
            for ri,recipe in enumerate(recipes):
                for seed in (2600,2601,2602):
                    model=train(base,seed,Xt,yt,X.shape[1],recipe);rows.append({'fold':fold,'recipe_index':ri,'recipe_id':recipe['id'],'seed':seed,'n_train':len(tr),'n_val':len(va),'validation_r2':r2(model,Xv,yv)})
    finally:restore()
    if len(rows)!=195 or counts['float32_sqrt_calls']<=0:raise RuntimeError('incomplete fixed exploration or profile not applied')
    ag=aggregate(rows,recipes);baseline,winner=choose(ag);loo=[]
    for omitted in range(5):
        x=aggregate(rows,recipes,[f for f in range(5) if f!=omitted]);b,w=choose(x);loo.append({'omitted_fold':omitted,'winner':None if w is None else w['recipe']['id'],'winner_min_fold_mean':None if w is None else w['min_fold_mean_validation_r2'],'baseline_mean':b['mean_validation_r2']})
    result={'experiment':'PHASE3C_NESTED_CV_TEACHER_SELECTION','evidence_class':'PROSPECTIVE_EXPLORATORY_TEACHER_SELECTION','stage_c_status':'PASS_SELECT_RECIPE' if winner else 'STOP_NO_STABLE_SAME_ARCHITECTURE_TEACHER',
            'outer_test_evaluated':False,'replacement_fitting_performed':False,'outer_train_n':len(outer_train),'outer_test_n_held_out':len(outer_test),
            'outer_train_indices_sha256':hashlib.sha256(np.asarray(sorted(outer_train),dtype=np.int64).tobytes()).hexdigest(),'outer_test_indices_sha256':hashlib.sha256(np.asarray(sorted(outer_test),dtype=np.int64).tobytes()).hexdigest(),
            'protocol_sha256':sha256(PROTOCOL),'amendment_sha256':sha256(AMENDMENT),'stagea_source_git_blob':git_blob_bytes(STAGEA),'base_runner_git_blob':git_blob_bytes(BASE),
            'numeric_profile':{'effective_cpu_capability':torch.backends.cpu.get_cpu_capability(),'MKL_CBWR':os.environ.get('MKL_CBWR'),'threads':torch.get_num_threads(),'mkldnn':torch.backends.mkldnn.enabled,'deterministic':torch.are_deterministic_algorithms_enabled(),'sqrt_intervention':counts,
                               'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__},
            'rows':rows,'recipe_aggregates':ag,'baseline_mean_validation_r2':baseline['mean_validation_r2'],'chosen_recipe':None if winner is None else winner['recipe'],
            'chosen_recipe_mean_validation_r2':None if winner is None else winner['mean_validation_r2'],'chosen_recipe_min_fold_mean_validation_r2':None if winner is None else winner['min_fold_mean_validation_r2'],
            'chosen_recipe_improvement_over_baseline':None if winner is None else winner['improvement_over_baseline_mean_r2'],'leave_one_fold_out_winners_descriptive':loo}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['stage_c_status','baseline_mean_validation_r2','chosen_recipe','chosen_recipe_mean_validation_r2','chosen_recipe_min_fold_mean_validation_r2','chosen_recipe_improvement_over_baseline']}),flush=True)
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--out',type=Path,required=True);x=q.parse_args();main(x.out)
