#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,platform
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
PROTOCOL=ROOT/'results/phase4/california_regression/STAGE_B_CONFIRMATORY_PROTOCOL.json'
BASE=ROOT/'scripts/phase3/regression_external_validity/run_seed.py'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
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
    def restore():torch.optim.AdamW.step=original_step;torch.Tensor.sqrt=original_sqrt
    return counts,restore

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise RuntimeError('AVX2 profile inactive')

def data():
    X,y=fetch_california_housing(return_X_y=True);X=np.asarray(X,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
    if X.shape!=(20640,8) or y.shape!=(20640,):raise RuntimeError('unexpected dataset shape')
    if sha_bytes(X.tobytes())!='ef53061b11adc508abf7f3d57bbeb8a90302caedaa3201b846cdd5f65caddfd5':raise RuntimeError('X dataset hash mismatch')
    if sha_bytes(y.tobytes())!='4fcec16744c2835e99b638e9ac95cde0619da1673dfdf7e638d31cd83f32fd14':raise RuntimeError('y dataset hash mismatch')
    idx=np.arange(len(y));pool,te=train_test_split(idx,test_size=.20,random_state=3119757572);tr,va=train_test_split(pool,test_size=.20,random_state=2895583027)
    xm=X[tr].mean(0,keepdims=True);xs=X[tr].std(0,keepdims=True);xs[xs<1e-8]=1.;ym=float(y[tr].mean());ys=float(y[tr].std());ys=1. if ys<1e-8 else ys
    def tx(ii):return torch.tensor((X[ii]-xm)/xs,dtype=torch.float32),torch.tensor((y[ii]-ym)/ys,dtype=torch.float32)
    Xt,yt=tx(tr);Xv,yv=tx(va);Xte,yte=tx(te)
    meta={'n_total':len(y),'n_train':len(tr),'n_val':len(va),'n_test':len(te),'input_dim':8,'X_sha256':sha_bytes(X.tobytes()),'y_sha256':sha_bytes(y.tobytes()),
          'train_indices_sha256':sha_bytes(np.asarray(sorted(tr),dtype=np.int64).tobytes()),'validation_indices_sha256':sha_bytes(np.asarray(sorted(va),dtype=np.int64).tobytes()),'test_indices_sha256':sha_bytes(np.asarray(sorted(te),dtype=np.int64).tobytes())}
    return Xt,yt,Xv,yv,Xte,yte,meta

def train_teacher(base,seed,Xt,yt):
    model=base.Net(seed,input_dim=Xt.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-3);gen=torch.Generator().manual_seed(seed+999)
    for _ in range(30):
        perm=torch.randperm(len(Xt),generator=gen)
        for i in range(0,len(Xt),256):
            ix=perm[i:i+256];opt.zero_grad();loss=F.mse_loss(model(Xt[ix]),yt[ix]);loss.backward();opt.step()
    return model

def main(seed,out):
    p=json.loads(PROTOCOL.read_text());allowed=p['fresh_model_seeds']+[2899]
    if seed not in allowed:raise ValueError('seed outside locked fresh/preflight sets')
    configure();base=load_base();counts,restore=install_sqrt64()
    try:
        Xt,yt,Xv,yv,Xte,yte,meta=data();teacher=train_teacher(base,seed,Xt,yt);ta=base.acts(teacher,Xt);va=base.acts(teacher,Xv)
        k=0;target=va[k+2];denom=float(((target-target.mean(0,keepdim=True))**2).mean())+1e-12;teacher_val=base.r2(teacher,Xv,yv);teacher_val_mse=base.mse(teacher,Xv,yv)
        rows=[];selected_sep=None;selected_comp=None
        for h in (2,4,6,8,12,16,20,24,32):
            budget=256*h
            r1=base.fit_map(base.TinyRes(64,h,seed+410000+h),ta[k],ta[k+1],600,seed+420000+h)
            r2m=base.fit_map(base.TinyRes(64,h,seed+430000+h),ta[k+1],ta[k+2],600,seed+440000+h)
            comp=base.fit_map(base.TinyRes(64,2*h,seed+450000+h),ta[k],ta[k+2],600,seed+460000+h)
            sepnet=base.PairReplacedNet(teacher,k,r1=r1,r2=r2m);compnet=base.PairReplacedNet(teacher,k,comp=comp)
            with torch.no_grad():
                sn=float(F.mse_loss(r2m(r1(va[k])),target))/denom;cn=float(F.mse_loss(comp(va[k]),target))/denom
            svr=base.r2(sepnet,Xv,yv);cvr=base.r2(compnet,Xv,yv)
            row={'h':h,'budget':budget,'sep_nmse':sn,'comp_nmse':cn,'sep_val_r2':svr,'comp_val_r2':cvr,
                 'sep_pass':bool(sn<=.12 and svr>=teacher_val-.05),'comp_pass':bool(cn<=.12 and cvr>=teacher_val-.05)}
            if row['sep_pass'] and selected_sep is None:
                row['sep_test_r2']=base.r2(sepnet,Xte,yte);row['sep_test_mse']=base.mse(sepnet,Xte,yte);selected_sep=dict(row)
            if row['comp_pass'] and selected_comp is None:
                row['comp_test_r2']=base.r2(compnet,Xte,yte);row['comp_test_mse']=base.mse(compnet,Xte,yte);selected_comp=dict(row)
            rows.append(row)
        teacher_test=base.r2(teacher,Xte,yte);teacher_test_mse=base.mse(teacher,Xte,yte)
    finally:restore()
    result={'experiment':'PHASE4_CALIFORNIA_CONFIRMATORY','evidence_class':'PROSPECTIVE_CONFIRMATORY' if seed in p['fresh_model_seeds'] else 'PREFLIGHT_ONLY','seed':seed,'dataset':'sklearn California Housing','task':'tabular_regression','data':meta,
            'teacher_recipe':p['selected_teacher_recipe'],'teacher_val_r2':teacher_val,'teacher_val_mse':teacher_val_mse,'teacher_test_r2':teacher_test,'teacher_test_mse':teacher_test_mse,
            'span':[0,1],'internal_width':64,'map_updates':600,'budget_grid_h':[2,4,6,8,12,16,20,24,32],'learned_parameter_budget_formula':'256*h for both conditions',
            'selection_rule':{'span_nmse_lte':.12,'val_r2_gte_teacher_minus':.05,'test_used_for_selection':False},'grid':rows,'selected_sep':selected_sep,'selected_comp':selected_comp,
            'numeric_profile':{'effective_cpu_capability':torch.backends.cpu.get_cpu_capability(),'MKL_CBWR':os.environ.get('MKL_CBWR'),'threads':torch.get_num_threads(),'mkldnn':torch.backends.mkldnn.enabled,'deterministic':torch.are_deterministic_algorithms_enabled(),'sqrt_intervention':counts},
            'provenance':{'protocol_sha256':sha_file(PROTOCOL),'base_runner_git_blob':'8dd3ddefb59ba250f5deca2d98d04fbff0055748','wrapper_sha256':sha_file(Path(__file__)),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__}}
    if selected_sep is not None and selected_comp is not None:
        result['log2_budget_ratio']=math.log2(selected_comp['budget']/selected_sep['budget']);result['test_r2_diff_comp_minus_sep']=selected_comp['comp_test_r2']-selected_sep['sep_test_r2']
    else:result['log2_budget_ratio']=None;result['test_r2_diff_comp_minus_sep']=None
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':seed,'teacher_test_r2':teacher_test,'sep':None if selected_sep is None else selected_sep['budget'],'comp':None if selected_comp is None else selected_comp['budget'],'log2':result['log2_budget_ratio']}),flush=True)
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--seed',type=int,required=True);q.add_argument('--out',type=Path,required=True);a=q.parse_args();main(a.seed,a.out)
