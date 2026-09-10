#!/usr/bin/env python3
"""R87D7: process-isolated numerical mitigation, not a production optimizer."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SEEDS=tuple(range(1200,1208))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def install_scoped_sqrt64(torch):
    original_step=torch.optim.AdamW.step;original_sqrt=torch.Tensor.sqrt;counts={'float32_sqrt_calls':0}
    def stable_sqrt(t):
        if t.dtype==torch.float32:
            counts['float32_sqrt_calls']+=1
            return original_sqrt(t.to(torch.float64)).to(torch.float32)
        return original_sqrt(t)
    def step(self,*a,**kw):
        if torch.Tensor.sqrt is not original_sqrt:raise RuntimeError('Unexpected concurrent Tensor.sqrt override')
        torch.Tensor.sqrt=stable_sqrt
        try:return original_step(self,*a,**kw)
        finally:torch.Tensor.sqrt=original_sqrt
    torch.optim.AdamW.step=step
    def restore():
        torch.optim.AdamW.step=original_step;torch.Tensor.sqrt=original_sqrt
    return counts,restore

def execute(seed,out,bridge=False):
    if seed not in SEEDS:raise ValueError('Locked known-cohort only')
    p=load('base_profile','scripts/reproduction_diagnostics/r87_profile.py');p.configure('portable_v1')
    import torch
    counts,restore=install_scoped_sqrt64(torch)
    try:p.execute(seed,'portable_v1',out,bridge)
    finally:restore()
    if counts['float32_sqrt_calls']==0:raise RuntimeError('Optimizer sqrt intervention did not execute')
    r=json.loads((out/'record.json').read_text());r.update(experiment='R87D7_OPTIMIZER_SQRT64_PROFILE',profile='portable_v2_sqrt64',sqrt_intervention=counts,sqrt_profile_source_sha256=sha(__file__))
    dump(out/'record.json',r)

def suite(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def one(seed,name,bridge=False):
        out=root/name;out.mkdir(exist_ok=True)
        cmd=[sys.executable,__file__,'--mode','seed','--seed',str(seed),'--out',str(out)]+(['--bridge'] if bridge else [])
        with (out/'process.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=360)
        dump(out/'EXIT.json',{'returncode':r.returncode,'argv':cmd})
        if r.returncode:raise RuntimeError('failed record: '+name)
        print('COMPLETE',name,flush=True)
    one(1202,'preflight',True)
    for s in SEEDS:one(s,f'seed_{s}')
    for s in (1202,1205):one(s,f'seed_{s}_repeat')
    p=load('profile_validator','scripts/reproduction_diagnostics/r87_profile.py');rows=[]
    for s in SEEDS:
        r,o=p.validate_record(root/f'seed_{s}');rows.append(o)
        if r['profile']!='portable_v2_sqrt64' or r['sqrt_intervention']['float32_sqrt_calls']==0:raise AssertionError('profile not applied')
    repeats=[]
    for s in (1202,1205):
        a,ao=p.validate_record(root/f'seed_{s}');b,bo=p.validate_record(root/f'seed_{s}_repeat')
        repeats.append({'seed':s,'outcome_exact':ao==bo,'traces_exact':a['trace']==b['trace']})
    summary={'experiment':'R87D7_OPTIMIZER_SQRT64_PROFILE','within_host_integrity':'PASS' if all(x['outcome_exact'] and x['traces_exact'] for x in repeats) else 'FAIL','seeds':list(SEEDS),'repeats':repeats,'rows':rows,'cross_host_status':'NOT_ASSESSED_IN_HOST_SUMMARY','historical_archive_recovered':False}
    dump(root/'SUMMARY.json',summary)
    dump(root/'FILES_SHA256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'})
    if summary['within_host_integrity']!='PASS':raise RuntimeError('Within-host repeat failure')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('seed','suite'),required=True);p.add_argument('--seed',type=int);p.add_argument('--out',type=Path,required=True);p.add_argument('--bridge',action='store_true');a=p.parse_args()
    if a.mode=='seed':execute(a.seed,a.out,a.bridge)
    else:suite(a.out)
