#!/usr/bin/env python3
"""Strict artifact/record/profile audit. Never changes a scientific outcome."""
from __future__ import annotations
import argparse,hashlib,importlib.util,itertools,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NEW=list(range(871200,871216))
GRID=[512,1024,1536,2048,3072,4096,6144]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def finite(x):
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite evidence')
    if isinstance(x,dict):
        for v in x.values():finite(v)
    if isinstance(x,list):
        for v in x:finite(v)
def check_record(p,source=False):
    r,o=read(p/'record.json'),read(p/'outcome.json');finite(o)
    if not (p/'trace.npz').is_file():raise ValueError('Missing numeric trace')
    if sha(p/'trace.npz')!=r['trace_npz_sha256'] or sha(p/'outcome.json')!=r['outcome_sha256']:raise ValueError('Record hash mismatch')
    if r['seed']!=o['seed']:raise ValueError('Seed mismatch')
    if [x['budget'] for x in o['grid']]!=GRID:raise ValueError('Changed grid')
    for side in ('sep','comp'):
        passing=[]
        for x in o['grid']:
            if x['budget']!=256*x['h'] or type(x[side+'_pass']) is not bool:raise ValueError('Budget or gate type')
            if not 0<=x[side+'_val_acc']<=1 or x[side+'_nmse']<0:raise ValueError('Metric domain')
            gate=x[side+'_nmse']<=.08 and x[side+'_val_acc']>=o['teacher_val_acc']-.02
            if x[side+'_pass']!=gate:raise ValueError('Candidate gate mismatch')
            if gate:passing.append(x)
        s=o['selected_'+side]
        if bool(passing)!=(s is not None):raise ValueError('Endpoint availability')
        if s:
            for k in ('h','budget','sep_nmse','comp_nmse','sep_val_acc','comp_val_acc','sep_pass','comp_pass'):
                if s[k]!=passing[0][k]:raise ValueError('Selected endpoint mismatch')
            if not 0<=s[side+'_test_acc']<=1:raise ValueError('Test domain')
    if o['selected_sep'] and o['selected_comp']:
        if math.log2(o['selected_comp']['budget']/o['selected_sep']['budget'])!=o['log2_budget_ratio']:raise ValueError('Budget statistic')
        if o['selected_comp']['comp_test_acc']-o['selected_sep']['sep_test_acc']!=o['test_acc_diff_comp_minus_sep']:raise ValueError('Test statistic')
    if source:
        for f,h in r['source_hashes'].items():
            if sha(ROOT/f)!=h:raise ValueError('Producing source mismatch '+f)
    return r,o

def fingerprint(root,names):
    h=hashlib.sha256()
    for name in names:
        r,o=check_record(root/name)
        h.update(name.encode()+b'\0')
        h.update(json.dumps({'outcome':o,'trace':r['trace']},sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        with np.load(root/name/'trace.npz',allow_pickle=False) as z:
            for k in sorted(z.files):
                a=np.ascontiguousarray(z[k]);h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()

def audit(root):
    original=(ROOT/'scripts/reproduce/core_discovery_digits/run_confirmatory.py').read_bytes()
    if hashlib.sha1(b'blob '+str(len(original)).encode()+b'\0'+original).hexdigest()!='8759933ed2bb95014c3afc835acafa1743e40ea6':raise ValueError('Original source changed')
    names={'d5':[f'{s}_{p}' for s in range(1200,1208) for p in ('avx2_baseline','portable_v1')]+[f'{s}_{p}_repeat' for s in (1202,1205) for p in ('avx2_baseline','portable_v1')],
           'd7':[f'seed_{s}' for s in range(1200,1208)]+['seed_1202_repeat','seed_1205_repeat'],
           'r2':[f'seed_{s}' for s in NEW]}
    files=records=0;fps={}
    for exp in names:
        for host in ('a','b'):
            base=root/(exp+'_'+host);mf=read(base/'FILES_SHA256.json')
            for name,h in mf.items():
                p=(base/name).resolve()
                if not p.is_relative_to(base.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError('Manifest mismatch '+str(p))
                files+=1
            for name in names[exp]:check_record(base/name,source=exp=='r2');records+=1
            if not read(base/'preflight/record.json')['trace'].get('bridge_original_outcome_exact'):raise ValueError('Bridge missing')
            for name in [x for x in names[exp] if x.endswith('_repeat')]:
                a,ao=check_record(base/name);b,bo=check_record(base/name[:-7])
                if ao!=bo or a['trace']!=b['trace']:raise ValueError('Within-host repeat differs')
            fps[exp+'_'+host]=fingerprint(base,names[exp])
        if fps[exp+'_a']!=fps[exp+'_b']:raise ValueError('Remote hosts differ '+exp)
    spec=importlib.util.spec_from_file_location('locked_r2',ROOT/'scripts/reproduction_diagnostics/r87r2_confirm.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    outcomes=[check_record(root/'r2_a'/name,True)[1] for name in names['r2']]
    d=m.evaluate(outcomes)
    if d!=read(root/'r2_a/DECISION.json') or d!=read(root/'r2_b/DECISION.json'):raise ValueError('Decision recomputation mismatch')
    logs=np.array([math.log2(o['selected_comp']['budget']/o['selected_sep']['budget']) for o in outcomes]);delta=np.array([o['selected_comp']['comp_test_acc']-o['selected_sep']['sep_test_acc'] for o in outcomes]);ix=np.random.default_rng(87122026).integers(0,16,size=(100000,16))
    cl=np.quantile(logs[ix].sum(1)/16,[.025,.975]).tolist();ct=np.quantile(delta[ix].sum(1)/16,[.025,.975]).tolist()
    if cl!=d['primary_log2_budget_ratio']['ci95'] or ct!=d['secondary_test_accuracy_difference']['ci95']:raise ValueError('Independent bootstrap mismatch')
    if not(cl[1]<0 and ct[0]>-.02):raise ValueError('Scientific gates not PASS')
    m.selftest()
    receipt=read(ROOT/'results/reproduction/r87_resolution/CURRENT_INTEL_RECEIPT.json')
    if receipt['scientific_fingerprint_sha256']!=fps['r2_a']:raise ValueError('Preserved local scientific fingerprint mismatch')
    return {'status':'PASS','remote_manifest_files_checked':files,'remote_primary_repeat_records_checked':records,'scientific_fingerprints':fps,'current_intel_receipt_matches_remote_scientific_bytes':True,'current_intel_raw_arrays_location':'Companion session bundle; receipt comparison is not a new execution in CI','independent_new_model_seeds':16,'scientific_decision':d['decision'],'original_evaluator_exact':True,'separate_primary_bootstrap_exact':True,'original_source_unchanged':True,'historical_archive_recovered':False,'scope':'Data/code audit, not independent scientific peer review or a general hardware theorem'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.evidence.resolve());a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
