#!/usr/bin/env python3
"""Read-only audit of preserved C75E/C76R/C77E. No model training."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b,diffs):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k],diffs)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y,diffs)
    elif isinstance(a,float):
        assert math.isfinite(a) and math.isfinite(b)
        diffs.append(abs(a-b));assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
    else:assert type(a) is type(b) and a==b

def check(experiment):
    from scripts.gaussian_shift_interface import c75e_geometry,c76r_head_rank,c77e_absolute
    module={'c75e':c75e_geometry,'c76r':c76r_head_rank,'c77e':c77e_absolute}[experiment]
    base=ROOT/'results/gaussian_shift_interface'/experiment
    rows=json.loads((base/'FRESH_ROWS.json').read_text())['rows'];stored=json.loads((base/'DECISION.json').read_text())
    computed=module.evaluate(rows);diffs=[];same(stored,computed,diffs)
    manifest=json.loads((base/'RUN_MANIFEST.json').read_text())
    for p,h in manifest['sha256'].items():assert sha(base/p)==h,p
    model_count=0
    for r in rows:
        module.validate(r)
        assert set(r['source_sha256'])==set(module.SOURCES)
        assert r['github_sha']==manifest['source_commit']
        assert json.loads((base/f"raw/seed_{r['seed']}.json").read_text())==r
        if experiment=='c75e':
            p=base/f"raw/geometry_{r['seed']}.npz";assert sha(p)==r['geometry_sha256']
            with np.load(p,allow_pickle=False) as z:
                for n,c in r['cells'].items():
                    assert (z[n+'_final_logits'].argmax(1)==z['validation_labels']).astype(int).tolist()==c['validation_correct']
                w=z['centered_head'];_,s,v=np.linalg.svd(w,full_matrices=False);rank=int((s>s[0]*1e-10).sum());assert rank==9
                q=v[:rank].T;rv=z['validation_residual'].astype(float)
                assert np.linalg.norm((rv-rv@q@q.T)@w.T)/np.linalg.norm(rv@w.T)<1e-10
        else:
            for c in r['cells'].values():
                logits=np.array(c['validation_logits']);y=np.array(r['validation_labels'])
                assert (logits.argmax(1)==y).astype(int).tolist()==c['validation_correct']
        model_count+=len(r['cells'])
    assert module.evaluate(rows[:-1])['decision'].startswith('STOP_INTEGRITY')
    assert module.evaluate(rows[:-1]+[rows[0]])['decision'].startswith('STOP_INTEGRITY')
    bad=copy.deepcopy(rows);name=next(iter(bad[0]['cells']));bad[0]['cells'][name]['validation_correct'][0]^=1
    try:module.evaluate(bad)
    except (ValueError,AssertionError):pass
    else:raise AssertionError('corrupt correctness record accepted')
    if experiment=='c76r':
        ordered=sorted(rows,key=lambda r:r['seed']);idx=np.random.default_rng(761384202).integers(0,16,(100000,16))
        x=np.array([r['cells']['head_rank9']['validation_accuracy']-r['cells']['p64']['validation_accuracy'] for r in ordered])*100
        y=np.array([math.log(r['cells']['head_rank9']['nmse']/r['cells']['p64']['nmse']) for r in ordered])
        np.testing.assert_allclose(np.percentile(np.take(x,idx).mean(1),[2.5,97.5]),stored['utility']['ci95'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.exp(np.percentile(np.take(y,idx).mean(1),[2.5,97.5])),stored['hidden_nmse_ratio']['ci95'],rtol=1e-12,atol=1e-12)
        assert abs(module.binomial_interval(16,16)[0]-.025**(1/16))<1e-12
    if experiment=='c77e':
        ordered=sorted(rows,key=lambda r:r['seed']);idx=np.random.default_rng(771384202).integers(0,16,(100000,16))
        x=np.array([r['cells']['N512_head_rank9']['validation_accuracy']-r['cells']['N384_head_rank9']['validation_accuracy'] for r in ordered])*100
        np.testing.assert_allclose(np.percentile(np.take(x,idx).mean(1),[2.5,97.5]),stored['gains']['head_rank9']['ci95'],rtol=1e-12,atol=1e-12)
        b=np.sort(np.random.default_rng(20260903).choice(1077,192,replace=False))
        e=np.sort(np.random.default_rng(20260906).choice(np.setdiff1d(np.arange(1077),b),192,replace=False))
        extra=np.sort(np.random.default_rng(20260908).choice(np.setdiff1d(np.arange(1077),np.r_[b,e]),128,replace=False))
        assert all(r['nesting']['extra_indices']==extra.tolist() for r in rows)
    return {'status':'PASS','experiment':experiment,'decision':stored['decision'],'records':len(rows),'final_models':model_count,
            'manifest_hashes_checked':len(manifest['sha256']),'exact_json_equal':computed==stored,'max_float_difference':max(diffs,default=0)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);a=ap.parse_args()
    report={'status':'PASS','numpy':np.__version__,'python':sys.version,'new_model_training':False,'independent_scientific_review':False,
            'scope':'source/raw/manifest/hash/logit integrity; separate numeric oracle and primary-statistic calculations; fixed-cohort negative integrity tests',
            'experiments':[check(e) for e in ['c75e','c76r','c77e']]}
    text=json.dumps(report,indent=2)+'\n'
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(text)
