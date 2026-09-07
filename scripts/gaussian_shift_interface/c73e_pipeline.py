#!/usr/bin/env python3
"""C73E: paired calibration repair; no outcomes are synthesized or imputed."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / 'results/gaussian_shift_interface/c73e/PROTOCOL.json'
EXPERIMENT = 'C73E_FULL_PIPELINE_CALIBRATION_REPAIR'
SEEDS = list(range(71400,71416))
CELLS = [f'N{n}_{s}' for n in (192,384) for s in ('H64','S64','D64')]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def run(seed: int, verification: bool=False) -> dict:
    if (verification and seed not in (70300,71300)) or (not verification and seed not in SEEDS):
        raise ValueError('seed outside locked cohort/mode')
    import torch
    import torch.nn.functional as F
    import sklearn
    from scripts.gaussian_shift_interface.run_c61r_seed import (
        adapt_anchored,build_base_hierarchy,canonical_nested_qr,sha256_int_array,sha256_tensor)
    from scripts.gaussian_shift_interface.run_c64r_seed import correction_for_k
    from scripts.gaussian_shift_interface.run_c68e_seed import train_paired_teachers,state_dict_sha256
    from scripts.gaussian_shift_interface.run_c72e_seed import (
        nested_calibration_indices,standard_normal_like,shifted_input)
    from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
        split_data,acts,accuracy,FullSpanReplacedNet,TinyRes,fit_map,
        compile_final_from_hierarchy,nmse,count_params)
    torch.set_num_threads(1)
    xt,yt,xv,yv=split_data()
    assert len(xt)==1077 and len(xv)==270
    clean,teacher,training_prov=train_paired_teachers(seed,xt,yt)
    base,budget=build_base_hierarchy(seed,*acts(teacher,xt))
    assert budget['exact_4096_each_level'] is True
    bi,ei,li=nested_calibration_indices(len(xt))
    xb=xt[torch.tensor(bi,dtype=torch.long)]
    xe=xt[torch.tensor(ei,dtype=torch.long)]
    xb=shifted_input(xb,standard_normal_like(xb,seed+710100))
    xe=shifted_input(xe,standard_normal_like(xe,seed+720101))
    xl=torch.cat([xb,xe],dim=0)
    xs=shifted_input(xv,standard_normal_like(xv,seed+710200))
    assert torch.equal(xl[:192],xb)
    ab=acts(teacher,xb); al=acts(teacher,xl); av=acts(teacher,xs)
    assert torch.equal(al[0][:192],ab[0]) and torch.equal(al[-1][:192],ab[-1])
    a0v,a4v=av[0],av[-1]
    dv=float(((a4v-a4v.mean(0,keepdim=True))**2).mean())+1e-12
    clean_acc=accuracy(teacher,xv,yv)
    shift_acc=accuracy(teacher,xs,yv)
    hashes={
        'teacher':state_dict_sha256(teacher),
        'base_hierarchy':state_dict_sha256(base),
        'base_indices':sha256_int_array(bi),
        'large_indices':sha256_int_array(li),
        'base_calibration':sha256_tensor(xb),
        'large_calibration':sha256_tensor(xl),
        'validation':sha256_tensor(xs),
    }
    cells={}; invariant={}
    for n,ac in ((192,ab),(384,al)):
        a0c,a4c=ac[0],ac[-1]
        dc=float(((a4c-a4c.mean(0,keepdim=True))**2).mean())+1e-12
        with torch.no_grad():
            bc=base(a0c).detach(); rc=a4c-bc
            q=canonical_nested_qr(rc)
            assert tuple(q.shape)==(64,64)
            corr=correction_for_k(rc,q,64)
            target=(bc+corr).detach()
            err=float(((rc-corr)**2).sum()/((rc**2).sum()+1e-30))
        assert math.isfinite(err) and err<=1e-10
        invariant[f'N{n}_full_basis_relative_sqerr']=err
        h=copy.deepcopy(base)
        adapt_anchored(h,a0c,target,600,seed+715000)
        s,_=compile_final_from_hierarchy(copy.deepcopy(h),a0c,a0v,a4v,dv,seed+716000,seed+717000)
        d=fit_map(TinyRes(64,32,seed+716000),a0c,a4c.detach(),600,seed+717000)
        for name,m in (('H64',h),('S64',s),('D64',d)):
            assert count_params(m)==4096
            with torch.no_grad():
                pc=m(a0c).detach(); pv=m(a0v).detach()
                mc=float(F.mse_loss(pc,a4c)); mv=float(F.mse_loss(pv,a4v))
                logits=teacher.head(pv)
                correct=(logits.argmax(-1)==yv).to(torch.int64).tolist()
            acc=accuracy(FullSpanReplacedNet(teacher,copy.deepcopy(m)),xs,yv)
            assert abs(acc-sum(correct)/len(correct))<1e-7
            cells[f'N{n}_{name}']={
                'calibration_samples':n,'parameters':4096,
                'validation_accuracy':float(acc),
                'validation_correct':correct,
                'activation_nmse_vs_teacher':float(nmse(pv,a4v,dv)),
                'calibration_nmse_vs_teacher':float(nmse(pc,a4c)),
                'calibration_mse':mc,'validation_mse':mv,
                'calibration_target_variance':dc,'validation_target_variance':dv,
                'calibration_mse_over_validation_variance':mc/dv,
                'state_sha256':state_dict_sha256(m),
            }
    source_paths=[Path(__file__).resolve(),PROTOCOL,
        ROOT/'scripts/gaussian_shift_interface/run_c61r_seed.py',
        ROOT/'scripts/gaussian_shift_interface/run_c64r_seed.py',
        ROOT/'scripts/gaussian_shift_interface/run_c68e_seed.py',
        ROOT/'scripts/gaussian_shift_interface/run_c72e_seed.py',
        ROOT/'scripts/recursive_composition/exploration/c10_boundary_signal_ablation.py']
    out={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY',
        'status':'IMPLEMENTATION_VERIFICATION' if verification else 'FRESH_SEED_OUTCOME',
        'seed':seed,'eligible':True,'test_evaluated':False,
        'github_sha':os.environ.get('GITHUB_SHA'),
        'teacher_clean_accuracy':float(clean_acc),'teacher_shifted_accuracy':float(shift_acc),
        'cells':cells,'invariants':invariant,'hashes':hashes,'training_provenance':training_prov,
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
            'sklearn':sklearn.__version__,'platform':platform.platform(),'machine':platform.machine(),
            'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0] if Path('/proc/cpuinfo').exists() else platform.processor(),
            'torch_threads':torch.get_num_threads(),'torch_build':torch.__config__.show(),
            'benchmark':False,'mapping_updates':600,'mapping_batch_size':128,'draws_per_fit':76800}}
    json.dumps(out,allow_nan=False)
    return out


def summary(values,idx,margin=None):
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('nonfinite statistic')
    boot=x[idx].mean(axis=1)
    lo,hi=np.percentile(boot,[2.5,97.5])
    out={'mean_pp':float(x.mean()),'median_pp':float(np.median(x)),
         'bootstrap95_pp':[float(lo),float(hi)],'seed_sd_pp':float(x.std(ddof=1)),
         'bootstrap_standard_error_pp':float(boot.std(ddof=1))}
    if margin is not None:
        out.update(margin_pp=margin,status='PASS' if lo>margin else ('FAIL' if hi<margin else 'UNCERTAIN'))
    return out


def evaluate(rows: list[dict]) -> dict:
    result={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY',
        'confirmatory_claim_allowed':False,'reduced_dimension_selected':None,
        'expected_seeds':SEEDS,'bootstrap_resamples':100000,'bootstrap_seed':731384202}
    if len(rows)!=16 or any(type(r.get('seed')) is not int for r in rows) or sorted(r['seed'] for r in rows)!=SEEDS:
        return dict(result,decision='STOP_INTEGRITY_OR_INCOMPLETE_COHORT',reason='missing/duplicate/unexpected seeds')
    rows=sorted(rows,key=lambda r:r['seed'])
    for r in rows:
        if r.get('eligible') is not True or r.get('test_evaluated') is not False or set(r.get('cells',{}))!=set(CELLS):
            raise ValueError('invalid row integrity')
        for value in (r['teacher_clean_accuracy'],r['teacher_shifted_accuracy']):
            if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('invalid teacher accuracy')
        for name,c in r['cells'].items():
            a=c['validation_accuracy']
            if not math.isfinite(a) or not 0<=a<=1 or c['parameters']!=4096 or c['calibration_samples']!=int(name[1:4]):
                raise ValueError('invalid cell')
            for key in ('activation_nmse_vs_teacher','calibration_mse','validation_mse','calibration_target_variance','validation_target_variance'):
                if not math.isfinite(c[key]) or c[key]<0: raise ValueError('invalid metric')
            if c['calibration_target_variance']<=0 or c['validation_target_variance']<=0:
                raise ValueError('invalid denominator')
    rng=np.random.default_rng(731384202)
    idx=rng.integers(0,16,size=(100000,16))
    t=np.array([r['teacher_shifted_accuracy'] for r in rows])
    clean=np.array([r['teacher_clean_accuracy'] for r in rows])
    result['eligible_count']=16
    result['target_validity']=summary(100*(t-clean),idx,-20.)
    result['teacher_clean_accuracy_mean']=float(clean.mean())
    result['teacher_shifted_accuracy_mean']=float(t.mean())
    result['cells']={}
    acc={}
    for name in CELLS:
        acc[name]=np.array([r['cells'][name]['validation_accuracy'] for r in rows])
        rec=summary(100*(acc[name]-t),idx,-5.)
        for metric in ('activation_nmse_vs_teacher','calibration_mse','validation_mse','calibration_target_variance','validation_target_variance','calibration_mse_over_validation_variance'):
            vals=np.array([r['cells'][name][metric] for r in rows])
            rec[metric+'_mean']=float(vals.mean())
            rec[metric+'_median']=float(np.median(vals))
        rec['accuracy_mean']=float(acc[name].mean())
        result['cells'][name]=rec
    contrasts={}
    for stage in ('H64','S64','D64'):
        contrasts[f'N384_minus_N192_{stage}']=summary(100*(acc[f'N384_{stage}']-acc[f'N192_{stage}']),idx)
    for n in (192,384):
        for a,b in (('D64','S64'),('S64','H64')):
            contrasts[f'N{n}_{a}_minus_{b}']=summary(100*(acc[f'N{n}_{a}']-acc[f'N{n}_{b}']),idx)
    result['descriptive_contrasts']=contrasts
    ok=lambda n:result['cells'][n]['status']=='PASS'
    if result['target_validity']['status']!='PASS':
        decision='STOP_TARGET_VALIDITY_NOT_ESTABLISHED'
    elif not ok('N384_D64'):
        decision='STOP_DIRECT_REPAIR_NOT_ESTABLISHED'
    elif ok('N384_H64') and ok('N384_S64'):
        decision='FULL_PIPELINE_REFERENCE_REPAIR'
    elif ok('N384_S64'):
        decision='FINAL_VALID_INTERMEDIATE_NOT_ESTABLISHED'
    elif ok('N384_H64'):
        decision='COMPILER_REPAIR_NOT_ESTABLISHED'
    else:
        decision='HIERARCHY_REPAIR_NOT_ESTABLISHED'
    result['decision']=decision
    result['uncertainty_scope']='seed bootstrap conditional on fixed dataset/splits/calibration subsets; no dataset-level or external-validity interval'
    return result


def validate_full_record(r):
    if r['experiment']!=EXPERIMENT or r['status']!='FRESH_SEED_OUTCOME' or r['test_evaluated'] is not False:
        raise ValueError('wrong experiment/evidence/test status')
    if r['source_sha256'].get(str(PROTOCOL.relative_to(ROOT)))!=digest(PROTOCOL):
        raise ValueError('protocol hash mismatch')
    if r['source_sha256'].get(str(Path(__file__).resolve().relative_to(ROOT)))!=digest(__file__):
        raise ValueError('runner hash mismatch')
    if any(not math.isfinite(v) or v>1e-10 for v in r['invariants'].values()):
        raise ValueError('invalid full-basis reconstruction')
    for c in r['cells'].values():
        y=c['validation_correct']
        if len(y)!=270 or any(type(v) is not int or v not in (0,1) for v in y):
            raise ValueError('invalid correctness ledger')
        if abs(sum(y)/270-c['validation_accuracy'])>1e-7:
            raise ValueError('accuracy/count mismatch')
        if not math.isclose(c['validation_mse']/c['validation_target_variance'],c['activation_nmse_vs_teacher'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('NMSE denominator mismatch')


def selftest():
    idx=np.tile(np.arange(16),(100,1))
    assert summary(np.repeat(-4.,16),idx,-5.)['status']=='PASS'
    assert summary(np.repeat(-6.,16),idx,-5.)['status']=='FAIL'
    assert summary(np.repeat(-5.,16),idx,-5.)['status']=='UNCERTAIN'
    def row(seed):
        cell={'parameters':4096,'validation_accuracy':.83,'activation_nmse_vs_teacher':.1,
              'calibration_mse':.05,'validation_mse':.1,'calibration_target_variance':1.,
              'validation_target_variance':1.,'calibration_mse_over_validation_variance':.05}
        return {'seed':seed,'eligible':True,'test_evaluated':False,'teacher_clean_accuracy':.97,'teacher_shifted_accuracy':.86,
            'cells':{name:dict(cell,calibration_samples=int(name[1:4])) for name in CELLS}}
    rows=[row(seed) for seed in SEEDS]
    assert evaluate(rows)['decision']=='FULL_PIPELINE_REFERENCE_REPAIR'
    assert evaluate(rows[:-1])['decision']=='STOP_INTEGRITY_OR_INCOMPLETE_COHORT'
    assert evaluate(rows[:-1]+[rows[0]])['decision']=='STOP_INTEGRITY_OR_INCOMPLETE_COHORT'
    r=copy.deepcopy(rows);r[0]['cells']['N384_D64']['validation_accuracy']=float('nan')
    try: evaluate(r)
    except ValueError: pass
    else: raise AssertionError('nonfinite input accepted')
    for r in rows: r['cells']['N384_D64']['validation_accuracy']=.80
    assert evaluate(rows)['decision']=='STOP_DIRECT_REPAIR_NOT_ESTABLISHED'
    for r in rows:
        r['cells']['N384_D64']['validation_accuracy']=.83
        r['cells']['N384_H64']['validation_accuracy']=.79
        r['cells']['N384_S64']['validation_accuracy']=.79
    assert evaluate(rows)['decision']=='HIERARCHY_REPAIR_NOT_ESTABLISHED'
    print('C73E SYNTHETIC TESTS PASS')


def bridge_check(outdir):
    from scripts.gaussian_shift_interface.run_c72e_seed import run as old_run
    old=old_run(70300,allow_verification_seed=True)
    new=run(70300,verification=True)
    assert old['provenance_hashes']['robust_teacher_final_state_sha256']==new['hashes']['teacher']
    for oldkey,newkey in [('shifted_base_calibration_tensor_sha256','base_calibration'),('shifted_large_calibration_tensor_sha256','large_calibration'),('shifted_validation_tensor_sha256','validation')]:
        assert old['provenance_hashes'][oldkey]==new['hashes'][newkey]
    for n in (192,384):
        a=old['cells'][f'N{n}_W32'];b=new['cells'][f'N{n}_D64']
        for k in ('validation_accuracy','activation_nmse_vs_teacher','calibration_nmse_vs_teacher'):
            assert a[k]==b[k],(n,k,a[k],b[k])
        assert a['final_state_sha256']==b['state_sha256']
    write_json(outdir/'bridge_70300.json',{'status':'EXACT_BRIDGE_PASS','seed':70300,'teacher_and_input_hashes_equal':True,'direct_metrics_and_state_hashes_equal':True})
    write_json(outdir/'verification_71300.json',run(71300,verification=True))
    print('C73E PREFLIGHT PASS; no fresh seeds consumed')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['preflight','seed','aggregate','selftest'],required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--input',type=Path)
    args=p.parse_args()
    if args.mode=='selftest': selftest()
    elif args.mode=='preflight':
        selftest();bridge_check(args.out)
    elif args.mode=='seed': write_json(args.out,run(args.seed))
    else:
        if args.input is None: raise ValueError('--input required')
        rows=[json.loads(f.read_text()) for f in sorted(args.input.glob('seed_*.json'))]
        for r in rows: validate_full_record(r)
        decision=evaluate(rows)
        write_json(args.out/'FRESH_ROWS.json',{'rows':rows})
        write_json(args.out/'DECISION.json',decision)
        print(json.dumps(decision,indent=2))

if __name__=='__main__': main()
