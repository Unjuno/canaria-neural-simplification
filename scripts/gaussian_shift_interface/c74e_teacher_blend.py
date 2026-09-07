#!/usr/bin/env python3
"""C74E tests one fixed target mixture, not a reduced teacher interface."""
from __future__ import annotations
import argparse,copy,json,math,os,platform,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.gaussian_shift_interface.c73e_pipeline import summary,digest,write_json
PROTOCOL=ROOT/'results/gaussian_shift_interface/c74e/PROTOCOL.json'
EXPERIMENT='C74E_FINAL_TEACHER_BLEND'
SEEDS=list(range(72400,72416))
CELLS=('H64','S0','B50','D64')


def run(seed:int,verification:bool=False)->dict:
    if (verification and seed not in (71300,72300)) or (not verification and seed not in SEEDS):
        raise ValueError('seed/mode outside locked cohort')
    import torch
    import torch.nn.functional as F
    import sklearn
    from scripts.gaussian_shift_interface.run_c61r_seed import adapt_anchored,build_base_hierarchy,canonical_nested_qr,sha256_tensor
    from scripts.gaussian_shift_interface.run_c64r_seed import correction_for_k
    from scripts.gaussian_shift_interface.run_c68e_seed import train_paired_teachers,state_dict_sha256
    from scripts.gaussian_shift_interface.run_c72e_seed import nested_calibration_indices,standard_normal_like,shifted_input
    from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import split_data,acts,accuracy,FullSpanReplacedNet,TinyRes,fit_map,nmse,count_params
    torch.set_num_threads(1)
    xt,yt,xv,yv=split_data()
    _,teacher,training_prov=train_paired_teachers(seed,xt,yt)
    base,budget=build_base_hierarchy(seed,*acts(teacher,xt))
    assert budget['exact_4096_each_level']
    bi,ei,_=nested_calibration_indices(len(xt))
    xb=xt[torch.tensor(bi,dtype=torch.long)];xe=xt[torch.tensor(ei,dtype=torch.long)]
    xb=shifted_input(xb,standard_normal_like(xb,seed+710100))
    xe=shifted_input(xe,standard_normal_like(xe,seed+720101))
    xl=torch.cat([xb,xe],dim=0)
    xs=shifted_input(xv,standard_normal_like(xv,seed+710200))
    ac=acts(teacher,xl);av=acts(teacher,xs)
    a0c,a4c=ac[0],ac[-1];a0v,a4v=av[0],av[-1]
    dc=float(((a4c-a4c.mean(0,keepdim=True))**2).mean())+1e-12
    dv=float(((a4v-a4v.mean(0,keepdim=True))**2).mean())+1e-12
    with torch.no_grad():
        bc=base(a0c).detach();rc=a4c-bc
        q=canonical_nested_qr(rc)
        assert tuple(q.shape)==(64,64)
        correction=correction_for_k(rc,q,64)
        target=(bc+correction).detach()
        err=float(((rc-correction)**2).sum()/((rc**2).sum()+1e-30))
    assert math.isfinite(err) and err<=1e-10
    h=copy.deepcopy(base)
    adapt_anchored(h,a0c,target,600,seed+715000)
    with torch.no_grad(): hc=h(a0c).detach()
    targets={'S0':hc,'B50':0.5*hc+0.5*a4c.detach(),'D64':a4c.detach()}
    models={'H64':h};initial={}
    for name,y in targets.items():
        m=TinyRes(64,32,seed+716000)
        initial[name]=state_dict_sha256(m)
        models[name]=fit_map(m,a0c,y,600,seed+717000)
    assert len(set(initial.values()))==1
    cells={}
    for name,m in models.items():
        assert count_params(m)==4096
        with torch.no_grad():
            pc=m(a0c).detach();pv=m(a0v).detach()
            correct=(teacher.head(pv).argmax(-1)==yv).to(torch.int64).tolist()
            mc=float(F.mse_loss(pc,a4c));mv=float(F.mse_loss(pv,a4v))
        acc=accuracy(FullSpanReplacedNet(teacher,copy.deepcopy(m)),xs,yv)
        assert abs(acc-sum(correct)/270)<1e-7
        cells[name]={'parameters':4096,'calibration_samples':384,
            'validation_accuracy':float(acc),'validation_correct':correct,
            'activation_nmse_vs_teacher':float(nmse(pv,a4v,dv)),
            'calibration_nmse_vs_teacher':float(nmse(pc,a4c)),
            'calibration_mse':mc,'validation_mse':mv,
            'calibration_target_variance':dc,'validation_target_variance':dv,
            'state_sha256':state_dict_sha256(m)}
    files=[Path(__file__).resolve(),PROTOCOL,
        ROOT/'scripts/gaussian_shift_interface/c73e_pipeline.py',
        ROOT/'scripts/gaussian_shift_interface/run_c61r_seed.py',
        ROOT/'scripts/gaussian_shift_interface/run_c64r_seed.py',
        ROOT/'scripts/gaussian_shift_interface/run_c68e_seed.py',
        ROOT/'scripts/gaussian_shift_interface/run_c72e_seed.py',
        ROOT/'scripts/recursive_composition/exploration/c10_boundary_signal_ablation.py']
    r={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY',
       'seed':seed,'status':'IMPLEMENTATION_VERIFICATION' if verification else 'FRESH_SEED_OUTCOME',
       'eligible':True,'test_evaluated':False,'github_sha':os.environ.get('GITHUB_SHA'),
       'teacher_clean_accuracy':accuracy(teacher,xv,yv),'teacher_shifted_accuracy':accuracy(teacher,xs,yv),
       'cells':cells,'full_basis_relative_sqerr':err,'final_initialization_shared':True,
       'hashes':{'teacher':state_dict_sha256(teacher),'base_hierarchy':state_dict_sha256(base),
                 'large_calibration':sha256_tensor(xl),'validation':sha256_tensor(xs),'final_initialization':initial['S0']},
       'training_provenance':training_prov,
       'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},
       'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
           'sklearn':sklearn.__version__,'platform':platform.platform(),'torch_threads':1,
           'torch_build':torch.__config__.show(),
           'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0] if Path('/proc/cpuinfo').exists() else platform.processor(),
           'benchmark':False,'mapping_updates':600,'mapping_batch_size':128}}
    json.dumps(r,allow_nan=False)
    return r


def evaluate(rows:list[dict])->dict:
    out={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY','expected_seeds':SEEDS,
         'confirmatory_claim_allowed':False,'reduced_dimension_selection_allowed':False,
         'bootstrap_seed':741384202,'bootstrap_resamples':100000}
    if len(rows)!=16 or any(type(r.get('seed')) is not int for r in rows) or sorted(r['seed'] for r in rows)!=SEEDS:
        return dict(out,decision='STOP_INTEGRITY_OR_INCOMPLETE_COHORT')
    rows=sorted(rows,key=lambda r:r['seed'])
    for r in rows:
        if r.get('eligible') is not True or r.get('test_evaluated') is not False or set(r.get('cells',{}))!=set(CELLS):
            raise ValueError('invalid integrity')
        for v in (r['teacher_clean_accuracy'],r['teacher_shifted_accuracy']):
            if not math.isfinite(v) or not 0<=v<=1: raise ValueError('invalid teacher accuracy')
        for c in r['cells'].values():
            if c['parameters']!=4096 or c['calibration_samples']!=384: raise ValueError('budget/sample mismatch')
            if not math.isfinite(c['validation_accuracy']) or not 0<=c['validation_accuracy']<=1: raise ValueError('invalid accuracy')
            for k in ('activation_nmse_vs_teacher','calibration_mse','validation_mse'):
                if not math.isfinite(c[k]) or c[k]<0: raise ValueError('invalid metric')
    idx=np.random.default_rng(741384202).integers(0,16,size=(100000,16))
    tc=np.array([r['teacher_clean_accuracy'] for r in rows]);ts=np.array([r['teacher_shifted_accuracy'] for r in rows])
    out['eligible_count']=16
    out['target_validity']=summary(100*(ts-tc),idx,-20.)
    out['teacher_clean_accuracy_mean']=float(tc.mean());out['teacher_shifted_accuracy_mean']=float(ts.mean())
    acc={k:np.array([r['cells'][k]['validation_accuracy'] for r in rows]) for k in CELLS}
    out['cells']={}
    for name in CELLS:
        cell=summary(100*(acc[name]-ts),idx,-5.)
        cell['accuracy_mean']=float(acc[name].mean())
        for key in ('activation_nmse_vs_teacher','calibration_mse','validation_mse'):
            vals=np.array([r['cells'][name][key] for r in rows])
            cell[key+'_mean']=float(vals.mean());cell[key+'_median']=float(np.median(vals))
        out['cells'][name]=cell
    out['blend_minus_standard']=summary(100*(acc['B50']-acc['S0']),idx,0.)
    out['descriptive_contrasts']={
        'B50_minus_D64':summary(100*(acc['B50']-acc['D64']),idx),
        'S0_minus_H64':summary(100*(acc['S0']-acc['H64']),idx),
        'D64_minus_S0':summary(100*(acc['D64']-acc['S0']),idx)}
    if out['target_validity']['status']!='PASS': decision='STOP_TARGET_VALIDITY_NOT_ESTABLISHED'
    elif out['cells']['D64']['status']!='PASS': decision='STOP_DIRECT_REFERENCE_NOT_ESTABLISHED'
    elif out['cells']['B50']['status']!='PASS': decision='HALF_BLEND_REFERENCE_NOT_ESTABLISHED'
    elif out['blend_minus_standard']['status']=='PASS': decision='HALF_BLEND_VALID_AND_IMPROVES_STANDARD'
    else: decision='HALF_BLEND_VALID_IMPROVEMENT_NOT_ESTABLISHED'
    out['decision']=decision
    out['boundary']='Full teacher supervision is used in final targets; no reduced teacher interface or universal optimal mixture claim. Seed bootstrap conditional on fixed data split.'
    return out


def validate_record(r):
    if r['experiment']!=EXPERIMENT or r['status']!='FRESH_SEED_OUTCOME': raise ValueError('wrong experiment/status')
    if not math.isfinite(r['full_basis_relative_sqerr']) or not 0<=r['full_basis_relative_sqerr']<=1e-10: raise ValueError('full basis failure')
    if r['final_initialization_shared'] is not True: raise ValueError('unpaired initialization')
    for p,h in r['source_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('source hash mismatch: '+p)
    for c in r['cells'].values():
        y=c['validation_correct']
        if len(y)!=270 or any(type(v) is not int or v not in (0,1) for v in y): raise ValueError('invalid correctness ledger')
        if abs(sum(y)/270-c['validation_accuracy'])>1e-7: raise ValueError('accuracy count mismatch')
        if c['validation_target_variance']<=0 or not math.isclose(c['validation_mse']/c['validation_target_variance'],c['activation_nmse_vs_teacher'],rel_tol=1e-12,abs_tol=1e-12): raise ValueError('NMSE mismatch')


def selftest():
    def row(seed):
        c={'parameters':4096,'calibration_samples':384,'activation_nmse_vs_teacher':.1,'calibration_mse':.05,'validation_mse':.1}
        return {'seed':seed,'eligible':True,'test_evaluated':False,'teacher_clean_accuracy':.97,'teacher_shifted_accuracy':.86,
                'cells':{k:dict(c,validation_accuracy=v) for k,v in zip(CELLS,(.80,.80,.83,.84))}}
    rows=[row(s) for s in SEEDS]
    assert evaluate(rows)['decision']=='HALF_BLEND_VALID_AND_IMPROVES_STANDARD'
    assert evaluate(rows[:-1])['decision']=='STOP_INTEGRITY_OR_INCOMPLETE_COHORT'
    assert evaluate(rows[:-1]+[rows[0]])['decision']=='STOP_INTEGRITY_OR_INCOMPLETE_COHORT'
    for r in rows:r['cells']['B50']['validation_accuracy']=.79
    assert evaluate(rows)['decision']=='HALF_BLEND_REFERENCE_NOT_ESTABLISHED'
    for r in rows:r['cells']['B50']['validation_accuracy']=r['cells']['S0']['validation_accuracy']=.83
    assert evaluate(rows)['decision']=='HALF_BLEND_VALID_IMPROVEMENT_NOT_ESTABLISHED'
    rows[0]['teacher_shifted_accuracy']=float('nan')
    try: evaluate(rows)
    except ValueError: pass
    else: raise AssertionError('nonfinite input accepted')
    print('C74E SYNTHETIC TESTS PASS')


def preflight(out):
    from scripts.gaussian_shift_interface.c73e_pipeline import run as oldrun
    old=oldrun(71300,verification=True);new=run(71300,verification=True)
    for key in ('teacher','base_hierarchy','large_calibration','validation'):
        assert old['hashes'][key]==new['hashes'][key],key
    for a,b in (('N384_H64','H64'),('N384_S64','S0'),('N384_D64','D64')):
        for key in ('validation_accuracy','activation_nmse_vs_teacher','calibration_mse','validation_mse','state_sha256'):
            assert old['cells'][a][key]==new['cells'][b][key],(a,b,key)
    write_json(out/'bridge_71300.json',{'status':'EXACT_BRIDGE_PASS','seed':71300,'teacher_data_and_stage_states_equal':True})
    write_json(out/'verification_72300.json',run(72300,verification=True))
    print('C74E PREFLIGHT PASS; fresh seeds untouched')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','seed','aggregate','selftest'],required=True)
    ap.add_argument('--seed',type=int);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--input',type=Path)
    a=ap.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='preflight':selftest();preflight(a.out)
    elif a.mode=='seed':write_json(a.out,run(a.seed))
    else:
        if a.input is None:raise ValueError('--input required')
        rows=[json.loads(p.read_text()) for p in sorted(a.input.glob('seed_*.json'))]
        for r in rows:validate_record(r)
        write_json(a.out/'FRESH_ROWS.json',{'rows':rows})
        result=evaluate(rows);write_json(a.out/'DECISION.json',result)
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
