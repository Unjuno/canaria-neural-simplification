#!/usr/bin/env python3
"""C75E: calibration-only geometry screens versus actual compiled outcomes."""
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
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
PROTOCOL=ROOT/'results/gaussian_shift_interface/c75e/PROTOCOL.json'
EXPERIMENT='C75E_TASK_OBSERVABLE_GEOMETRY'
SEEDS=list(range(73400,73416))
DIMS=[2,4,8,9,16]
FAMILIES=['qr_identity','qr_reverse','qr_permutation','residual_svd','centered_head_svd']
CELL_NAMES=['p0','p64','direct']+[f'{f}_{k}' for f in FAMILIES for k in ([2,4,8,9] if f=='centered_head_svd' else DIMS)]
SOURCES=['scripts/gaussian_shift_interface/c75e_geometry.py','results/gaussian_shift_interface/c75e/PROTOCOL.json','scripts/gaussian_shift_interface/run_c61r_seed.py','scripts/gaussian_shift_interface/run_c64r_seed.py','scripts/gaussian_shift_interface/run_c68e_seed.py','scripts/gaussian_shift_interface/run_c72e_seed.py','scripts/recursive_composition/exploration/c10_boundary_signal_ablation.py']

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')

def prepare(seed):
    import torch
    from scripts.gaussian_shift_interface.run_c61r_seed import build_base_hierarchy,sha256_tensor,sha256_int_array
    from scripts.gaussian_shift_interface.run_c68e_seed import train_paired_teachers,state_dict_sha256
    from scripts.gaussian_shift_interface.run_c72e_seed import nested_calibration_indices,standard_normal_like,shifted_input
    from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import split_data,acts,accuracy
    torch.set_num_threads(1)
    xt,yt,xv,yv=split_data()
    assert (len(xt),len(xv))==(1077,270)
    clean,t,tp=train_paired_teachers(seed,xt,yt)
    base,budget=build_base_hierarchy(seed,*acts(t,xt))
    assert budget['exact_4096_each_level']
    bi,ei,li=nested_calibration_indices(len(xt))
    xb=xt[torch.tensor(bi)];xe=xt[torch.tensor(ei)]
    xb=shifted_input(xb,standard_normal_like(xb,seed+710100))
    xe=shifted_input(xe,standard_normal_like(xe,seed+720101))
    xc=torch.cat([xb,xe],dim=0)
    xs=shifted_input(xv,standard_normal_like(xv,seed+710200))
    ac=acts(t,xc);av=acts(t,xs)
    a0c,a4c=ac[0],ac[-1];a0v,a4v=av[0],av[-1]
    with torch.no_grad(): bc=base(a0c).detach();bv=base(a0v).detach()
    return dict(t=t,base=base,a0c=a0c,a4c=a4c,a0v=a0v,a4v=a4v,bc=bc,bv=bv,rc=a4c-bc,rv=a4v-bv,yv=yv,xs=xs,
        clean_acc=accuracy(t,xv,yv),shift_acc=accuracy(t,xs,yv),training_provenance=tp,
        hashes={'teacher':state_dict_sha256(t),'base_hierarchy':state_dict_sha256(base),'large_indices':sha256_int_array(li),'large_calibration':sha256_tensor(xc),'validation':sha256_tensor(xs)})

def construct_bases(g):
    import torch
    from scripts.gaussian_shift_interface.run_c61r_seed import canonical_nested_qr
    rc=g['rc']
    perm=torch.tensor(np.random.default_rng(750123).permutation(len(rc)),dtype=torch.long)
    wc=g['t'].head.weight.detach().double();wc=wc-wc.mean(0,keepdim=True)
    uh,sh,vh=torch.linalg.svd(wc,full_matrices=False)
    rank=int((sh>sh[0]*1e-10).sum())
    assert rank==9,('unexpected centered-head rank',rank)
    _,sr,vr=torch.linalg.svd(rc.double(),full_matrices=False)
    bases={'qr_identity':canonical_nested_qr(rc),'qr_reverse':canonical_nested_qr(rc.flip(0)),
           'qr_permutation':canonical_nested_qr(rc[perm]),'residual_svd':vr.T.float(),'centered_head_svd':vh[:9].T.float()}
    sv=[]
    for r in (rc,rc.flip(0),rc[perm]):
        sv.append({str(k):float(torch.linalg.cond(r[:k].double())) for k in DIMS})
    p=vh[:9].T
    oracle={}
    for name,r in [('calibration',g['rc']),('validation',g['rv'])]:
        r=r.double();err=(r-(r@p)@p.T)@wc.T
        rel=float(torch.linalg.vector_norm(err)/(torch.linalg.vector_norm(r@wc.T)+1e-30))
        assert math.isfinite(rel) and rel<=1e-10,rel
        oracle[name+'_relative_centered_logit_error_float64']=rel
    return bases,dict(centered_head_rank=rank,head_singular_values=sh.tolist(),residual_singular_values=sr.tolist(),prefix_condition_numbers=sv,**oracle)

def ideal_record(g,q,k):
    import torch
    t=g['t']
    with torch.no_grad():
        qc=q[:,:k]
        oc=g['bc']+(g['rc']@qc)@qc.T if k else g['bc']
        ov=g['bv']+(g['rv']@qc)@qc.T if k else g['bv']
        zt=t.head(g['a4c']).double();zt=zt-zt.mean(1,keepdim=True)
        zo=t.head(oc).double();zo=zo-zo.mean(1,keepdim=True)
        e=zo-zt
        top=torch.topk(zt,2,dim=1).values
        margin=top[:,0]-top[:,1]
        unsafe=float((2*e.abs().max(1).values>=margin).double().mean())
        left=float(((g['a4c']-oc).double()**2).sum()/((g['rc'].double()**2).sum()+1e-30))
        rel=float((e**2).sum()/((zt-zt.mean(0,keepdim=True))**2).sum().clamp_min(1e-30))
        disag=float((t.head(oc).argmax(1)!=t.head(g['a4c']).argmax(1)).double().mean())
        correct=(t.head(ov).argmax(1)==g['yv']).long().tolist()
        va=float((t.head(ov).argmax(1)==g['yv']).float().mean())
        vd=float((t.head(ov).argmax(1)!=t.head(g['a4v']).argmax(1)).double().mean())
    return oc.detach(),ov.detach(),dict(calibration_remaining_energy_fraction=left,calibration_margin_unsafe_fraction=unsafe,
        calibration_centered_logit_relative_mse=rel,calibration_teacher_disagreement=disag,
        oracle_validation_accuracy=va,oracle_validation_correct=correct,oracle_validation_teacher_disagreement=vd)

def run(seed,verification=False,controls_only=False,outdir=None):
    if seed not in ((71300,73300) if verification else SEEDS): raise ValueError('seed outside locked cohort')
    import torch
    import torch.nn.functional as F
    import sklearn
    from scripts.gaussian_shift_interface.run_c61r_seed import adapt_anchored,sha256_tensor
    from scripts.gaussian_shift_interface.run_c68e_seed import state_dict_sha256
    from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import TinyRes,fit_map,compile_final_from_hierarchy,count_params
    g=prepare(seed);bases,invariants=construct_bases(g)
    t=g['t'];a0c=g['a0c'];a4c=g['a4c'];a0v=g['a0v'];a4v=g['a4v']
    dv=float(((a4v-a4v.mean(0,keepdim=True))**2).mean())+1e-12
    dc=float(((a4c-a4c.mean(0,keepdim=True))**2).mean())+1e-12
    ideals={};pairs={}
    for name in CELL_NAMES:
        if name=='direct': continue
        if name=='p0': q,k=bases['qr_identity'],0
        elif name=='p64': q,k=bases['qr_identity'],64
        else:
            f,ks=name.rsplit('_',1);q,k=bases[f],int(ks)
        oc,ov,rec=ideal_record(g,q,k)
        ideals[name]=dict(rec,k=k,basis_sha256=sha256_tensor(q[:,:k]));pairs[name]=(oc,ov)
    predictions={}
    for f in FAMILIES:
        ks=[2,4,8,9] if f=='centered_head_svd' else DIMS
        predictions[f]={}
        for screen,metric,threshold in [('energy','calibration_remaining_energy_fraction',.05),('margin','calibration_margin_unsafe_fraction',.01)]:
            chosen=next((k for k in ks if ideals[f'{f}_{k}'][metric]<=threshold),64)
            predictions[f][screen]=chosen
    prefit={'seed':seed,'predictions':predictions,'calibration_only_screens':{n:{k:v for k,v in r.items() if k.startswith('calibration_') or k in ('k','basis_sha256')} for n,r in ideals.items()},'source_sha256':{p:digest(ROOT/p) for p in SOURCES}}
    if outdir is not None: save(Path(outdir)/f'prefit_{seed}.json',prefit)
    prefit_sha=hashlib.sha256(json.dumps(prefit,sort_keys=True).encode()).hexdigest()
    print('C75E PRE-FIT PREDICTIONS',seed,prefit_sha,json.dumps(predictions,sort_keys=True),flush=True)
    names=['p64','direct'] if controls_only else CELL_NAMES
    cells={};arrays={}
    w=t.head.weight.detach().double();wc=w-w.mean(0,keepdim=True)
    with torch.no_grad(): teacher_center=a4v.double()@wc.T
    for name in names:
        if name=='direct':
            h=None;f=fit_map(TinyRes(64,32,seed+716000),a0c,a4c.detach(),600,seed+717000)
            ov=a4v;info={'k':64}
        else:
            oc,ov=pairs[name];info=ideals[name];h=copy.deepcopy(g['base'])
            if name!='p0': adapt_anchored(h,a0c,oc,600,seed+715000)
            f,_=compile_final_from_hierarchy(copy.deepcopy(h),a0c,a0v,a4v,dv,seed+716000,seed+717000)
        assert count_params(f)==4096
        with torch.no_grad():
            pc=f(a0c).detach();pv=f(a0v).detach()
            zv=t.head(pv);correct=(zv.argmax(1)==g['yv']).long().tolist()
            hv=h(a0v).detach() if h is not None else a4v
            hc=(t.head(hv).argmax(1)==g['yv']).long().tolist()
            acc=float((zv.argmax(1)==g['yv']).float().mean())
            mc=float(F.mse_loss(pc,a4c));mv=float(F.mse_loss(pv,a4v))
            ep=(ov.double()-a4v.double())@wc.T
            ef=(pv.double()-ov.double())@wc.T
            a=float((ep**2).mean());b=float((ef**2).mean());cross=float((2*ep*ef).mean())
            total=float(((pv.double()@wc.T)-teacher_center).square().mean())
            assert math.isclose(a+b+cross,total,rel_tol=1e-10,abs_tol=1e-12)
        cells[name]=dict(info,validation_accuracy=acc,validation_correct=correct,hierarchy_validation_accuracy=sum(hc)/270,
            hierarchy_validation_correct=hc,validation_mse=mv,calibration_mse=mc,validation_variance=dv,calibration_variance=dc,
            nmse=mv/dv,calibration_nmse=mc/dc,final_parameters=4096,final_state_sha256=state_dict_sha256(f),
            hierarchy_state_sha256=state_dict_sha256(h) if h is not None else None,
            centered_logit_projection_mse=a,centered_logit_fit_mse=b,centered_logit_cross_term=cross,centered_logit_total_mse=total)
        arrays[name+'_final_logits']=zv.cpu().numpy()
        print('C75E CELL',seed,name,acc,mv/dv,flush=True)
    geometry_path=None
    if outdir is not None:
        geometry_path=Path(outdir)/f'geometry_{seed}.npz'
        np.savez_compressed(geometry_path,calibration_residual=g['rc'].numpy(),validation_residual=g['rv'].numpy(),
            centered_head=wc.numpy(),validation_labels=g['yv'].numpy(),teacher_validation_logits=t.head(a4v).detach().numpy(),
            **{f'basis_{k}':v.numpy() for k,v in bases.items()},**arrays)
    result={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY','seed':seed,
        'status':'IMPLEMENTATION_VERIFICATION' if verification else 'FRESH_SEED_OUTCOME','eligible':True,'test_evaluated':False,
        'github_sha':os.environ.get('GITHUB_SHA'),'teacher_clean_accuracy':g['clean_acc'],'teacher_shifted_accuracy':g['shift_acc'],
        'hashes':g['hashes'],'invariants':invariants,'cells':cells,'prefit':prefit,'prefit_content_sha256':prefit_sha,
        'geometry_sha256':digest(geometry_path) if geometry_path else None,'source_sha256':{p:digest(ROOT/p) for p in SOURCES},
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,
            'platform':platform.platform(),'cpuinfo':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
            'torch_threads':torch.get_num_threads(),'torch_build':torch.__config__.show(),'timing_benchmark':False},'training_provenance':g['training_provenance']}
    json.dumps(result,allow_nan=False)
    return result

def summary(x,idx,margin=None,ratio=False):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all(): raise ValueError('nonfinite values')
    if ratio:
        if np.any(x<=0): raise ValueError('nonpositive ratio')
        boots=np.exp(np.log(x)[idx].mean(1));mean=float(np.exp(np.log(x).mean()))
    else: boots=x[idx].mean(1);mean=float(x.mean())
    lo,hi=np.percentile(boots,[2.5,97.5]);se=float(boots.std(ddof=1))
    r={'mean':mean,'median':float(np.median(x)),'ci95':[float(lo),float(hi)],'bootstrap_se':se}
    if margin is not None:
        status=('PASS' if hi<margin else 'FAIL' if lo>margin else 'UNCERTAIN') if ratio else ('PASS' if lo>margin else 'FAIL' if hi<margin else 'UNCERTAIN')
        r.update(margin=margin,status=status)
    return r

def validate(r):
    if r['experiment']!=EXPERIMENT or r['status']!='FRESH_SEED_OUTCOME' or r['test_evaluated'] is not False or r['eligible'] is not True: raise ValueError('invalid provenance/status')
    if set(r['cells'])!=set(CELL_NAMES): raise ValueError('incomplete cells')
    for p,h in r['source_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('source drift '+p)
    if hashlib.sha256(json.dumps(r['prefit'],sort_keys=True).encode()).hexdigest()!=r['prefit_content_sha256']: raise ValueError('prefit drift')
    if r['invariants']['centered_head_rank']!=9: raise ValueError('rank')
    for part in ['calibration','validation']:
        if not 0<=r['invariants'][part+'_relative_centered_logit_error_float64']<=1e-10: raise ValueError('oracle invariance')
    for c in r['cells'].values():
        y=c['validation_correct']
        if len(y)!=270 or any(type(v) is not int or v not in (0,1) for v in y): raise ValueError('correctness ledger')
        if abs(sum(y)/270-c['validation_accuracy'])>1e-7: raise ValueError('accuracy/count')
        if c['final_parameters']!=4096 or not c['validation_variance']>0 or not math.isfinite(c['nmse']): raise ValueError('invalid parameters/metric')
        if not math.isclose(c['validation_mse']/c['validation_variance'],c['nmse'],rel_tol=1e-12,abs_tol=1e-12): raise ValueError('NMSE mismatch')
        if not math.isclose(c['centered_logit_projection_mse']+c['centered_logit_fit_mse']+c['centered_logit_cross_term'],c['centered_logit_total_mse'],rel_tol=1e-10,abs_tol=1e-12): raise ValueError('error decomposition')

def evaluate(rows):
    d={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY','confirmatory_claim_allowed':False,'bootstrap_resamples':100000,'bootstrap_seed':751384202,'attempted':len(rows)}
    if len(rows)!=16 or sorted(r.get('seed',-1) for r in rows)!=SEEDS:
        return dict(d,decision='STOP_INTEGRITY_OR_INCOMPLETE_COHORT')
    rows=sorted(rows,key=lambda r:r['seed'])
    for r in rows: validate(r)
    idx=np.random.default_rng(751384202).integers(0,16,size=(100000,16))
    a=lambda name:np.array([r['cells'][name]['validation_accuracy'] for r in rows])
    n=lambda name:np.array([r['cells'][name]['nmse'] for r in rows])
    t=np.array([r['teacher_shifted_accuracy'] for r in rows]);cl=np.array([r['teacher_clean_accuracy'] for r in rows])
    d['teacher_target']=summary(100*(t-cl),idx,-20)
    d['references']={name:summary(100*(a(name)-t),idx,-5) for name in ['p64','direct']}
    d['teacher_mean_accuracy']=float(t.mean());d['eligible_count']=16;d['cells']={}
    for name in CELL_NAMES:
        c={'accuracy_mean':float(a(name).mean()),'nmse_mean':float(n(name).mean()),'utility_vs_p64_pp':summary(100*(a(name)-a('p64')),idx,-2),
            'hidden_nmse_ratio_vs_p64':summary(n(name)/n('p64'),idx,1.25,ratio=True)}
        c['joint_pass']=c['utility_vs_p64_pp']['status']==c['hidden_nmse_ratio_vs_p64']['status']=='PASS'
        for k in ['centered_logit_projection_mse','centered_logit_fit_mse','centered_logit_cross_term','centered_logit_total_mse']:
            c[k]=summary([r['cells'][name][k] for r in rows],idx)
        if name!='direct':
            for k in ['oracle_validation_accuracy','oracle_validation_teacher_disagreement','calibration_remaining_energy_fraction','calibration_margin_unsafe_fraction']:
                c[k]=summary([r['cells'][name][k] for r in rows],idx)
        d['cells'][name]=c
    d['order_contrasts']={f'{f}_{k}_minus_identity':summary(100*(a(f'{f}_{k}')-a(f'qr_identity_{k}')),idx) for f in ['qr_reverse','qr_permutation'] for k in DIMS}
    d['predictions']={}
    for f in FAMILIES:
        ks=([2,4,8,9] if f=='centered_head_svd' else DIMS)+[64]
        get=lambda r,k:r['cells']['p64' if k==64 else f'{f}_{k}']
        ok=lambda r,k:(get(r,k)['validation_accuracy']-r['cells']['p64']['validation_accuracy']>=-.02 and get(r,k)['nmse']/r['cells']['p64']['nmse']<=1.25)
        for screen in ['energy','margin']:
            selected=[r['prefit']['predictions'][f][screen] for r in rows]
            successes=[int(ok(r,k)) for r,k in zip(rows,selected)]
            actual=[next(k for k in ks if ok(r,k)) for r in rows]
            d['predictions'][f+'_'+screen]={'selected_dimensions':selected,'empirical_seed_successes':successes,'success_fraction':summary(successes,idx),
                'retrospective_smallest_tested_success':actual,'dimension_regret':[k-v for k,v in zip(selected,actual)],
                'scope':'paired seed empirical margin success, not individual statistical confirmation; all basis cells within seed dependent'}
    h=d['cells']['centered_head_svd_9']
    if d['teacher_target']['status']!='PASS' or any(r['status']!='PASS' for r in d['references'].values()): decision='REFERENCE_NOT_ESTABLISHED_GEOMETRY_ONLY'
    elif h['joint_pass']: decision='HEAD9_TASK_AND_HIDDEN_NI'
    elif h['utility_vs_p64_pp']['status']=='PASS': decision='HEAD9_TASK_NI_HIDDEN_NOT_ESTABLISHED'
    else: decision='HEAD9_LEARNED_TASK_NI_NOT_ESTABLISHED'
    d['decision']=decision
    d['uncertainty']='Exploratory model-seed bootstrap conditional on fixed digits data/splits/subsets. No multiplicity-corrected frontier or dataset-level uncertainty.'
    return d

def selftest():
    import torch
    torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(750000)
    w=torch.randn(10,64,generator=gen,dtype=torch.float64);w-=w.mean(0,keepdim=True)
    r=torch.randn(31,64,generator=gen,dtype=torch.float64)
    _,_,v=torch.linalg.svd(w,full_matrices=False);q=v[:9].T
    err=(r-(r@q)@q.T)@w.T
    assert float(torch.linalg.vector_norm(err))<1e-10
    idx=np.tile(np.arange(16),(100,1))
    for x,status in [(-4,'PASS'),(-5,'UNCERTAIN'),(-6,'FAIL')]: assert summary([x]*16,idx,-5)['status']==status
    for x,status in [(1.,'PASS'),(1.25,'UNCERTAIN'),(1.5,'FAIL')]: assert summary([x]*16,idx,1.25,True)['status']==status
    assert evaluate([])['decision']=='STOP_INTEGRITY_OR_INCOMPLETE_COHORT'
    print('C75E SYNTHETIC TESTS PASS')

def preflight(out):
    from scripts.gaussian_shift_interface.c73e_pipeline import run as oldrun
    selftest()
    old=oldrun(71300,verification=True);new=run(71300,verification=True,controls_only=True,outdir=out)
    for k in ['teacher','base_hierarchy','large_indices','large_calibration','validation']: assert old['hashes'][k]==new['hashes'][k],k
    for name,oname in [('p64','N384_S64'),('direct','N384_D64')]:
        a=old['cells'][oname];b=new['cells'][name]
        assert a['state_sha256']==b['final_state_sha256'],name
        assert a['validation_accuracy']==b['validation_accuracy'],name
        assert a['activation_nmse_vs_teacher']==b['nmse'],name
    assert old['cells']['N384_H64']['state_sha256']==new['cells']['p64']['hierarchy_state_sha256']
    save(out/'bridge_71300.json',{'status':'EXACT_BRIDGE_PASS','seed':71300,'source_sha256':new['source_sha256'],'hashes':new['hashes']})
    save(out/'verification_73300.json',run(73300,verification=True,outdir=out))
    print('C75E PREFLIGHT PASS; zero fresh seeds consumed')

def aggregate(inp,out):
    paths=sorted(inp.glob('seed_*.json'));rows=[json.loads(p.read_text()) for p in paths]
    decision=evaluate(rows);save(out/'FRESH_ROWS.json',{'rows':rows});save(out/'DECISION.json',decision)
    for p in paths:
        r=json.loads(p.read_text());geom=inp/f"geometry_{r['seed']}.npz"
        if digest(geom)!=r['geometry_sha256']: raise ValueError('geometry digest mismatch')
    save(out/'AUDIT.json',{'record_count':len(rows),'decision_recomputed':evaluate(rows)==decision,'status':'PASS' if len(rows)==16 else 'INCOMPLETE','independent_scientific_review':False})
    print(json.dumps(decision,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['selftest','preflight','seed','aggregate'],required=True)
    ap.add_argument('--seed',type=int);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--input',type=Path)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='selftest': selftest()
    elif args.mode=='preflight': preflight(args.out)
    elif args.mode=='seed': save(args.out/f'seed_{args.seed}.json',run(args.seed,outdir=args.out))
    else: aggregate(args.input,args.out)
