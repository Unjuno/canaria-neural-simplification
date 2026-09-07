#!/usr/bin/env python3
"""Confirm the head-rank rule without fitting a reduced-dimension grid."""
from __future__ import annotations
import argparse,copy,json,math,os,platform,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.gaussian_shift_interface.c75e_geometry import prepare,ideal_record,summary,digest,save,SOURCES as OLD_SOURCES
EXPERIMENT='C76R_HEAD_RANK_CONFIRMATION'
PROTOCOL=ROOT/'results/gaussian_shift_interface/c76r/PROTOCOL.json'
SEEDS=list(range(74400,74416))
CELLS=['head_rank9','p64','direct']
SOURCES=OLD_SOURCES+['scripts/gaussian_shift_interface/c76r_head_rank.py','results/gaussian_shift_interface/c76r/PROTOCOL.json']

def run(seed,verification=False):
    if seed not in ((73300,74300) if verification else SEEDS): raise ValueError('seed outside locked cohort')
    import torch
    import torch.nn.functional as F
    import sklearn
    from scripts.gaussian_shift_interface.run_c61r_seed import canonical_nested_qr,adapt_anchored,sha256_tensor
    from scripts.gaussian_shift_interface.run_c68e_seed import state_dict_sha256
    from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import TinyRes,fit_map,compile_final_from_hierarchy,count_params
    g=prepare(seed);t=g['t']
    a0c=g['a0c'];a4c=g['a4c'];a0v=g['a0v'];a4v=g['a4v']
    wc=t.head.weight.detach().double();wc=wc-wc.mean(0,keepdim=True)
    _,sing,vh=torch.linalg.svd(wc,full_matrices=False)
    rank=int((sing>sing[0]*1e-10).sum());assert rank==9
    head_q=vh[:rank].T.float();full_q=canonical_nested_qr(g['rc'])
    print('C76R PRE-FIT HEAD RANK',seed,rank,flush=True)
    dv=float(((a4v-a4v.mean(0,keepdim=True))**2).mean())+1e-12
    dc=float(((a4c-a4c.mean(0,keepdim=True))**2).mean())+1e-12
    cells={}
    for name in CELLS:
        if name=='direct':
            ov=a4v;h=None;rec={'k':64}
            f=fit_map(TinyRes(64,32,seed+716000),a0c,a4c.detach(),600,seed+717000)
        else:
            q,k=(head_q,rank) if name=='head_rank9' else (full_q,64)
            oc,ov,rec=ideal_record(g,q,k);rec=dict(rec,k=k)
            h=copy.deepcopy(g['base']);adapt_anchored(h,a0c,oc,600,seed+715000)
            f,_=compile_final_from_hierarchy(copy.deepcopy(h),a0c,a0v,a4v,dv,seed+716000,seed+717000)
        assert count_params(f)==4096
        with torch.no_grad():
            pc=f(a0c).detach();pv=f(a0v).detach();z=t.head(pv)
            correct=(z.argmax(1)==g['yv']).long().tolist()
            acc=float((z.argmax(1)==g['yv']).float().mean())
            mc=float(F.mse_loss(pc,a4c));mv=float(F.mse_loss(pv,a4v))
            ep=(ov.double()-a4v.double())@wc.T;ef=(pv.double()-ov.double())@wc.T
            a=float(ep.square().mean());b=float(ef.square().mean());cross=float((2*ep*ef).mean())
            total=float(((pv.double()-a4v.double())@wc.T).square().mean())
            assert math.isclose(a+b+cross,total,rel_tol=1e-10,abs_tol=1e-12)
        cells[name]=dict(rec,validation_accuracy=acc,validation_correct=correct,validation_logits=z.detach().tolist(),
            validation_mse=mv,calibration_mse=mc,validation_variance=dv,calibration_variance=dc,nmse=mv/dv,
            final_state_sha256=state_dict_sha256(f),hierarchy_state_sha256=state_dict_sha256(h) if h is not None else None,
            parameters=4096,centered_logit_projection_mse=a,centered_logit_fit_mse=b,centered_logit_cross_term=cross,centered_logit_total_mse=total)
    return {'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_CONFIRMATORY','seed':seed,'eligible':True,'test_evaluated':False,
        'status':'IMPLEMENTATION_VERIFICATION' if verification else 'FRESH_SEED_OUTCOME','github_sha':os.environ.get('GITHUB_SHA'),
        'predicted_rank_before_fits':rank,'head_singular_values':sing.tolist(),'head_basis_sha256':sha256_tensor(head_q),
        'teacher_clean_accuracy':g['clean_acc'],'teacher_shifted_accuracy':g['shift_acc'],'validation_labels':g['yv'].tolist(),
        'hashes':g['hashes'],'cells':cells,'source_sha256':{p:digest(ROOT/p) for p in SOURCES},
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,
            'platform':platform.platform(),'cpuinfo':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
            'torch_threads':torch.get_num_threads(),'torch_build':torch.__config__.show(),'timing_benchmark':False}}

def binomial_interval(s,n,alpha=.05):
    if not 0<=s<=n or n<1: raise ValueError('invalid binomial count')
    def upper_tail(p): return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(s,n+1))
    def lower_tail(p): return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(s+1))
    lo=0.
    if s:
        a,b=0.,1.
        for _ in range(80):
            mid=(a+b)/2
            if upper_tail(mid)<alpha/2: a=mid
            else: b=mid
        lo=(a+b)/2
    hi=1.
    if s<n:
        a,b=0.,1.
        for _ in range(80):
            mid=(a+b)/2
            if lower_tail(mid)>alpha/2: a=mid
            else: b=mid
        hi=(a+b)/2
    return [lo,hi]

def validate(r):
    if r['experiment']!=EXPERIMENT or r['evidence_class']!='PROSPECTIVE_CONFIRMATORY' or r['status']!='FRESH_SEED_OUTCOME': raise ValueError('wrong experiment/status')
    if r['eligible'] is not True or r['test_evaluated'] is not False or r['predicted_rank_before_fits']!=9 or set(r['cells'])!=set(CELLS): raise ValueError('invalid integrity')
    for p,h in r['source_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('source drift '+p)
    for c in r['cells'].values():
        y=c['validation_correct'];logits=np.array(c['validation_logits'])
        if len(y)!=270 or any(type(v) is not int or v not in (0,1) for v in y): raise ValueError('invalid correctness')
        if logits.shape!=(270,10) or not np.isfinite(logits).all(): raise ValueError('invalid logits')
        if (logits.argmax(1)==np.array(r['validation_labels'])).astype(int).tolist()!=y: raise ValueError('logit/count discrepancy')
        if abs(sum(y)/270-c['validation_accuracy'])>1e-7 or c['parameters']!=4096: raise ValueError('accuracy/parameters')
        if not math.isclose(c['validation_mse']/c['validation_variance'],c['nmse'],rel_tol=1e-12,abs_tol=1e-12): raise ValueError('NMSE discrepancy')
        if not math.isclose(c['centered_logit_projection_mse']+c['centered_logit_fit_mse']+c['centered_logit_cross_term'],c['centered_logit_total_mse'],rel_tol=1e-10,abs_tol=1e-12): raise ValueError('decomposition discrepancy')

def evaluate(rows):
    d={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_CONFIRMATORY','bootstrap_resamples':100000,'bootstrap_seed':761384202,'attempted':len(rows)}
    if len(rows)!=16 or sorted(r.get('seed',-1) for r in rows)!=SEEDS: return dict(d,decision='STOP_INTEGRITY')
    rows=sorted(rows,key=lambda r:r['seed'])
    for r in rows: validate(r)
    idx=np.random.default_rng(761384202).integers(0,16,(100000,16))
    t=np.array([r['teacher_shifted_accuracy'] for r in rows]);clean=np.array([r['teacher_clean_accuracy'] for r in rows])
    a=lambda n:np.array([r['cells'][n]['validation_accuracy'] for r in rows])
    nm=lambda n:np.array([r['cells'][n]['nmse'] for r in rows])
    d['eligible_count']=16;d['teacher_shifted_accuracy_mean']=float(t.mean());d['teacher_clean_accuracy_mean']=float(clean.mean())
    d['target_validity']=summary(100*(t-clean),idx,-20)
    d['references']={n:summary(100*(a(n)-t),idx,-5) for n in ['p64','direct']}
    d['utility']=summary(100*(a('head_rank9')-a('p64')),idx,-2)
    d['hidden_nmse_ratio']=summary(nm('head_rank9')/nm('p64'),idx,1.25,ratio=True)
    d['conditions']={n:{'accuracy_mean':float(a(n).mean()),'nmse_mean':float(nm(n).mean()),'teacher_gap_pp':summary(100*(a(n)-t),idx)} for n in CELLS}
    d['head_minus_direct_pp']=summary(100*(a('head_rank9')-a('direct')),idx)
    success=(a('head_rank9')-a('p64')>=-.02)&(nm('head_rank9')/nm('p64')<=1.25)
    s=int(success.sum());d['empirical_seed_success']={'successes':s,'n':16,'two_sided95_clopper_pearson':binomial_interval(s,16),'interpretation':'conditional iid model-seed assumption, not independent dataset validation'}
    if d['target_validity']['status']!='PASS' or any(v['status']!='PASS' for v in d['references'].values()): dec='STOP_REFERENCE_NOT_ESTABLISHED'
    elif d['utility']['status']==d['hidden_nmse_ratio']['status']=='PASS': dec='C76R_CONFIRMATORY_PASS'
    elif 'FAIL' in [d['utility']['status'],d['hidden_nmse_ratio']['status']]: dec='C76R_CONFIRMATORY_FAIL'
    else: dec='C76R_CONFIRMATORY_UNCERTAIN'
    d['decision']=dec;d['scope']='Sufficient head-derived dimension only; fixed digits split; no minimum-dimension, universal compression, hardware or independent-dataset claim.'
    return d

def preflight(out):
    from scripts.gaussian_shift_interface.c75e_geometry import run as oldrun,selftest
    selftest()
    assert abs(binomial_interval(16,16)[0]-.025**(1/16))<1e-12
    assert abs(binomial_interval(0,16)[1]-(1-.025**(1/16)))<1e-12
    assert evaluate([])['decision']=='STOP_INTEGRITY'
    old=oldrun(73300,verification=True);new=run(73300,verification=True)
    assert old['hashes']==new['hashes']
    for n,o in [('head_rank9','centered_head_svd_9'),('p64','p64'),('direct','direct')]:
        for k in ['validation_accuracy','nmse','final_state_sha256','hierarchy_state_sha256']:
            assert new['cells'][n][k]==old['cells'][o][k],(n,k)
    save(out/'bridge_73300.json',{'status':'EXACT_BRIDGE_PASS','seed':73300,'source_sha256':new['source_sha256']})
    save(out/'verification_74300.json',run(74300,verification=True))
    print('C76R PREFLIGHT PASS; no fresh outcomes')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','seed','aggregate'],required=True)
    ap.add_argument('--seed',type=int);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--input',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='preflight': preflight(a.out)
    elif a.mode=='seed': save(a.out/f'seed_{a.seed}.json',run(a.seed))
    else:
        rows=[json.loads(p.read_text()) for p in sorted(a.input.glob('seed_*.json'))]
        d=evaluate(rows);save(a.out/'FRESH_ROWS.json',{'rows':rows});save(a.out/'DECISION.json',d)
        save(a.out/'AUDIT.json',{'status':'PASS' if len(rows)==16 else 'INCOMPLETE','record_count':len(rows),'recomputation_equal':evaluate(rows)==d,'independent_scientific_review':False})
        print(json.dumps(d,indent=2))
