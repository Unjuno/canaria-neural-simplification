#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json,math,os,platform,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.gaussian_shift_interface.c75e_geometry import prepare,ideal_record,summary,digest,save
from scripts.gaussian_shift_interface.c76r_head_rank import SOURCES as OLD_SOURCES
EXPERIMENT='C77E_HEAD_RANK_ABSOLUTE_UTILITY'
SEEDS=list(range(75400,75416))
CELLS=[f'N{n}_{c}' for n in (384,512) for c in ('head_rank9','p64','direct')]
SOURCES=OLD_SOURCES+['scripts/gaussian_shift_interface/c77e_absolute.py','results/gaussian_shift_interface/c77e/PROTOCOL.json']

def run(seed,verification=False):
    if seed not in ((74300,75300) if verification else SEEDS): raise ValueError('seed outside cohort')
    import torch
    import torch.nn.functional as F
    import sklearn
    from scripts.gaussian_shift_interface.run_c61r_seed import canonical_nested_qr,adapt_anchored,sha256_tensor,sha256_int_array
    from scripts.gaussian_shift_interface.run_c68e_seed import state_dict_sha256
    from scripts.gaussian_shift_interface.run_c72e_seed import nested_calibration_indices,standard_normal_like,shifted_input
    from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import split_data,acts,TinyRes,fit_map,compile_final_from_hierarchy,count_params
    g=prepare(seed);t=g['t'];xt,yt,xv,yv=split_data()
    _,_,old_idx=nested_calibration_indices(len(xt))
    complement=np.setdiff1d(np.arange(len(xt)),old_idx,assume_unique=True)
    extra=np.sort(np.random.default_rng(20260908).choice(complement,128,replace=False))
    assert len(np.unique(np.r_[old_idx,extra]))==512
    xe=xt[torch.tensor(extra,dtype=torch.long)]
    xe=shifted_input(xe,standard_normal_like(xe,seed+770101));ae=acts(t,xe)
    g512=dict(g)
    with torch.no_grad():
        g512['a0c']=torch.cat([g['a0c'],ae[0]],0);g512['a4c']=torch.cat([g['a4c'],ae[-1]],0)
        g512['bc']=torch.cat([g['bc'],g['base'](ae[0]).detach()],0);g512['rc']=g512['a4c']-g512['bc']
    assert torch.equal(g512['a0c'][:384],g['a0c']) and torch.equal(g512['a4c'][:384],g['a4c'])
    w=t.head.weight.detach().double();wc=w-w.mean(0,keepdim=True)
    _,s,vh=torch.linalg.svd(wc,full_matrices=False);rank=int((s>s[0]*1e-10).sum());assert rank==9
    hq=vh[:rank].T.float();cells={}
    print('C77E PRE-FIT HEAD RANK',seed,rank,flush=True)
    dv=float(((g['a4v']-g['a4v'].mean(0,keepdim=True))**2).mean())+1e-12
    for n,data in [(384,g),(512,g512)]:
        a0c=data['a0c'];a4c=data['a4c'];fq=canonical_nested_qr(data['rc'])
        for kind in ['head_rank9','p64','direct']:
            if kind=='direct':
                h=None;ov=data['a4v'];ideal={'k':64}
                f=fit_map(TinyRes(64,32,seed+716000),a0c,a4c.detach(),600,seed+717000)
            else:
                q,k=(hq,rank) if kind=='head_rank9' else (fq,64)
                oc,ov,ideal=ideal_record(data,q,k);ideal=dict(ideal,k=k)
                h=copy.deepcopy(data['base']);adapt_anchored(h,a0c,oc,600,seed+715000)
                f,_=compile_final_from_hierarchy(copy.deepcopy(h),a0c,data['a0v'],data['a4v'],dv,seed+716000,seed+717000)
            assert count_params(f)==4096
            with torch.no_grad():
                pc=f(a0c).detach();pv=f(data['a0v']).detach();logits=t.head(pv)
                correct=(logits.argmax(1)==yv).long().tolist();acc=float((logits.argmax(1)==yv).float().mean())
                mse=float(F.mse_loss(pv,data['a4v']));cmse=float(F.mse_loss(pc,a4c))
            cells[f'N{n}_{kind}']=dict(ideal,calibration_samples=n,parameters=4096,validation_accuracy=acc,validation_correct=correct,
                validation_logits=logits.detach().tolist(),validation_mse=mse,validation_variance=dv,nmse=mse/dv,calibration_mse=cmse,
                final_state_sha256=state_dict_sha256(f),hierarchy_state_sha256=state_dict_sha256(h) if h is not None else None)
    return {'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY','seed':seed,'eligible':True,'test_evaluated':False,
        'status':'IMPLEMENTATION_VERIFICATION' if verification else 'FRESH_SEED_OUTCOME','github_sha':os.environ.get('GITHUB_SHA'),
        'predicted_rank_before_fits':rank,'teacher_clean_accuracy':g['clean_acc'],'teacher_shifted_accuracy':g['shift_acc'],
        'validation_labels':yv.tolist(),'hashes':g['hashes'],'cells':cells,
        'nesting':{'exact_prefix':True,'unique_indices':512,'extension_index_seed':20260908,'extension_noise_seed':seed+770101,
            'extra_indices':extra.tolist(),'extra_indices_sha256':sha256_int_array(extra),'extended_input_activations_sha256':sha256_tensor(g512['a0c']),
            'extended_target_activations_sha256':sha256_tensor(g512['a4c'])},
        'source_sha256':{p:digest(ROOT/p) for p in SOURCES},
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,
            'platform':platform.platform(),'cpuinfo':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
            'torch_threads':torch.get_num_threads(),'torch_build':torch.__config__.show(),'timing_benchmark':False}}

def validate(r):
    if r['experiment']!=EXPERIMENT or r['status']!='FRESH_SEED_OUTCOME' or r['eligible'] is not True or r['test_evaluated'] is not False: raise ValueError('provenance')
    if set(r['cells'])!=set(CELLS) or r['predicted_rank_before_fits']!=9 or not r['nesting']['exact_prefix'] or r['nesting']['unique_indices']!=512: raise ValueError('integrity')
    if set(r['source_sha256'])!=set(SOURCES): raise ValueError('missing sources')
    for p,h in r['source_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('source drift '+p)
    for name,c in r['cells'].items():
        y=c['validation_correct'];z=np.array(c['validation_logits'])
        if len(y)!=270 or any(type(v) is not int or v not in (0,1) for v in y): raise ValueError('ledger')
        if z.shape!=(270,10) or not np.isfinite(z).all() or (z.argmax(1)==r['validation_labels']).astype(int).tolist()!=y: raise ValueError('logits')
        if abs(sum(y)/270-c['validation_accuracy'])>1e-7 or c['parameters']!=4096 or c['calibration_samples']!=int(name[1:4]): raise ValueError('accuracy/params')
        if not c['validation_variance']>0 or not math.isclose(c['validation_mse']/c['validation_variance'],c['nmse'],rel_tol=1e-12,abs_tol=1e-12): raise ValueError('NMSE')

def evaluate(rows):
    d={'experiment':EXPERIMENT,'evidence_class':'PROSPECTIVE_EXPLORATORY','confirmatory_claim_allowed':False,'attempted':len(rows),'bootstrap_resamples':100000,'bootstrap_seed':771384202}
    if len(rows)!=16 or sorted(r.get('seed',-1) for r in rows)!=SEEDS:return dict(d,decision='STOP_INTEGRITY')
    rows=sorted(rows,key=lambda r:r['seed'])
    for r in rows:validate(r)
    idx=np.random.default_rng(771384202).integers(0,16,(100000,16))
    t=np.array([r['teacher_shifted_accuracy'] for r in rows]);clean=np.array([r['teacher_clean_accuracy'] for r in rows])
    acc=lambda n:np.array([r['cells'][n]['validation_accuracy'] for r in rows])
    nm=lambda n:np.array([r['cells'][n]['nmse'] for r in rows])
    d['eligible_count']=16;d['teacher_accuracy_mean']=float(t.mean());d['target']=summary(100*(t-clean),idx,-20)
    d['cells']={n:{'accuracy_mean':float(acc(n).mean()),'nmse_mean':float(nm(n).mean()),'absolute_teacher_gap_pp':summary(100*(acc(n)-t),idx,-5)} for n in CELLS}
    d['relative']={str(n):{'utility_pp':summary(100*(acc(f'N{n}_head_rank9')-acc(f'N{n}_p64')),idx,-2),
        'hidden_nmse_ratio':summary(nm(f'N{n}_head_rank9')/nm(f'N{n}_p64'),idx,1.25,True)} for n in [384,512]}
    d['gains']={k:summary(100*(acc(f'N512_{k}')-acc(f'N384_{k}')),idx,0) for k in ['head_rank9','p64','direct']}
    if d['target']['status']!='PASS' or any(d['cells'][f'N512_{k}']['absolute_teacher_gap_pp']['status']!='PASS' for k in ['p64','direct']): dec='STOP_TARGET_OR_REFERENCE_NOT_ESTABLISHED'
    elif d['cells']['N512_head_rank9']['absolute_teacher_gap_pp']['status']!='PASS':dec='N512_HEAD_ABSOLUTE_NOT_ESTABLISHED'
    elif any(v['status']!='PASS' for v in d['relative']['512'].values()):dec='N512_HEAD_RELATIVE_NOT_ESTABLISHED'
    elif d['gains']['head_rank9']['status']=='PASS':dec='N512_ABSOLUTE_AND_RELATIVE_VALID_WITH_GAIN'
    else:dec='N512_VALID_GAIN_NOT_ESTABLISHED'
    d['decision']=dec;d['scope']='Exploratory; do not assume N384 is intrinsically invalid; no universal sample threshold, minimum dimension or new dataset claim.'
    return d

def preflight(out):
    from scripts.gaussian_shift_interface.c76r_head_rank import run as oldrun
    idx=np.tile(np.arange(16),(100,1))
    assert summary([-5]*16,idx,-5)['status']=='UNCERTAIN'
    assert evaluate([])['decision']=='STOP_INTEGRITY'
    old=oldrun(74300,verification=True);new=run(74300,verification=True)
    assert old['hashes']==new['hashes']
    for n in ['head_rank9','p64','direct']:
        for k in ['validation_accuracy','nmse','final_state_sha256','hierarchy_state_sha256']:assert old['cells'][n][k]==new['cells']['N384_'+n][k],(n,k)
    save(out/'bridge_74300.json',{'status':'EXACT_BRIDGE_PASS','seed':74300,'source_sha256':new['source_sha256']})
    save(out/'verification_75300.json',run(75300,verification=True))
    print('C77E PREFLIGHT PASS; no fresh seeds used')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','seed','aggregate'],required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int);ap.add_argument('--input',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='preflight':preflight(a.out)
    elif a.mode=='seed':save(a.out/f'seed_{a.seed}.json',run(a.seed))
    else:
        rows=[json.loads(p.read_text()) for p in sorted(a.input.glob('seed_*.json'))];d=evaluate(rows)
        save(a.out/'FRESH_ROWS.json',{'rows':rows});save(a.out/'DECISION.json',d)
        save(a.out/'AUDIT.json',{'status':'PASS' if len(rows)==16 else 'INCOMPLETE','record_count':len(rows),'recomputation_equal':evaluate(rows)==d,'independent_scientific_review':False})
        print(json.dumps(d,indent=2))
