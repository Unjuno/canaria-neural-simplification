"""Post-outcome raw-data audit; not an independent scientific replication."""
from __future__ import annotations
import argparse, ast, contextlib, copy, hashlib, json, math, runpy, tempfile, time
from pathlib import Path
import numpy as np
SEEDS=list(range(9200,9216)); WIDTHS=['id','8','16','32','64']
def need(ok,msg):
    if not ok: raise ValueError(msg)
def near(a,b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
def budget(c):
    if isinstance(c,list): return sum(budget(x) for x in c)
    return 0 if c=='id' else 4384+65*int(c)
def check(rows):
    need(len(rows)==16 and sorted(r['seed'] for r in rows)==SEEDS,'cohort')
    eligible=[]
    for r in sorted(rows,key=lambda r:r['seed']):
        v=r['baseline_val_acc']; tp=r['test_policy']
        need(type(r['eligible']) is bool and math.isfinite(v) and 0<=v<=1 and r['eligible']==(v>=.95),'eligibility')
        need(tp['candidate_test_metrics_computed_preselection'] is False and tp['test_evaluations_before_selection']==0,'test firewall')
        need(r['span']==[1,2] and r['widths']==[8,16,32,64] and r['fit_epochs']==40,'configuration')
        if not r['eligible']:
            need(r['decision_state']=='INELIGIBLE_BASELINE' and r['baseline_test_acc'] is None and 'componentwise' not in r,'ineligible record')
            continue
        need(r['decision_state']=='COMPLETE' and tp['actual_test_evaluations_after_selection']==3,'completeness')
        for g in ['componentwise','composed']:
            cs=r[g]; expected=[(a,b) for a in WIDTHS for b in WIDTHS] if g=='componentwise' else WIDTHS
            actual=[tuple(c['choice']) if isinstance(c['choice'],list) else c['choice'] for c in cs]
            need(len(actual)==len(expected) and set(actual)==set(expected),'grid')
            for c in cs:
                need(c['replacement_params']==budget(c['choice']),'parameter count')
                need(c['test_acc'] is None and 'test_utility' not in c,'grid test leak')
                need(0<=c['val_acc']<=1 and near(c['val_utility'],c['val_acc']/v),'validation utility')
                need(c['hold_mse']>=0 and c['hold_den']>0 and near(c['hold_nmse'],c['hold_mse']/c['hold_den']),'NMSE')
                need(c['compiler_updates'] in [0,320,640],'updates')
            passing=[c for c in cs if c['hold_nmse']<=.12 and c['val_utility']>=.95]
            need(bool(passing),'no endpoint')
            chosen=min(passing,key=lambda c:(c['replacement_params'],c['hold_nmse'],str(c['choice'])))
            need(r['selected_'+g+'_pretest']==chosen,'selection')
            post=r['selected_'+g]
            need(all(post[k]==chosen[k] for k in chosen if k!='test_acc'),'post-test selection drift')
            need(0<r['baseline_test_acc']<=1 and 0<=post['test_acc']<=1 and near(post['test_utility'],post['test_acc']/r['baseline_test_acc']),'test values')
        eligible.append(r)
    need(len(eligible)>=12,'eligible count')
    return eligible

def control_flow(path):
    tree=ast.parse(path.read_text()); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['run','choose']]
    need(len(nodes)==2,'run/choose AST')
    class Map:
        def __call__(self,x): return x
    ns={'Path':Path,'json':json,'time':time,'torch':type('Torch',(),{'no_grad':staticmethod(contextlib.nullcontext)})(),'set_seed':lambda _:None,'data_split':lambda:('train','val','test'),'SmallViT':Map,'Identity':Map,'train_cls':lambda *a,**k:None,'collect_span':lambda *a,**k:((1,1,1),(1,1,1)),'fit_map':lambda *a:(Map(),320),'count_params':lambda _:4904,'build_candidate':lambda *a:Map(),'nmse':lambda *a:(.05,.05,1.)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    original_choose=ns['choose']; result={}
    for case in ['complete','ineligible','missing']:
        events=[]
        def choose(cs):
            events.append('select'); return None if case=='missing' else original_choose(cs)
        def accuracy(model,ds):
            if ds=='test': need(events.count('select')==2,'test before selection')
            events.append(ds); return .90 if case=='ineligible' else .96
        ns['choose']=choose; ns['accuracy']=accuracy
        with tempfile.TemporaryDirectory() as td: ns['run'](9199,td)
        need(events.count('test')==(3 if case=='complete' else 0),'actual test call count')
        result[case]={'selections':events.count('select'),'test_calls':events.count('test')}
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; base=root/'results/replication/vit_hidden_test_confirm'
    manifest=json.loads((base/'RUN_MANIFEST.json').read_text()); protocol=json.loads((base/'PROTOCOL.json').read_text())
    paths=sorted(a.input.glob('**/seed_*.json')); rows=[json.loads(p.read_text()) for p in paths]
    eligible=check(rows)
    for p in paths:
        s=json.loads(p.read_text())['seed']; need(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['seed_files'][str(s)],'raw hash')
    dp=a.input/'DECISION.json'; need(hashlib.sha256(dp.read_bytes()).hexdigest()==manifest['decision_sha256'],'decision hash'); d=json.loads(dp.read_text())
    for filename,key in [('vit_hidden_test_confirm.py','runner_git_blob_sha1'),('evaluate_vit_hidden_test_confirm.py','evaluator_git_blob_sha1')]:
        raw=(root/'scripts/replication'/filename).read_bytes(); need(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==protocol[key],'locked source')
    ev=runpy.run_path(str(root/'scripts/replication/evaluate_vit_hidden_test_confirm.py'))
    need(ev['evaluate'](paths[0].parent)==d,'original evaluator exact')
    ratios=[r['selected_composed']['replacement_params']/r['selected_componentwise']['replacement_params'] for r in eligible]
    values=[[math.log2(x) for x in ratios],[r['selected_composed']['test_acc']-r['selected_componentwise']['test_acc'] for r in eligible],[r['selected_composed']['test_utility'] for r in eligible]]
    target=[(d['primary_budget']['mean_log2_ratio'],d['primary_budget']['bootstrap95_mean_log2_ratio']),(d['postselection_test_safeguard']['mean_composed_minus_componentwise_accuracy'],d['postselection_test_safeguard']['bootstrap95_mean_difference']),(d['postselection_test_safeguard']['mean_composed_test_utility'],d['postselection_test_safeguard']['bootstrap95_mean_composed_test_utility'])]
    rng=np.random.default_rng(20260912); stats=[]
    for x,(m,ci) in zip(values,target):
        v=np.asarray(x,dtype=np.float64); bs=v[rng.integers(0,len(v),(100000,len(v)))].mean(axis=1); interval=np.quantile(bs,[.025,.975]).tolist()
        need(near(v.mean(),m) and all(near(x,y) for x,y in zip(interval,ci)),'independent bootstrap')
        stats.append({'mean':float(v.mean()),'ci95':interval})
    passed=stats[0]['ci95'][1]<0 and sum(x<1 for x in ratios)>=math.ceil(.75*len(ratios)) and stats[1]['ci95'][0]>-.03
    need(passed and d['decision']=='SMALLVIT_HIDDEN_TEST_CONFIRMATORY_PASS','independent decision')
    mutations=[]
    for c in ['missing','duplicate','eligibility','budget','test','selection','nmse']:
        bad=copy.deepcopy(rows); i=next(j for j,r in enumerate(bad) if r['eligible'])
        if c=='missing': bad.pop()
        elif c=='duplicate': bad[-1]=copy.deepcopy(bad[0])
        elif c=='eligibility': bad[i]['eligible']=False
        elif c=='budget': bad[i]['componentwise'][0]['replacement_params']+=1
        elif c=='test': bad[i]['composed'][0]['test_acc']=.9
        elif c=='selection': bad[i]['selected_composed']['replacement_params']+=1
        else: bad[i]['composed'][0]['hold_nmse']=float('nan')
        try: check(bad)
        except (ValueError,KeyError): mutations.append(c)
        else: raise ValueError('accepted mutation: '+c)
    result={'status':'PASS','original_evaluator_exact':True,'raw_seed_count':16,'eligible':len(eligible),'mutation_tests':mutations,'independent_statistics':stats,'geometric_budget_ratio':float(2**np.mean(values[0])),'two_block_floor_count':sum(r['selected_componentwise']['replacement_params']==9808 for r in eligible),'single_block_floor_count':sum(r['selected_composed']['replacement_params']==4904 for r in eligible),'code_control_flow_regression':control_flow(root/'scripts/replication/vit_hidden_test_confirm.py'),'new_independent_seeds':0,'scope':'Post-outcome record/code audit, not external review or historical process access logging. Floor analysis is descriptive and does not invalidate the locked grid-specific result.'}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
