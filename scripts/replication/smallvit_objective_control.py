"""Prospectively locked SmallViT objective control; fixed arms, no model/grid selection."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, platform, sys, time, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'results/replication/smallvit_objective_control/PROTOCOL.json'
SEEDS=list(range(194000,194016)); VERIFY=193999
ARMS=['local_pair','joint_pair','direct_single']
COUNTS={'local_pair':9808,'joint_pair':9808,'direct_single':4904}

def need(ok,msg):
    if not ok: raise ValueError(msg)
def filehash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def arrayhash(a):
    a=np.ascontiguousarray(a); h=hashlib.sha256(); h.update(str((a.shape,str(a.dtype))).encode()); h.update(a.tobytes()); return h.hexdigest()
def statehash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(arrayhash(v.detach().cpu().numpy()).encode())
    return h.hexdigest()
def load_base():
    p=ROOT/'scripts/replication/vit_hidden_test_confirm.py'; raw=p.read_bytes()
    need(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='cef2281914aaebc55dc6d92505c0ae28bb67193d','inherited source drift')
    sp=importlib.util.spec_from_file_location('_svht_locked',p); b=importlib.util.module_from_spec(sp); sp.loader.exec_module(b); return b

def environment(torch):
    import sklearn
    cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown') if Path('/proc/cpuinfo').exists() else platform.processor()
    return {'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'cpu':cpu,'threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),'aten_capability':torch.backends.cpu.get_cpu_capability(),'torch_build':torch.__config__.show(),'clock_control':'not_fixed; no runtime/energy comparison'}

def fit_arms(b,seed,X,M,Y,epochs=40):
    torch=b.torch
    b.set_seed(seed+100008); l0=b.Block(32,4,8)
    b.set_seed(seed+200008); l1=b.Block(32,4,8)
    j0,j1=copy.deepcopy(l0),copy.deepcopy(l1)
    b.set_seed(seed+300008); single=b.Block(32,4,8)
    initial={'local0':statehash(l0),'local1':statehash(l1),'joint0':statehash(j0),'joint1':statehash(j1),'single':statehash(single)}
    need(initial['local0']==initial['joint0'] and initial['local1']==initial['joint1'],'paired initialization')
    models={'local_pair':torch.nn.Sequential(l0,l1),'joint_pair':torch.nn.Sequential(j0,j1),'direct_single':single}
    need({k:b.count_params(v) for k,v in models.items()}==COUNTS,'parameter budget mismatch')
    opts=[torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.00001) for m in [l0,l1,models['joint_pair'],single]]
    schedule=[]; updates=0; losses=None
    for epoch in range(epochs):
        g=torch.Generator().manual_seed(seed+410000+epoch); order=torch.randperm(len(X),generator=g); schedule.append(order.numpy())
        for idx in order.split(64):
            tasks=[(l0,X[idx],M[idx]),(l1,M[idx],Y[idx]),(models['joint_pair'],X[idx],Y[idx]),(single,X[idx],Y[idx])]
            losses=[]
            for opt,(model,inputs,target) in zip(opts,tasks):
                model.train(); opt.zero_grad(); loss=((model(inputs)-target)**2).mean(); need(bool(torch.isfinite(loss)),'nonfinite training loss'); loss.backward(); opt.step(); losses.append(float(loss.detach()))
            updates+=1
    for model in models.values(): model.eval()
    return models,initial,{'per_parameter_updates':updates,'batch_size':64,'schedule_sha256':arrayhash(np.stack(schedule)),'schedule_shape':list(np.stack(schedule).shape),'last_batch_losses_descriptive':losses}

class TestGate:
    def __init__(self,predict,certificate): self.predict=predict; self.certificate=certificate; self.calls=0; self.expected_hash=None
    def arm(self): self.expected_hash=filehash(self.certificate)
    def evaluate(self,m,ds):
        need(self.expected_hash is not None and self.certificate.exists() and filehash(self.certificate)==self.expected_hash,'test before fixed-arm freeze')
        self.calls+=1; need(self.calls<=4,'extra test evaluation'); return self.predict(m,ds)

def run_seed(seed,out):
    need(seed in SEEDS+[VERIFY],'unexpected seed')
    import torch
    import sklearn
    need(sys.version_info[:2]==(3,11) and torch.__version__.startswith('2.13.0') and np.__version__=='2.4.6' and sklearn.__version__=='1.9.0','locked software environment required')
    b=load_base(); start=time.time(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    b.set_seed(seed); tr,va,te=b.data_split(); teacher=b.SmallViT(); teacher_init=statehash(teacher)
    b.train_cls(teacher,tr,epochs=45,seed=seed+50000)
    def predict(model,ds):
        model.eval(); logits=[]; labels=[]
        with torch.no_grad():
            for x,y in torch.utils.data.DataLoader(ds,batch_size=256,shuffle=False): logits.append(model(x).cpu().numpy()); labels.append(y.numpy())
        a=np.concatenate(logits); y=np.concatenate(labels); need(np.isfinite(a).all(),'nonfinite logits'); return a,y,float(np.mean(a.argmax(1)==y))
    val_logits,val_y,val_acc=predict(teacher,va)
    d={'experiment':'SMALLVIT_OBJECTIVE_CONTROL_E1','evidence_class':'PROSPECTIVE_EXPLORATORY_MECHANISM','seed':seed,'protocol_sha256':filehash(PROTOCOL),'runner_sha256':filehash(__file__),'inherited_runner_blob':'cef2281914aaebc55dc6d92505c0ae28bb67193d','environment':environment(torch),'teacher_val_acc':val_acc,'eligible':val_acc>=.95,'teacher_initial_state_hash':teacher_init,'teacher_state_hash':statehash(teacher),'data_hashes':{name:[arrayhash(t.numpy()) for t in ds.tensors] for name,ds in [('train',tr),('validation',va),('test',te)]},'test_calls_before_freeze':0,'test_calls_after_freeze':0,'arms':{}}
    raw={'teacher_val_logits':val_logits,'validation_labels':val_y}
    if not d['eligible']:
        d['state']='INELIGIBLE_BASELINE'; d['runtime_sec']=time.time()-start; np.savez_compressed(out/'predictions.npz',**raw); d['predictions_sha256']=filehash(out/'predictions.npz'); write(out/'record.json',d); return d
    (X,M,Y),(Xh,Mh,Yh)=b.collect_span(teacher,tr,start=1)
    need(tuple(X.shape)==(512,17,32) and tuple(Xh.shape)==(256,17,32),'calibration shapes')
    models,initial,training=fit_arms(b,seed,X,M,Y)
    need(training['per_parameter_updates']==320,'locked updates')
    d['initial_states']=initial; d['training']=training
    for name,model in models.items():
        with torch.no_grad(): hp=model(Xh); e,mse,den=b.nmse(hp,Yh)
        full=b.build_candidate(teacher,[model,b.Identity()],1)
        vl,vy,acc=predict(full,va); need(np.array_equal(vy,val_y),'validation labels')
        d['arms'][name]={'parameters':b.count_params(model),'hold_nmse':e,'hold_mse':mse,'hold_den':den,'val_acc':acc,'val_utility':acc/val_acc,'state_hash':statehash(model)}
        raw[name+'_val_logits']=vl
        for k,t in model.state_dict().items(): raw[name+'_state_'+k]=t.detach().cpu().numpy()
    for k,t in teacher.state_dict().items(): raw['teacher_state_'+k]=t.detach().cpu().numpy()
    certificate=out/'PRETEST_FREEZE.json'; write(certificate,{'seed':seed,'protocol_sha256':d['protocol_sha256'],'arms':copy.deepcopy(d['arms']),'teacher_state_hash':d['teacher_state_hash'],'test_metrics_computed':False})
    gate=TestGate(predict,certificate); gate.arm(); tl,ty,ta=gate.evaluate(teacher,te)
    d['teacher_test_acc']=ta; raw['teacher_test_logits']=tl; raw['test_labels']=ty
    for name,model in models.items():
        full=b.build_candidate(teacher,[model,b.Identity()],1); tl,labels,acc=gate.evaluate(full,te); need(np.array_equal(labels,ty),'test labels')
        d['arms'][name]['test_acc']=acc; raw[name+'_test_logits']=tl
        need(d['arms'][name]['state_hash']==statehash(model),'post-freeze parameter drift')
    need(gate.calls==4,'test call count'); d['test_calls_after_freeze']=gate.calls; d['freeze_sha256']=gate.expected_hash
    d['state']='COMPLETE'; d['runtime_sec']=time.time()-start; np.savez_compressed(out/'predictions.npz',**raw); d['predictions_sha256']=filehash(out/'predictions.npz'); write(out/'record.json',d); return d

def validate(rows):
    need(len(rows)==16 and sorted(r['seed'] for r in rows)==SEEDS,'missing/duplicate seed')
    good=[]
    for r in sorted(rows,key=lambda x:x['seed']):
        need(r['protocol_sha256']==filehash(PROTOCOL) and r['runner_sha256']==filehash(__file__),'provenance')
        need(type(r['eligible']) is bool and math.isfinite(r['teacher_val_acc']) and 0<=r['teacher_val_acc']<=1 and r['eligible']==(r['teacher_val_acc']>=.95),'eligibility')
        need(r['test_calls_before_freeze']==0,'pre-freeze test')
        if not r['eligible']:
            need(r['state']=='INELIGIBLE_BASELINE' and r['test_calls_after_freeze']==0 and not r['arms'],'ineligible fitted/tested'); continue
        need(r['state']=='COMPLETE' and r['test_calls_after_freeze']==4 and len(r['freeze_sha256'])==64,'incomplete/freeze')
        s=r['initial_states']; need(s['local0']==s['joint0'] and s['local1']==s['joint1'],'unpaired initialization')
        need(r['training']['per_parameter_updates']==320 and r['training']['batch_size']==64 and r['training']['schedule_shape']==[40,512],'training schedule')
        need(set(r['arms'])==set(ARMS),'missing/extra arm')
        for name,x in r['arms'].items():
            need(x['parameters']==COUNTS[name],'wrong budget')
            need(all(math.isfinite(x[k]) for k in ['hold_nmse','hold_mse','hold_den','val_acc','test_acc']),'nonfinite metric')
            need(x['hold_nmse']>0 and x['hold_den']>0 and math.isclose(x['hold_nmse'],x['hold_mse']/x['hold_den'],rel_tol=1e-12,abs_tol=1e-12),'invalid NMSE')
            need(0<=x['val_acc']<=1 and 0<=x['test_acc']<=1,'accuracy range')
        good.append(r)
    need(len(good)>=12,'insufficient eligible')
    return good

def evaluate(inputdir,out):
    paths=sorted(Path(inputdir).glob('seed_*/record.json')); rows=[json.loads(p.read_text()) for p in paths]
    d={'experiment':'SMALLVIT_OBJECTIVE_CONTROL_E1','evidence_class':'PROSPECTIVE_EXPLORATORY_MECHANISM','attempted':len(rows),'new_scientific_model_seed_ids':SEEDS,'errors':[]}
    try:
        good=validate(rows)
        for p,r in zip(paths,rows):
            need(filehash(p.parent/'predictions.npz')==r['predictions_sha256'],'predictions hash')
            with np.load(p.parent/'predictions.npz',allow_pickle=False) as a:
                need(float(np.mean(a['teacher_val_logits'].argmax(1)==a['validation_labels']))==r['teacher_val_acc'],'teacher validation reconstruction')
                if r['eligible']:
                    need(filehash(p.parent/'PRETEST_FREEZE.json')==r['freeze_sha256'],'freeze hash')
                    f=json.loads((p.parent/'PRETEST_FREEZE.json').read_text()); need(f['test_metrics_computed'] is False,'freeze test flag')
                    for name,x in r['arms'].items():
                        need({k:v for k,v in x.items() if k!='test_acc'}==f['arms'][name],'endpoint changed after freeze')
                        need(float(np.mean(a[name+'_test_logits'].argmax(1)==a['test_labels']))==x['test_acc'],'test correctness')
                        need(float(np.mean(a[name+'_val_logits'].argmax(1)==a['validation_labels']))==x['val_acc'],'validation correctness')
        a=np.asarray([math.log2(r['arms']['joint_pair']['hold_nmse']/r['arms']['local_pair']['hold_nmse']) for r in good]); b=np.asarray([r['arms']['joint_pair']['test_acc']-r['arms']['local_pair']['test_acc'] for r in good]); rng=np.random.default_rng(20260925)
        stats=[]
        for v in [a,b]:
            means=v[rng.integers(0,len(v),(100000,len(v)))].mean(axis=1); stats.append({'mean':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()})
        passes={'nmse_improvement':stats[0]['ci95'][1]<0,'test_noninferiority':stats[1]['ci95'][0]>-.03}
        d.update({'decision':'SMALLVIT_OBJECTIVE_CONTROL_EXPLORATORY_PASS' if all(passes.values()) else 'SMALLVIT_OBJECTIVE_CONTROL_UNCERTAIN','eligible':len(good),'ineligible_seeds':[r['seed'] for r in rows if not r['eligible']],'paired_log2_nmse_ratio':stats[0],'geometric_joint_over_local_nmse':float(2**a.mean()),'joint_minus_local_test_accuracy':stats[1],'gates':passes,'descriptive_arm_means':{name:{key:float(np.mean([r['arms'][name][key] for r in good])) for key in ['parameters','hold_nmse','val_acc','test_acc']} for name in ARMS},'bootstrap':{'resamples':100000,'seed':20260925},'test_and_prediction_audit':'PASS','scope':'One reused digits split, eligible model seeds, fixed width8 grammar; objective/input-distribution bundle intervention, not unique mechanism or new budget frontier.'})
    except (ValueError,KeyError,FileNotFoundError) as exc:
        d.update({'decision':'STOP_INTEGRITY_OR_COVERAGE','errors':[str(exc)]})
    out=Path(out); out.mkdir(parents=True,exist_ok=True); write(out/'DECISION.json',d); write(out/'FRESH_ROWS.json',rows); print(json.dumps(d,indent=2)); return d

def selftest(out):
    b=load_base(); torch=b.torch; b.set_seed(VERIFY)
    X=torch.randn(64,17,32); M=X+.1*torch.randn_like(X); Y=M+.1*torch.randn_like(X)
    models,initial,info=fit_arms(b,VERIFY,X,M,Y,epochs=1)
    need(info['per_parameter_updates']==1 and {k:b.count_params(v) for k,v in models.items()}==COUNTS,'synthetic update/budget')
    cert=Path(out)/'guard.json'; cert.parent.mkdir(parents=True,exist_ok=True); gate=TestGate(lambda *a:None,cert)
    rejected=False
    try: gate.evaluate(None,None)
    except ValueError: rejected=True
    need(rejected,'test before freeze accepted'); write(cert,{'test':False}); gate.arm(); gate.evaluate(None,None); write(cert,{'test':True})
    rejected=False
    try: gate.evaluate(None,None)
    except ValueError: rejected=True
    need(rejected,'freeze modification accepted')
    seedrow={'seed':194000,'protocol_sha256':filehash(PROTOCOL),'runner_sha256':filehash(__file__),'eligible':True,'teacher_val_acc':.96,'test_calls_before_freeze':0,'test_calls_after_freeze':4,'freeze_sha256':'0'*64,'state':'COMPLETE','initial_states':initial,'training':{'per_parameter_updates':320,'batch_size':64,'schedule_shape':[40,512]},'arms':{n:{'parameters':COUNTS[n],'hold_nmse':.1,'hold_mse':.1,'hold_den':1.,'val_acc':.95,'test_acc':.94} for n in ARMS}}
    rows=[dict(copy.deepcopy(seedrow),seed=s) for s in SEEDS]; validate(rows)
    tests=[]
    for case in ['missing','duplicate','budget','initialization','test','nan','missing_arm']:
        bad=copy.deepcopy(rows)
        if case=='missing': bad.pop()
        elif case=='duplicate': bad[-1]=copy.deepcopy(bad[0])
        elif case=='budget': bad[0]['arms']['joint_pair']['parameters']=4904
        elif case=='initialization': bad[0]['initial_states']['joint0']='wrong'
        elif case=='test': bad[0]['test_calls_before_freeze']=1
        elif case=='nan': bad[0]['arms']['joint_pair']['hold_nmse']=float('nan')
        else: del bad[0]['arms']['joint_pair']
        try: validate(bad)
        except ValueError: tests.append(case)
        else: raise ValueError('mutation accepted: '+case)
    d={'status':'PASS','synthetic_paired_step':True,'test_guard_rejections':2,'mutation_rejections':tests,'fresh_seeds_used':0,'environment':environment(torch)}; write(Path(out)/'PREFLIGHT.json',d); print(json.dumps(d,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['seed','evaluate','selftest'],required=True); ap.add_argument('--seed',type=int); ap.add_argument('--input',type=Path); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.mode=='selftest': selftest(a.out); return 0
    if a.mode=='evaluate':
        d=evaluate(a.input,a.out); return 2 if d['decision']=='STOP_INTEGRITY_OR_COVERAGE' else 0
    try:
        d=run_seed(a.seed,a.out); print(json.dumps({k:d[k] for k in ['seed','eligible','state','teacher_val_acc']},indent=2)); return 0
    except Exception as exc:
        write(a.out/'ERROR.json',{'seed':a.seed,'error':str(exc),'traceback':traceback.format_exc(),'protocol_sha256':filehash(PROTOCOL),'runner_sha256':filehash(__file__)}); raise
if __name__=='__main__': raise SystemExit(main())
