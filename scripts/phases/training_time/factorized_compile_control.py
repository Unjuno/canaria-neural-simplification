import copy,sys,json,time
from pathlib import Path
import torch
sys.path.insert(0,'/mnt/data'); import run_g7_training_consolidation_v2 as g; import base_realtext_v23 as b
sys.path.insert(0,'/mnt/data/g10_work'); from run_g10_inheritance import structured_compiler, fit_given_init

torch.set_num_threads(1)

def fit_hybrid(m,tr,d,mlp,epochs,seed):
    c,inh=structured_compiler(m,d,mlp,seed); c,fit=fit_given_init(m,tr,c,epochs,seed+10000,'hybrid'); return c,{'inheritance':inh,'fit':fit}

def finish(m,state,tr,va,te,seed):
    m,o,curve=g.continue_from(m,state,tr,va,4,12,seed); return {'curve':curve,'final':g.eval_model(m,va,te)}

def run(seed):
    tr,va,te=b.datasets(); m=g.new_model(4,48,seed);o=g.task_optimizer(m,g.epoch_lr(0));snap=state=None;large_curve=[]
    for ep in range(12):
        g.set_lr(o,g.epoch_lr(ep));tm=g.train_one_epoch(m,tr,o,seed+50000+ep);large_curve.append({'epoch':ep+1,'train':tm,'val':b.tf_metrics(m,va)})
        if ep+1==4:snap=copy.deepcopy(m);state=g.named_opt_state(m,o)
    # Direct: 4->2, 12 compiler fit epochs = 192 updates.
    dm=copy.deepcopy(snap);dc,df=fit_hybrid(dm,tr,2,24,12,seed+960000);g.replace_inplace(dm,dc);dtau=g.eval_model(dm,va,te);dres=finish(dm,state,tr,va,te,seed)
    # Factorized: 4->3 (6 epochs) then immediately 3->2 (6 epochs), no task learning between; same 192 updates.
    fm=copy.deepcopy(snap);c1,f1=fit_hybrid(fm,tr,3,36,6,seed+970000);g.replace_inplace(fm,c1);mid=g.eval_model(fm,va,te)
    c2,f2=fit_hybrid(fm,tr,2,24,6,seed+980000);g.replace_inplace(fm,c2);ftau=g.eval_model(fm,va,te);fres=finish(fm,state,tr,va,te,seed)
    return {'seed':seed,'protocol':'G16-factorized-compilation-exploratory-v1','large':{'final':g.eval_model(m,va,te),'curve':large_curve},
      'direct':{'compiler_updates':192,'after_compile':dtau,'fit':df,**dres},
      'factorized':{'compiler_updates':192,'after_first':mid,'after_compile':ftau,'fit1':f1,'fit2':f2,**fres}}
if __name__=='__main__':
    s=int(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);t=time.time();r=run(s);(out/f'seed_{s}.json').write_text(json.dumps(r,indent=2));print(s,round(time.time()-t,2),'direct',round(r['direct']['final']['test']['ppl'],4),'factor',round(r['factorized']['final']['test']['ppl'],4),'large',round(r['large']['final']['test']['ppl'],4),flush=True)
