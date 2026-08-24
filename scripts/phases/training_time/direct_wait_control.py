import sys,json
from dataclasses import dataclass,asdict
import torch
sys.path.insert(0,'/mnt/data'); import run_g7_training_consolidation_v2 as g; import base_realtext_v23 as b
sys.path.insert(0,'/mnt/data/g10_work'); from run_g10_inheritance import structured_compiler
sys.path.insert(0,'/mnt/data/g11_auto_canaria'); from canaria_controller import collect_calibration_split

torch.set_num_threads(1)
@dataclass
class Config:
    initial_check_epoch:int=4; check_interval:int=2; fit_epochs_per_check:int=3
    base_nmse_threshold:float=0.010; mid_nmse_threshold:float=0.0125; late_nmse_threshold:float=0.020
    mid_epoch:int=8; late_epoch:int=10; nfit:int=512; nhold:int=256; fit_lr:float=3e-3; fit_weight_decay:float=1e-5

def th(c,e): return c.late_nmse_threshold if e>=c.late_epoch else (c.mid_nmse_threshold if e>=c.mid_epoch else c.base_nmse_threshold)
class Pending:
    def __init__(self,m,seed,c):
        self.comp,self.inh=structured_compiler(m,2,24,seed); self.seed=seed; self.c=c; self.ep=0;self.updates=0
        self.opt=torch.optim.AdamW(self.comp.parameters(),lr=c.fit_lr,weight_decay=c.fit_weight_decay)
    def update(self,m,tr):
        X,Y,Xh,Yh=collect_calibration_split(m,tr,self.c.nfit,self.c.nhold); ds=torch.utils.data.TensorDataset(X,Y)
        den=float(((Yh-Yh.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12; hist=[]
        for j in range(self.c.fit_epochs_per_check):
            dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(self.seed+self.ep+j))
            for a,y in dl:
                self.opt.zero_grad();loss=((self.comp(a)-y)**2).mean();loss.backward();self.opt.step();self.updates+=1
            self.ep+=1
            with torch.no_grad(): mse=float(((self.comp(Xh)-Yh)**2).mean()); nmse=mse/den
            hist.append({'fit_epoch':self.ep,'hold_nmse':nmse,'hold_mse':mse})
        return hist

def run(seed,c=None):
    c=c or Config(); tr,va,te=b.datasets();m=g.new_model(4,48,seed);o=g.task_optimizer(m,g.epoch_lr(0));p=None;events=[];curve=[];next_check=c.initial_check_epoch;total=0
    for ep0 in range(12):
        g.set_lr(o,g.epoch_lr(ep0));tm=g.train_one_epoch(m,tr,o,seed+50000+ep0); ep=ep0+1; vm=b.tf_metrics(m,va);curve.append({'epoch':ep,'train':tm,'val':vm,'shape':g.shape(m)})
        if g.shape(m)['depth']<=2 or ep<next_check:continue
        if p is None:p=Pending(m,seed+930000,c)
        h=p.update(m,tr);total+=48; threshold=th(c,ep);passed=h[-1]['hold_nmse']<=threshold
        ev={'task_epoch':ep,'threshold':threshold,'history':h,'passed':passed,'pre_val':vm,'pending_fit_epochs':p.ep,'pending_updates':p.updates}
        if passed:
            st=g.named_opt_state(m,o);g.replace_inplace(m,p.comp);o=g.task_optimizer(m,g.epoch_lr(ep0));g.restore_named_state(m,o,st);ev['committed']=True;ev['post_val']=b.tf_metrics(m,va);ev['new_shape']=g.shape(m);p=None
        else:ev['committed']=False;next_check=ep+c.check_interval
        events.append(ev)
    return {'seed':seed,'protocol':'G13-direct-wait-exploratory-v1','config':asdict(c),'events':events,'curve':curve,'final_test':b.tf_metrics(m,te),'final_val':b.tf_metrics(m,va),'final_shape':g.shape(m),'final_params':b.nparams(m),'compiler_updates_total':total}
if __name__=='__main__':
    from pathlib import Path
    import time
    s=int(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);t=time.time();r=run(s);(out/f'seed_{s}.json').write_text(json.dumps(r,indent=2));print(s,round(time.time()-t,2),round(r['final_test']['ppl'],4),r['final_shape'],'updates',r['compiler_updates_total'],[(e['task_epoch'],e['committed'],round(e['history'][-1]['hold_nmse'],4)) for e in r['events']],flush=True)
