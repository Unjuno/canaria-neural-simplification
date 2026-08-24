import copy, sys, json
from dataclasses import dataclass, asdict
import torch
sys.path.insert(0,'/mnt/data')
import run_g7_training_consolidation_v2 as g
import base_realtext_v23 as b
sys.path.insert(0,'/mnt/data/g10_work')
from run_g10_inheritance import structured_compiler
from canaria_controller import collect_calibration_split

torch.set_num_threads(1)

@dataclass
class AutoConfig:
    initial_check_epoch:int=4
    min_dwell:int=2
    check_interval:int=2
    fit_epochs_per_check:int=3
    base_nmse_threshold:float=0.010
    mid_nmse_threshold:float=0.0125
    late_nmse_threshold:float=0.020
    mid_epoch:int=8
    late_epoch:int=10
    nfit:int=512
    nhold:int=256
    fit_lr:float=3e-3
    fit_weight_decay:float=1e-5

class Pending:
    def __init__(self,model,target_depth,target_mlp,seed,cfg):
        self.depth=target_depth;self.mlp=target_mlp;self.seed=seed;self.cfg=cfg
        self.comp,self.inherit=structured_compiler(model,target_depth,target_mlp,seed)
        self.opt=torch.optim.AdamW(self.comp.parameters(),lr=cfg.fit_lr,weight_decay=cfg.fit_weight_decay)
        self.total_fit_epochs=0;self.total_updates=0
    def update(self,teacher,tr,task_epoch):
        X,Y,Xh,Yh=collect_calibration_split(teacher,tr,self.cfg.nfit,self.cfg.nhold)
        ds=torch.utils.data.TensorDataset(X,Y)
        denom=float(((Yh-Yh.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
        hist=[]
        for local in range(self.cfg.fit_epochs_per_check):
            dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(self.seed+self.total_fit_epochs+local))
            for a,y in dl:
                self.opt.zero_grad();loss=((self.comp(a)-y)**2).mean();loss.backward();self.opt.step();self.total_updates+=1
            self.total_fit_epochs+=1
            with torch.no_grad():mse=float(((self.comp(Xh)-Yh)**2).mean());nmse=mse/denom
            hist.append({'total_fit_epoch':self.total_fit_epochs,'hold_mse':mse,'hold_nmse':nmse})
        return hist

def threshold(cfg,ep):
    if ep>=cfg.late_epoch:return cfg.late_nmse_threshold
    if ep>=cfg.mid_epoch:return cfg.mid_nmse_threshold
    return cfg.base_nmse_threshold

def next_shape(model):
    d=g.shape(model)['depth']
    return (3,36) if d==4 else ((2,24) if d==3 else None)

def run(seed,cfg=None):
    cfg=cfg or AutoConfig();tr,va,te=b.datasets();m=g.new_model(4,48,seed);opt=g.task_optimizer(m,g.epoch_lr(0))
    events=[];curve=[];pending=None;last_commit=0;next_check=cfg.initial_check_epoch
    total_comp_updates=0
    for ep0 in range(12):
        g.set_lr(opt,g.epoch_lr(ep0));tm=g.train_one_epoch(m,tr,opt,seed+50000+ep0);ep=ep0+1;vm=b.tf_metrics(m,va)
        curve.append({'epoch':ep,'train':tm,'val':vm,'shape':g.shape(m)})
        if g.shape(m)['depth']<=2: continue
        if ep < next_check or ep-last_commit < cfg.min_dwell: continue
        shp=next_shape(m)
        if shp is None:continue
        if pending is None or (pending.depth,pending.mlp)!=shp:
            pending=Pending(m,shp[0],shp[1],seed+800000+ep*100+shp[0],cfg)
        hist=pending.update(m,tr,ep);total_comp_updates+=cfg.fit_epochs_per_check*16
        th=threshold(cfg,ep);passed=hist[-1]['hold_nmse']<=th
        ev={'task_epoch':ep,'source_shape':g.shape(m),'target_depth':shp[0],'target_mlp':shp[1],'threshold':th,'history':hist,
            'pending_total_fit_epochs':pending.total_fit_epochs,'pending_total_updates':pending.total_updates,'passed':passed,'pre_val':vm}
        if passed:
            state=g.named_opt_state(m,opt);g.replace_inplace(m,pending.comp);last_commit=ep;pending=None
            opt=g.task_optimizer(m,g.epoch_lr(ep0));g.restore_named_state(m,opt,state);ev['committed']=True;ev['post_val']=b.tf_metrics(m,va);next_check=ep+cfg.min_dwell
        else:
            ev['committed']=False;next_check=ep+cfg.check_interval
        events.append(ev)
    return {'seed':seed,'protocol':'G11-auto-canaria-v2-exploratory','config':asdict(cfg),'events':events,'curve':curve,'final_test':b.tf_metrics(m,te),'final_val':b.tf_metrics(m,va),
            'final_shape':g.shape(m),'final_params':b.nparams(m),'compiler_updates_total':total_comp_updates}
