import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import run_realtext_lm_v23 as R
import torch

def fit_comp(base,tr,seed,epochs=20):
    R.set_seed(seed+700024); comp=R.Compiler(mlp=24); X,Y=R.collect_core_io(base,tr,512)
    ds=torch.utils.data.TensorDataset(X,Y); g=torch.Generator().manual_seed(seed+700025)
    dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True,generator=g); opt=torch.optim.AdamW(comp.parameters(),lr=3e-3,weight_decay=1e-5)
    for _ in range(epochs):
        for a,b in dl:
            opt.zero_grad(); loss=((comp(a)-b)**2).mean(); loss.backward(); opt.step()
    return comp

def run(seed,outdir):
    R.set_seed(seed); tr,va,te=R.datasets(); base=R.DecoderLM(); R.train_lm(base,tr,12,seed=seed)
    vm=R.tf_metrics(base,va); tm=R.tf_metrics(base,te); rec={'seed':seed,'baseline_val':vm,'baseline_test':tm,'baseline_params':R.nparams(base)}
    if vm['ppl']>20.0 or vm['token_acc']<.20: rec['eligible']=False
    else:
        rec['eligible']=True; comp=fit_comp(base,tr,seed); cand=R.with_compiler(base,comp); rec['compiled_params']=R.nparams(cand); rec['param_reduction']=1-rec['compiled_params']/rec['baseline_params']
        c0=R.tf_metrics(cand,te); rec['tau0_compiled']=c0; rec['tau0_ppl_utility']=tm['ppl']/c0['ppl']; rec['tau0_generation']=R.generation_agreement(cand,base,te)
        ctrl=copy.deepcopy(base); R.train_lm(ctrl,tr,8,lr=7e-4,seed=seed+10008); ct=R.tf_metrics(ctrl,te)
        cm=copy.deepcopy(cand); R.train_lm(cm,tr,8,lr=3e-4,trainable=[cm.tok,cm.pos,cm.norm,cm.lm_head,cm.compiler],seed=seed+30008); mt=R.tf_metrics(cm,te)
        rec['tau8_control']=ct; rec['tau8_compiled']=mt; rec['tau8_ppl_utility']=ct['ppl']/mt['ppl']; rec['tau8_generation']=R.generation_agreement(cm,ctrl,te)
    Path(outdir).mkdir(parents=True,exist_ok=True); (Path(outdir)/f'seed_{seed}.json').write_text(json.dumps(rec,indent=2)); print(json.dumps(rec,indent=2))

if __name__=='__main__':
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); run(a.seed,a.out)
