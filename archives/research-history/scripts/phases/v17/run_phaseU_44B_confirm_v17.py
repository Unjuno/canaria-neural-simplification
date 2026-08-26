import os, argparse, importlib.util, pandas as pd, torch, json
from copy import deepcopy
SRC='/mnt/data/canaria_v17_work/scripts/run_phaseT_pattern_share_v17.py'
spec=importlib.util.spec_from_file_location('T',SRC);T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
OUT='/mnt/data/canaria_v17_work/raw_experiments/phaseU_44B_confirm_v17';os.makedirs(OUT,exist_ok=True);TAU=8

def train8(m,X,y,seed):
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=7e-4,weight_decay=1e-4)
    for ep in range(1,9):
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,y),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed*10000+ep));m.train()
        for xb,yb in dl:opt.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(m(xb),yb).backward();opt.step()
    return m

def evalcond(base,W,A,B,Xa,yv,ba,refa,RX,RY,ca,group_size,seed):
    mask,pbits=T.shared_nm1_mask(W,group_size);Wr=T.P.refit_mask(A,B,mask);Q,_=T.P.quant_group_fp16(Wr,A,B,1,2);core=T.P.matrix_to_core(Q);pre=T.P.replace_full(base,core);prea=T.P.acc(pre,Xa,yv);bits=int(mask.sum())*2+pbits+16
    mm=T.P.set_full_shell(T.P.replace_full(base,core));mm=train8(mm,RX,RY,seed+70000);au=T.P.acc(mm,Xa,yv);u=au/(ca+1e-12)
    return dict(output_share_group=group_size,bytes=bits/8,pre_utility=prea/(ba+1e-12),pre_retention=prea/(refa+1e-12),repair_utility=u,pass95=u>=.95,repair_aug=au,control_aug=ca)

def run(seed):
    base,data=T.P.train_base(seed);Xtr,ytr,Xv,yv=data;Xa=T.P.make_aug(Xv,123);bc=T.P.acc(base,Xv,yv);ba=T.P.acc(base,Xa,yv);od=os.path.join(OUT,f'seed_{seed}');os.makedirs(od,exist_ok=True)
    if bc<.95:json.dump({'seed':seed,'eligible':False,'baseline_clean':bc},open(os.path.join(od,'meta.json'),'w'),indent=2);print('INELIGIBLE',seed,bc,flush=True);return
    with torch.no_grad():Z=T.P.span_input(base,Xtr[:T.P.CALN]);Y=T.P.span_output(base,Z)
    A=T.P.design(Z);B=T.P.targets(Y);ref=T.P.fit_full(Z,Y);W=T.P.core_to_matrix(ref);refm=T.P.replace_full(base,ref);refa=T.P.acc(refm,Xa,yv);RX,RY=T.P.repair_data(data,seed)
    ctrl=deepcopy(base)
    for p in ctrl.parameters():p.requires_grad_(True)
    ctrl=train8(ctrl,RX,RY,seed+70000);ca=T.P.acc(ctrl,Xa,yv)
    rows=[]
    for gs in [8,1]:
        row=evalcond(base,W,A,B,Xa,yv,ba,refa,RX,RY,ca,gs,seed);row['seed']=seed;rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(od,'results.csv'),index=False);json.dump({'seed':seed,'eligible':True,'baseline_clean':bc,'baseline_aug':ba,'reference_aug':refa,'control_aug_tau8':ca},open(os.path.join(od,'meta.json'),'w'),indent=2);print('DONE',seed,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();run(a.seed)
