import os, argparse, importlib.util, pandas as pd, torch, json, struct, numpy as np
SRC='/mnt/data/canaria_v17_work/scripts/run_phaseT_pattern_share_v17.py'
spec=importlib.util.spec_from_file_location('T',SRC);T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
OUT='/mnt/data/canaria_v17_work/raw_experiments/phaseV_ternary_pack_v17';os.makedirs(OUT,exist_ok=True)

def pack_trits(vals):
    d=[int(v)+1 for v in vals];out=bytearray()
    for i in range(0,len(d),5):
        chunk=d[i:i+5]; x=0; mul=1
        for z in chunk:x+=z*mul;mul*=3
        out.append(x)
    return bytes(out)
def unpack_trits(buf,n):
    d=[]
    for x in buf:
        for _ in range(5):d.append((x%3)-1);x//=3
    return d[:n]
def pack_pattern(indices):
    x=0
    for i,v in enumerate(indices):x|=(int(v)&3)<<(2*i)
    return x.to_bytes(5,'little')
def unpack_pattern(buf,n=18):
    x=int.from_bytes(buf,'little');return [int((x>>(2*i))&3) for i in range(n)]
def encode(Wq,mask,scale):
    idx=[]; tr=[]
    for g in range(18):
        loc=torch.where(mask[0,g*4:(g+1)*4])[0]; assert len(loc)==1;idx.append(int(loc[0]))
    for o in range(8):
        for g in range(18):tr.append(int(round(float(Wq[o,g*4+idx[g]]/scale))))
        tr.append(int(round(float(Wq[o,72]/scale))))
    assert len(tr)==152 and all(z in (-1,0,1) for z in tr)
    p=pack_pattern(idx); t=pack_trits(tr); s=np.float16(scale).tobytes()
    return p+t+s,idx,tr
def decode(buf):
    p=buf[:5];t=buf[5:36];s=buf[36:38];assert len(buf)==38
    idx=unpack_pattern(p);tr=unpack_trits(t,152);scale=float(np.frombuffer(s,dtype=np.float16)[0]);W=torch.zeros((8,73));k=0
    for o in range(8):
        for g in range(18):W[o,g*4+idx[g]]=tr[k]*scale;k+=1
        W[o,72]=tr[k]*scale;k+=1
    return W,scale,idx,tr

def run(seed):
    base,data=T.P.train_base(seed);Xtr,ytr,Xv,yv=data;Xa=T.P.make_aug(Xv,123);bc=T.P.acc(base,Xv,yv);ba=T.P.acc(base,Xa,yv);od=os.path.join(OUT,f'seed_{seed}');os.makedirs(od,exist_ok=True)
    if bc<.95:print('INELIGIBLE',seed,bc);return
    with torch.no_grad():Z=T.P.span_input(base,Xtr[:T.P.CALN]);Y=T.P.span_output(base,Z)
    A=T.P.design(Z);B=T.P.targets(Y);ref=T.P.fit_full(Z,Y);W=T.P.core_to_matrix(ref);mask,pbits=T.shared_nm1_mask(W,8);Wr=T.P.refit_mask(A,B,mask);Q,scales=T.P.quant_group_fp16(Wr,A,B,1,2);scale=scales[0]
    buf,idx,tr=encode(Q,mask,scale);Q2,sc2,idx2,tr2=decode(buf);maxdiff=float((Q-Q2).abs().max());assert idx==idx2 and tr==tr2 and maxdiff==0.0
    c1=T.P.matrix_to_core(Q);c2=T.P.matrix_to_core(Q2);a1=T.P.acc(T.P.replace_full(base,c1),Xa,yv);a2=T.P.acc(T.P.replace_full(base,c2),Xa,yv);assert a1==a2
    row=dict(seed=seed,bytes=len(buf),ternary_bytes=31,pattern_bytes=5,scale_bytes=2,max_abs_decode_error=maxdiff,aug_original=a1,aug_decoded=a2,utility=a2/(ba+1e-12),zero_fraction=sum(z==0 for z in tr)/len(tr))
    pd.DataFrame([row]).to_csv(os.path.join(od,'results.csv'),index=False);open(os.path.join(od,'core.bin'),'wb').write(buf);json.dump({'seed':seed,'eligible':True,'baseline_clean':bc},open(os.path.join(od,'meta.json'),'w'),indent=2);print('DONE',seed,'bytes',len(buf),'zero',round(row['zero_fraction'],3),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();run(a.seed)
