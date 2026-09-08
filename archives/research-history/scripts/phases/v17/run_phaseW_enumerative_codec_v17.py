import os, math, argparse, importlib.util, pandas as pd
SRC='/mnt/data/canaria_v17_work/scripts/run_phaseV_ternary_pack_v17.py'
spec=importlib.util.spec_from_file_location('V',SRC);V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
IN='/mnt/data/canaria_v17_work/raw_experiments/phaseV_ternary_pack_v17'
OUT='/mnt/data/canaria_v17_work/raw_experiments/phaseW_enumerative_codec_v17';os.makedirs(OUT,exist_ok=True)
N=152

def rank_comb(pos):
    return sum(math.comb(p,i+1) for i,p in enumerate(pos))
def unrank_comb(rank,k,n=N):
    pos=[0]*k; r=rank; x=n-1
    for i in range(k,0,-1):
        while math.comb(x,i)>r:x-=1
        pos[i-1]=x; r-=math.comb(x,i); x-=1
    return pos

def pack_bits(vals):
    x=0
    for i,v in enumerate(vals):x|=(int(v)&1)<<i
    return x.to_bytes((len(vals)+7)//8,'little')
def unpack_bits(buf,n):
    x=int.from_bytes(buf,'little');return [(x>>i)&1 for i in range(n)]

def encode(pattern5,trits,scale2):
    nz=[i for i,t in enumerate(trits) if t!=0];k=len(nz);rank=rank_comb(nz)
    comb_n=math.comb(N,k); rank_bits=0 if comb_n<=1 else math.ceil(math.log2(comb_n)); rank_bytes=(rank_bits+7)//8
    signs=[1 if trits[i]>0 else 0 for i in nz]; sb=pack_bits(signs)
    out=bytearray(pattern5);out.append(k);out.extend(rank.to_bytes(rank_bytes,'little'));out.extend(sb);out.extend(scale2)
    return bytes(out),rank_bytes,len(sb),k

def decode(buf):
    pattern=buf[:5];k=buf[5];comb_n=math.comb(N,k);rank_bits=0 if comb_n<=1 else math.ceil(math.log2(comb_n));rank_bytes=(rank_bits+7)//8;sign_bytes=(k+7)//8
    p=6; rank=int.from_bytes(buf[p:p+rank_bytes],'little');p+=rank_bytes; signs=unpack_bits(buf[p:p+sign_bytes],k);p+=sign_bytes;scale=buf[p:p+2];p+=2
    assert p==len(buf);nz=unrank_comb(rank,k);tr=[0]*N
    for j,i in enumerate(nz):tr[i]=1 if signs[j] else -1
    return pattern,tr,scale

def run(seed):
    raw=open(os.path.join(IN,f'seed_{seed}','core.bin'),'rb').read();assert len(raw)==38
    pattern=raw[:5];trits=V.unpack_trits(raw[5:36],152);scale=raw[36:38]
    enc,rb,sb,k=encode(pattern,trits,scale);p2,t2,s2=decode(enc);assert p2==pattern and t2==trits and s2==scale
    od=os.path.join(OUT,f'seed_{seed}');os.makedirs(od,exist_ok=True);open(os.path.join(od,'core_enum.bin'),'wb').write(enc)
    row=dict(seed=seed,original_fixed_ternary_bytes=38,enumerative_bytes=len(enc),nonzero_count=k,zero_fraction=1-k/N,rank_bytes=rb,sign_bytes=sb,saving_fraction=1-len(enc)/38)
    pd.DataFrame([row]).to_csv(os.path.join(od,'results.csv'),index=False);print('DONE',seed,'bytes',len(enc),'k',k,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();run(a.seed)
