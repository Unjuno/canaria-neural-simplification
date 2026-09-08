import os,argparse,importlib.util,struct,itertools
from copy import deepcopy
import numpy as np,pandas as pd,torch
ABP='/mnt/data/canaria_v19_head/canaria_v19_head_compression/scripts/run_phaseAB_conv3q4_sparsehead_v19.py'
spec=importlib.util.spec_from_file_location('AB',ABP);AB=importlib.util.module_from_spec(spec);spec.loader.exec_module(AB)
A=AB.A;A2=importlib.util.module_from_spec(importlib.util.spec_from_file_location('A2','/mnt/data/canaria_v19_head/canaria_v19_head_compression/scripts/run_phaseAA2_head_2to4_norefit_v19.py')); A2.__spec__.loader.exec_module(A2)
Q=AB.Q;Z=AB.Z;X=AB.X;OUT='/mnt/data/canaria_v19_head/canaria_v19_head_compression/raw_AD';os.makedirs(OUT,exist_ok=True)
ALPHAS=Q.ALPHAS; COMB=list(itertools.combinations(range(4),2)); C2I={c:i for i,c in enumerate(COMB)}

def q_tensor(x,bits=4):
 x=x.detach().cpu().float();qmax=2**(bits-1)-1;mx=float(x.abs().max());base=mx/qmax if mx>1e-12 else 1.;best=None
 for a in ALPHAS:
  sc=np.float16(base*a);sf=float(np.float32(sc));q=torch.clamp(torch.round(x/sf),-qmax,qmax).to(torch.int8);xr=q.float()*sf;mse=float(((xr-x)**2).mean())
  if best is None or mse<best[0]:best=(mse,sc,q)
 return best[1],best[2]
def q_rows(W,bits=4,mask=None):
 W=W.detach().cpu().float();qmax=2**(bits-1)-1; qs=[];scales=[]
 for o in range(W.shape[0]):
  vals=W[o].reshape(-1) if mask is None else W[o][mask[o]]
  mx=float(vals.abs().max());base=mx/qmax if mx>1e-12 else 1.;best=None
  for a in ALPHAS:
   sc=np.float16(base*a);sf=float(np.float32(sc));q=torch.clamp(torch.round(vals/sf),-qmax,qmax).to(torch.int8);xr=q.float()*sf;mse=float(((xr-vals)**2).mean())
   if best is None or mse<best[0]:best=(mse,sc,q)
  scales.append(best[1]);qs.append(best[2])
 return np.asarray(scales,dtype=np.float16),qs

def pack_q4(vals):
 a=np.asarray(vals,dtype=np.int8).reshape(-1);out=bytearray()
 for i in range(0,len(a),2):
  lo=int(a[i]) & 0xF; hi=(int(a[i+1]) & 0xF) if i+1<len(a) else 0;out.append(lo|(hi<<4))
 return bytes(out)
def unpack_q4(buf,n):
 out=np.empty(n,dtype=np.int8)
 for i in range(n):
  v=(buf[i//2]>>(4*(i%2)))&0xF;out[i]=v-16 if v>=8 else v
 return out

def pack3(codes):
 out=bytearray();acc=0;nb=0
 for c in codes:
  acc|=(int(c)&7)<<nb;nb+=3
  while nb>=8:out.append(acc&255);acc>>=8;nb-=8
 if nb:out.append(acc&255)
 return bytes(out)
def unpack3(buf,n):
 out=[];acc=0;nb=0;j=0
 for _ in range(n):
  while nb<3:acc|=buf[j]<<nb;j+=1;nb+=8
  out.append(acc&7);acc>>=3;nb-=3
 return out

def enc_core(conv):
 sw,qw=q_tensor(conv.weight);sb,qb=q_tensor(conv.bias);return sw.tobytes()+pack_q4(qw.numpy())+sb.tobytes()+pack_q4(qb.numpy())
def dec_core(conv,b,off):
 sw=np.frombuffer(b[off:off+2],dtype=np.float16)[0];off+=2;qw=unpack_q4(b[off:off+288],576);off+=288;sb=np.frombuffer(b[off:off+2],dtype=np.float16)[0];off+=2;qb=unpack_q4(b[off:off+4],8);off+=4
 with torch.no_grad():conv.weight.copy_(torch.tensor(qw,dtype=torch.float32).reshape_as(conv.weight)*float(sw));conv.bias.copy_(torch.tensor(qb,dtype=torch.float32)*float(sb))
 return off

def enc_dense(mod):
 sc,qs=q_rows(mod.weight); raw=sc.tobytes()+pack_q4(np.concatenate([q.numpy() for q in qs])); raw+=mod.bias.detach().cpu().numpy().astype(np.float16).tobytes();return raw
def dec_dense(mod,b,off):
 o=mod.weight.shape[0];n=mod.weight.numel();sc=np.frombuffer(b[off:off+2*o],dtype=np.float16).copy();off+=2*o;q=unpack_q4(b[off:off+(n+1)//2],n);off+=(n+1)//2;bias=np.frombuffer(b[off:off+2*mod.bias.numel()],dtype=np.float16).copy();off+=2*mod.bias.numel();q=q.reshape(o,-1).astype(np.float32);W=q*sc.astype(np.float32)[:,None]
 with torch.no_grad():mod.weight.copy_(torch.tensor(W).reshape_as(mod.weight));mod.bias.copy_(torch.tensor(bias.astype(np.float32)))
 return off

def enc_sparse_fc(fc,mask):
 codes=[];vals=[]
 for o in range(48):
  for g in range(128):
   pos=tuple(torch.where(mask[o,g*4:(g+1)*4])[0].tolist());codes.append(C2I[pos])
 sc,qs=q_rows(fc.weight,4,mask);vals=np.concatenate([q.numpy() for q in qs])
 return pack3(codes)+sc.tobytes()+pack_q4(vals)+fc.bias.detach().cpu().numpy().astype(np.float16).tobytes()
def dec_sparse_fc(fc,b,off):
 ngrp=48*128; plen=(ngrp*3+7)//8;codes=unpack3(b[off:off+plen],ngrp);off+=plen;sc=np.frombuffer(b[off:off+96],dtype=np.float16).copy();off+=96;nvals=48*256;q=unpack_q4(b[off:off+nvals//2],nvals);off+=nvals//2;bias=np.frombuffer(b[off:off+96],dtype=np.float16).copy();off+=96
 W=np.zeros((48,512),np.float32);k=0;c=0
 for o in range(48):
  for g in range(128):
   pos=COMB[codes[c]];c+=1
   for p in pos:W[o,g*4+p]=float(q[k])*float(sc[o]);k+=1
 with torch.no_grad():fc.weight.copy_(torch.tensor(W));fc.bias.copy_(torch.tensor(bias.astype(np.float32)))
 return off

def encode(compiled):
 sparse,mask=A2.make_sparse(compiled);parts=[b'V1'];parts.append(enc_core(sparse.blocks[0].conv));parts.append(enc_dense(sparse.stem[0]));parts.append(enc_dense(sparse.b_in[0]));parts.append(enc_dense(sparse.b_out[0]));parts.append(enc_sparse_fc(sparse.head[1],mask));parts.append(enc_dense(sparse.head[3]));return b''.join(parts),sparse

def decode(template,buf):
 m=deepcopy(template);off=2;off=dec_core(m.blocks[0].conv,buf,off);off=dec_dense(m.stem[0],buf,off);off=dec_dense(m.b_in[0],buf,off);off=dec_dense(m.b_out[0],buf,off);off=dec_sparse_fc(m.head[1],buf,off);off=dec_dense(m.head[3],buf,off);assert off==len(buf),(off,len(buf));return m

def run(seed):
 od=os.path.join(OUT,f'seed_{seed}');os.makedirs(od,exist_ok=True);built=AB.build(seed)
 if built is None:print('INELIGIBLE',seed);return
 base,data,control,compiled=built;Xtr,ytr,Xv,yv=data;Xa=X.make_aug(Xv,123);buf,sparse=encode(compiled);open(os.path.join(od,'model.bin'),'wb').write(buf);dec=decode(sparse,buf)
 a=X.acc(dec,Xa,yv); logits1=dec(Xv[:32]).detach(); dec2=decode(sparse,buf);logits2=dec2(Xv[:32]).detach();maxdiff=float((logits1-logits2).abs().max())
 row=dict(seed=seed,bytes=len(buf),aug_acc=a,roundtrip_logit_maxdiff=maxdiff,roundtrip_exact=bool(maxdiff==0.0));pd.DataFrame([row]).to_csv(os.path.join(od,'codec.csv'),index=False);print('DONE',seed,row,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();run(a.seed)
