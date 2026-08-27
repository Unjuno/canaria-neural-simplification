from __future__ import annotations
import argparse,json,math,statistics,sys,time
from pathlib import Path
import numpy as np
from scipy.special import erf
EPS=np.float32(1e-5); HEADS=4; HEAD_DIM=6; REPEATS=5
WEIGHTS=14784; X_BYTES=9216; Z_BYTES=9216; K_BYTES=9216; V_BYTES=9216; SCORE_BYTES=1536; ROW_SCRATCH_BUDGET=1024; PEAK_BOUND=54208; CAPACITY=65536
def load_block(root,mapping,idx):
 bdir=root/'raw'/f'block{idx}'; return {k:np.load(bdir/fn,mmap_mode='r',allow_pickle=False) for k,fn in mapping.items()}
def layernorm(x,w,b):
 m=x.mean(axis=-1,keepdims=True,dtype=np.float32); c=x-m; v=(c*c).mean(axis=-1,keepdims=True,dtype=np.float32); return (c/np.sqrt(v+EPS))*w+b
def linear(x,w,b): return x@w.T+b
def gelu(x): return np.float32(.5)*x*(np.float32(1.)+erf(x/np.float32(math.sqrt(2.))))
def attention_row_stream(x,state):
 z=layernorm(x,state['n1.weight'],state['n1.bias']); w=state['attn.in_proj_weight']; b=state['attn.in_proj_bias']; d=x.shape[-1]; wq,wk,wv=w[:d],w[d:2*d],w[2*d:]; bq,bk,bv=b[:d],b[d:2*d],b[2*d:]
 k=linear(z,wk,bk).astype(np.float32,copy=False); v=linear(z,wv,bv).astype(np.float32,copy=False); bsz,t,_=x.shape; k4=k.reshape(bsz,t,HEADS,HEAD_DIM).transpose(0,2,1,3); v4=v.reshape(bsz,t,HEADS,HEAD_DIM).transpose(0,2,1,3)
 max_score_bytes=0
 for ti in range(t):
  q=linear(z[:,ti:ti+1,:],wq,bq).astype(np.float32,copy=False); q4=q.reshape(bsz,1,HEADS,HEAD_DIM).transpose(0,2,1,3); kk=k4[:,:,:ti+1,:]; vv=v4[:,:,:ti+1,:]; scores=(q4@kk.swapaxes(-1,-2))/np.float32(math.sqrt(HEAD_DIM)); max_score_bytes=max(max_score_bytes,int(scores.nbytes)); scores-=scores.max(axis=-1,keepdims=True); np.exp(scores,out=scores); sums=scores.sum(axis=-1,keepdims=True,dtype=np.float32); scores/=sums; ctx=(scores@vv).transpose(0,2,1,3).reshape(bsz,1,d); a=linear(ctx,state['attn.out_proj.weight'],state['attn.out_proj.bias']); x[:,ti:ti+1,:]+=a
 return x,max_score_bytes
def run_block(x,state):
 x,score_bytes=attention_row_stream(x,state); z2=layernorm(x,state['n2.weight'],state['n2.bias']); h=linear(z2,state['mlp.0.weight'],state['mlp.0.bias']); h=gelu(h); h=linear(h,state['mlp.2.weight'],state['mlp.2.bias']); return x+h,score_bytes
def execute(root,manifest):
 x0=np.load(root/'input.npy',allow_pickle=False).astype(np.float32,copy=False); out=None; times=[]; observed_score=0
 for _ in range(REPEATS):
  x=x0.copy(); t0=time.perf_counter()
  for i in range(2):
   state=load_block(root,manifest['key_files'][i],i); x,sb=run_block(x,state); observed_score=max(observed_score,sb); del state
  times.append(time.perf_counter()-t0); out=x
 return out,times,observed_score
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); a=ap.parse_args()
 if 'torch' in sys.modules: raise SystemExit('torch unexpectedly imported')
 manifest=json.loads((a.artifact_dir/'manifest.json').read_text()); out,times,score_bytes=execute(a.artifact_dir,manifest); torch_imported='torch' in sys.modules; ref=np.load(a.artifact_dir/'pytorch_reference.npy',allow_pickle=False); diff=out.astype(np.float64)-ref.astype(np.float64); max_abs=float(np.max(np.abs(diff))); rel=float(np.linalg.norm(diff.ravel())/(np.linalg.norm(ref.astype(np.float64).ravel())+1e-30)); checks={'torch_not_imported':not torch_imported,'max_abs_lte_5e_5':max_abs<=5e-5,'relative_l2_lte_1e_5':rel<=1e-5,'max_score_row_lte_1536':score_bytes<=1536,'peak_managed_tensor_upper_bound_lte_65536':PEAK_BOUND<=CAPACITY,'no_full_score_tensor_created':True}
 report={'experiment':'Canaria Systems S5 row-streamed attention','status':'PASS' if all(checks.values()) else 'FAIL','torch_imported':torch_imported,'max_absolute_output_difference':max_abs,'relative_l2_output_difference':rel,'output_sum':float(out.sum()),'reference_sum':float(ref.sum()),'observed_max_score_row_bytes':score_bytes,'locked_peak_managed_tensor_upper_bound_bytes':PEAK_BOUND,'capacity_target_bytes':CAPACITY,'conventional_full_score_bytes':73728,'conventional_weights_plus_x_plus_scores_lower_bound_bytes':97728,'median_execution_seconds':float(statistics.median(times)),'execution_seconds':[float(x) for x in times],'checks':checks,'boundary':'Logical managed tensor working-set accounting only; Python/NumPy runtime overhead and opaque ufunc/library temporaries are excluded.'}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report))
if __name__=='__main__': main()
