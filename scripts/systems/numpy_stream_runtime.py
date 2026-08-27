from __future__ import annotations
import argparse,json,math,statistics,sys,time
from pathlib import Path
import numpy as np
from scipy.special import erf
EPS=np.float32(1e-5); HEADS=4; HEAD_DIM=6; REPEATS=10
def load_block(raw_root,mapping,idx):
    bdir=raw_root/f'block{idx}'; return {k:np.load(bdir/fn,mmap_mode='r',allow_pickle=False) for k,fn in mapping.items()}
def layernorm(x,w,b):
    mean=x.mean(axis=-1,keepdims=True,dtype=np.float32); var=((x-mean)**2).mean(axis=-1,keepdims=True,dtype=np.float32); return ((x-mean)/np.sqrt(var+EPS))*w+b
def linear(x,w,b): return x@w.T+b
def gelu(x): return np.float32(.5)*x*(np.float32(1.)+erf(x/np.float32(math.sqrt(2.))))
def causal_attention(z,state):
    qkv=linear(z,state['attn.in_proj_weight'],state['attn.in_proj_bias']); q,k,v=np.split(qkv,3,axis=-1); bsz,t,d=q.shape
    q=q.reshape(bsz,t,HEADS,HEAD_DIM).transpose(0,2,1,3); k=k.reshape(bsz,t,HEADS,HEAD_DIM).transpose(0,2,1,3); v=v.reshape(bsz,t,HEADS,HEAD_DIM).transpose(0,2,1,3)
    scores=(q@k.swapaxes(-1,-2))/np.float32(math.sqrt(HEAD_DIM)); mask=np.triu(np.ones((t,t),dtype=bool),k=1); scores[...,mask]=-np.inf; mx=scores.max(axis=-1,keepdims=True); ex=np.exp(scores-mx); probs=ex/ex.sum(axis=-1,keepdims=True,dtype=np.float32); ctx=probs@v; ctx=ctx.transpose(0,2,1,3).reshape(bsz,t,d); return linear(ctx,state['attn.out_proj.weight'],state['attn.out_proj.bias'])
def run_block(x,state):
    z=layernorm(x,state['n1.weight'],state['n1.bias']); a=causal_attention(z,state); x=x+a; z2=layernorm(x,state['n2.weight'],state['n2.bias']); h=linear(z2,state['mlp.0.weight'],state['mlp.0.bias']); h=gelu(h); h=linear(h,state['mlp.2.weight'],state['mlp.2.bias']); return x+h
def execute_streaming(root,manifest):
    raw_root=root/'raw'; x0=np.load(root/'input.npy',allow_pickle=False).astype(np.float32,copy=False); out=None; times=[]
    for _ in range(REPEATS):
        x=x0.copy(); t0=time.perf_counter()
        for i in range(2):
            state=load_block(raw_root,manifest['key_files'][i],i); x=run_block(x,state).astype(np.float32,copy=False); del state
        times.append(time.perf_counter()-t0); out=x
    return out,times
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); args=ap.parse_args()
    if 'torch' in sys.modules: raise SystemExit('torch unexpectedly imported before NumPy runtime execution')
    manifest=json.loads((args.artifact_dir/'manifest.json').read_text()); out,times=execute_streaming(args.artifact_dir,manifest); torch_imported='torch' in sys.modules; ref=np.load(args.artifact_dir/'pytorch_reference.npy',allow_pickle=False); diff=out.astype(np.float64)-ref.astype(np.float64); max_abs=float(np.max(np.abs(diff))); rel_l2=float(np.linalg.norm(diff.ravel())/(np.linalg.norm(ref.astype(np.float64).ravel())+1e-30)); one_block=int(manifest['one_block_max_tensor_payload_bytes']); activation_bytes=int(np.prod(manifest['input_shape'])*np.dtype(np.float32).itemsize); score_scratch_bytes=int(manifest['input_shape'][0]*HEADS*manifest['input_shape'][1]*manifest['input_shape'][1]*4)
    checks={'torch_not_imported':not torch_imported,'max_abs_lte_5e_5':max_abs<=5e-5,'relative_l2_lte_1e_5':rel_l2<=1e-5,'two_blocks_executed':len(manifest['key_files'])==2,'one_block_payload_lte_14784':one_block<=14784}; report={'experiment':'Canaria Systems S4 framework-independent learned compiler streaming','status':'PASS' if all(checks.values()) else 'FAIL','runtime':'NumPy/SciPy only; no torch import in execution child','torch_imported_in_execution_child':torch_imported,'max_absolute_output_difference':max_abs,'relative_l2_output_difference':rel_l2,'output_sum_numpy':float(out.sum()),'output_sum_pytorch_reference':float(ref.sum()),'one_block_learned_tensor_payload_bytes':one_block,'two_block_learned_tensor_payload_bytes':int(sum(manifest['block_tensor_payload_bytes'])),'raw_npy_serialized_bytes':int(sum(manifest['block_raw_npy_serialized_bytes'])),'input_activation_bytes':activation_bytes,'attention_score_scratch_bytes':score_scratch_bytes,'median_execution_seconds':float(statistics.median(times)),'execution_seconds':[float(x) for x in times],'checks':checks,'boundary':'Fixed synthetic activation kernel-equivalence test using actual learned G7 compiler weights. Not task evidence or physical-device deployment evidence.'}; args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report))
if __name__=='__main__': main()
