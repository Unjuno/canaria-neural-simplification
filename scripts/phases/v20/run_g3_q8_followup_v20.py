import sys, json, copy, struct, zlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_vit_generalization_v20 as r


def q8_pack_and_dequantize(model):
    chunks=[]; deq={}
    for name,t in model.state_dict().items():
        a=t.detach().cpu().numpy()
        nameb=name.encode('utf-8')
        chunks.append(struct.pack('<H',len(nameb))+nameb)
        chunks.append(struct.pack('<B',a.ndim)+b''.join(struct.pack('<I',int(x)) for x in a.shape))
        if np.issubdtype(a.dtype,np.floating):
            mx=float(np.max(np.abs(a))) if a.size else 0.0
            scale=mx/127.0 if mx>0 else 1.0
            q=np.clip(np.rint(a/scale),-127,127).astype(np.int8)
            chunks.append(b'F'+struct.pack('<f',scale)+q.tobytes(order='C'))
            deq[name]=torch.from_numpy((q.astype(np.float32)*scale).reshape(a.shape)).to(t.dtype)
        else:
            raw=a.tobytes(order='C'); chunks.append(b'R'+struct.pack('<I',len(raw))+raw)
            deq[name]=t.detach().clone()
    raw=b''.join(chunks); comp=zlib.compress(raw,9)
    m=copy.deepcopy(model); m.load_state_dict(deq,strict=True)
    return len(raw),len(comp),m

def run(seed,outdir):
    r.set_seed(seed); tr,te=r.data_loaders()
    base=r.SmallViT(depth=4,d=32,heads=4,mlp=64)
    r.train_cls(base,tr,40,lr=2e-3,wd=1e-4,seed=seed)
    ba=r.acc(base,te)
    if ba<.95: raise RuntimeError('seed not eligible')
    comp=r.fit_compiler(base,tr,mlp_dim=32,depth=2,epochs=50,seed=seed)
    cand=r.clone_baseline_with_compiler(base,comp)
    ctrl=copy.deepcopy(base); r.train_cls(ctrl,tr,8,lr=8e-4,wd=1e-4,seed=seed+10008)
    cm=copy.deepcopy(cand); shell=[cm.patch,cm.norm,cm.head,cm.cls,cm.pos]
    r.train_cls(cm,tr,8,lr=8e-4,wd=1e-4,trainable=shell,seed=seed+20008)
    cr,cz,cq=q8_pack_and_dequantize(ctrl); mr,mz,mq=q8_pack_and_dequantize(cm)
    out={
      'seed':seed,'baseline_acc':ba,
      'control_fp32_acc':r.acc(ctrl,te),'compiled_fp32_acc':r.acc(cm,te),
      'control_q8_acc':r.acc(cq,te),'compiled_q8_acc':r.acc(mq,te),
      'control_q8_raw_bytes':cr,'compiled_q8_raw_bytes':mr,
      'control_q8_zlib_bytes':cz,'compiled_q8_zlib_bytes':mz,
      'q8_raw_reduction':1-mr/cr,'q8_zlib_reduction':1-mz/cz,
      'q8_utility':r.acc(mq,te)/r.acc(cq,te),
    }
    Path(outdir).mkdir(parents=True,exist_ok=True); (Path(outdir)/f'seed_{seed}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__':
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); run(a.seed,a.out)
