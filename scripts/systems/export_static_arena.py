from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import torch

EXPECTED = {
    'compiler_block0.pt':'78fd7f52ef6f019f6ede72c73b4928b0482d61e7f7914de532ebc084779fce56',
    'compiler_block1.pt':'d70098af827694a42e7bb31958cf9959ec3aceef0d432941960c15d0cb5091c8',
}
KEYS = [
    'n1.weight','n1.bias','attn.in_proj_weight','attn.in_proj_bias',
    'attn.out_proj.weight','attn.out_proj.bias','n2.weight','n2.bias',
    'mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias',
]
EXPECTED_FLOATS = 3696

def tensor_hash(sd):
    h=hashlib.sha256()
    for k in sorted(sd):
        t=sd[k].detach().cpu().contiguous()
        h.update(k.encode()+b'\0'); h.update(str(t.dtype).encode()+b'\0'); h.update(str(tuple(t.shape)).encode()+b'\0'); h.update(t.numpy().tobytes(order='C'))
    return h.hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def write_f32(path, arrays):
    count=0
    with path.open('wb') as f:
        for a in arrays:
            x=np.asarray(a,dtype='<f4',order='C')
            f.write(x.tobytes(order='C')); count += x.size
    return count

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-dir',type=Path,required=True); ap.add_argument('--s4-dir',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    blocks=[]; tensor_hashes={}; float_sums={}; layout=[]
    offset=0
    first=None
    for i in range(2):
        name=f'compiler_block{i}.pt'; sd=torch.load(a.source_dir/name,map_location='cpu',weights_only=True); th=tensor_hash(sd)
        if th != EXPECTED[name]: raise SystemExit(f'tensor hash mismatch {name}: {th}')
        if list(sd.keys()) != KEYS: raise SystemExit(f'unexpected state-dict order: {list(sd.keys())}')
        tensor_hashes[name]=th; float_sums[name]=sum(float(sd[k].sum()) for k in KEYS)
        arrays=[]
        if first is None:
            for k in KEYS:
                v=sd[k].detach().cpu().contiguous().numpy().astype('<f4',copy=False)
                layout.append({'key':k,'offset_floats':offset,'shape':list(v.shape),'count':int(v.size)})
                offset += int(v.size)
            first=True
        arrays=[sd[k].detach().cpu().contiguous().numpy() for k in KEYS]
        out=a.out_dir/f'block{i}.bin'; n=write_f32(out,arrays)
        if n != EXPECTED_FLOATS: raise SystemExit(f'packed float count {n}')
        blocks.append(out)
    x=np.load(a.s4_dir/'input.npy',allow_pickle=False).astype('<f4',copy=False)
    ref=np.load(a.s4_dir/'pytorch_reference.npy',allow_pickle=False).astype('<f4',copy=False)
    if tuple(x.shape)!=(2,48,24) or tuple(ref.shape)!=(2,48,24): raise SystemExit('shape mismatch')
    write_f32(a.out_dir/'input.bin',[x]); write_f32(a.out_dir/'reference.bin',[ref])
    files=blocks+[a.out_dir/'input.bin',a.out_dir/'reference.bin']
    manifest={
        'format':'canaria-s6-static-arena-v1',
        'source_tensor_content_sha256':tensor_hashes,
        'source_block_float_sums':float_sums,
        'packing_order':KEYS,
        'layout':layout,
        'block_float_count':EXPECTED_FLOATS,
        'block_bytes':EXPECTED_FLOATS*4,
        'input_shape':[2,48,24],
        'input_bytes':int(x.nbytes),
        'reference_bytes':int(ref.nbytes),
        'reference_sum':float(ref.sum()),
        'raw_sha256':{p.name:sha256(p) for p in files},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
    }
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))
if __name__=='__main__': main()
