"""Historical graph and loader derivative. Import only in an authorized worker.
No training, acquisition, selection, tuning, or alternative model graph.
"""
import gzip,io,pathlib,hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model_definitions import ResBlock,TinyRes
from timing_core import ARMS
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

class Deploy(nn.Module):
 def __init__(self,arm,budget):
  super().__init__();self.stem=nn.Linear(64,64);self.head=nn.Linear(64,10)
  self.blocks=nn.ModuleDict({str(i):ResBlock() for i in (range(4) if arm=='teacher' else (2,3))})
  if arm=='C':self.replacement=TinyRes(64,budget//128,0)
  elif arm=='J':self.replacement=nn.Sequential(TinyRes(64,budget//256,0),TinyRes(64,budget//256,0))
 def forward(self,x):
  h=F.gelu(self.stem(x))
  if hasattr(self,'replacement'):h=self.replacement(h)
  for block in self.blocks.values():h=block(h)
  return self.head(h)


def load_models(selection, artifacts, expected, storage_root, source_paths):
    """Same deployment graph, strict NPZ tensor loading and parameter census."""
    s=selection
    mods={}
    assert sha(source_paths['teacher'])==s['teacher']['file_sha256']
    for arm in ARMS:
        if arm!='teacher':assert sha(source_paths[arm])==s['arms'][arm]['file_sha256']
        p=pathlib.Path(storage_root)/f'seed_{s["id"]}_{arm}_stored.npz'
        assert sha(p)==artifacts[p.name]['sha256']
        m=Deploy(arm,0 if arm=='teacher' else s['arms'][arm]['budget'])
        with np.load(p,allow_pickle=False) as z:
            expected_state=m.state_dict()
            assert len(z.files)==len(expected_state) and set(z.files)==set(expected_state)
            state={}
            for k,v in expected_state.items():
                a=z[k]
                assert a.shape==tuple(v.shape) and a.dtype==np.float32 and np.isfinite(a).all()
                state[k]=torch.from_numpy(a.copy())
            m.load_state_dict(state,strict=True)
        assert sum(p.numel() for p in m.parameters())==expected[arm+'_parameters']
        m.eval()
        for p in m.parameters():p.requires_grad_(False)
        mods[arm]=m
    return mods


def validation(selection,mods,validation_row,csv_path,split_path,reference_paths):
    """Only 270 validation feature rows, no labels; original 256+14 chunking."""
    assert sha(csv_path)=='09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22'
    with np.load(split_path,allow_pickle=False) as z:ids=z['validation'].copy()
    assert ids.ndim==1 and np.issubdtype(ids.dtype,np.integer)
    assert len(ids)==270 and len(set(ids))==270 and np.all((0<=ids)&(ids<1797))
    wanted=set(map(int,ids));lines={}
    with gzip.open(csv_path,'rt') as f:
        for i,line in enumerate(f):
            if i in wanted:lines[i]=line
    X=np.loadtxt(io.StringIO(''.join(lines[int(i)] for i in ids)),delimiter=',',usecols=tuple(range(64)),dtype=np.float32)/16
    errors={};references={}
    with torch.no_grad():
        for arm,m in mods.items():
            p=reference_paths[arm];key=validation_row['logits'][arm]['key']
            with np.load(p,allow_pickle=False) as z:ref=z[key].copy()
            got=np.concatenate([m(torch.from_numpy(X[start:start+256])).numpy() for start in range(0,270,256)])
            assert got.shape==ref.shape==(270,10) and np.isfinite(got).all()
            errors[arm]=float(np.max(np.abs(got-ref)));assert errors[arm]<=1e-5
            references[arm]={'path':validation_row['logits'][arm]['path'],'sha256':sha(p),'key':key}
    return {'id':selection['id'],'n':270,'max_abs':errors,'references':references,'split_sha256':sha(split_path)}
