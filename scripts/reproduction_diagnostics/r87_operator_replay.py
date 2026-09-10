"""R87D2: post-hoc first-batch operator replay, known seed1202, no fitted models."""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from r87_dispatch import load_runner,metadata

def capture(out):
 r=load_runner();torch.set_num_threads(1);xt,yt,*_=r.datasets();m=r.Net(1202)
 g=torch.Generator().manual_seed(2201);ix=torch.randperm(len(xt),generator=g)[:64]
 values={};spec=[];orig=F.gelu
 def save(kind,name,x,y,**kw):
  i=len(spec);d={'kind':kind,'name':name,'index':i,'eps':kw.get('eps')};spec.append(d)
  values[f'{i}_x']=x.detach().numpy().copy()
  for k in ('weight','bias'):
   if kw.get(k) is not None:values[f'{i}_{k}']=kw[k].detach().numpy().copy()
  values[f'{i}_y']=y.detach().numpy().copy()
  def hook(gy):values[f'{i}_gy']=gy.detach().numpy().copy()
  y.register_hook(hook)
 def gelu(x,*args,**kw):
  y=orig(x,*args,**kw);save('gelu','gelu_'+str(sum(d['kind']=='gelu' for d in spec)),x,y);return y
 F.gelu=gelu
 for name,mod in m.named_modules():
  if isinstance(mod,torch.nn.Linear):
   mod.register_forward_hook(lambda mod,args,y,name=name:save('linear',name,args[0],y,weight=mod.weight,bias=mod.bias))
  elif isinstance(mod,torch.nn.LayerNorm):
   mod.register_forward_hook(lambda mod,args,y,name=name:save('layer_norm',name,args[0],y,weight=mod.weight,bias=mod.bias,eps=mod.eps))
 logits=m(xt[ix]);loss=F.cross_entropy(logits,yt[ix]);save('cross_entropy','loss',logits,loss)
 values[str(spec[-1]['index'])+'_labels']=yt[ix].numpy().copy();loss.backward();F.gelu=orig
 out.mkdir(exist_ok=True,parents=True);np.savez_compressed(out/'common_inputs.npz',**values)
 (out/'operations.json').write_text(json.dumps({'evidence_class':'POST_HOC_OPERATOR_REPLAY','seed':1202,'spec':spec,'capture_environment':metadata(),'common_inputs_sha256':hashlib.sha256((out/'common_inputs.npz').read_bytes()).hexdigest()},indent=2))
 print('captured',len(spec),'operations')

def replay(root,out):
 torch.set_num_threads(1);spec=json.loads((root/'operations.json').read_text());d=np.load(root/'common_inputs.npz',allow_pickle=False)
 assert hashlib.sha256((root/'common_inputs.npz').read_bytes()).hexdigest()==spec['common_inputs_sha256']
 values={}
 for s in spec['spec']:
  i=str(s['index']);kind=s['kind'];keys=[k for k in ('x','weight','bias') if f'{i}_{k}' in d]
  t={k:torch.from_numpy(d[f'{i}_{k}'].copy()).requires_grad_(True) for k in keys}
  if kind=='linear':y=F.linear(t['x'],t['weight'],t.get('bias'))
  elif kind=='gelu':y=F.gelu(t['x'])
  elif kind=='layer_norm':y=F.layer_norm(t['x'],(64,),t['weight'],t['bias'],s['eps'])
  else:y=F.cross_entropy(t['x'],torch.from_numpy(d[f'{i}_labels'].copy()))
  grads=torch.autograd.grad(y,[t[k] for k in keys],torch.from_numpy(d[f'{i}_gy'].copy()))
  values[f'{i}_y']=y.detach().numpy().copy()
  for k,v in zip(keys,grads):values[f'{i}_g_{k}']=v.numpy().copy()
 out.mkdir(exist_ok=True,parents=True);np.savez_compressed(out/'outputs.npz',**values)
 (out/'metadata.json').write_text(json.dumps({'environment':metadata(),'common_inputs_sha256':spec['common_inputs_sha256']},indent=2))
 print('replayed',len(spec['spec']),torch.backends.cpu.get_cpu_capability())

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');ap.add_argument('--root',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.capture:capture(a.out)
 else:replay(a.root,a.out)
