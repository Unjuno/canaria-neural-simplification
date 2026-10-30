"""Modified extraction of historical measure.py: same arrays/NPY/ZIP semantics.
Changed archive opening from overwrite to exclusive creation; see TRANSFORMATION.md.
No filesystem writes, alarms, limits, or model reads occur at import.
"""
import hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def shapes_teacher():
 d={'stem.weight':(64,64),'stem.bias':(64,), 'head.weight':(10,64),'head.bias':(10,)}
 for i in range(4):
  for k,s in {'ln.weight':(64,),'ln.bias':(64,),'fc1.weight':(128,64),'fc1.bias':(128,),'fc2.weight':(64,128),'fc2.bias':(64,)}.items():d[f'blocks.{i}.{k}']=s
 return d

def state_hash(d):
 q={}
 for k,a in d.items():q[k]=hashlib.sha256(b'torch.float32'+str(tuple(a.shape)).encode()+a.tobytes(order='C')).hexdigest()
 return hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()

def serialize(path,d,codec):
 with zipfile.ZipFile(path,'x',compression=codec,compresslevel=6 if codec==zipfile.ZIP_DEFLATED else None) as z:
  for k,a in sorted(d.items()):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(a),version=(1,0),allow_pickle=False)
   info=zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0));info.create_system=3;info.external_attr=0o600<<16;info.compress_type=codec
   z.writestr(info,b.getvalue(),compress_type=codec,compresslevel=6 if codec==zipfile.ZIP_DEFLATED else None)
 with np.load(path,allow_pickle=False) as z:
  if set(z.files)!=set(d) or len(z.files)!=len(d):raise ValueError('archive readback keys')
  for k,a in d.items():
   if z[k].dtype!=a.dtype or z[k].shape!=a.shape or z[k].tobytes()!=a.tobytes():raise ValueError('archive readback tensor bytes')
 return path.stat().st_size


def shapes_arm(arm, budget):
 if arm not in ('C','J') or type(budget) is not int or budget <= 0 or budget % (128 if arm == 'C' else 256):
  raise ValueError('arm budget')
 width = budget // (128 if arm == 'C' else 256)
 return {prefix+key:shape for prefix in ([''] if arm == 'C' else ['0.','1.'])
         for key,shape in {'w1.weight':(width,64),'w2.weight':(64,width)}.items()}


def load_source(path, expected_sha256, shapes):
 # Verify and decode the same bounded bytes, avoiding a hash/read TOCTOU gap.
 raw = Path(path).read_bytes()
 if len(raw) > 128 * 1024**2 or hashlib.sha256(raw).hexdigest() != expected_sha256:
  raise ValueError('source file SHA256 or byte ceiling')
 # Before allocating arrays, bound member count and total expanded ZIP bytes.
 with zipfile.ZipFile(io.BytesIO(raw)) as archive:
  members = archive.infolist()
  if len(members) != len(shapes) or {m.filename for m in members} != {k+'.npy' for k in shapes}:
   raise ValueError('source ZIP member census')
  if sum(m.file_size for m in members) > 2 * 1024**2:
   raise ValueError('expanded source byte ceiling')
 with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
  if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(shapes):
   raise ValueError('source tensor keys')
  arrays = {key:archive[key].copy() for key in archive.files}
 for key, array in arrays.items():
  if array.shape != shapes[key] or array.dtype != np.dtype('<f4') or not np.isfinite(array).all():
   raise ValueError('source tensor shape/dtype/finite contract')
 return arrays


def assemble(teacher, replacement, arm, budget):
 expected = shapes_teacher()
 if set(teacher) != set(expected) or set(replacement) != set(shapes_arm(arm,budget)):
  raise ValueError('assembly tensor keys')
 kept = {k:v for k,v in teacher.items() if not k.startswith(('blocks.0.','blocks.1.'))}
 whole = {**kept, **{'replacement.'+k:v for k,v in replacement.items()}}
 if len(whole) != len(kept)+len(replacement) or sum(v.size for v in replacement.values()) != budget:
  raise ValueError('replacement parameter count')
 return whole
