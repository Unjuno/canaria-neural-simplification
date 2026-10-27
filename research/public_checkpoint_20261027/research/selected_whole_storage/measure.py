import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
import numpy as np
import json,hashlib,pathlib,zipfile,io,time,signal,platform,subprocess,csv,math
signal.alarm(600)
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def shapes_teacher():
 d={'stem.weight':(64,64),'stem.bias':(64,), 'head.weight':(10,64),'head.bias':(10,)}
 for i in range(4):
  for k,s in {'ln.weight':(64,),'ln.bias':(64,),'fc1.weight':(128,64),'fc1.bias':(128,),'fc2.weight':(64,128),'fc2.bias':(64,)}.items():d[f'blocks.{i}.{k}']=s
 return d
def load(path,expected,shapes):
 assert sha(path)==expected,'source file SHA'
 with np.load(path,allow_pickle=False) as z:
  assert len(z.files)==len(set(z.files)) and set(z.files)==set(shapes),'keys'
  d={k:z[k].copy() for k in z.files}
 for k,a in d.items():assert a.shape==shapes[k] and a.dtype==np.dtype('<f4') and np.isfinite(a).all(),(k,a.shape,a.dtype)
 return d
def state_hash(d):
 q={}
 for k,a in d.items():q[k]=hashlib.sha256(b'torch.float32'+str(tuple(a.shape)).encode()+a.tobytes(order='C')).hexdigest()
 return hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
def serialize(path,d,codec):
 with zipfile.ZipFile(path,'w',compression=codec,compresslevel=6 if codec==zipfile.ZIP_DEFLATED else None) as z:
  for k,a in sorted(d.items()):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(a),version=(1,0),allow_pickle=False)
   info=zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0));info.create_system=3;info.external_attr=0o600<<16;info.compress_type=codec
   z.writestr(info,b.getvalue(),compress_type=codec,compresslevel=6 if codec==zipfile.ZIP_DEFLATED else None)
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(d) and len(z.files)==len(d)
  for k,a in d.items():assert z[k].dtype==a.dtype and z[k].shape==a.shape and z[k].tobytes()==a.tobytes()
 return path.stat().st_size
manifest=json.loads((ROOT/'SELECTION.json').read_text());assert len(manifest)==16 and {s['id'] for s in manifest}==set(range(16))
env={'python':platform.python_version(),'numpy':np.__version__,'zlib':__import__('zlib').ZLIB_VERSION,'platform':platform.platform(),'runtime_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':sha(__file__),'selection_sha256':sha(ROOT/'SELECTION.json'),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
dump(OUT/'ENV.json',env)
rows=[];census=[];artifacts=[];sources=[];mapping=[]
for s in manifest:
 try:
  t=load(s['teacher_path'],s['teacher_file_sha256'],shapes_teacher());models={'teacher':t}
  sources.append({'id':s['id'],'arm':'teacher','path':s['teacher_path'],'sha256':s['teacher_file_sha256']})
  for arm in ['C','J']:
   a=s['arms'][arm];w=a['budget']//(128 if arm=='C' else 256)
   shapes={f'{prefix}{k}':shape for prefix in ([''] if arm=='C' else ['0.','1.']) for k,shape in {'w1.weight':(w,64),'w2.weight':(64,w)}.items()}
   r=load(a['path'],a['file_sha256'],shapes);assert state_hash(r)==a['state_sha256']
   kept={k:v for k,v in t.items() if not k.startswith(('blocks.0.','blocks.1.'))};d={**kept,**{'replacement.'+k:v for k,v in r.items()}}
   assert len(d)==len(kept)+len(r) and sum(v.size for v in r.values())==a['budget'];models[arm]=d
   sources.append({'id':s['id'],'arm':arm,'path':a['path'],'sha256':a['file_sha256'],'logical_sha256':state_hash(r)})
  rec={'id':s['id'],'C_budget':s['arms']['C']['budget'],'J_budget':s['arms']['J']['budget']}
  for arm,d in models.items():
   rec[arm+'_parameters']=sum(a.size for a in d.values());rec[arm+'_tensors']=len(d);rec[arm+'_payload_bytes']=sum(a.nbytes for a in d.values())
   for k,a in sorted(d.items()):mapping.append({'id':s['id'],'arm':arm,'destination':k,'source_key':k.removeprefix('replacement.'),'source_arm':arm if k.startswith('replacement.') else 'teacher','shape':list(a.shape),'dtype':a.dtype.str,'sha256':hashlib.sha256(a.tobytes()).hexdigest()})
   for label,codec in [('stored',zipfile.ZIP_STORED),('deflated',zipfile.ZIP_DEFLATED)]:
    path=OUT/f'seed_{s["id"]}_{arm}_{label}.npz';rec[arm+'_'+label+'_bytes']=serialize(path,d,codec);artifacts.append({'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path)})
  for metric in ['parameters','stored_bytes','deflated_bytes']:
   for a,b in [('C','J'),('C','teacher'),('J','teacher')]:rec[a+'_over_'+b+'_'+metric]=rec[a+'_'+metric]/rec[b+'_'+metric]
  rows.append(rec);census.append({'id':s['id'],'status':'PASS'})
 except Exception as e:census.append({'id':s['id'],'status':'FAIL','error':repr(e)})
assert sum(x['bytes'] for x in artifacts)<128*1024**2
summary={'cohort':16,'complete':len(rows),'failures':[x for x in census if x['status']!='PASS'],'ratios':{}}
if rows:
 for k in rows[0]:
  if '_over_' in k:
   v=[r[k] for r in rows];summary['ratios'][k]={'geomean':math.exp(sum(map(math.log,v))/len(v)),'min':min(v),'max':max(v),'smaller':sum(x<1 for x in v),'tie':sum(x==1 for x in v),'larger':sum(x>1 for x in v)}
for name,x in [('ROWS',rows),('CENSUS',census),('ARTIFACTS',artifacts),('SOURCES',sources),('TENSORS',mapping),('SUMMARY',summary)]:dump(OUT/(name+'.json'),x)
if rows:
 with (OUT/'ROWS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2));assert len(rows)==16
