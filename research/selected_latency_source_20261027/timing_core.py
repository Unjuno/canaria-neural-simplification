"""Historical scheduling, timing arithmetic and bounded evidence writer. No ML imports."""
import itertools,json,math,pathlib,statistics,time
ARMS=('teacher','C','J');ORDERS=list(itertools.permutations(ARMS));ROUNDS=30;ITERS=20;WARMUP=20

def ordered(seed,round_id):return ORDERS[(seed+round_id)%6]

def timed_block(fn,x,clock=time.perf_counter_ns,iterations=ITERS):
 start=clock()
 for _ in range(iterations):fn(x)
 return clock()-start

def summarize(raw):
 medians=[]
 for seed in range(16):
  for batch in (1,128):
   row={'id':seed,'batch':batch}
   for arm in ARMS:
    values=[r['ns_per_forward'] for r in raw if r['id']==seed and r['batch']==batch and r['arm']==arm]
    if len(values)!=ROUNDS:return {'status':'INCOMPLETE','paired':[]}
    row[arm+'_ns']=statistics.median(values)
   for a,b in [('C','J'),('C','teacher'),('J','teacher')]:row[a+'_over_'+b]=row[a+'_ns']/row[b+'_ns']
   medians.append(row)
 ratios={}
 for batch in (1,128):
  for key in ('C_over_J','C_over_teacher','J_over_teacher'):
   vals=[r[key] for r in medians if r['batch']==batch]
   ratios[f'batch{batch}_{key}']={'geomean':math.exp(statistics.mean(map(math.log,vals))),'min':min(vals),'max':max(vals),'smaller':sum(v<1 for v in vals),'tie':sum(v==1 for v in vals),'larger':sum(v>1 for v in vals)}
 return {'status':'COMPLETE','paired':medians,'ratios':ratios,'limitation':'Conditional shared-host eager CPU steady-state latency; contention and sequential-seed temporal drift possible; not portable hardware or training-time claim.'}

class EvidenceWriter:
 def __init__(self,roots,cap=128*1024**2,reserve=1024**2):self.roots=list(map(pathlib.Path,roots));self.cap=cap;self.reserve=reserve
 def used(self):return sum(p.stat().st_size for root in self.roots if root.exists() for p in root.rglob('*') if p.is_file())
 def write(self,path,data,append=False,final=False):
  data=data.encode() if isinstance(data,str) else data
  limit=self.cap-65536 if final else self.cap-self.reserve
  if self.used()+len(data)>limit:raise RuntimeError('128MiB aggregate evidence cap')
  with pathlib.Path(path).open('ab' if append else 'xb') as f:f.write(data)
 def json(self,path,value,final=False):self.write(path,json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',final=final)

def failure_census(census):
 for c in census:
  if c['status']=='STARTED':c['status']='FAILED'
 return census


def full_census():
    return [{'id':i,'batch':b,'arm':a,'status':'NOT_STARTED'}
            for i in range(16) for b in (1,128) for a in ARMS]


def validate_raw(raw):
    """Strict additive handoff guard; no exclusion, trimming or winsorization."""
    expected=[(s,b,r,p,a) for s in range(16)
              for b in ((1,128) if s%2==0 else (128,1))
              for r in range(ROUNDS) for p,a in enumerate(ordered(s,r))]
    if len(raw)!=2880:
        raise ValueError('exactly 2880 blocks required')
    for row,identity in zip(raw,expected):
        if (row['id'],row['batch'],row['round'],row['position'],row['arm'])!=identity:
            raise ValueError('block order, duplicate, or cohort mismatch')
        if type(row['total_ns']) is not int or row['total_ns']<=0 or row['iterations']!=ITERS:
            raise ValueError('positive elapsed integer and 20 forwards required')
        if row['ns_per_forward']!=row['total_ns']/ITERS:
            raise ValueError('per-forward arithmetic mismatch')
    return raw
