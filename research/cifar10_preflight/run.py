import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import sys, json, time, hashlib, urllib.request, tarfile, pickle, io, pathlib, datetime, resource, math, traceback, signal, platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'artifacts'; OUT.mkdir(exist_ok=True)
EXPIRY=datetime.datetime(2026,10,26,9,33,38,tzinfo=datetime.timezone.utc).timestamp()
FIRST=datetime.datetime(2026,10,26,8,48,38,tzinfo=datetime.timezone.utc).timestamp()
def check():
    if time.time()>=EXPIRY: raise TimeoutError('45min preflight expiry')
def emit(event, **kw):
    row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw)
    print(json.dumps(row),flush=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def sha(x): return hashlib.sha256(x).hexdigest()
def memory():
    return dict(proc_meminfo=pathlib.Path('/proc/meminfo').read_text(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
class SafeUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'Forbidden global {module}.{name}')
        return allowed[module,name]
class Block(nn.Module):
    def __init__(self,ci,co,stride):
        super().__init__(); self.stride=stride; self.co=co
        self.c1=nn.Conv2d(ci,co,3,stride,1,bias=False); self.b1=nn.BatchNorm2d(co)
        self.c2=nn.Conv2d(co,co,3,1,1,bias=False); self.b2=nn.BatchNorm2d(co)
    def forward(self,x):
        y=F.relu(self.b1(self.c1(x))); y=self.b2(self.c2(y))
        z=x[:,:,::self.stride,::self.stride]
        if z.shape[1]!=self.co:
            pad=(self.co-z.shape[1])//2; z=F.pad(z,(0,0,0,0,pad,pad))
        return F.relu(y+z)
class ResNet20(nn.Module):
    def __init__(self):
        super().__init__(); self.stem=nn.Conv2d(3,16,3,1,1,bias=False); self.bn=nn.BatchNorm2d(16)
        blocks=[]; ci=16
        for co in (16,32,64):
            for b in range(3):
                blocks.append(Block(ci,co,2 if b==0 and ci!=co else 1));ci=co
        self.blocks=nn.Sequential(*blocks); self.fc=nn.Linear(64,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x): return self.fc(self.blocks(F.relu(self.bn(self.stem(x)))).mean((2,3)))
def main():
    check();signal.signal(signal.SIGALRM,lambda *a: (_ for _ in ()).throw(TimeoutError('Hard expiry')));signal.alarm(max(1,int(EXPIRY-time.time())))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(260026)
    emit('start',python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),affinity=sorted(os.sched_getaffinity(0)),memory=memory(),first_data_action=FIRST,expiry=EXPIRY)
    archive=OUT/'cifar-10-python.tar.gz';url='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
    if not archive.exists():
        emit('download_start',url=url)
        with urllib.request.urlopen(url,timeout=60) as src,archive.with_suffix('.part').open('wb') as dst:
            while True:
                check();chunk=src.read(1024*1024)
                if not chunk:break
                dst.write(chunk)
        archive.with_suffix('.part').rename(archive)
    raw=archive.read_bytes(); md5=hashlib.md5(raw).hexdigest()
    assert md5=='c58f30108f718f92721af3b95e74349a',md5
    emit('archive_verified',bytes=len(raw),md5=md5,sha256=sha(raw)); del raw
    xx=[];yy=[];hashes={};read_members=[]
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers(); names=[m.name for m in members];assert len(names)==len(set(names))
        allowed={'cifar-10-batches-py','cifar-10-batches-py/batches.meta','cifar-10-batches-py/readme.html','cifar-10-batches-py/test_batch'}|{f'cifar-10-batches-py/data_batch_{i}' for i in range(1,6)}
        for m in members:
            assert m.name.rstrip('/') in allowed and (m.isfile() or m.isdir()) and 0<=m.size<40000000,(m.name,m.size)
        for i in range(1,6):
            check();name=f'cifar-10-batches-py/data_batch_{i}';blob=tf.extractfile(name).read();hashes[name]=sha(blob);read_members.append(name)
            d=SafeUnpickler(io.BytesIO(blob),encoding='bytes').load();x=d[b'data'];y=np.array(d[b'labels'])
            assert type(x) is np.ndarray and x.dtype==np.uint8 and x.shape==(10000,3072)
            assert y.shape==(10000,) and y.dtype.kind in 'iu' and y.min()>=0 and y.max()<=9
            xx.append(x.reshape(-1,3,32,32));yy.append(y)
    x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy,blob,d
    assert np.array_equal(np.bincount(y),np.full(10,5000))
    rng=np.random.Generator(np.random.PCG64(20261026));parts=[[],[],[]]
    for c in range(10):
        ids=rng.permutation(np.flatnonzero(y==c));parts[0].extend(ids[:4000]);parts[1].extend(ids[4000:4500]);parts[2].extend(ids[4500:])
    splits={name:np.array(ids,dtype=np.int64) for name,ids in zip(('teachertrain','calibration','validation'),parts)}
    assert len(np.unique(np.concatenate(list(splits.values()))))==50000
    splitmeta={}
    for name,ids in splits.items():
        np.save(OUT/f'{name}_indices.npy',ids,allow_pickle=False);splitmeta[name]={'n':len(ids),'index_bytes_sha256':sha(ids.tobytes()),'file_sha256':sha((OUT/f'{name}_indices.npy').read_bytes()),'class_counts':np.bincount(y[ids]).tolist()}
    train=splits['teachertrain'];s=np.zeros(3);ss=np.zeros(3);n=0
    for j in range(0,len(train),256):
        a=x[train[j:j+256]].astype(np.float64)/255;s+=a.sum((0,2,3));ss+=(a*a).sum((0,2,3));n+=a.shape[0]*1024
    mean=s/n;std=np.sqrt(ss/n-mean**2);del a
    metadata=dict(archive_md5=md5,train_member_sha256=hashes,read_members=read_members,test_batch_deserialized=False,split_seed=20261026,splits=splitmeta,normalization_mean=mean.tolist(),normalization_std=std.tolist(),normalization_source='teachertrain only')
    (OUT/'data_manifest.json').write_text(json.dumps(metadata,indent=2));emit('data_ready',**metadata)
    mean=torch.tensor(mean,dtype=torch.float32)[None,:,None,None];std=torch.tensor(std,dtype=torch.float32)[None,:,None,None]
    model=ResNet20();params=sum(p.numel() for p in model.parameters());assert params==269722,params
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
    rng=np.random.Generator(np.random.PCG64(260026));order=rng.permutation(train);cursor=0;times=[];model.train()
    emit('timing_begin',parameters=params,warmup_updates=30,timed_updates=200,batch_size=128,memory=memory())
    for step in range(230):
        check();t=time.perf_counter()
        if cursor+128>len(order):order=rng.permutation(train);cursor=0
        ids=order[cursor:cursor+128];cursor+=128
        a=np.pad(x[ids],((0,0),(0,0),(4,4),(4,4)));crops=np.empty((128,3,32,32),dtype=np.uint8)
        offsets=rng.integers(0,9,(128,2));flips=rng.random(128)<.5
        for k,(r,c) in enumerate(offsets):
            crop=a[k,:,r:r+32,c:c+32];crops[k]=crop[:,:,::-1] if flips[k] else crop
        inputs=(torch.from_numpy(crops).float()/255-mean)/std;targets=torch.from_numpy(y[ids]).long()
        opt.zero_grad(set_to_none=True);logits=model(inputs);assert logits.shape==(128,10) and torch.isfinite(logits).all()
        loss=F.cross_entropy(logits,targets);assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
        dt=time.perf_counter()-t
        if step>=30:times.append(dt)
        if (step+1)%10==0:emit('progress',updates=step+1,timed=len(times),last_step_s=dt,loss=float(loss.detach()))
    elapsed=sum(times);projection=313*120*3*elapsed/200;preflight=time.time()-FIRST
    result=dict(status='complete',parameters=params,finite_checks_passed=True,timed_steps=200,timed_seconds=elapsed,steps_per_second=200/elapsed,images_per_second=25600/elapsed,step_seconds_quantiles=np.quantile(times,[0,.5,.9,1]).tolist(),teacher_120epoch_3seed_updates=112680,projected_teacher_3seed_seconds=projection,projected_teacher_per_seed_seconds=projection/3,projection_excludes='validation/checkpoint/epoch-boundary overhead; timings include finite checks',preflight_elapsed_seconds=preflight,remaining_72h_after_preflight_and_projected_teachers_seconds=72*3600-preflight-projection,memory=memory(),no_quality_inference=True,no_validation_or_test_evaluation=True)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2));np.save(OUT/'step_seconds.npy',times,allow_pickle=False);emit('complete',**result)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        emit('blocked',error=repr(e),traceback=traceback.format_exc());(OUT/'blocked.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise
