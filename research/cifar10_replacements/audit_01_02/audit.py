"""Independent, read-only audit: no imports of production experiment code."""
import os, sys, json, hashlib, io, pickle, tarfile, subprocess, platform, math, time
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '4'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
ROOT=Path('/workspace/shared/canaria_cifar_replacement_runs_20261026')
OUT=Path('/workspace/shared/canaria_cifar_replacement_audit_20261026')
EX=ROOT/'research/cifar10_replacements/execution'
COMMIT='bb30713cca06519a000251a26668309b1a4ef82b'
SEEDS=(26102601,26102602)
ARMS=('Comp-joint','Sep-local','Sep-joint')
evidence={}
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def read(p):
    p=Path(p);b=p.read_bytes();h=sha(b)
    if str(p) in evidence: assert evidence[str(p)]==h, 'Evidence changed during audit: '+str(p)
    evidence[str(p)]=h
    return b
def loadj(p):return json.loads(read(p))
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def finite(obj):
    if isinstance(obj,torch.Tensor):assert bool(torch.isfinite(obj).all())
    elif isinstance(obj,dict):
        for v in obj.values():finite(v)
    elif isinstance(obj,(tuple,list)):
        for v in obj:finite(v)
    elif isinstance(obj,float):assert math.isfinite(obj)
def thash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode()+b'\0'+str(t.dtype).encode()+str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def envelope(path,magic):
    b=read(path);m,h,payload=b.split(b'\n',2)
    assert m==magic and h.decode()==sha(payload)
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    finite(state)
    return state,dict(file_sha256=sha(b),payload_sha256=sha(payload),bytes=len(b))
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
class Map(nn.Module):
    def __init__(self,r):
        super().__init__();self.down=nn.Conv2d(32,r,3,padding=1,bias=False);self.up=nn.Conv2d(r,32,1,bias=False)
def init(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sep=nn.Sequential(Map(8),Map(8)); comp=nn.Sequential(Map(16))
    return {'Comp-joint':comp.state_dict(),'Sep-local':sep.state_dict(),'Sep-joint':sep.state_dict()}
def schedule(ids,seed):
    g=torch.Generator().manual_seed(seed);batches=[]
    for _ in range(25):
        order=torch.randperm(5000,generator=g).tolist()
        batches.extend([[int(ids[i]) for i in order[j:j+128]] for j in range(0,5000,128)])
    assert len(batches)==1000 and sum(map(len,batches))==125000
    for j in range(0,1000,40):
        assert sorted(sum(batches[j:j+40],[]))==sorted(ids.tolist())
        assert list(map(len,batches[j:j+40]))==[128]*39+[8]
    return batches
# Separate functional implementation: never instantiate/import the production teacher or evaluator.
def bn(x,s,p):
    return F.batch_norm(x,s[p+'.running_mean'],s[p+'.running_var'],s[p+'.weight'],s[p+'.bias'],training=False,eps=1e-5)
def block(x,s,i):
    p=f'blocks.{i}';stride=2 if i in (3,6) else 1
    y=F.relu(bn(F.conv2d(x,s[p+'.c1.weight'],stride=stride,padding=1),s,p+'.b1'))
    y=bn(F.conv2d(y,s[p+'.c2.weight'],padding=1),s,p+'.b2')
    skip=x[:,:,::stride,::stride]
    d=y.shape[1]-skip.shape[1]
    if d:
        assert d%2==0
        zeros=torch.zeros((len(x),d//2,*skip.shape[2:]),dtype=x.dtype)
        skip=torch.cat((zeros,skip,zeros),dim=1)
    return F.relu(y+skip)
def endpoints(x,s):
    h=F.relu(bn(F.conv2d(x,s['stem.weight'],padding=1),s,'bn'))
    for i in range(4):h=block(h,s,i)
    return h,block(block(h,s,4),s,5)
def suffix(h,s):
    for i in range(6,9):h=block(h,s,i)
    return F.linear(h.mean((2,3)),s['fc.weight'],s['fc.bias'])
def replace(x,s):
    maps=len(s)//2
    for i in range(maps):
        x=F.relu(x+F.conv2d(F.relu(F.conv2d(x,s[f'{i}.down.weight'],padding=1)),s[f'{i}.up.weight']))
    return x
class Restricted(pickle.Unpickler):
    def find_class(self,m,n):
        allowed={('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
        if (m,n) not in allowed:raise pickle.UnpicklingError(f'Forbidden {m}.{n}')
        return allowed[m,n]
started=time.time()
assert git('rev-parse','HEAD')==COMMIT
assert git('status','--porcelain')==''
protocol=loadj(EX/'EXECUTION_PROTOCOL.json')
assert sha(read(EX/'ENVIRONMENT.json'))==protocol['environment_manifest_sha256']
env=loadj(EX/'ENVIRONMENT.json')
assert sha(read(Path(sys.executable).resolve()))==env['python_binary_sha256']
assert platform.python_version()==env['python_version']
for p,h in protocol['source_sha256'].items():assert sha(read(ROOT/p))==h
for earlier,later in [('7a3046da9033d2b87b634d791585f480925edccc','9e48fbb00abf9baa9ad777a2828f6eb92066db94'),('9e48fbb00abf9baa9ad777a2828f6eb92066db94','0229f2a'),('0229f2a','2e7656a'),('2e7656a',COMMIT)]:
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',earlier,later],check=True)
for p,h in protocol['source_sha256'].items():
    assert sha(subprocess.check_output(['git','-C',str(ROOT),'show','7a3046da9033d2b87b634d791585f480925edccc:'+p]))==h
assert sha(subprocess.check_output(['git','-C',str(ROOT),'show','9e48fbb00abf9baa9ad777a2828f6eb92066db94:research/cifar10_replacements/execution/EXECUTION_PROTOCOL.json']))==sha(read(EX/'EXECUTION_PROTOCOL.json'))
records={}
# Check final teacher admission and all file/hash identities before any dataset decoding.
for seed in SEEDS:
    planpath=EX/f'plan_seed_{seed}.APPROVED.json';slotpath=EX/f'slot_seed_{seed}.APPROVED.json'
    plan=loadj(planpath);slot=loadj(slotpath);run=plan['runs'][str(seed)]
    assert plan['status']=='parent-approved-and-user-announced' and plan['official_test']=='sealed'
    assert plan['planned_seeds']==[26102601,26102602,26102603] and plan['updates']==1000
    assert plan['threads']==4 and plan['interop_threads']==1 and plan['global_science_threads']==8
    assert plan['source_sha256']==protocol['source_sha256']
    assert plan['execution_protocol_sha256']==sha(read(EX/'EXECUTION_PROTOCOL.json'))
    assert slot['plan_sha256']==sha(read(planpath)) and slot['seed']==seed
    assert slot['replacement_threads']==4 and slot['other_science_threads']<=4 and slot['global_science_threads']==8
    for k in ('teacher_export','teacher_metrics','teacher_checkpoint'):assert sha(read(run[k]))==run[k+'_sha256']
    export=loadj(run['teacher_export']);metrics=loadj(run['teacher_metrics'])
    teacher,teacher_hashes=envelope(run['teacher_checkpoint'],b'CANARIA1')
    assert teacher['schema']==2 and teacher['seed']==seed and teacher['epoch']==120 and teacher['cursor']==0 and teacher['order']==[]
    assert metrics['status']=='complete' and metrics['adequate'] is True and metrics['correct']>=4250 and metrics['n']==5000
    assert metrics['accuracy']==metrics['correct']/5000
    assert export['calibration_used'] is False and export['official_test_decoded'] is False
    assert metrics['calibration_used'] is False and metrics['test_batch_decoded'] is False
    for r in (export,metrics):
        assert r['checkpoint_sha256']==teacher_hashes['file_sha256'] and r['seed']==seed and r['epoch']==120
        assert r['environment']==teacher['environment'] and r['data_manifest_sha256']==teacher['binding']
    artifact=Path(run['output']);lock=loadj(artifact/'recipe_lock.json');validation=loadj(artifact/'validation.json')
    launch=loadj(artifact/'launch.json')
    initial,ih=envelope(artifact/'initial.ckpt',b'CANARIA-REPLACEMENT1')
    final,fh=envelope(artifact/'final_update1000.ckpt',b'CANARIA-REPLACEMENT1')
    last,lh=envelope(artifact/'last.ckpt',b'CANARIA-REPLACEMENT1')
    assert initial['cursor']==0 and final['cursor']==last['cursor']==1000
    assert equal(last,final)
    assert canonical(initial['lock'])==canonical(final['lock'])==canonical(last['lock'])==canonical(lock)
    assert initial['lock_sha256']==final['lock_sha256']==canonical(lock)==sha(read(artifact/'recipe_lock.json'))
    assert lock['seed']==seed and lock['teacher_sha256']==teacher_hashes['file_sha256']
    assert lock['recipe']==dict(updates=1000,batch_size=128,optimizer='Adam',lr=0.001,betas=[0.9,0.999],eps=1e-8,weight_decay=0.0,schedule='constant')
    assert lock['span']==[4,6] and lock['activation']=='inner-and-post-add-ReLU'
    assert lock['ranks']=={'Comp-joint':[16],'Sep-local':[8,8],'Sep-joint':[8,8]}
    approval_commit='0229f2a' if seed==26102601 else '2e7656a'
    relplan=str(planpath.relative_to(ROOT))
    assert sha(subprocess.check_output(['git','-C',str(ROOT),'show',approval_commit+':'+relplan]))==sha(read(planpath))
    assert int(git('show','-s','--format=%ct',approval_commit))<launch['started']
    assert lock['source_identity']==plan['source_sha256'] and lock['launch']==launch
    assert launch['approved_plan_sha256']==sha(read(planpath))
    assert launch['expiry']==min(launch['started']+7200,plan['absolute_deadline'])
    assert launch['started']<slot['expires_at']
    assert lock['teacher_validation_correct']==metrics['correct']
    assert validation['checkpoint_sha256']==fh['file_sha256'] and validation['lock_sha256']==canonical(lock)
    assert validation['teacher_sha256']==teacher_hashes['file_sha256'] and validation['seed']==seed and validation['updates']==1000
    assert validation['status']=='complete' and validation['official_test_decoded'] is False and validation['split']=='validation'
    receipts=list(artifact.glob('exit_*.json'));assert len(receipts)==1
    receipt=loadj(receipts[0]);assert receipt['status']=='complete' and receipt['exit_code']==0 and receipt['resume'] is False
    for k,h in [('plan_sha256',sha(read(planpath))),('slot_sha256',sha(read(slotpath))),('recipe_sha256',canonical(lock)),('teacher_sha256',teacher_hashes['file_sha256']),('replacement_sha256',fh['file_sha256']),('validation_sha256',sha(read(artifact/'validation.json')))]:assert receipt[k]==h
    expected=init(seed)
    initial_hashes={}
    for name in ARMS:
        assert equal(initial['initial'][name],initial['arms'][name])
        assert equal(initial['initial'][name],final['initial'][name])
        assert equal(initial['initial'][name],expected[name])
        initial_hashes[name]=thash(initial['initial'][name]);assert initial_hashes[name]==lock['initialization'][name]
        assert sum(t.numel() for t in final['arms'][name].values())==5120
        assert set(final['arms'][name])==set(initial['arms'][name])
        assert len(final['arms'][name])==(2 if name=='Comp-joint' else 4)
        for key,t in final['arms'][name].items():
            assert key.endswith('.weight') and t.dtype==torch.float32 and t.shape==initial['arms'][name][key].shape
        opt=final['optimizers'][name];iopt=initial['optimizers'][name]
        assert iopt['state']=={} and len(opt['param_groups'])==1
        assert opt['param_groups']==iopt['param_groups']
        pg=opt['param_groups'][0]
        assert pg['lr']==0.001 and tuple(pg['betas'])==(0.9,0.999) and pg['eps']==1e-8 and pg['weight_decay']==0
        assert pg['amsgrad'] is False and pg['maximize'] is False
        assert pg['params']==list(range(len(final['arms'][name])))
        assert set(opt['state'])==set(pg['params'])
        for pid,(key,t) in zip(pg['params'],final['arms'][name].items()):
            st=opt['state'][pid]
            assert set(st)=={'step','exp_avg','exp_avg_sq'} and float(st['step'])==1000
            assert st['exp_avg'].shape==t.shape and st['exp_avg_sq'].shape==t.shape
            assert bool((st['exp_avg_sq']>=0).all())
    assert equal(initial['initial']['Sep-local'],initial['initial']['Sep-joint'])
    assert all(initial['arms']['Sep-local'][k].data_ptr()!=initial['arms']['Sep-joint'][k].data_ptr() for k in initial['arms']['Sep-local'])
    records[seed]=dict(plan=plan,run=run,teacher=teacher,export=export,metrics=metrics,lock=lock,initial=initial,final=final,validation=validation,
                       checks=dict(teacher=teacher_hashes,initial=ih,final=fh,last=lh,initialization=initial_hashes,all_optimizer_steps=1000,parameters_per_arm=5120,
                       independent_sep_storage=True,seed_initialization_exact=True,final_equals_last=True))
print('PRELIMINARY_ARTIFACT_HASH_INIT_OPTIMIZER_CHECKS_PASS',flush=True)
pins=protocol['teacher_training_manifest_identity']['preflight_sha256']
PRE=Path(records[SEEDS[0]]['plan']['preflight'])
manifest=loadj(PRE/'data_manifest.json');assert evidence[str(PRE/'data_manifest.json')]==pins['data_manifest.json']
pre_metrics=loadj(PRE/'metrics.json');assert evidence[str(PRE/'metrics.json')]==pins['metrics.json']
assert pre_metrics['status']=='complete' and pre_metrics['finite_checks_passed'] is True
assert manifest['test_batch_deserialized'] is False and manifest['normalization_source']=='teachertrain only'
train_names=[f'cifar-10-batches-py/data_batch_{i}' for i in range(1,6)]
assert manifest['read_members']==train_names and manifest['split_seed']==20261026
archive=PRE/'cifar-10-python.tar.gz';extracted=[]
with archive.open('rb') as f:
    h=hashlib.sha256();m=hashlib.md5()
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk);m.update(chunk)
    assert h.hexdigest()==pins[archive.name] and m.hexdigest()=='c58f30108f718f92721af3b95e74349a'
    evidence[str(archive)]=h.hexdigest();f.seek(0)
    xx=[];yy=[]
    with tarfile.open(fileobj=f,mode='r:gz') as tf:
        members=tf.getmembers();assert len({x.name for x in members})==len(members)
        for name in train_names:
            assert name in train_names and not name.endswith('test_batch')
            member=tf.getmember(name);assert member.isfile() and 0<member.size<40000000
            b=tf.extractfile(member).read();extracted.append(name)
            assert sha(b)==manifest['train_member_sha256'][name]
            p=Restricted(io.BytesIO(b),encoding='bytes').load();x=p[b'data'];y=np.asarray(p[b'labels'],dtype=np.int64)
            assert type(x) is np.ndarray and x.dtype==np.uint8 and x.shape==(10000,3072) and y.shape==(10000,)
            xx.append(x.reshape(-1,3,32,32));yy.append(y)
images=np.concatenate(xx);labels=np.concatenate(yy);del xx,yy
assert np.array_equal(np.bincount(labels),np.full(10,5000)) and extracted==train_names
rng=np.random.Generator(np.random.PCG64(20261026));parts=[[],[],[]]
for cls in range(10):
    ids=rng.permutation(np.flatnonzero(labels==cls))
    for p,b in zip(parts,(ids[:4000],ids[4000:4500],ids[4500:])):p.extend(b)
splits={k:np.array(v,dtype=np.int64) for k,v in zip(('teachertrain','calibration','validation'),parts)}
for name,ids in splits.items():
    b=read(PRE/(name+'_indices.npy'));assert sha(b)==pins[name+'_indices.npy']
    assert np.array_equal(np.load(io.BytesIO(b),allow_pickle=False),ids)
    assert sha(ids.astype('<i8').tobytes())==manifest['splits'][name]['index_bytes_sha256']
assert len(set(np.concatenate(list(splits.values())).tolist()))==50000
sums=np.zeros(3);squares=np.zeros(3);pixels=0
for j in range(0,40000,256):
    x=images[splits['teachertrain'][j:j+256]].astype(np.float64)/255
    sums+=x.sum((0,2,3));squares+=(x*x).sum((0,2,3));pixels+=len(x)*1024
mean=sums/pixels;std=np.sqrt(squares/pixels-mean**2)
assert np.array_equal(mean,manifest['normalization_mean']) and np.array_equal(std,manifest['normalization_std'])
normalization_hash=sha(np.array([mean,std],dtype='<f8').tobytes())
for seed,r in records.items():
    identity=r['lock']['data_identity'];assert identity==r['teacher']['evaluator_identity']
    assert r['teacher']['binding']==pins['data_manifest.json']
    assert r['teacher']['train']==splits['teachertrain'].tolist()
    for k,v in identity.items():assert r['export'][k]==v
    assert identity['model_config_sha256']==canonical(identity['model_config'])
    assert identity['normalization']['sha256']==normalization_hash
    for name,ids in splits.items():assert identity['splits'][name]==dict(n=len(ids),sha256=sha(ids.astype('<i8').tobytes()))
    for key,v in [('mean',mean),('std',std)]:assert torch.equal(r['teacher'][key],torch.tensor(v,dtype=torch.float32)[None,:,None,None])
    batches=schedule(splits['calibration'],seed);assert canonical(batches)==r['lock']['batch_schedule_sha256']
    r['checks']['schedule']=dict(sha256=canonical(batches),updates=1000,passes=25,presentations_per_arm=125000,last_batch_each_pass=8)
    model=r['teacher']['model'];expected_shapes={'stem.weight':(16,3,3,3),'fc.weight':(10,64),'fc.bias':(10,)}
    def add_bn(prefix,width):
        for name in ('weight','bias','running_mean','running_var'):expected_shapes[prefix+'.'+name]=(width,)
        expected_shapes[prefix+'.num_batches_tracked']=()
    add_bn('bn',16);ci=16
    for i,co in enumerate([16]*3+[32]*3+[64]*3):
        expected_shapes[f'blocks.{i}.c1.weight']=(co,ci,3,3)
        expected_shapes[f'blocks.{i}.c2.weight']=(co,co,3,3)
        add_bn(f'blocks.{i}.b1',co);add_bn(f'blocks.{i}.b2',co);ci=co
    assert set(model)==set(expected_shapes)
    for key,shape in expected_shapes.items():
        assert tuple(model[key].shape)==shape
        assert model[key].dtype==(torch.int64 if key.endswith('num_batches_tracked') else torch.float32)
print('INDEPENDENT_DATA_SPLIT_NORMALIZATION_SCHEDULE_CHECKS_PASS',flush=True)
results={}
with torch.inference_mode():
    for seed,r in records.items():
        s=r['teacher']['model'];before=thash(s)
        assert sum(t.numel() for k,t in s.items() if not k.endswith(('running_mean','running_var','num_batches_tracked')))==269722
        predictions={n:[] for n in ('Teacher',)+ARMS};logits_saved={n:[] for n in predictions}
        sum_y=np.zeros((32,16,16),dtype=np.float64);sum_y2=np.zeros_like(sum_y);sse={n:0. for n in ARMS}
        for j in range(0,5000,128):
            ids=splits['validation'][j:j+128]
            x=torch.from_numpy(images[ids].copy()).float()/255
            x=(x-torch.tensor(mean,dtype=torch.float32)[None,:,None,None])/torch.tensor(std,dtype=torch.float32)[None,:,None,None]
            h,y=endpoints(x,s);assert h.shape[1:]==y.shape[1:]==(32,16,16)
            a=y.numpy().astype(np.float64);sum_y+=a.sum(axis=0);sum_y2+=(a*a).sum(axis=0)
            ys={'Teacher':y}
            for name in ARMS:
                p=replace(h,r['final']['arms'][name]);ys[name]=p
                d=p.numpy().astype(np.float64)-a;sse[name]+=float((d*d).sum())
            for name,p in ys.items():
                z=suffix(p,s);assert bool(torch.isfinite(z).all())
                predictions[name].append(z.argmax(dim=1).numpy());logits_saved[name].append(z.numpy())
        assert thash(s)==before
        truth=labels[splits['validation']];predictions={n:np.concatenate(v) for n,v in predictions.items()}
        logits_saved={n:np.concatenate(v) for n,v in logits_saved.items()}
        counts={n:int(np.count_nonzero(v==truth)) for n,v in predictions.items()}
        assert counts['Teacher']==r['metrics']['correct']
        centered=float((sum_y2-sum_y*sum_y/5000).sum());assert centered>=0
        denominator=centered+1e-12*5000*32*16*16
        arms={}
        for name in ARMS:
            original=r['validation']['arms'][name]
            nmse=sse[name]/denominator;uncentered=sse[name]/float(sum_y2.sum())
            drop=(counts['Teacher']-counts[name])/50
            arms[name]=dict(correct=counts[name],accuracy=counts[name]/5000,drop_pp=drop,utility_drop_le_2pp=counts[name]>=counts['Teacher']-100,
                            centered_span_nmse=nmse,sse=sse[name],centered_teacher_ss=centered,uncentered_energy_nmse=uncentered,
                            nmse_absolute_difference_from_original=abs(nmse-original['centered_span_nmse']))
            assert counts[name]==original['correct'] and counts[name]/5000==original['accuracy']
            assert arms[name]['utility_drop_le_2pp']==original['utility_drop_le_2pp']
            assert math.isclose(nmse,original['centered_span_nmse'],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(sse[name],original['sse'],rel_tol=1e-12,abs_tol=1e-7)
            assert math.isclose(centered,original['centered_teacher_ss'],rel_tol=1e-12,abs_tol=1e-7)
        np.savez_compressed(OUT/f'predictions_seed_{seed}.npz',validation_ids=splits['validation'],truth=truth,
                            **{n.replace('-','_')+'_predicted':v for n,v in predictions.items()},
                            **{n.replace('-','_')+'_logits':v for n,v in logits_saved.items()})
        np.savez_compressed(OUT/f'endpoint_statistics_seed_{seed}.npz',sum_teacher=sum_y,sum_squared_teacher=sum_y2)
        result=dict(seed=seed,n=5000,teacher_correct=counts['Teacher'],teacher_accuracy=counts['Teacher']/5000,arms=arms,
                    checks=r['checks'],teacher_state_unchanged_during_audit=True,teacher_state_sha256=before)
        results[str(seed)]=result
        (OUT/f'result_seed_{seed}.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps(dict(seed=seed,counts=counts,arms=arms)),flush=True)
# Prove evidence bytes used by this audit remained unchanged; hash only allowed files.
for path,expected_hash in evidence.items():
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    assert h.hexdigest()==expected_hash, 'Evidence changed: '+path
assert git('status','--porcelain')=='' and git('rev-parse','HEAD')==COMMIT
report=dict(status='PASS',source_commit=COMMIT,elapsed_seconds=time.time()-started,
            environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),deterministic=torch.are_deterministic_algorithms_enabled()),
            official_test_member_opened=False,official_test_decoded=False,read_members=extracted,
            training_replayed=False,new_optimizer_steps=0,source_evidence_unchanged=True,normalization_sha256=normalization_hash,
            git_chronology=git('log','-6','--format=%H %aI %s'),results=results,
            limits=['Independent functional inference and aggregate metrics reproduced; production evaluator was not imported.',
                    'Actual historical 1000-update trajectory and per-update IDs not replayed; final counters, deterministic schedule, persisted initialization, source and hash bindings verified.',
                    'Teacher is frozen by reviewed source and excluded from saved optimizers. No before/after teacher tensors were persisted by fit, so historical state immutability cannot be independently proved from snapshots; original teacher used unchanged in this audit.',
                    'Audit confirms only two planned seeds; third remains required. Validation is not official-test evidence and descriptive <=2pp failure is not a population noninferiority analysis.',
                    'Cryptographic hashes establish local integrity/binding, not independent artifact authorship or proof that no unlogged activity occurred.'])
(OUT/'audit_result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
(OUT/'evidence_sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
print('AUDIT_COMPLETE_PASS',flush=True)
