"""Future-only admitted evaluation. Import does not read data or checkpoints."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import torch
from research.cifar10_replacements import core as c, data as d, runtime as r
from .entry import canonical, CLAIM_SCOPE

ARMS=('Comp-joint','Sep-local','Sep-joint')
TEST='cifar-10-batches-py/test_batch'

def check_time(launch):
    if time.time()>=launch['expiry']: raise TimeoutError('Heldout cap expired')

def load_test(preflight):
    """Called only after ALL seed admissions. Same descriptor hash then decode."""
    path=Path(preflight)/'cifar-10-python.tar.gz'
    with path.open('rb') as stream:
        md5=hashlib.md5(); sha=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''): md5.update(block);sha.update(block)
        if md5.hexdigest()!=d.OFFICIAL_MD5 or sha.hexdigest()!=d.APPROVED_MANIFEST['preflight_sha256'][path.name]:
            raise ValueError('Official archive MD5/SHA256 mismatch')
        stream.seek(0)
        with tarfile.open(fileobj=stream,mode='r:gz') as tf:
            allowed={TEST,'cifar-10-batches-py','cifar-10-batches-py/batches.meta','cifar-10-batches-py/readme.html'}|{f'cifar-10-batches-py/data_batch_{i}' for i in range(1,6)}
            members=tf.getmembers()
            if len({m.name for m in members})!=len(members): raise ValueError('Duplicate tar member')
            for m in members:
                if not (m.name.rstrip('/') in allowed and (m.isfile() or m.isdir()) and 0<=m.size<40000000):
                    raise ValueError('Unsafe tar member')
            member=tf.getmember(TEST)
            if not member.isfile(): raise ValueError('Test member must be regular file')
            blob=tf.extractfile(member).read()
    record=d.SafeUnpickler(io.BytesIO(blob),encoding='bytes').load()
    x=record[b'data']; y=np.asarray(record[b'labels'])
    if not (type(x) is np.ndarray and x.dtype==np.uint8 and x.shape==(10000,3072)
            and y.shape==(10000,) and y.dtype.kind in 'iu' and np.array_equal(np.bincount(y),np.full(10,1000))):
        raise ValueError('Malformed test population')
    return x.reshape(10000,3,32,32),y,dict(archive_md5=md5.hexdigest(),archive_sha256=sha.hexdigest(),test_member_sha256=c.sha(blob))

def verify_frozen_fit(state,lock,calibration_ids,seed):
    """Reconstruct frozen invariants, not just a self-asserted final cursor."""
    if state.get('cursor')!=1000: raise ValueError('Final update1000 required')
    schedule=c.calibration_batches(tuple(int(i) for i in calibration_ids),seed)
    if lock.get('batch_schedule_sha256')!=c.canonical_hash(schedule):
        raise ValueError('Frozen calibration schedule mismatch')
    arms=c.make_arms(seed)
    expected_initial={name:r.tensor_hash(a.state_dict()) for name,a in arms.items()}
    if lock.get('initialization')!=expected_initial or set(state.get('initial',{}))!=set(arms):
        raise ValueError('Frozen initialization identity mismatch')
    if {n:r.tensor_hash(v) for n,v in state['initial'].items()}!=expected_initial:
        raise ValueError('Initial snapshot differs from fixed constructor')
    if set(state.get('optimizers',{}))!=set(arms): raise ValueError('All Adam states required')
    for name,arm in arms.items():
        expected=c.make_optimizer(arm).state_dict()['param_groups']
        saved=state['optimizers'][name]
        if saved.get('param_groups')!=expected: raise ValueError('Frozen Adam recipe mismatch')
        ids=[i for group in expected for i in group['params']]
        if set(saved.get('state',{}))!=set(ids): raise ValueError('Missing/extra Adam parameter state')
        for ident,param in zip(ids,arm.parameters()):
            item=saved['state'][ident]
            if set(item)!={'step','exp_avg','exp_avg_sq'}: raise ValueError('Unexpected Adam state')
            step=item['step']
            if not (isinstance(step,torch.Tensor) and step.ndim==0 and torch.isfinite(step) and step.item()==1000):
                raise ValueError('Every Adam parameter must have exactly1000 steps')
            for key in ('exp_avg','exp_avg_sq'):
                value=item[key]
                if not (isinstance(value,torch.Tensor) and value.shape==param.shape and value.dtype==param.dtype and torch.isfinite(value).all()):
                    raise ValueError('Invalid Adam moments')
            if (item['exp_avg_sq']<0).any(): raise ValueError('Negative Adam squared moment')
    return arms

def admit_all(plan,qualified,blobs):
    # Independently pinned preflight reconstructs normalization and split identity;
    # it opens training members only, never test_batch.
    data=d.load_approved_data(plan['preflight'],execution_authorized=True)
    if canonical(data.identity)!=plan.get('data_identity_sha256') or data.binding!=plan.get('data_manifest_sha256'):
        raise ValueError('Release normalization/split/data identity mismatch')
    admitted={}
    for seed in qualified:
        b=blobs[str(seed)]; row=plan['cohort'][str(seed)]
        export=json.loads(b['export']);metrics=json.loads(b['metrics']); lock=json.loads(b['recipe']); validation=json.loads(b['validation'])
        teacher=d.admit_teacher_bundle(export,metrics,b['teacher'],data,seed).teacher
        state=r.decode(b['replacement']);r.finite(state)
        if not (state.get('schema')==1 and state.get('cursor')==1000
                and canonical(state.get('lock'))==canonical(lock)==state.get('lock_sha256')
                and lock.get('teacher_sha256')==row['teacher_sha256'] and lock.get('seed')==seed
                and canonical(lock.get('recipe'))==canonical(c.RECIPE)
                and lock.get('span')==list(c.SPAN)
                and lock.get('activation')=='inner-and-post-add-ReLU'
                and lock.get('ranks')=={'Comp-joint':[16],'Sep-local':[8,8],'Sep-joint':[8,8]}
                and lock.get('teacher_validation_correct')==metrics['correct']
                and lock.get('environment',{}).get('threads')==4
                and lock.get('environment',{}).get('interop')==1
                and lock.get('environment',{}).get('deterministic') is True
                and lock.get('data_identity')==data.identity
                and lock.get('source_identity')==plan['replacement_source_sha256']
                and validation.get('status')=='complete' and validation.get('seed')==seed
                and validation.get('updates')==1000 and validation.get('split')=='validation'
                and validation.get('official_test_decoded') is False
                and validation.get('checkpoint_sha256')==row['replacement_sha256']
                and validation.get('teacher_sha256')==row['teacher_sha256']
                and validation.get('lock_sha256')==state['lock_sha256']):
            raise ValueError('Final student/recipe/source/validation binding mismatch')
        arms=verify_frozen_fit(state,lock,data.splits['calibration'],seed)
        if set(state['arms'])!=set(ARMS): raise ValueError('All three arms required')
        for name,a in arms.items(): a.load_state_dict(state['arms'][name],strict=True); a.requires_grad_(False).eval()
        teacher.eval()
        admitted[seed]=(teacher,arms)
    return data,admitted

def summarize(predictions,labels,stats):
    n=len(labels)
    correct={k:int(np.count_nonzero(v==labels)) for k,v in predictions.items()}
    arms={k:dict(correct=correct[k],accuracy=correct[k]/n,teacher_correct=correct['teacher'],
                 paired_degradation_pp=100*(correct['teacher']-correct[k])/n,
                 descriptive_drop_le_2pp=correct[k]>=correct['teacher']-n*.02,
                 teacher_only_correct=int(np.count_nonzero((predictions['teacher']==labels)&(predictions[k]!=labels))),
                 student_only_correct=int(np.count_nonzero((predictions['teacher']!=labels)&(predictions[k]==labels))),
                 **stats[k].result()) for k in ARMS}
    contrasts={a+' minus '+b:dict(correct_delta=correct[a]-correct[b],accuracy_delta_pp=100*(correct[a]-correct[b])/n,
                 centered_nmse_delta=arms[a]['centered_span_nmse']-arms[b]['centered_span_nmse'])
                 for a,b in (('Comp-joint','Sep-local'),('Comp-joint','Sep-joint'),('Sep-joint','Sep-local'))}
    return dict(n=n,teacher_correct=correct['teacher'],teacher_accuracy=correct['teacher']/n,arms=arms,contrasts=contrasts)

def run(plan,qualified,blobs,output,launch):
    r.cohort_release_gate({int(k):v for k,v in plan['cohort'].items()},plan)
    data,admitted=admit_all(plan,qualified,blobs)
    check_time(launch)
    images,labels,data_hashes=load_test(plan['preflight']) # LAST: only after all admission checks
    results={}
    with torch.inference_mode():
        for seed,(teacher,arms) in admitted.items():
            check_time(launch)
            teacher.eval()
            for arm in arms.values(): arm.eval()
            if any(m.training for obj in (teacher,*arms.values()) for m in obj.modules()): raise ValueError('Every module must be EVAL')
            before={n:d.model_state_hash(obj) for n,obj in [('teacher',teacher),*arms.items()]}
            predictions={n:[] for n in ('teacher',*ARMS)};stats={n:c.EndpointStats() for n in ARMS}
            for start in range(0,10000,128):
                check_time(launch)
                x=torch.from_numpy(images[start:start+128].copy()).float()/255
                x=(x-torch.tensor(data.mean,dtype=torch.float32)[None,:,None,None])/torch.tensor(data.std,dtype=torch.float32)[None,:,None,None]
                r.finite(x)
                logits=teacher.model(x);r.finite(logits)
                predictions['teacher'].append(logits.argmax(1).cpu().numpy())
                endpoint,_,target=teacher.features(x);r.finite(target)
                for name,arm in arms.items():
                    estimate=arm(endpoint);stats[name].update(estimate,target)
                    logits=teacher.installed_logits(x,arm);r.finite(logits)
                    predictions[name].append(logits.argmax(1).cpu().numpy())
            predictions={k:np.concatenate(v) for k,v in predictions.items()}
            if any(v.shape!=(10000,) for v in predictions.values()): raise ValueError('Incomplete common test population')
            if before!={n:d.model_state_hash(obj) for n,obj in [('teacher',teacher),*arms.items()]}: raise ValueError('Model state adapted')
            raw=io.BytesIO();np.savez(raw,image_id=np.arange(10000),labels=labels,**predictions)
            rawbytes=raw.getvalue();r.atomic_bytes(output/f'{seed}_predictions.npz',rawbytes)
            results[str(seed)]=dict(**summarize(predictions,labels,stats),predictions_sha256=c.sha(rawbytes))
    result=dict(status='complete',claim_scope=CLAIM_SCOPE,cohort=plan['cohort'],qualified_seeds=qualified,
                source_sha256=plan['source_sha256'],data_identity_sha256=canonical(data.identity),
                data_manifest_sha256=data.binding,data_hashes=data_hashes,results=results)
    r.finite(result);check_time(launch)
    encoded=json.dumps(result,sort_keys=True,allow_nan=False).encode();r.atomic_bytes(output/'results.json',encoded)
    return c.sha(encoded)
