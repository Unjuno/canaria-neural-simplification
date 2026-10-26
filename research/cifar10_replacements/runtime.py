"""Preparation-only fit plumbing. No CLI, data download, or test evaluation.

Caller MUST obtain parent plan approval and teacher admission before real use.
Call open_launch BEFORE loading data or numerical teacher artifacts on resume.
"""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import platform
import signal
import time

import numpy as np
import torch
from . import core as c

from .launch import DEADLINE, CAP_SECONDS, open_launch, arm_deadline


def atomic_bytes(path, blob):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        f.write(blob); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def tensor_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode()+b'\0'+str(t.dtype).encode()+str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def finite(value):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite tensor')
    elif isinstance(value,dict):
        for v in value.values(): finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value: finite(v)
    elif isinstance(value,float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite scalar')


def envelope(state):
    buffer=io.BytesIO(); torch.save(state,buffer); payload=buffer.getvalue()
    return b'CANARIA-REPLACEMENT1\n'+c.sha(payload).encode()+b'\n'+payload


def decode(blob):
    magic,digest,payload=blob.split(b'\n',2)
    if magic != b'CANARIA-REPLACEMENT1' or digest != c.sha(payload).encode():
        raise ValueError('Checkpoint integrity mismatch')
    return torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)


class FitSession:
    """All three arms share a single teacher cap, admitted ID schedule and final step."""
    def __init__(self, teacher, teacher_hash, calibration_ids, seed, launch, data_identity,
                 source_identity, now=None, teacher_validation_correct=None):
        if seed not in c.SEEDS: raise ValueError('Unplanned teacher seed')
        self.teacher=teacher
        if teacher_validation_correct is not None and not 4250<=teacher_validation_correct<=5000:
            raise ValueError('Teacher quality gate invalid')
        self.teacher_validation_correct=teacher_validation_correct
        self.launch=copy.deepcopy(launch)
        self.check_time(now)
        self.arms=c.make_arms(seed)
        self.optimizers={name:c.make_optimizer(arm) for name,arm in self.arms.items()}
        self.batches=c.calibration_batches(tuple(calibration_ids),seed)
        self.initial={name:copy.deepcopy(arm.state_dict()) for name,arm in self.arms.items()}
        self.lock=dict(environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                       platform=platform.platform(),threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
                       deterministic=torch.are_deterministic_algorithms_enabled()),schema=1,teacher_sha256=teacher_hash,seed=seed,teacher_validation_correct=teacher_validation_correct,
                       recipe=copy.deepcopy(c.RECIPE),span=list(c.SPAN),activation='inner-and-post-add-ReLU',
                       ranks={'Comp-joint':[16],'Sep-local':[8,8],'Sep-joint':[8,8]},
                       initialization={name:tensor_hash(state) for name,state in self.initial.items()},
                       batch_schedule_sha256=c.canonical_hash(self.batches),
                       data_identity=copy.deepcopy(data_identity),source_identity=copy.deepcopy(source_identity),
                       launch=copy.deepcopy(launch),primary_nmse='centered per-feature teacher variance; eps=1e-12')
        self.lock_sha256=c.canonical_hash(self.lock)
        self.cursor=0
        self.failed=False

    def check_time(self,now=None):
        now=time.time() if now is None else now
        if now >= self.launch['expiry']: raise TimeoutError('Original shared fit cap expired')

    def step(self, ids, images):
        """Images must correspond exactly to batches[cursor], via approved calibration accessor."""
        self.check_time()
        if self.failed: raise RuntimeError('Failed session requires last-good restore')
        if self.cursor >= c.RECIPE['updates']: raise ValueError('Exactly1000 updates; already complete')
        if tuple(ids) != self.batches[self.cursor]: raise ValueError('Incorrect scheduled batch IDs')
        if len(images) != len(self.batches[self.cursor]): raise ValueError('Incorrect scheduled batch size')
        try:
            endpoints=self.teacher.features(images); finite(endpoints)
            for name,arm in self.arms.items():
                self.check_time(); arm.train()
                optimizer=self.optimizers[name]; optimizer.zero_grad(set_to_none=True)
                loss=c.objective(name,arm,endpoints); finite(loss); loss.backward()
                if any(p.grad is None for p in arm.parameters()): raise ValueError('Missing replacement gradient')
                finite([p.grad for p in arm.parameters()]); optimizer.step()
                finite(arm.state_dict()); finite(optimizer.state_dict())
            self.check_time()
        except BaseException:
            self.failed=True
            raise
        self.cursor+=1

    def state(self):
        if self.failed: raise RuntimeError('Never serialize partial/nonfinite update')
        state=dict(schema=1,lock=self.lock,lock_sha256=self.lock_sha256,cursor=self.cursor,
                   initial=self.initial,arms={n:a.state_dict() for n,a in self.arms.items()},
                   optimizers={n:o.state_dict() for n,o in self.optimizers.items()},
                   torch_rng=torch.get_rng_state(),python_rng=random.getstate())
        nr=np.random.get_state()
        state['numpy_rng']=[nr[0],nr[1].tolist(),int(nr[2]),int(nr[3]),float(nr[4])]
        finite(state)
        return state

    def save(self,path):
        # Saving a completed finite step is allowed after cap; no new computation.
        atomic_bytes(path,envelope(self.state()))

    def restore(self,path):
        self.check_time()  # reject before checkpoint decode
        state=decode(Path(path).read_bytes())
        if not (state.get('schema') == 1 and state.get('lock') == self.lock
                and state.get('lock_sha256') == self.lock_sha256
                and type(state.get('cursor')) is int and 0 <= state['cursor'] <=1000):
            raise ValueError('Resume recipe/teacher/data/source/launch/cursor mismatch')
        finite(state)
        if {n:tensor_hash(v) for n,v in state['initial'].items()} != self.lock['initialization']:
            raise ValueError('Initial snapshot mismatch')
        for n,o in self.optimizers.items():
            if state['optimizers'][n]['param_groups'] != o.state_dict()['param_groups']:
                raise ValueError('Optimizer recipe mismatch')
        for n,a in self.arms.items():
            a.load_state_dict(state['arms'][n],strict=True)
            self.optimizers[n].load_state_dict(state['optimizers'][n])
        torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
        nr=state['numpy_rng']; np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
        self.cursor=state['cursor']; self.failed=False

    @torch.no_grad()
    def final_validation(self, batches, expected_ids):
        """Validation-only (IDs checked exactly); never accepts test designation."""
        self.check_time()
        if self.cursor !=1000 or self.failed: raise ValueError('Final1000-update checkpoint required')
        expected=tuple(expected_ids)
        if len(expected)!=5000 or len(set(expected))!=5000: raise ValueError('Fixed validation size required')
        if set(expected)&set(sum((list(b) for b in self.batches[:40]),[])):
            raise ValueError('Validation/calibration overlap')
        if c.sha(np.asarray(expected,dtype='<i8').tobytes()) != self.lock['data_identity']['splits']['validation']['sha256']:
            raise ValueError('Unapproved validation IDs')
        accum={n:c.EndpointStats() for n in self.arms}; correct={n:0 for n in self.arms}; seen=[]
        for ids,images,labels in batches:
            self.check_time(); ids=tuple(ids)
            if ids != expected[len(seen):len(seen)+len(ids)]: raise ValueError('Validation order/membership mismatch')
            if len(images)!=len(ids) or len(labels)!=len(ids): raise ValueError('Validation batch mismatch')
            x,_,target=self.teacher.features(images)
            for n,a in self.arms.items():
                a.eval(); predicted=a(x); accum[n].update(predicted,target)
                logits=self.teacher.installed_logits(images,a); finite(logits)
                correct[n]+=int((logits.argmax(1)==labels).sum())
            seen.extend(ids)
        if tuple(seen)!=expected: raise ValueError('Incomplete validation')
        return dict(lock_sha256=self.lock_sha256,updates=self.cursor,split='validation',official_test_decoded=False,
                    arms={n:dict(correct=correct[n],accuracy=correct[n]/5000,
                        utility_drop_le_2pp=(correct[n]>=self.teacher_validation_correct-100) if self.teacher_validation_correct is not None else None,
                        teacher_correct=self.teacher_validation_correct,**accum[n].result()) for n in self.arms})


def cohort_release_gate(records, approved_release_manifest):
    """Eligibility check ONLY. No test loader/evaluator and no automatic release.

    Separate approval must bind all3 planned statuses, recipes, final artifact
    hashes and validation outcomes; failed teachers remain visible.
    """
    if approved_release_manifest.get('action') != 'separately-reviewed-official-test-release':
        raise ValueError('Separate cohort release approval required')
    if set(records) != set(c.SEEDS): raise ValueError('All planned seeds including failures required')
    for record in records.values():
        if record.get('status') not in ('complete','teacher-quality-failed','teacher-incomplete','fit-failed'):
            raise ValueError('Cohort outcome not locked')
        if record['status']=='complete' and not all(record.get(k) for k in ('recipe_sha256','teacher_sha256','replacement_sha256','validation_sha256')):
            raise ValueError('Final artifacts/outcomes not locked')
    if approved_release_manifest.get('cohort_sha256') != c.canonical_hash(records):
        raise ValueError('Cohort release binding mismatch')
    return {'eligible':True,'cohort_sha256':c.canonical_hash(records),'test_evaluation_implemented':False}


def fit_admitted_teacher(data, admission, output, launch, source_identity, *,
                         worker_lock, execution_authorized=False, resume=False):
    """Minimum future orchestration; NOT RUN during preparation.

    Required order: stdlib launch.open_launch + arm_deadline; numerical imports;
    production pinned data loader and teacher admission; this function. Caller
    must supply actual announced plan hash, not invent one. No automatic retries.
    """
    if execution_authorized is not True:
        raise PermissionError('Parent-announced plan and admitted teacher required')
    from .data import AdmissionBundle,model_state_hash
    if not isinstance(admission,AdmissionBundle): raise ValueError('Production admission bundle required')
    if (admission.data_binding!=data.binding or admission.identity_sha256!=c.canonical_hash(data.identity)
            or admission.model_state_sha256!=model_state_hash(admission.teacher.model)):
        raise ValueError('Admitted teacher/data binding changed')
    if worker_lock.closed or worker_lock.name!='/workspace/shared/canaria_cifar_replacement_worker.lock':
        raise ValueError('Held single-worker bootstrap lock required')
    if json.loads((Path(output)/'launch.json').read_text())!=launch:
        raise ValueError('Launch must equal durable bootstrap record')
    if (torch.get_num_threads()!=4 or torch.get_num_interop_threads()!=1
            or not torch.are_deterministic_algorithms_enabled()
            or any(os.environ.get(k)!='4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))):
        raise ValueError('Configure4intra-op/1inter-op deterministic threads after stdlib bootstrap')
    teacher=admission.teacher; teacher_hash=admission.checkpoint_sha256; seed=admission.seed
    arm_deadline(launch)
    output=Path(output)
    ids=data.splits['calibration']; data.require_calibration_ids(ids)
    session=FitSession(teacher,teacher_hash,tuple(int(i) for i in ids),seed,launch,
                       data.identity,source_identity,teacher_validation_correct=admission.teacher_validation_correct)
    lockfile=output/'recipe_lock.json'
    lockbytes=json.dumps(session.lock,sort_keys=True,separators=(',',':')).encode()
    checkpoint=output/'last.ckpt'
    if resume:
        if lockfile.read_bytes()!=lockbytes: raise ValueError('Immutable recipe lock mismatch')
        session.restore(checkpoint)
    else:
        # Exclusive lock; no silent overwrite of earlier runs or initialization.
        with lockfile.open('xb') as f:
            f.write(lockbytes); f.flush(); os.fsync(f.fileno())
        session.save(output/'initial.ckpt'); session.save(checkpoint)
    while session.cursor<1000:
        batch_ids=session.batches[session.cursor]
        images=data.image_batch(batch_ids,split='calibration')
        session.step(batch_ids,images)
        session.save(checkpoint)
    finalpath=output/'final_update1000.ckpt'; session.save(finalpath)
    validation_ids=tuple(int(i) for i in data.splits['validation'])
    def validation_batches():
        for start in range(0,5000,128):
            ids=validation_ids[start:start+128]
            yield ids,data.image_batch(ids,split='validation'),torch.from_numpy(data.labels[list(ids)].copy()).long()
    result=session.final_validation(validation_batches(),validation_ids)
    result.update(status='complete',seed=seed,teacher_sha256=teacher_hash,
                  checkpoint_sha256=c.sha(finalpath.read_bytes()),selection='fixed update1000; no best selection')
    atomic_bytes(output/'validation.json',json.dumps(result,sort_keys=True,allow_nan=False).encode())
    signal.setitimer(signal.ITIMER_REAL,0)
    return result
