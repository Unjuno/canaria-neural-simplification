"""PROVISIONAL future entry. No default paths, release inference, resume or retry."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from research.cifar10_replacements import entry as prior
from research.cifar10_replacements.launch import bootstrap, DEADLINE, CAP_SECONDS

REGISTRY=Path('/workspace/shared/canaria_cifar10_official_test_one_run.json')
ROOT=prior.ROOT
SEEDS=prior.SEEDS
SOURCE_FILES=prior.SOURCE_FILES+tuple('research/cifar10_heldout/'+n+'.py' for n in ('entry','evaluate'))
CLAIM_SCOPE='descriptive qualified-seed results only; no statistical noninferiority or generalization claim; all planned failures reported'

def canonical(value):
    return prior.digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())

def release_guard(plan):
    """Pure guard: invalid release cannot open any file (including source/data)."""
    if not (plan.get('status')=='parent-approved-and-user-announced'
            and plan.get('action')=='separately-reviewed-official-test-release'
            and plan.get('planned_seeds')==list(SEEDS) and plan.get('updates')==1000
            and plan.get('threads')==4 and plan.get('interop_threads')==1
            and plan.get('cap_seconds')==CAP_SECONDS and plan.get('absolute_deadline')==DEADLINE
            and plan.get('claim_scope')==CLAIM_SCOPE and plan.get('no_adaptation') is True
            and plan.get('one_run') is True):
        raise ValueError('Separate announced release and fixed protocol required')
    if not all(valid_sha(plan.get(k)) for k in ('data_identity_sha256','data_manifest_sha256')):
        raise ValueError('Frozen normalization/split/data binding required')
    records=plan.get('cohort',{})
    if set(records)!=set(map(str,SEEDS)) or canonical(records)!=plan.get('cohort_sha256'):
        raise ValueError('All three locked outcomes required')
    policy=plan.get('predeclared_failure_policy',{})
    qualified=[]
    for seed,r in records.items():
        if r.get('status')=='complete':
            for k in ('recipe','teacher','replacement','validation','export','metrics'):
                if not r.get(k+'_path') or not valid_sha(r.get(k+'_sha256')):
                    raise ValueError('Final artifact binding missing')
            qualified.append(int(seed))
        else:
            # No fallback exclusion, retries, or unresolved statuses.
            if not (r.get('status') in ('teacher-quality-failed','teacher-incomplete','fit-failed')
                    and r.get('resolved') is True and r.get('resolution')=='explicitly-excluded-from-qualified-descriptive-analysis'
                    and r.get('failure_class') in policy.get('allowed_classes',[])
                    and policy.get('status')=='parent-predeclared-before-outcomes'
                    and valid_sha(policy.get('declaration_sha256'))
                    and r.get('evidence_path') and valid_sha(r.get('evidence_sha256'))):
                raise ValueError('Unresolved/unapproved failure blocks whole cohort')
    if not qualified: raise ValueError('No qualified seeds')
    if set(plan.get('source_sha256',{}))!=set(SOURCE_FILES): raise ValueError('Complete source pins required')
    if plan.get('replacement_source_sha256')!={k:plan['source_sha256'][k] for k in prior.SOURCE_FILES}:
        raise ValueError('Replacement source must match reviewed code')
    return qualified

def valid_sha(s):
    return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)

def verified_bytes(path,digest):
    blob=Path(path).read_bytes()
    if not valid_sha(digest) or prior.digest(blob)!=digest: raise ValueError('Artifact/source SHA256 mismatch')
    return blob

def verify(plan,plan_hash,slot):
    qualified=release_guard(plan)
    if not (slot.get('status')=='parent-reserved-free-slot' and slot.get('plan_sha256')==plan_hash
            and slot.get('output')==plan.get('output') and slot.get('threads')==4
            and type(slot.get('other_science_threads')) is int and 0<=slot['other_science_threads']<=4
            and slot.get('global_science_threads')==8 and time.time()<slot.get('expires_at',0)):
        raise ValueError('Current parent CPU slot required')
    for name,h in plan['source_sha256'].items(): verified_bytes(ROOT/name,h)
    if any(r['status']!='complete' for r in plan['cohort'].values()):
        policy=plan['predeclared_failure_policy']
        declaration=json.loads(verified_bytes(policy['declaration_path'],policy['declaration_sha256']))
        if declaration!={k:v for k,v in policy.items() if k not in ('declaration_path','declaration_sha256')}:
            raise ValueError('Failure declaration mismatch')
    return qualified

def numerical_modules():
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    from . import evaluate
    return evaluate

def claim_one_run(output,plan_hash):
    """Exclusive durable claim; existing/partial records are never adopted or reset."""
    with REGISTRY.open('x') as f:
        json.dump(dict(plan_sha256=plan_hash,output=str(output),started=time.time()),f)
        f.flush();os.fsync(f.fileno())
    fd=os.open(REGISTRY.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def execute(plan_path,plan_hash,slot_path,slot_hash):
    plan=prior.checked_json(plan_path,plan_hash); slot=prior.checked_json(slot_path,slot_hash)
    qualified=verify(plan,plan_hash,slot)
    output=Path(plan['output']); output.mkdir(parents=True,exist_ok=False)
    claim_one_run(output,plan_hash)
    worker=None
    try:
        launch,worker=bootstrap(output,plan_hash)
        # Hash every qualified and failed-seed artifact before any data/checkpoint decode.
        blobs={}
        for seed,r in plan['cohort'].items():
            keys=('recipe','teacher','replacement','validation','export','metrics') if r['status']=='complete' else ('evidence',)
            blobs[seed]={k:verified_bytes(r[k+'_path'],r[k+'_sha256']) for k in keys}
        result=numerical_modules().run(plan,qualified,blobs,output,launch)
        prior.write_status(output,dict(status='complete',plan_sha256=plan_hash,result_sha256=result))
        return result
    except BaseException as exc:
        prior.write_status(output,dict(status='failed',plan_sha256=plan_hash,error_type=type(exc).__name__,error=str(exc),
                                      test_may_have_been_opened=True,retry_authorized=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if worker is not None: worker.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','plan-sha256','slot','slot-sha256'): p.add_argument('--'+n,required=True)
    a=p.parse_args();execute(a.plan,a.plan_sha256,a.slot,a.slot_sha256)
