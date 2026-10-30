"""Manifest-driven future entry. PREPARED, NOT AUTHORIZED OR RUN ON REAL DATA.

Only standard-library imports until source/plan/slot checks and timed bootstrap.
No test decoder or automatic cohort release. Parent supplies trusted manifest hashes.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import signal
import time

from .launch import bootstrap, CAP_SECONDS, DEADLINE

SEEDS=(26102601,26102602,26102603)
ROOT=Path(__file__).resolve().parents[2]
SOURCE_FILES=tuple('research/cifar10_replacements/'+name+'.py' for name in
                   ('core','data','launch','runtime','entry'))+('research/cifar10_teacher/model.py',)


def digest(blob): return hashlib.sha256(blob).hexdigest()


def checked_json(path,expected_hash):
    blob=Path(path).read_bytes()
    if digest(blob)!=expected_hash: raise ValueError('Manifest/artifact SHA256 mismatch: '+str(path))
    return json.loads(blob)


def verify_plan(plan_path,plan_hash,slot_path,slot_hash,seed,resume):
    plan=checked_json(plan_path,plan_hash); slot=checked_json(slot_path,slot_hash)
    if not (plan.get('status')=='parent-approved-and-user-announced'
            and plan.get('planned_seeds')==list(SEEDS) and seed in SEEDS
            and plan.get('threads')==4 and plan.get('interop_threads')==1
            and plan.get('max_replacement_workers')==1 and plan.get('global_science_threads')==8
            and plan.get('per_teacher_cap_seconds')==CAP_SECONDS and plan.get('absolute_deadline')==DEADLINE
            and plan.get('official_test')=='sealed' and plan.get('updates')==1000):
        raise ValueError('Plan not approved or fixed protocol mismatch')
    hashes=plan.get('source_sha256',{})
    if set(hashes)!=set(SOURCE_FILES): raise ValueError('Complete reviewed source manifest required')
    for name in SOURCE_FILES:
        if digest((ROOT/name).read_bytes())!=hashes[name]: raise ValueError('Reviewed source changed: '+name)
    runs=plan['runs']
    if not set(runs)<=set(str(s) for s in SEEDS) or len({v['output'] for v in runs.values()})!=len(runs):
        raise ValueError('Unplanned seed or duplicate output directories')
    run=runs[str(seed)]
    if not (slot.get('status')=='parent-reserved-free-slot' and slot.get('plan_sha256')==plan_hash
            and slot.get('seed')==seed and slot.get('output')==run['output']
            and slot.get('replacement_threads')==4 and type(slot.get('other_science_threads')) is int
            and 0<=slot['other_science_threads']<=4 and slot.get('global_science_threads')==8
            and slot.get('resume_authorized') is resume and time.time()<slot.get('expires_at',0)):
        raise ValueError('Current parent CPU slot/resume authorization required')
    return plan,run


def preliminary_teacher(run,seed):
    """Gate JSON only, before numerical/data/checkpoint loading."""
    export=checked_json(run['teacher_export'],run['teacher_export_sha256'])
    metrics=checked_json(run['teacher_metrics'],run['teacher_metrics_sha256'])
    for record in (export,metrics):
        if (record.get('seed')!=seed or record.get('epoch')!=120
                or record.get('checkpoint_sha256')!=run['teacher_checkpoint_sha256']):
            raise ValueError('Teacher seed/final checkpoint identity mismatch')
    if not (metrics.get('status')=='complete' and metrics.get('adequate') is True
            and metrics.get('n')==5000 and type(metrics.get('correct')) is int
            and 4250<=metrics['correct']<=5000 and metrics.get('accuracy')==metrics['correct']/5000
            and metrics.get('test_batch_decoded') is False and metrics.get('calibration_used') is False
            and export.get('official_test_decoded') is False and export.get('calibration_used') is False):
        raise ValueError('Final teacher quality/isolation gate failed')
    return export,metrics


def numerical_modules():
    torch=importlib.import_module('torch')
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    data=importlib.import_module('research.cifar10_replacements.data')
    runtime=importlib.import_module('research.cifar10_replacements.runtime')
    return data,runtime


def write_status(output,status):
    """Append-only attempt receipt; ordinary failures cannot replace previous receipt."""
    import os
    path=Path(output)/('exit_'+str(time.time_ns())+'.json')
    with path.open('x') as f:
        json.dump(status,f,sort_keys=True,allow_nan=False); f.flush(); os.fsync(f.fileno())


def execute(plan_path,plan_hash,slot_path,slot_hash,seed,*,resume=False):
    plan,run=verify_plan(plan_path,plan_hash,slot_path,slot_hash,seed,resume)
    output=Path(run['output'])
    if resume:
        if not output.is_dir(): raise ValueError('Explicit resume requires existing run directory')
        if (output/'validation.json').exists(): raise ValueError('Completed run cannot be resumed/reselected')
    else:
        output.mkdir(parents=True,exist_ok=False)  # never silently adopt/overwrite old run
    worker=None
    status=dict(seed=seed,plan_sha256=plan_hash,slot_sha256=slot_hash,resume=resume,official_test_decoded=False,
                teacher_sha256=run['teacher_checkpoint_sha256'])
    try:
        launch,worker=bootstrap(output,plan_hash,resume=resume)
        export,metrics=preliminary_teacher(run,seed)
        data_module,runtime=numerical_modules()
        data=data_module.load_approved_data(plan['preflight'],execution_authorized=True)
        blob=Path(run['teacher_checkpoint']).read_bytes()
        if digest(blob)!=run['teacher_checkpoint_sha256']: raise ValueError('Teacher checkpoint file mismatch')
        admitted=data_module.admit_teacher_bundle(export,metrics,blob,data,seed)
        result=runtime.fit_admitted_teacher(data,admitted,output,launch,plan['source_sha256'],
                                            worker_lock=worker,execution_authorized=True,resume=resume)
        status.update(status='complete',exit_code=0,teacher_sha256=run['teacher_checkpoint_sha256'],
                      replacement_sha256=result['checkpoint_sha256'],
                      validation_sha256=digest((output/'validation.json').read_bytes()),
                      recipe_sha256=digest((output/'recipe_lock.json').read_bytes()))
        write_status(output,status)
        return status
    except BaseException as exc:
        status.update(status='failed',exit_code=1,error_type=type(exc).__name__,error=str(exc))
        if (output/'last.ckpt').exists(): status['last_checkpoint_sha256']=digest((output/'last.ckpt').read_bytes())
        write_status(output,status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if worker is not None: worker.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True); parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--slot-manifest',required=True); parser.add_argument('--slot-sha256',required=True)
    parser.add_argument('--seed',required=True,type=int,choices=SEEDS)
    parser.add_argument('--parent-approved-resume',action='store_true')
    args=parser.parse_args()
    try:
        result=execute(args.plan,args.plan_sha256,args.slot_manifest,args.slot_sha256,args.seed,
                       resume=args.parent_approved_resume)
    except Exception as exc:
        print(json.dumps(dict(status='failed',seed=args.seed,error_type=type(exc).__name__,error=str(exc))))
        return 1
    print(json.dumps(result,sort_keys=True)); return 0


if __name__=='__main__': raise SystemExit(main())
