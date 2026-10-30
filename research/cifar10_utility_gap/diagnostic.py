"""Single private reporting phase. No original validation/test or selection API."""
import copy
import json
from pathlib import Path

import torch
from . import core as c
from . import protocol as p
from . import runtime as r
from . import report


def checkpoint_models(seed, frozen_teacher_record, blobs, expected_lock, *, fit_metric_blobs=None):
    """Verify every available checkpoint byte/lock before any diagnostic access."""
    models={}; metadata={}
    for name in p.CELLS:
        for step in p.BUDGETS:
            key=f'{name}@{step}'; record=frozen_teacher_record['trajectories'][name]['checkpoints'][str(step)]
            if record['status']!='saved':
                if key in blobs: raise ValueError('Unexpected blob for missing/failed checkpoint')
                continue
            blob=blobs[key]
            if p.sha(blob)!=record['checkpoint_sha256']: raise ValueError('Frozen checkpoint full-file hash mismatch')
            state=r.decode(blob)
            r.validate_checkpoint_state(state,expected_lock=expected_lock,expected_cell=name,expected_updates=step)
            if state.get('schema')!=1 or state.get('cell')!=name or state.get('updates')!=step:
                raise ValueError('Checkpoint does not match fixed cell/budget')
            if state.get('lock')!=expected_lock or state.get('lock_sha256')!=p.digest(expected_lock):
                raise ValueError('Checkpoint exact session binding mismatch')
            if expected_lock['seed']!=seed or expected_lock['teacher_sha256']!=frozen_teacher_record['teacher_sha256']:
                raise ValueError('Checkpoint teacher identity mismatch')
            if state['teacher_state_after_sha256']!=frozen_teacher_record['teacher_state_before_sha256']:
                raise ValueError('Teacher changed during fit')
            if not isinstance(fit_metric_blobs,dict) or key not in fit_metric_blobs:
                raise ValueError('Exact frozen full-fit metric bytes required')
            metric_blob=fit_metric_blobs[key]
            if p.sha(metric_blob)!=record['fit_metrics_sha256']:
                raise ValueError('Frozen full-fit metric full-file hash mismatch')
            metrics=json.loads(metric_blob)
            if p.canonical(metrics)!=metric_blob:
                raise ValueError('Canonical full-fit metric record required')
            r.validate_fit_metrics(metrics,state)
            model=c.make_cells(seed)[name]; model.load_state_dict(state['model'],strict=True); model.eval()
            c.finite(model.state_dict()); models[key]=model
            metadata[key]=dict(checkpoint_sha256=record['checkpoint_sha256'],fit_metrics_sha256=record['fit_metrics_sha256'])
    if set(blobs)!=set(models): raise ValueError('Exact saved-checkpoint blob census required')
    if models and set(fit_metric_blobs)!=set(models): raise ValueError('Exact saved-checkpoint metric census required')
    return models,metadata


@torch.no_grad()
def predict_teacher(teacher, models, vault, metadata, check_clock):
    """Labels cannot be read here; teacher and all available students in one pass."""
    teacher.verify_immutable(); ids=tuple(vault.ids)
    if len(ids)!=1000: raise ValueError('Exactly 1000 diagnostic IDs required')
    teacher_logits=[]; logits={key:[] for key in models}; stats={key:c.EndpointStats() for key in models}; kl={key:0. for key in models}
    for start in range(0,1000,128):
        check_clock(); batch=ids[start:start+128]; images=vault.image_batch(batch)
        x,target,tlogits=teacher.targets(images); c.finite(tlogits); teacher_logits.extend(tlogits.double().tolist())
        for key,model in models.items():
            check_clock(); endpoint=model(x); slogits=teacher.suffix_logits(endpoint); c.finite(slogits)
            stats[key].update(endpoint,target); kl[key]+=float(c.logit_kl(slogits,tlogits,'none').double().sum())
            logits[key].extend(slogits.double().tolist())
    teacher.verify_immutable()
    return dict(status='predicted', teacher_logits=teacher_logits,
        teacher_predictions=[max(range(10),key=row.__getitem__) for row in teacher_logits],
        checkpoints={key:dict(status='predicted',**metadata[key],student_logits=values,
            student_predictions=[max(range(10),key=row.__getitem__) for row in values],
            logit_kl=kl[key]/1000,**stats[key].result()) for key,values in logits.items()})


def freeze_predictions(directory, frozen, partition_record, teachers):
    """Every planned checkpoint must appear, including explicit missing outcomes."""
    if set(teachers)!={str(seed) for seed in p.SEEDS}: raise ValueError('Every planned teacher required')
    expected={f'{name}@{step}' for name in p.CELLS for step in p.BUDGETS}
    for seed,teacher in teachers.items():
        if set(teacher.get('checkpoints',{}))!=expected: raise ValueError('Entire 24-checkpoint prediction/status census required')
        for key,value in teacher['checkpoints'].items():
            if value.get('status') not in ('predicted','missing','failed'): raise ValueError('Prediction status not frozen')
            name,step=key.split('@'); source=frozen['teachers'][seed]['trajectories'][name]['checkpoints'][step]
            if value['status']=='predicted':
                if source['status']!='saved' or value['checkpoint_sha256']!=source['checkpoint_sha256']:
                    raise ValueError('Prediction differs from frozen checkpoint')
                if len(value['student_predictions'])!=1000 or len(value['student_logits'])!=1000:
                    raise ValueError('Partial predictions cannot be released as complete')
                if len(teacher.get('teacher_predictions',[]))!=1000 or len(teacher.get('teacher_logits',[]))!=1000:
                    raise ValueError('Teacher predictions missing')
            elif not value.get('reason'): raise ValueError('Explicit prediction failure/missing reason required')
    record=dict(status='all-planned-predictions-frozen',frozen_cohort_sha256=p.digest(frozen),
        diagnostic_ids=list(partition_record['diagnostic_ids']),teachers=copy.deepcopy(teachers))
    p.write_exclusive_json(Path(directory)/'predictions_frozen.json',record)
    return record,p.digest(record)


def accuracy_report(predictions, labels):
    """Report all available cells after labels; never choose a best cell or budget."""
    result=dict(exploratory=True, formal_noninferiority=False, pristine_study_holdout=False,
        original_validation_accessed=False,official_test_accessed=False,
        predictions_sha256=p.digest(predictions), diagnostic_labels=[int(v) for v in labels],teachers={})
    for seed,teacher in predictions['teachers'].items():
        entries={}
        for key,value in teacher['checkpoints'].items():
            entries[key]=dict(status=value['status'])
            if value['status']=='predicted':
                entries[key].update(report.paired_counts(teacher['teacher_predictions'],value['student_predictions'],labels))
                entries[key].update(centered_span_nmse=value['centered_span_nmse'],logit_kl=value['logit_kl'])
            else: entries[key]['reason']=value['reason']
        result['teachers'][seed]=entries
    return result
