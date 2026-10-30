"""In-memory fit/checkpoint primitives; not a scientific launch command.

No resume or automatic retry API. Last finite states persist for audit only.
A production orchestrator/calibration-only producer still require review.
"""
import copy
import io
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from . import core as c
from . import protocol as p


def atomic_bytes(path, blob):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        handle.write(blob); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def envelope(state):
    c.finite(state); buffer = io.BytesIO(); torch.save(state, buffer); payload = buffer.getvalue()
    return b'CANARIA-UTILITY-GAP1\n'+p.sha(payload).encode()+b'\n'+payload


def decode(blob):
    magic, checksum, payload = blob.split(b'\n', 2)
    if magic != b'CANARIA-UTILITY-GAP1' or checksum != p.sha(payload).encode():
        raise ValueError('Checkpoint envelope integrity mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    c.finite(state)
    return state


def _number(value, label, *, nonnegative=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError('Invalid finite numeric '+label)
    return value


def validate_checkpoint_state(state, *, expected_lock=None, expected_cell=None, expected_updates=None):
    """Check internal training evidence, not just a caller-supplied update counter.

    This establishes consistency, not cryptographic proof of actual optimization.
    Source/producer custody and independently trusted artifact hashes still apply.
    """
    c.finite(state)
    if type(state.get('schema')) is not int or state['schema'] != 1 or state.get('cell') not in p.CELLS:
        raise ValueError('Invalid checkpoint schema/cell')
    name=state['cell']; updates=state.get('updates'); lock=state.get('lock',{})
    if type(updates) is not int or not 0 <= updates <= 2048:
        raise ValueError('Invalid actual update count')
    if (expected_lock is not None and lock != expected_lock) or (expected_cell is not None and name != expected_cell) or (expected_updates is not None and updates != expected_updates):
        raise ValueError('Checkpoint session/cell/update binding mismatch')
    if state.get('lock_sha256') != p.digest(lock) or lock.get('design_sha256') != p.digest(p.DESIGN):
        raise ValueError('Checkpoint locked design mismatch')
    if lock.get('seed') not in p.SEEDS or state.get('teacher_state_after_sha256') != lock.get('teacher_state_before_sha256'):
        raise ValueError('Checkpoint teacher seed/state mismatch')
    for field in ('teacher_sha256','teacher_state_before_sha256','partition_sha256','fit_ids_sha256','batch_schedule_sha256'):
        p.require_hash(lock.get(field))
    model=c.make_cells(lock['seed'])[name]
    expected_parameters=model.state_dict()
    for key in ('model','initial'):
        values=state.get(key,{})
        if set(values)!=set(expected_parameters): raise ValueError('Checkpoint parameter names mismatch')
        for parameter,tensor in values.items():
            expected=expected_parameters[parameter]
            if not isinstance(tensor,torch.Tensor) or tensor.dtype!=expected.dtype or tensor.shape!=expected.shape:
                raise ValueError('Checkpoint parameter shape/dtype mismatch')
    initial_hash=c.state_dict_hash(state['initial'])
    if initial_hash!=lock.get('initialization',{}).get(name) or initial_hash!=c.state_hash(model):
        raise ValueError('Checkpoint initial state not the exact fresh locked initialization')
    if updates==0 and c.state_dict_hash(state['model'])!=initial_hash:
        raise ValueError('Warm-started zero-update checkpoint rejected')
    log=state.get('training_log')
    if not isinstance(log,list) or len(log)!=updates:
        raise ValueError('Complete per-update training log required')
    required_log={'update','examples','objective_mean','objective_sum','gradient_norm','relative_parameter_update_norm'}
    for index,row in enumerate(log):
        examples=32 if (index+1)%32==0 else 128
        if not isinstance(row,dict) or set(row)!=required_log or type(row['update']) is not int or row['update']!=index+1 or type(row['examples']) is not int or row['examples']!=examples:
            raise ValueError('Training log update/schedule/sample count mismatch')
        mean=_number(row['objective_mean'],'objective_mean'); total=_number(row['objective_sum'],'objective_sum')
        if not math.isclose(total,mean*examples,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Training log is not sample weighted')
        if _number(row['gradient_norm'],'gradient_norm')<=0:
            raise ValueError('Recorded replacement gradient must be nonzero')
        _number(row['relative_parameter_update_norm'],'relative_parameter_update_norm',nonnegative=True)
    optimizer=state.get('optimizer',{}); expected_optimizer=c.make_optimizer(model).state_dict()
    if optimizer.get('param_groups')!=expected_optimizer['param_groups']:
        raise ValueError('Checkpoint Adam recipe/parameter groups mismatch')
    parameter_ids=expected_optimizer['param_groups'][0]['params']; adam=optimizer.get('state')
    if not isinstance(adam,dict) or set(adam)!=(set(parameter_ids) if updates else set()):
        raise ValueError('Complete fresh/updated Adam state census required')
    for identifier,param in zip(parameter_ids,model.parameters()):
        if not updates: break
        record=adam[identifier]
        if set(record)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Unexpected/missing Adam state')
        step=record['step']
        if not isinstance(step,torch.Tensor) or step.dtype!=torch.float32 or step.shape!=torch.Size([]) or float(step)!=updates:
            raise ValueError('Adam step does not equal actual checkpoint budget')
        for key in ('exp_avg','exp_avg_sq'):
            tensor=record[key]
            if not isinstance(tensor,torch.Tensor) or tensor.shape!=param.shape or tensor.dtype!=param.dtype:
                raise ValueError('Adam tensor shape/dtype mismatch')
        if (record['exp_avg_sq']<0).any(): raise ValueError('Negative Adam second moment')
    rng=state.get('torch_rng')
    if not isinstance(rng,torch.Tensor) or rng.dtype!=torch.uint8 or rng.shape!=torch.get_rng_state().shape:
        raise ValueError('Missing/invalid Torch RNG snapshot')
    try:
        random.Random().setstate(state['python_rng'])
        nr=state['numpy_rng']
        np.random.RandomState().set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
    except (KeyError,TypeError,ValueError,IndexError,OverflowError) as exc:
        raise ValueError('Missing/invalid Python/NumPy RNG snapshot') from exc
    return state


def validate_fit_metrics(metrics, state):
    required={'schema','split','n','cell','updates','session_lock_sha256','model_state_sha256',
              'fit_ids_sha256','teacher_state_sha256','logit_kl','centered_span_nmse','sse',
              'centered_teacher_ss','epsilon','uncentered_energy_nmse'}
    if not isinstance(metrics,dict) or set(metrics)!=required or type(metrics['schema']) is not int or metrics['schema']!=1 or metrics['split']!='fit' or type(metrics['n']) is not int or metrics['n']!=4000 or type(metrics['updates']) is not int:
        raise ValueError('Complete full-fit NMSE/KL metric schema required')
    lock=state['lock']
    expected=dict(cell=state['cell'],updates=state['updates'],session_lock_sha256=p.digest(lock),
        model_state_sha256=c.state_dict_hash(state['model']),fit_ids_sha256=lock['fit_ids_sha256'],
        teacher_state_sha256=lock['teacher_state_before_sha256'])
    if any(metrics.get(key)!=value for key,value in expected.items()):
        raise ValueError('Metrics not bound to exact session/cell/update/model/fit IDs/teacher')
    for key in ('logit_kl','centered_span_nmse','sse','centered_teacher_ss'):
        _number(metrics[key],key,nonnegative=True)
    if type(metrics['epsilon']) is not float or metrics['epsilon']!=1e-12:
        raise ValueError('Centered NMSE epsilon changed')
    expected_nmse=metrics['sse']/(metrics['centered_teacher_ss']+1e-12*4000*32*16*16)
    if not math.isclose(metrics['centered_span_nmse'],expected_nmse,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Centered NMSE does not match full-fit sufficient statistics')
    if metrics['uncentered_energy_nmse'] is not None:
        _number(metrics['uncentered_energy_nmse'],'uncentered_energy_nmse',nonnegative=True)
    return p.digest(metrics)


class FitSession:
    def __init__(self, admitted_teacher, teacher_sha256, partition_record, seed,
                 teacher_clock, cohort_clock, source_hashes, *, now=None):
        p.check_teacher_clock(teacher_clock, cohort_clock, now=now)
        if seed != teacher_clock['seed']: raise ValueError('Seed/launch mismatch')
        self.teacher = c.DifferentiableFrozenTeacher(admitted_teacher)
        self.teacher_before = self.teacher.verify_immutable()
        self.partition = copy.deepcopy(partition_record)
        self.teacher_clock = copy.deepcopy(teacher_clock); self.cohort_clock = copy.deepcopy(cohort_clock)
        self.cells = c.make_cells(seed)
        self.optimizers = {name:c.make_optimizer(model) for name,model in self.cells.items()}
        self.batches = c.schedule(partition_record, seed)
        self.initial = {name:copy.deepcopy(model.state_dict()) for name,model in self.cells.items()}
        self.lock = dict(schema=1, seed=seed, teacher_sha256=p.require_hash(teacher_sha256),
            teacher_state_before_sha256=self.teacher_before, design_sha256=p.digest(p.DESIGN),
            partition_sha256=p.verify_partition(partition_record), fit_ids_sha256=partition_record['fit_sha256'], source_sha256=copy.deepcopy(source_hashes),
            teacher_clock=copy.deepcopy(teacher_clock), cohort_clock=copy.deepcopy(cohort_clock),
            initialization={name:c.state_hash(model) for name,model in self.cells.items()},
            batch_schedule_sha256=p.digest(self.batches), fresh_adam=True, warm_start=False)
        self.cursor = {name:0 for name in p.CELLS}; self.failure = {name:None for name in p.CELLS}; self.failure_status = {name:None for name in p.CELLS}
        self.logs = {name:[] for name in p.CELLS}
        self.checkpoints = {name:{str(step):dict(status='missing', reason='not reached') for step in p.BUDGETS} for name in p.CELLS}
        self.closed = False
        self._metric_receipts = {}
        self._terminal = None

    def check_time(self, now=None):
        p.check_teacher_clock(self.teacher_clock, self.cohort_clock, now=now)

    def step(self, name, ids, images, *, now=None):
        if self.closed or name not in p.CELLS or self.failure[name] is not None:
            raise RuntimeError('Closed/failed/unplanned trajectory; retry denied')
        self.check_time(now)
        cursor = self.cursor[name]
        if cursor >= 2048: raise ValueError('Exactly 2048 updates; no continuation')
        if cursor in p.BUDGETS and self.checkpoints[name][str(cursor)]['status'] != 'saved':
            raise ValueError('Freeze prespecified checkpoint and full-fit metrics before continuing')
        if tuple(ids) != self.batches[cursor] or len(images) != len(ids):
            raise ValueError('Exact shared fit-only schedule batch required')
        model = self.cells[name]; optimizer = self.optimizers[name]
        try:
            self.teacher.verify_immutable(); model.train()
            x, target, teacher_logits = self.teacher.targets(images); c.finite((x,target,teacher_logits))
            predicted = model(x)
            # MSE need not backpropagate through suffix, but KL must retain it.
            student_logits = self.teacher.suffix_logits(predicted) if name.endswith('_kl') else None
            loss = c.objective(name.rsplit('_',1)[1], predicted, target, student_logits, teacher_logits)
            c.finite(loss); optimizer.zero_grad(set_to_none=True); loss.backward()
            gradients = [param.grad for param in model.parameters()]
            if any(gradient is None for gradient in gradients): raise ValueError('Missing replacement gradient')
            c.finite(gradients)
            gradient_norm = float(torch.sqrt(sum(g.detach().double().square().sum() for g in gradients)))
            if not gradient_norm > 0: raise ValueError('Zero replacement gradient')
            before = [param.detach().clone() for param in model.parameters()]
            old_norm = float(torch.sqrt(sum(value.double().square().sum() for value in before)))
            optimizer.step(); c.finite(model.state_dict()); c.finite(optimizer.state_dict())
            delta = float(torch.sqrt(sum((param.detach().double()-value.double()).square().sum() for param,value in zip(model.parameters(),before))))
            self.teacher.verify_immutable()
            entry = dict(update=cursor+1, examples=len(ids), objective_mean=float(loss.detach()),
                         objective_sum=float(loss.detach())*len(ids), gradient_norm=gradient_norm,
                         relative_parameter_update_norm=delta/max(old_norm,1e-30))
            c.finite(entry)
        except BaseException as exc:
            self.failure[name] = type(exc).__name__+': '+str(exc)
            self.failure_status[name] = 'cap-exhausted' if isinstance(exc,p.DeadlineExceeded) else 'failed'
            raise
        self.cursor[name] += 1; self.logs[name].append(entry)
        # Saving this finite update remains allowed if the clock has just expired;
        # every further computation, including full-fit metrics, checks the clock.
        return entry

    def state(self, name):
        if name not in p.CELLS or self.failure[name] is not None:
            raise RuntimeError('Never serialize failed/nonfinite/partial trajectory')
        self.teacher.verify_immutable()
        nr = np.random.get_state()
        state = dict(schema=1, lock=copy.deepcopy(self.lock), lock_sha256=p.digest(self.lock),
            cell=name, updates=self.cursor[name], model=copy.deepcopy(self.cells[name].state_dict()),
            initial=copy.deepcopy(self.initial[name]), optimizer=copy.deepcopy(self.optimizers[name].state_dict()),
            torch_rng=torch.get_rng_state().clone(), python_rng=random.getstate(),
            numpy_rng=[nr[0],nr[1].tolist(),int(nr[2]),int(nr[3]),float(nr[4])],
            training_log=copy.deepcopy(self.logs[name]), teacher_state_after_sha256=self.teacher.verify_immutable())
        validate_checkpoint_state(state)
        return state

    def save_last(self, name, directory):
        # One durable last-finite artifact per trajectory. No warm-start reader.
        blob = envelope(self.state(name)); atomic_bytes(Path(directory)/(name+'_last.ckpt'), blob)
        return p.sha(blob)

    @torch.no_grad()
    def full_fit_metrics(self, name, access, *, now=None):
        if self.closed or name not in p.CELLS or self.failure[name] is not None:
            raise RuntimeError('Closed/failed/unplanned trajectory; metric retry denied')
        step=self.cursor[name]
        if step not in p.BUDGETS or self.checkpoints[name][str(step)]['status']!='missing' or (name,step) in self._metric_receipts:
            raise ValueError('Metrics only once at a new prespecified finite checkpoint')
        try:
            self.check_time(now)
            if tuple(access.ids) != tuple(self.partition['fit_ids']): raise ValueError('Exact 4000-fit capability required')
            self.teacher.verify_immutable(); stats = c.EndpointStats(); kl_sum=0.; model=self.cells[name].eval()
            for start in range(0,4000,128):
                self.check_time(now); ids=access.ids[start:start+128]; images=access.image_batch(ids)
                x,target,teacher_logits=self.teacher.targets(images); predicted=model(x)
                logits=self.teacher.suffix_logits(predicted); c.finite(logits)
                stats.update(predicted,target); kl_sum+=float(c.logit_kl(logits,teacher_logits,'none').double().sum())
            self.teacher.verify_immutable(); self.check_time(now)
            metrics=dict(schema=1,split='fit',cell=name,updates=step,session_lock_sha256=p.digest(self.lock),
                model_state_sha256=c.state_hash(model),fit_ids_sha256=self.lock['fit_ids_sha256'],
                teacher_state_sha256=self.teacher_before,logit_kl=kl_sum/4000,**stats.result())
            validate_fit_metrics(metrics,dict(cell=name,updates=step,lock=self.lock,model=model.state_dict()))
            self._metric_receipts[(name,step)]=copy.deepcopy(metrics)
            return metrics
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc)
            self.failure[name]=reason
            self.failure_status[name]='cap-exhausted' if isinstance(exc,p.DeadlineExceeded) else 'failed'
            self.checkpoints[name][str(step)]=dict(status='failed',reason=reason)
            raise

    def freeze_checkpoint(self, name, directory, fit_metrics):
        if self.closed or name not in p.CELLS or self.failure[name] is not None:
            raise RuntimeError('Closed/failed/unplanned trajectory; checkpoint freeze denied')
        step=self.cursor[name]
        if step not in p.BUDGETS or self.checkpoints[name][str(step)]['status']!='missing':
            raise ValueError('Checkpoint must be new and prespecified; no overwrite')
        # Validate fields/bindings before any write, then require this session's
        # internally produced one-use receipt, not caller-invented measurements.
        current=dict(cell=name,updates=step,lock=self.lock,model=self.cells[name].state_dict())
        validate_fit_metrics(fit_metrics,current)
        if self._metric_receipts.get((name,step))!=fit_metrics:
            raise ValueError('Exact internally produced full-fit metric receipt required')
        blob=envelope(self.state(name)); directory=Path(directory); prefix=f'{name}_update{step}'
        try:
            with (directory/(prefix+'.ckpt')).open('xb') as handle:
                handle.write(blob); handle.flush(); os.fsync(handle.fileno())
            p.write_exclusive_json(directory/(prefix+'_fit.json'),fit_metrics)
        except BaseException as exc:
            self.failure[name]=type(exc).__name__+': '+str(exc); self.failure_status[name]='failed'
            self.checkpoints[name][str(step)]=dict(status='failed',reason=self.failure[name])
            raise
        record=dict(status='saved',checkpoint_sha256=p.sha(blob),fit_metrics_sha256=p.digest(fit_metrics))
        self.checkpoints[name][str(step)]=record
        del self._metric_receipts[(name,step)]
        return copy.deepcopy(record)

    def terminal_record(self, *, reason=None, now=None):
        if self._terminal is not None: return copy.deepcopy(self._terminal)
        now=time.time() if now is None else now
        p.require_finite_time(now,'terminal time')
        self.teacher.verify_immutable()
        trajectories={}
        for name in p.CELLS:
            complete=self.cursor[name]==2048 and all(v['status']=='saved' for v in self.checkpoints[name].values())
            status='complete' if complete else self.failure_status[name] or ('cap-exhausted' if now>=self.teacher_clock['expiry'] else 'failed')
            trajectories[name]=dict(status=status, updates=self.cursor[name], checkpoints=copy.deepcopy(self.checkpoints[name]))
            if not complete: trajectories[name]['reason']=self.failure[name] or reason or 'incomplete terminal trajectory; no lower checkpoint substituted'
            p.verify_trajectory(trajectories[name])
        self._terminal=dict(admission='admitted',teacher_sha256=self.lock['teacher_sha256'],
            teacher_state_before_sha256=self.teacher_before,teacher_state_after_sha256=self.teacher.verify_immutable(),
            session_lock_sha256=p.digest(self.lock),trajectories=trajectories)
        self.closed=True
        return copy.deepcopy(self._terminal)


def not_admitted_record(reason):
    if not reason: raise ValueError('Explicit admission failure required')
    return dict(admission='not-admitted', trajectories={name:dict(status='not-admitted',updates=0,reason=reason,
        checkpoints={str(step):dict(status='missing',reason=reason) for step in p.BUDGETS}) for name in p.CELLS})


def weighted_training_summary(log):
    count=sum(row['examples'] for row in log)
    if not count: raise ValueError('No training observations')
    return dict(examples=count, objective=sum(row['objective_sum'] for row in log)/count,
                updates=len(log), adaptation=False)
