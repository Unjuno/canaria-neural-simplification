"""Negative tests from independent review; synthetic counters are not real fits."""
import copy
import tempfile
import unittest
from unittest.mock import Mock, patch

from .test_synthetic import fixtures, synthetic_teacher, NOW, H
from . import protocol as p, core as c, runtime as r, diagnostic, report
import torch


def consistent_synthetic_state(session,name,updates):
    """Construct internally consistent audit fixtures, NOT measured optimization."""
    state=session.state(name); state['updates']=updates
    state['training_log']=[dict(update=index+1,examples=32 if (index+1)%32==0 else 128,
        objective_mean=.25,objective_sum=.25*(32 if (index+1)%32==0 else 128),
        gradient_norm=1.,relative_parameter_update_norm=.001) for index in range(updates)]
    ids=state['optimizer']['param_groups'][0]['params']
    state['optimizer']['state']={identifier:dict(step=torch.tensor(float(updates)),
        exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
        for identifier,parameter in zip(ids,session.cells[name].parameters())} if updates else {}
    return state


def bound_metrics(state):
    lock=state['lock']; sse=4.; centered=20.
    return dict(schema=1,split='fit',n=4000,cell=state['cell'],updates=state['updates'],
        session_lock_sha256=p.digest(lock),model_state_sha256=c.state_dict_hash(state['model']),
        fit_ids_sha256=lock['fit_ids_sha256'],teacher_state_sha256=lock['teacher_state_before_sha256'],
        logit_kl=.1,centered_span_nmse=sse/(centered+1e-12*4000*32*16*16),sse=sse,
        centered_teacher_ss=centered,epsilon=1e-12,uncentered_energy_nmse=.2)


def teacher_record_for_state(session,state,metric_blob):
    teacher=dict(admission='admitted',teacher_sha256=H,teacher_state_before_sha256=session.teacher_before,
        teacher_state_after_sha256=session.teacher_before,trajectories={name:dict(status='failed',updates=0,
        reason='synthetic missing',checkpoints={str(step):dict(status='missing',reason='synthetic missing')
        for step in p.BUDGETS}) for name in p.CELLS})
    blob=r.envelope(state); tr=teacher['trajectories'][state['cell']]; tr['updates']=state['updates']
    tr['checkpoints'][str(state['updates'])]=dict(status='saved',checkpoint_sha256=p.sha(blob),fit_metrics_sha256=p.sha(metric_blob))
    return teacher,blob


class ReviewRegressionTests(unittest.TestCase):
    def session(self):
        part,_,source,_,_,cohort,clock=fixtures()
        return r.FitSession(synthetic_teacher(),H,part,p.SEEDS[0],clock,cohort,source,now=NOW+1)

    def test_malformed_metrics_never_create_checkpoint(self):
        session=self.session(); session.cursor['w8_mse']=1024
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(ValueError):
            session.freeze_checkpoint('w8_mse',directory,dict(split='fit',n=4000))

    def test_terminal_session_rejects_checkpoint_and_metrics(self):
        session=self.session(); session.cursor['w8_mse']=1024
        session.terminal_record(now=NOW+1)
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(RuntimeError):
            session.freeze_checkpoint('w8_mse',directory,dict(split='fit',n=4000))
        access=Mock(); access.ids=tuple(session.partition['fit_ids'])
        with self.assertRaises(RuntimeError): session.full_fit_metrics('w8_mse',access,now=NOW+1)
        access.image_batch.assert_not_called()

    def test_full_fit_failure_is_latched_and_not_relabelled_cap(self):
        session=self.session(); session.cursor['w8_mse']=1024
        access=Mock(); access.ids=tuple(session.partition['fit_ids'])
        access.image_batch.side_effect=FloatingPointError('synthetic metric failure')
        with self.assertRaises(FloatingPointError): session.full_fit_metrics('w8_mse',access,now=NOW+1)
        self.assertIsNotNone(session.failure['w8_mse'])
        self.assertEqual(session.terminal_record(now=NOW+1)['trajectories']['w8_mse']['status'],'failed')

    def test_counter_only_checkpoint_not_admitted_for_diagnostics(self):
        session=self.session(); state=session.state('w8_mse'); state['updates']=1024
        blob=r.envelope(state)
        teacher=r.not_admitted_record('synthetic missing')
        teacher.update(admission='admitted',teacher_sha256=H,teacher_state_before_sha256=session.teacher_before,
                       teacher_state_after_sha256=session.teacher_before)
        tr=teacher['trajectories']['w8_mse']; tr.update(status='failed',updates=1024)
        tr['checkpoints']['1024']=dict(status='saved',checkpoint_sha256=p.sha(blob),fit_metrics_sha256=H)
        with self.assertRaises(ValueError):
            diagnostic.checkpoint_models(p.SEEDS[0],teacher,{'w8_mse@1024':blob},session.lock)

    def test_nonfinite_and_boolean_clocks_rejected(self):
        _,_,_,_,_,cohort,clock=fixtures()
        for now in (float('nan'),float('inf'),float('-inf'),True):
            with self.subTest(now=now),self.assertRaises((ValueError,TimeoutError)):
                p.check_teacher_clock(clock,cohort,now=now)

    def test_fractional_boolean_and_string_classes_rejected(self):
        for invalid in (.9,True,'0'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                report.paired_counts([invalid]*1000,[0]*1000,[0]*1000)
        # Actual numpy integer labels remain valid and are not silently truncated.
        import numpy as np
        self.assertEqual(report.paired_counts([0]*1000,[0]*1000,np.zeros(1000,dtype=np.int64))['student_correct'],1000)

    def test_resource_slot_must_cover_full_teacher_allocation(self):
        part,original,source,_,approval,_,_=fixtures()
        short=copy.deepcopy(approval); short['resource_slot']['expires']=NOW+2
        with self.assertRaises((PermissionError,ValueError)):
            p.execution_gate(short,p.digest(short),original,part,source,execution_authorized=True,now=NOW+1)

    def test_checkpoint_log_adam_initial_and_rng_consistency(self):
        session=self.session(); state=consistent_synthetic_state(session,'w8_mse',1024)
        r.validate_checkpoint_state(state)
        for kind in ('empty-log','bad-log-order','bad-sample-weight','empty-adam','bad-adam-step','initial-drift','missing-rng'):
            bad=copy.deepcopy(state)
            if kind=='empty-log': bad['training_log']=[]
            elif kind=='bad-log-order': bad['training_log'][1]['update']=1
            elif kind=='bad-sample-weight': bad['training_log'][31]['examples']=128
            elif kind=='empty-adam': bad['optimizer']['state']={}
            elif kind=='bad-adam-step': next(iter(bad['optimizer']['state'].values()))['step']=torch.tensor(1.)
            elif kind=='initial-drift': next(iter(bad['initial'].values())).add_(.01)
            else: bad.pop('torch_rng')
            with self.subTest(kind=kind),self.assertRaises(ValueError): r.validate_checkpoint_state(bad)

    def test_metrics_exact_bindings_and_one_use_internal_receipt(self):
        session=self.session(); state=consistent_synthetic_state(session,'w8_mse',1024)
        session.cursor['w8_mse']=1024; session.logs['w8_mse']=state['training_log']
        session.optimizers['w8_mse'].load_state_dict(state['optimizer'])
        metrics=bound_metrics(state)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): session.freeze_checkpoint('w8_mse',directory,metrics)
            session._metric_receipts[('w8_mse',1024)]=copy.deepcopy(metrics)
            for field,value in (('cell','w8_kl'),('updates',2048),('model_state_sha256','b'*64),
                ('fit_ids_sha256','b'*64),('session_lock_sha256','b'*64),('logit_kl',float('nan'))):
                bad=dict(metrics,**{field:value})
                with self.subTest(field=field),self.assertRaises(ValueError): session.freeze_checkpoint('w8_mse',directory,bad)
            session.freeze_checkpoint('w8_mse',directory,metrics)
            self.assertNotIn(('w8_mse',1024),session._metric_receipts)
            with self.assertRaises(ValueError): session.freeze_checkpoint('w8_mse',directory,metrics)

    def test_diagnostic_checkpoint_requires_verified_bound_metric_bytes(self):
        session=self.session(); state=consistent_synthetic_state(session,'w8_mse',1024)
        metrics=bound_metrics(state); metric_blob=p.canonical(metrics)
        teacher,blob=teacher_record_for_state(session,state,metric_blob)
        args=(p.SEEDS[0],teacher,{'w8_mse@1024':blob},session.lock)
        with self.assertRaises(ValueError): diagnostic.checkpoint_models(*args)
        with self.assertRaises(ValueError): diagnostic.checkpoint_models(*args,fit_metric_blobs={'w8_mse@1024':b'changed'})
        models,_=diagnostic.checkpoint_models(*args,fit_metric_blobs={'w8_mse@1024':metric_blob})
        self.assertEqual(set(models),{'w8_mse@1024'})
        wrong=dict(metrics,cell='w8_kl'); wrong_blob=p.canonical(wrong)
        teacher['trajectories']['w8_mse']['checkpoints']['1024']['fit_metrics_sha256']=p.sha(wrong_blob)
        with self.assertRaises(ValueError): diagnostic.checkpoint_models(*args,fit_metric_blobs={'w8_mse@1024':wrong_blob})

    def test_full_fit_receipt_generated_internally_once(self):
        session=self.session(); session.cursor['w8_mse']=1024
        access=Mock(); access.ids=tuple(session.partition['fit_ids'])
        access.image_batch.side_effect=lambda ids: torch.zeros(len(ids),3,1,1)
        class StubStats:
            def update(self,*args): pass
            def result(self):
                return dict(n=4000,centered_span_nmse=4/(20+1e-12*4000*32*16*16),sse=4.,
                    centered_teacher_ss=20.,epsilon=1e-12,uncentered_energy_nmse=.2)
        def targets(images):
            endpoint=images[:,:1].expand(-1,32,1,1)
            return endpoint,endpoint,torch.zeros(len(images),10)
        with patch.object(session.teacher,'targets',side_effect=targets), \
             patch.object(session.teacher,'suffix_logits',side_effect=lambda endpoint:torch.zeros(len(endpoint),10)), \
             patch.object(c,'EndpointStats',StubStats):
            metrics=session.full_fit_metrics('w8_mse',access,now=NOW+1)
        self.assertEqual(access.image_batch.call_count,32)
        self.assertEqual(metrics,session._metric_receipts[('w8_mse',1024)])
        self.assertEqual(metrics['fit_ids_sha256'],session.partition['fit_sha256'])
        with self.assertRaises(ValueError): session.full_fit_metrics('w8_mse',access,now=NOW+1)

    def test_metric_timeout_and_terminal_record_are_stable(self):
        session=self.session(); session.cursor['w8_mse']=1024
        access=Mock(); access.ids=tuple(session.partition['fit_ids'])
        with self.assertRaises(p.DeadlineExceeded): session.full_fit_metrics('w8_mse',access,now=session.teacher_clock['expiry'])
        access.image_batch.assert_not_called()
        record=session.terminal_record(now=session.teacher_clock['expiry'])
        self.assertEqual(record['trajectories']['w8_mse']['status'],'cap-exhausted')
        self.assertEqual(session.terminal_record(now=NOW+1),record)
        record['trajectories']['w8_mse']['status']='tampered caller copy'
        self.assertEqual(session.terminal_record()['trajectories']['w8_mse']['status'],'cap-exhausted')

    def test_reader_timeout_is_failure_not_owned_deadline(self):
        session=self.session(); session.cursor['w8_mse']=1024
        access=Mock(); access.ids=tuple(session.partition['fit_ids'])
        access.image_batch.side_effect=TimeoutError('non-deadline reader failure')
        self.assertEqual(session.teacher_clock['expiry']-(NOW+1),7199)
        with self.assertRaises(TimeoutError): session.full_fit_metrics('w8_mse',access,now=NOW+1)
        record=session.terminal_record(now=NOW+1)
        self.assertEqual(record['trajectories']['w8_mse']['status'],'failed')
        self.assertIn('non-deadline reader failure',record['trajectories']['w8_mse']['reason'])

    def test_update_timeout_is_failure_not_owned_deadline(self):
        session=self.session(); ids=session.batches[0]
        with patch.object(session.teacher,'targets',side_effect=TimeoutError('non-deadline target failure')):
            with self.assertRaises(TimeoutError): session.step('w8_mse',ids,torch.zeros(len(ids),3,1,1),now=NOW+1)
        record=session.terminal_record(now=NOW+1)
        self.assertEqual(record['trajectories']['w8_mse']['status'],'failed')
        self.assertIn('non-deadline target failure',record['trajectories']['w8_mse']['reason'])


if __name__=='__main__': unittest.main(verbosity=2)
