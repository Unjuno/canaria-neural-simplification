"""One-thread synthetic-only preparation checks. No real artifact/model/data reads."""
import ast
import copy
import datetime
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from research.cifar10_teacher.model import ResNet20
from research.cifar10_replacements.core import FrozenTeacher
from . import protocol as p, core as c, data as d, runtime as r, report, diagnostic

torch.set_num_threads(1)
if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
H='a'*64
NOW=1793000000.


def fixtures():
    part=p.partition(list(range(5000)),p.ids_hash(range(5000)))
    original=dict(status='frozen',base_commit=p.BASE_COMMIT,records={str(seed):dict(status='complete',
        terminal_record_sha256=H,teacher_sha256=H,replacement_sha256=H,validation_sha256=H,
        recipe_sha256=H,original_negative_results_preserved=True) for seed in p.SEEDS})
    source={'synthetic_fixture_only':H}
    normalization=dict(source='teachertrain only',mean=[.5,.5,.5],std=[.2,.2,.2])
    normalization['sha256']=p.sha(np.asarray([normalization['mean'],normalization['std']],dtype='<f8').tobytes())
    manifest=dict(partition_sha256=p.digest(part),scope='calibration-only; original validation/test never decoded',
                  payload_sha256={name:H for name in d.NAMES},normalization=normalization)
    identity=dict(normalization=normalization,splits={'calibration':dict(n=5000,sha256=part['calibration_sha256'])})
    original['evaluator_identity_sha256']=p.digest(identity)
    approval=dict(action='approved-utility-gap-execution',design=copy.deepcopy(p.DESIGN),
        original_cohort_sha256=p.digest(original),partition_sha256=p.digest(part),source_sha256=source,
        parent_announced=True,independent_review='GO',fees=False,network_writes=False,ci=False,
        artifact_manifest_sha256=p.digest(manifest),evaluator_identity=identity,
        resource_slot=dict(reservation_id='synthetic-only',threads=4,interop=1,other_science_threads=0,
            original_cohort_has_priority=True,single_worker_lock_required=True,starts=NOW,expires=NOW+14400))
    cohort=dict(schema=1,approval_sha256=p.digest(approval),started=NOW,expiry=NOW+21600,fit_cutoff=NOW+19800)
    teacher_clock=dict(seed=p.SEEDS[0],started=NOW,expiry=NOW+7200,cohort_launch_sha256=p.digest(cohort))
    return part,original,source,manifest,approval,cohort,teacher_clock


def frozen_fixture(part,original,source):
    trajectory=dict(status='complete',updates=2048,checkpoints={str(step):dict(status='saved',checkpoint_sha256=H,fit_metrics_sha256=H) for step in p.BUDGETS})
    return dict(status='all-trajectories-and-checkpoints-frozen',design_sha256=p.digest(p.DESIGN),
        partition_sha256=p.digest(part),original_cohort_sha256=p.digest(original),source_sha256=source,
        teachers={str(seed):dict(admission='admitted',teacher_sha256=H,teacher_state_before_sha256=H,
            teacher_state_after_sha256=H,trajectories={name:copy.deepcopy(trajectory) for name in p.CELLS}) for seed in p.SEEDS})


def synthetic_teacher():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        return FrozenTeacher(ResNet20())


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.part,self.original,self.source,self.manifest,self.approval,self.cohort,self.clock=fixtures()

    def test_threads_and_deadline(self):
        self.assertEqual(torch.get_num_threads(),1); self.assertEqual(torch.get_num_interop_threads(),1)
        self.assertEqual(datetime.datetime.fromtimestamp(p.GLOBAL_DEADLINE,datetime.timezone.utc).isoformat(),'2026-10-29T06:50:00+00:00')
        self.assertEqual(p.DESIGN['primary_contrast'],'w8_mse:1024->2048')
        self.assertFalse(p.DESIGN['formal_noninferiority'])

    def test_exact_hash_partition_without_labels(self):
        self.assertEqual(len(self.part['fit_ids']),4000); self.assertEqual(len(self.part['diagnostic_ids']),1000)
        self.assertFalse(set(self.part['fit_ids'])&set(self.part['diagnostic_ids']))
        expected=sorted(range(5000),key=lambda i:(p.sha(f'canaria-utility-gap-v1:{i}'.encode()),i))
        self.assertEqual(self.part['diagnostic_ids'],expected[:1000])
        self.assertEqual(self.part['fit_ids'],expected[1000:])
        self.assertEqual(p.verify_partition(self.part),p.digest(self.part))
        for bad in ([True]+list(range(1,5000)),[0]*5000,list(range(4999)),list(range(50000,55000))):
            with self.assertRaises(ValueError): p.partition(bad,H)
        wrong=copy.deepcopy(self.part); wrong['fit_ids'][0],wrong['diagnostic_ids'][0]=wrong['diagnostic_ids'][0],wrong['fit_ids'][0]
        with self.assertRaises(ValueError): p.verify_partition(wrong)

    def test_execution_default_denies_before_payload(self):
        reader=Mock(side_effect=AssertionError('No payload should be read'))
        args=(reader,self.manifest,self.approval,p.digest(self.approval),self.original,self.part,self.source)
        with self.assertRaises(PermissionError): d.open_fit_access(*args)
        reader.assert_not_called()
        self.assertEqual(p.execution_gate(self.approval,p.digest(self.approval),self.original,self.part,self.source,execution_authorized=True,now=NOW+1),p.digest(self.approval))

    def test_approval_and_original_lock_fail_closed(self):
        for key,value in [('action','proposal'),('parent_announced',False),('independent_review','pending'),('ci',True),('fees',True)]:
            bad=copy.deepcopy(self.approval); bad[key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,PermissionError)):
                p.execution_gate(bad,p.digest(bad),self.original,self.part,self.source,execution_authorized=True,now=NOW+1)
        original=copy.deepcopy(self.original); original['records'][str(p.SEEDS[-1])]['status']='pending'
        with self.assertRaises(ValueError): p.verify_original_lock(original)
        original=copy.deepcopy(self.original); original['records'].pop(str(p.SEEDS[-1]))
        with self.assertRaises(ValueError): p.verify_original_lock(original)
        with self.assertRaises(ValueError): p.execution_gate(self.approval,p.digest(self.approval),self.original,self.part,{'drift':H},execution_authorized=True,now=NOW+1)

    def test_new_cohort_and_teacher_caps_no_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort=p.open_cohort(directory,H,now=NOW)
            self.assertEqual(cohort['expiry']-NOW,21600); self.assertEqual(cohort['fit_cutoff']-NOW,19800)
            with self.assertRaises(FileExistsError): p.open_cohort(directory,H,now=NOW+1)
            launch=p.teacher_launch(directory,cohort,p.SEEDS[0],now=NOW+10)
            self.assertEqual(launch['expiry'],NOW+7210)
            p.check_teacher_clock(launch,cohort,now=NOW+100)
            with self.assertRaises(TimeoutError): p.check_teacher_clock(launch,cohort,now=launch['expiry'])
            changed=dict(launch,expiry=launch['expiry']+1)
            with self.assertRaises(ValueError): p.check_teacher_clock(changed,cohort,now=NOW+100)
            with self.assertRaises(FileExistsError): p.teacher_launch(directory,cohort,p.SEEDS[0],now=NOW+100)
            with self.assertRaises(TimeoutError): p.teacher_launch(directory,cohort,p.SEEDS[1],now=cohort['fit_cutoff'])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError): p.open_cohort(directory,H,now=p.GLOBAL_DEADLINE-1900)
            cohort=p.open_cohort(directory,H,now=p.GLOBAL_DEADLINE-9000)
            self.assertEqual(cohort['fit_cutoff'],p.GLOBAL_DEADLINE-1800)
            launch=p.teacher_launch(directory,cohort,p.SEEDS[0],now=p.GLOBAL_DEADLINE-9000)
            self.assertEqual(launch['expiry'],p.GLOBAL_DEADLINE-1800)
            with self.assertRaises(TimeoutError): p.teacher_launch(directory,cohort,p.SEEDS[1],now=p.GLOBAL_DEADLINE-8999)

    def test_full_24_checkpoint_census_before_release(self):
        frozen=frozen_fixture(self.part,self.original,self.source)
        self.assertEqual(p.verify_frozen_cohort(frozen,self.part,self.original,self.source),p.digest(frozen))
        for kind in ('pending','missing-teacher','missing-cell','missing-budget','mutated-teacher'):
            bad=copy.deepcopy(frozen); teacher=bad['teachers'][str(p.SEEDS[0])]
            if kind=='pending': teacher['trajectories']['w8_mse']['status']='running'
            elif kind=='missing-teacher': bad['teachers'].pop(str(p.SEEDS[2]))
            elif kind=='missing-cell': teacher['trajectories'].pop('w16_kl')
            elif kind=='missing-budget': teacher['trajectories']['w8_mse']['checkpoints'].pop('2048')
            else: teacher['teacher_state_after_sha256']='b'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError): p.verify_frozen_cohort(bad,self.part,self.original,self.source)
        bad=copy.deepcopy(frozen); tr=bad['teachers'][str(p.SEEDS[0])]['trajectories']['w8_mse']
        tr['updates']=1024
        with self.assertRaises(ValueError): p.verify_frozen_cohort(bad,self.part,self.original,self.source)

    def test_failure_and_not_admitted_remain_visible(self):
        frozen=frozen_fixture(self.part,self.original,self.source)
        teacher=frozen['teachers'][str(p.SEEDS[0])]
        teacher['trajectories']['w8_mse']=dict(status='failed',updates=1024,reason='synthetic nonfinite failure',
            checkpoints={'1024':dict(status='saved',checkpoint_sha256=H,fit_metrics_sha256=H),'2048':dict(status='missing',reason='failed before fixed final')})
        frozen['teachers'][str(p.SEEDS[2])]=r.not_admitted_record('original teacher did not meet admission')
        p.verify_frozen_cohort(frozen,self.part,self.original,self.source)
        self.assertEqual(len(frozen['teachers']),3)
        with self.assertRaises(ValueError): r.not_admitted_record('')

    def test_one_shot_release_and_separate_diagnostic_window(self):
        frozen=frozen_fixture(self.part,self.original,self.source)
        approval=dict(action='approved-private-utility-gap-diagnostic-release',frozen_cohort_sha256=p.digest(frozen),
                      official_test=False,original_validation=False,exploratory=True)
        args=(approval,p.digest(approval),frozen,self.part,self.original,self.source)
        with self.assertRaises(PermissionError): p.release_gate(*args)
        p.release_gate(*args,release_authorized=True)
        with tempfile.TemporaryDirectory() as directory:
            consumed=p.begin_release(directory,self.cohort,p.digest(frozen),now=NOW+19800)
            self.assertEqual(consumed['expiry'],NOW+21600)
            p.check_release_clock(consumed,self.cohort,now=NOW+20000)
            with self.assertRaises(FileExistsError): p.begin_release(directory,self.cohort,p.digest(frozen),now=NOW+19801)
            with self.assertRaises(TimeoutError): p.check_release_clock(consumed,self.cohort,now=NOW+21600)
            changed=dict(consumed,expiry=NOW+21601)
            with self.assertRaises(ValueError): p.check_release_clock(changed,self.cohort,now=NOW+20000)


class CoreTests(unittest.TestCase):
    def test_architecture_init_isolation_and_fresh_adam(self):
        before=torch.get_rng_state().clone(); cells=c.make_cells(p.SEEDS[0])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for width,count in ((8,5120),(16,10240)):
            mse,kl=cells[f'w{width}_mse'],cells[f'w{width}_kl']
            self.assertEqual(sum(param.numel() for param in mse.parameters()),count)
            self.assertEqual(len(mse),2); self.assertEqual(c.state_hash(mse),c.state_hash(kl))
            self.assertFalse(any(isinstance(module,nn.BatchNorm2d) for module in mse.modules()))
            for a,b in zip(mse.parameters(),kl.parameters()):
                self.assertNotEqual(a.data_ptr(),b.data_ptr()); self.assertTrue(torch.equal(a,b))
            optimizer=c.make_optimizer(mse)
            self.assertEqual(optimizer.state_dict()['state'],{})
            group=optimizer.param_groups[0]
            self.assertEqual((group['lr'],group['betas'],group['eps'],group['weight_decay']),(.001,(.9,.999),1e-8,0.))
        self.assertNotEqual(c.state_hash(cells['w8_mse']),c.state_hash(cells['w16_mse']))

    def test_exact_schedule_64_complete_passes(self):
        part,*_=fixtures(); before=torch.get_rng_state().clone(); schedule=c.schedule(part,p.SEEDS[0])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(schedule,c.schedule(part,p.SEEDS[0])); self.assertEqual(len(schedule),2048)
        for index in range(64):
            epoch=schedule[index*32:(index+1)*32]
            self.assertEqual([len(batch) for batch in epoch],[128]*31+[32])
            self.assertEqual(sorted(i for batch in epoch for i in batch),sorted(part['fit_ids']))
        self.assertEqual(sum(map(len,schedule))*4*3,3072000)
        self.assertEqual(2048*4*3,24576)
        self.assertEqual(len(schedule[:1024]),32*32)

    def test_kl_t1_manual_equivalence_and_detached_teacher(self):
        student=torch.tensor([[1.,2.,-1.],[3.,1.,0.]],requires_grad=True)
        teacher=torch.tensor([[2.,1.,0.],[1.,1.,1.]],requires_grad=True)
        q=teacher.detach().softmax(1)
        expected=(q*(teacher.detach().log_softmax(1)-student.log_softmax(1))).sum(1).mean()
        self.assertTrue(torch.equal(c.logit_kl(student,teacher),expected))
        c.logit_kl(student,teacher).backward()
        self.assertIsNone(teacher.grad); self.assertGreater(float(student.grad.norm()),0)

    def test_nonzero_gradients_through_frozen_suffix_and_hash_parity(self):
        teacher=c.DifferentiableFrozenTeacher(synthetic_teacher()); before=teacher.verify_immutable()
        teacher.train(True); self.assertTrue(all(not m.training for m in teacher.modules()))
        images=torch.rand(2,3,8,8)
        x,target,tlogits=teacher.targets(images)
        self.assertTrue(all(not tensor.requires_grad for tensor in (x,target,tlogits)))
        with torch.no_grad(): self.assertTrue(torch.equal(tlogits,teacher.teacher.model(images)))
        for name,model in c.make_cells(p.SEEDS[0]).items():
            predicted=model(x); logits=teacher.suffix_logits(predicted)
            self.assertTrue(logits.requires_grad)
            loss=c.objective(name.split('_')[1],predicted,target,logits,tlogits); loss.backward()
            self.assertTrue(all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters()))
            self.assertGreater(sum(float(param.grad.abs().sum()) for param in model.parameters()),0)
            c.make_optimizer(model).step(); self.assertEqual(teacher.verify_immutable(),before)
        self.assertTrue(all(param.grad is None and not param.requires_grad for param in teacher.parameters()))
        teacher.teacher.model.bn.running_mean.add_(1)
        with self.assertRaises(ValueError): teacher.verify_immutable()

    def test_centered_nmse_sample_aggregation(self):
        target=torch.tensor([0.,2.,4.]).reshape(3,1,1,1); predicted=target+1
        stats=c.EndpointStats(); stats.update(predicted[:2],target[:2]); stats.update(predicted[2:],target[2:])
        self.assertAlmostEqual(stats.result()['centered_span_nmse'],3/(8+3e-12))
        log=[dict(examples=128,objective_sum=256.),dict(examples=32,objective_sum=320.)]
        self.assertEqual(r.weighted_training_summary(log)['objective'],576/160)


class RuntimeTests(unittest.TestCase):
    def session(self):
        part,_,source,_,_,cohort,clock=fixtures()
        return r.FitSession(synthetic_teacher(),H,part,p.SEEDS[0],clock,cohort,source,now=NOW+1)

    def test_finite_step_checkpoint_adam_rng_and_atomic_last_good(self):
        session=self.session(); ids=session.batches[0]
        entry=session.step('w8_kl',ids,torch.rand(len(ids),3,8,8),now=NOW+1)
        self.assertEqual(entry['update'],1); self.assertEqual(entry['examples'],128)
        self.assertGreater(entry['gradient_norm'],0); self.assertGreater(entry['relative_parameter_update_norm'],0)
        with tempfile.TemporaryDirectory() as directory:
            checksum=session.save_last('w8_kl',directory); path=Path(directory)/'w8_kl_last.ckpt'; blob=path.read_bytes()
            self.assertEqual(p.sha(blob),checksum); state=r.decode(blob)
            self.assertEqual(state['updates'],1); self.assertEqual(state['teacher_state_after_sha256'],session.teacher_before)
            self.assertTrue(state['optimizer']['state']); self.assertIn('torch_rng',state); self.assertIn('python_rng',state); self.assertIn('numpy_rng',state)
            self.assertTrue(all(float(value['step'])==1 for value in state['optimizer']['state'].values()))
            with self.assertRaises(ValueError): r.decode(blob+b'corrupt')
            with self.assertRaises(FloatingPointError): session.step('w8_kl',session.batches[1],torch.full((128,3,8,8),float('nan')),now=NOW+1)
            with self.assertRaises(RuntimeError): session.save_last('w8_kl',directory)
            self.assertEqual(path.read_bytes(),blob)
            with self.assertRaises(RuntimeError): session.step('w8_kl',session.batches[1],torch.rand(128,3,8,8),now=NOW+1)

    def test_reject_wrong_batch_no_continuation_and_intermediate_observation(self):
        session=self.session()
        with self.assertRaises(ValueError): session.step('w8_mse',tuple(reversed(session.batches[0])),torch.rand(128,3,8,8),now=NOW+1)
        session.cursor['w8_mse']=1024
        with self.assertRaises(ValueError): session.step('w8_mse',session.batches[1024],torch.rand(128,3,8,8),now=NOW+1)
        session.cursor['w8_mse']=2048
        with self.assertRaises(ValueError): session.step('w8_mse',(),torch.rand(1,3,8,8),now=NOW+1)
        self.assertFalse(hasattr(session,'restore')); self.assertFalse(hasattr(session,'final_validation'))

    def test_fixed_checkpoint_is_exclusive_not_best_selection(self):
        from .test_review_regressions import consistent_synthetic_state, bound_metrics
        session=self.session(); state=consistent_synthetic_state(session,'w8_mse',1024)
        session.cursor['w8_mse']=1024; session.logs['w8_mse']=state['training_log']
        session.optimizers['w8_mse'].load_state_dict(state['optimizer'])
        metrics=bound_metrics(state); session._metric_receipts[('w8_mse',1024)]=copy.deepcopy(metrics)
        with tempfile.TemporaryDirectory() as directory:
            record=session.freeze_checkpoint('w8_mse',directory,metrics)
            self.assertEqual(record['status'],'saved')
            self.assertEqual(record['fit_metrics_sha256'],p.digest(metrics))
            with self.assertRaises(ValueError): session.freeze_checkpoint('w8_mse',directory,metrics)
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),['w8_mse_update1024.ckpt','w8_mse_update1024_fit.json'])
        terminal=session.terminal_record(reason='synthetic cap exercise',now=session.teacher_clock['expiry'])
        self.assertEqual(terminal['trajectories']['w8_mse']['status'],'cap-exhausted')
        self.assertEqual(terminal['trajectories']['w8_mse']['checkpoints']['2048']['status'],'missing')


class AccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads={}
        for name,array in [('fit_images.npy',np.zeros((4000,3,32,32),dtype=np.uint8)),
                           ('diagnostic_images.npy',np.zeros((1000,3,32,32),dtype=np.uint8)),
                           ('diagnostic_labels.npy',np.zeros(1000,dtype=np.int64))]:
            buffer=io.BytesIO(); np.save(buffer,array,allow_pickle=False); cls.payloads[name]=buffer.getvalue()

    def setup_payloads(self):
        part,original,source,manifest,approval,cohort,clock=fixtures()
        manifest['payload_sha256']={name:p.sha(blob) for name,blob in self.payloads.items()}
        approval['artifact_manifest_sha256']=p.digest(manifest)
        return part,original,source,manifest,approval,cohort,clock

    def test_fit_reads_only_fit_payload_and_has_no_label_capability(self):
        part,original,source,manifest,approval,_,_=self.setup_payloads(); seen=[]
        def reader(name): seen.append(name); return self.payloads[name]
        access=d.open_fit_access(reader,manifest,approval,p.digest(approval),original,part,source,execution_authorized=True,now=NOW+1)
        self.assertEqual(seen,['fit_images.npy']); self.assertEqual(tuple(access.ids),tuple(part['fit_ids']))
        self.assertEqual(access.image_batch(part['fit_ids'][:2]).shape,(2,3,32,32))
        with self.assertRaises(PermissionError): access.image_batch(part['diagnostic_ids'][:1])
        self.assertFalse(hasattr(access,'labels')); self.assertFalse(hasattr(access,'reveal_labels'))

    def test_diagnostic_default_denied_and_labels_after_predictions_only(self):
        part,original,source,manifest,approval,cohort,_=self.setup_payloads(); frozen=frozen_fixture(part,original,source)
        release=dict(action='approved-private-utility-gap-diagnostic-release',frozen_cohort_sha256=p.digest(frozen),official_test=False,original_validation=False,exploratory=True)
        seen=[]
        def reader(name): seen.append(name); return self.payloads[name]
        with tempfile.TemporaryDirectory() as directory:
            consumed=p.begin_release(directory,cohort,p.digest(frozen),now=NOW+100)
            args=(reader,manifest,part,p.digest(manifest),frozen,original,source,release,p.digest(release),consumed,cohort)
            with self.assertRaises(PermissionError): d.open_diagnostic_vault(*args,directory=directory,now=NOW+101)
            self.assertEqual(seen,[])
            vault=d.open_diagnostic_vault(*args,directory=directory,release_authorized=True,now=NOW+101)
            self.assertEqual(seen,['diagnostic_images.npy'])
            with self.assertRaises(FileExistsError): d.open_diagnostic_vault(*args,directory=directory,release_authorized=True,now=NOW+102)
            self.assertEqual(seen,['diagnostic_images.npy'])
            with self.assertRaises(PermissionError): vault.image_batch(part['fit_ids'][:1])
            with self.assertRaises(ValueError): vault.reveal_labels({},p.digest({}))
            self.assertEqual(seen,['diagnostic_images.npy'])
            teachers={str(seed):dict(status='failed',checkpoints={f'{name}@{step}':dict(status='missing',reason='synthetic missing') for name in p.CELLS for step in p.BUDGETS}) for seed in p.SEEDS}
            predictions,_=diagnostic.freeze_predictions(directory,frozen,part,teachers)
            labels=vault.reveal_labels(predictions,p.digest(predictions),now=NOW+102); self.assertEqual(len(labels),1000)
            self.assertEqual(seen,['diagnostic_images.npy','diagnostic_labels.npy'])
            with self.assertRaises(PermissionError): vault.reveal_labels(predictions,p.digest(predictions))

    def test_payload_tamper_and_normalization_tamper(self):
        part,_,_,manifest,_,_,_=self.setup_payloads()
        with self.assertRaises(ValueError): d.read_array(lambda _:b'wrong',manifest,'fit_images.npy',(4000,3,32,32),np.dtype('uint8'))
        manifest['normalization']['std'][0]=0
        with self.assertRaises(ValueError): d.verify_artifact_manifest(manifest,part,p.digest(manifest))

    def test_source_has_no_actual_dataset_decoder_cli_or_network(self):
        root=Path(__file__).parent
        forbidden={'load_approved_data','decode_train_members','final_validation','urlopen','requests','subprocess'}
        for name in ('protocol.py','core.py','data.py','runtime.py','diagnostic.py','report.py'):
            tree=ast.parse((root/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    called=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                    self.assertNotIn(called,forbidden)


class ReportingTests(unittest.TestCase):
    def test_paired_counts_are_exact_not_reconstructed_aggregates(self):
        labels=[0]*1000; teacher=[0]*900+[1]*100; student=[0]*890+[1]*10+[0]*20+[1]*80
        result=report.paired_counts(teacher,student,labels)
        self.assertEqual(result['teacher_correct'],900); self.assertEqual(result['student_correct'],910)
        self.assertEqual(result['teacher_correct_student_wrong'],10); self.assertEqual(result['teacher_wrong_student_correct'],20)
        self.assertEqual(result['prediction_disagreements'],30); self.assertEqual(result['teacher_drop'],-.01)

    def test_flags_boundaries_zero_baseline_and_missing(self):
        cells={f'{name}@{step}':dict(fit_nmse=1.,diagnostic_nmse=1.,diagnostic_kl=1.,correct=800) for name in p.CELLS for step in p.BUDGETS}
        cells['w8_mse@2048']=dict(fit_nmse=.9,diagnostic_nmse=.9,diagnostic_kl=1.,correct=810)
        result=report.contrasts(cells)
        self.assertTrue(result['budget']['w8_mse']['budget_sensitivity']); self.assertTrue(result['budget']['w8_mse']['task_relevant'])
        self.assertFalse(result['formal_noninferiority']); self.assertTrue(result['complete_matrix'])
        self.assertIsNone(report.relative_reduction(0,0)); self.assertFalse(report.drops_ten_percent(0,0))
        cells['w8_kl@2048']=dict(fit_nmse=1.,diagnostic_nmse=1.,diagnostic_kl=.9,correct=820)
        self.assertTrue(report.contrasts(cells)['alignment']['8']['alignment_evidence'])
        cells['w8_kl@2048']['diagnostic_nmse']=.8
        result=report.contrasts(cells)['alignment']['8']
        self.assertFalse(result['alignment_evidence']); self.assertTrue(result['alternative_objective_helps_without_isolated_alignment'])
        cells['w16_kl@2048']=None
        self.assertFalse(report.contrasts(cells)['complete_matrix'])
        self.assertEqual(report.contrasts(cells)['budget']['w16_kl']['status'],'missing/inconclusive')

    def test_prediction_freeze_requires_entire_matrix_and_exact_checkpoint(self):
        part,original,source,*_=fixtures(); frozen=frozen_fixture(part,original,source)
        teachers={str(seed):dict(status='failed',checkpoints={f'{name}@{step}':dict(status='missing',reason='synthetic missing') for name in p.CELLS for step in p.BUDGETS}) for seed in p.SEEDS}
        with tempfile.TemporaryDirectory() as directory:
            predictions,checksum=diagnostic.freeze_predictions(directory,frozen,part,teachers)
            self.assertEqual(p.digest(predictions),checksum)
            self.assertTrue((Path(directory)/'predictions_frozen.json').exists())
            with self.assertRaises(FileExistsError): diagnostic.freeze_predictions(directory,frozen,part,teachers)
        teachers[str(p.SEEDS[0])]['checkpoints'].pop('w8_mse@1024')
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(ValueError): diagnostic.freeze_predictions(directory,frozen,part,teachers)


if __name__=='__main__': unittest.main(verbosity=2)
