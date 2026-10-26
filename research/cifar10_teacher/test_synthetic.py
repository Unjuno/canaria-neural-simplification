"""Synthetic-only tests; never download, decode or evaluate real CIFAR."""
import copy
import pathlib
import random
import tempfile
import unittest
import numpy as np
import torch
from .run import Trainer, partition, normalization, evaluator_identity, canonical_hash

class SyntheticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(714)
        self.x = rng.integers(0,256,(10,3,32,32), dtype=np.uint8)
        self.y = np.arange(10, dtype=np.int64)
        self.ids = np.arange(7, dtype=np.int64)
        self.mean, self.std = normalization(self.x, self.ids)

    def trainer(self, binding='synthetic-only'):
        return Trainer(self.x, self.y, self.ids, self.mean, self.std, 26102601, binding)

    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a): self.assertTrue(torch.equal(a,b))
        elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a,b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a: self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b): self.assert_nested_equal(x,y)
        else: self.assertEqual(a,b)

    def test_portable_identity(self):
        import hashlib
        splits = dict(teachertrain=self.ids, calibration=np.array([7]), validation=np.array([8,9]))
        identity = evaluator_identity(splits, self.mean, self.std)
        self.assertEqual(identity['model_config_sha256'], canonical_hash(identity['model_config']))
        self.assertEqual(identity['splits']['validation']['sha256'],
                         hashlib.sha256(np.array([8,9], dtype='<i8').tobytes()).hexdigest())
        self.assertEqual(identity['normalization']['sha256'],
                         hashlib.sha256(np.array([self.mean,self.std], dtype='<f8').tobytes()).hexdigest())
        self.assertEqual(set(identity['splits']), {'teachertrain','calibration','validation'})

    def test_architecture_shape_and_pinned_source(self):
        t = self.trainer()
        self.assertEqual(sum(p.numel() for p in t.model.parameters()),269722)
        self.assertEqual(tuple(t.model(torch.zeros(2,3,32,32)).shape),(2,10))
        root = pathlib.Path(__file__).parent
        source = (root.parent/'cifar10_preflight/run.py').read_text()
        self.assertEqual((root/'model.py').read_text().split('class Block',1)[1],
                         source[source.index('class Block'):source.index('def main')].split('class Block',1)[1])

    def test_partition_and_train_only_normalization(self):
        y = np.repeat(np.arange(10),6)
        s = partition(y,4,1)
        self.assertEqual([len(v) for v in s.values()],[40,10,10])
        self.assertEqual(len(np.unique(np.concatenate(list(s.values())))),60)
        for k,v in s.items():
            self.assert_nested_equal(v,partition(y,4,1)[k])
        changed = self.x.copy(); changed[7:] = 0
        mean,std = normalization(changed,self.ids)
        np.testing.assert_array_equal(mean,self.mean); np.testing.assert_array_equal(std,self.std)

    def test_exact_mid_epoch_and_epoch_boundary_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)/'resume.ckpt'
            t = self.trainer(); t.step(3); self.assertEqual(t.cursor,3); t.save(path)
            for _ in range(4): t.step(3)
            expected = copy.deepcopy(t.state())
            expected_draws = (random.random(), np.random.random(), torch.rand(2))
            r = self.trainer(); r.restore(path)
            for _ in range(4): r.step(3)
            self.assert_nested_equal(expected,r.state())
            self.assert_nested_equal(expected_draws,(random.random(),np.random.random(),torch.rand(2)))
            r.step(3); self.assertEqual(r.cursor,0); self.assertEqual(r.epoch,2); r.save(path)
            r.step(3); expected = copy.deepcopy(r.state())
            q = self.trainer(); q.restore(path); q.step(3)
            self.assert_nested_equal(expected,q.state())

    def test_integrity_and_binding_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)/'state.ckpt'; t = self.trainer(); t.save(path)
            with self.assertRaisesRegex(ValueError,'mismatch'): self.trainer('wrong-data').restore(path)
            data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'integrity'): t.restore(path)

    def test_final_only_validation_and_lr(self):
        t = self.trainer()
        with self.assertRaisesRegex(ValueError,'only after'): t.validate_final(np.array([8,9]))
        for epoch,lr in ((59,.1),(60,.01),(89,.01),(90,.001),(119,.001)):
            t.epoch = epoch; t.cursor = 0; t.order = t.train.copy(); t.step(3)
            self.assertEqual(t.opt.param_groups[0]['lr'],lr)
        t.epoch = 120
        with self.assertRaisesRegex(ValueError,'overlap'): t.validate_final(np.array([0,1]))
        # No validation accuracy is computed even on synthetic data.
        with self.assertRaisesRegex(ValueError,'already complete'): t.step(3)

if __name__ == '__main__': unittest.main()
