"""Synthetic-only validation; no model execution or new measurements."""
import copy
import math
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import check_checkpoint
import analyze_saved as a


def fixture(grid=False):
    rows = []
    for i in range(16):
        cell = dict(n=270, correct=265, nmse=.05, accuracy=265/270, quality_pass=True)
        row = dict(id=i, status='COMPLETE', teacher_correct=265)
        if grid:
            row['cells'] = [dict(cell, arm=arm, budget=b, reused=(b == 2048)) for arm in ('C', 'J') for b in a.BUDGETS]
        else:
            row['arms'] = {arm: dict(cell) for arm in ('C', 'J', 'L')}
        rows.append(row)
    return rows


def timing():
    return [dict(id=i, batch=b, round=r, position=p, arm=arm, iterations=20, total_ns=2000*(p+1), ns_per_forward=100*(p+1))
            for i in range(16) for b in (1, 128) for r in range(30) for p, arm in enumerate(('teacher', 'C', 'J'))]

class SavedTests(unittest.TestCase):
    def test_missing_expected_summary_field_rejected(self):
        rows = fixture()
        expected = a.matched(rows)
        expected.pop("mean_nmse")
        expected.pop("primary_ci95")
        expected["all_rows"] = rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "research").mkdir()
            (root / "research" / "summary.json").write_text(json.dumps(expected))
            with patch.dict(check_checkpoint.FILES, {"matched": ("summary.json", "summary.json")}, clear=True):
                with self.assertRaisesRegex(AssertionError, "missing expected summary fields"):
                    check_checkpoint.check(root)

    def test_matched_equal(self):
        out = a.matched(fixture())
        self.assertEqual(out['primary_ci95'], [0, 0])
        self.assertEqual(out['quality_counts'], dict(C=16,J=16,L=16))
        self.assertEqual(out['feature_gate'], 'UNCERTAIN')
    def test_grid_equal(self):
        out = a.grid(fixture(True))
        self.assertEqual(out['geometric_ratio'], 1)
        self.assertEqual(out['minima'][0]['C'], 512)
    def test_censor_no_complete_case_ratio(self):
        rows = fixture(True)
        for c in rows[0]['cells']:
            if c['arm'] == 'J':
                c.update(nmse=1,quality_pass=False)
        out = a.grid(rows)
        self.assertEqual(out['decision'], 'CENSORED_NO_COHORT_RATIO')
        self.assertNotIn('geometric_ratio', out)
        self.assertEqual(out['coverage']['C_only'], 1)
    def test_nonmonotonic(self):
        rows = fixture(True)
        rows[0]['cells'][1].update(nmse=1,quality_pass=False)
        out = a.grid(rows)
        self.assertTrue(out['patterns'][0]['nonmonotone'])
        self.assertEqual(out['minima'][0]['C'], 512)
    def test_cohort_rejections(self):
        for func, grid in ((a.matched, False), (a.grid, True)):
            rows = fixture(grid)
            for invalid in (rows[:-1], rows+[rows[0]], [rows[0]]*16):
                with self.assertRaises(a.InvalidInput): func(invalid)
            rows[0]['status'] = 'FAILED'
            with self.assertRaises(a.InvalidInput): func(rows)
    def test_bad_metrics(self):
        for field, value in [('nmse', math.nan),('nmse', math.inf),('nmse', -.1),('correct', 271),('correct', True),('n', 269),('quality_pass', False),('accuracy', .1)]:
            rows=fixture(); rows[0]['arms']['C'][field]=value
            with self.subTest(field=field,value=value), self.assertRaises(a.InvalidInput): a.matched(rows)
    def test_missing(self):
        rows=fixture(); del rows[0]['arms']['J']['nmse']
        with self.assertRaises(KeyError): a.matched(rows)
    def test_grid_duplicate_and_reuse(self):
        rows=fixture(True); rows[0]['cells'][0]=copy.deepcopy(rows[0]['cells'][1])
        with self.assertRaises(a.InvalidInput): a.grid(rows)
        rows=fixture(True); rows[0]['cells'][0]['reused']=True
        with self.assertRaises(a.InvalidInput): a.grid(rows)
    def test_storage(self):
        rows=[dict(id=i, **{arm+'_'+m: n for arm,n in [('C',2),('J',4),('teacher',8)] for m in ('parameters','stored_bytes','deflated_bytes')}) for i in range(16)]
        self.assertEqual(a.storage(rows)['ratios']['C_over_J_parameters']['geomean'], .5)
        rows[0]['C_parameters']=math.nan
        with self.assertRaises(a.InvalidInput): a.storage(rows)
    def test_latency(self):
        raw=timing(); out=a.latency(raw)
        self.assertEqual(out['paired'][0]['C_ns'],200)
        self.assertEqual(out['paired'][0]['C_over_J'],2/3)
        raw[0]=copy.deepcopy(raw[1])
        with self.assertRaises(a.InvalidInput): a.latency(raw)
    def test_latency_incomplete_arithmetic_nonfinite(self):
        with self.assertRaises(a.InvalidInput): a.latency(timing()[:-1])
        for field,value in [('total_ns',0),('ns_per_forward',math.nan),('iterations',19),('position',3),('round',30),('ns_per_forward',999)]:
            raw=timing(); raw[0][field]=value
            with self.subTest(field=field),self.assertRaises(a.InvalidInput): a.latency(raw)

if __name__ == '__main__': unittest.main()
