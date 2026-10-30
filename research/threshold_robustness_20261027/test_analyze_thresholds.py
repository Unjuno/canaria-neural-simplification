import copy
import unittest
import analyze_thresholds as a


def fixture():
    rows = []
    for i in range(16):
        cells = []
        for arm in a.ARMS:
            for budget in a.BUDGETS:
                nmse = .08 if budget >= (1536 if arm == 'C' else 2048) else .10
                cells.append(dict(arm=arm, budget=budget, correct=260, n=270,
                                  nmse=nmse, quality_pass=nmse <= .08, reused=budget == 2048))
        rows.append(dict(id=i, status='COMPLETE', teacher_correct=265, cells=cells))
    return rows

class Tests(unittest.TestCase):
    def test_surface_and_original(self):
        out = a.analyze(fixture())
        self.assertEqual(len(out['settings']), 25)
        self.assertEqual(len(out['selected_margins']), 32)
        original = next(s for s in out['settings'] if s['original_gate'])
        self.assertEqual(original['direction_counts'], dict(smaller=16, tied=0, larger=0))
        self.assertAlmostEqual(original['geometric_C_over_J'], .75)
    def test_inclusive_boundaries(self):
        c = {'nmse': .08, 'correct': 260}
        self.assertTrue(a.passes(c, 265, .08, 5))
        self.assertFalse(a.passes(c, 265, .08, 4))
        self.assertFalse(a.passes(c, 265, .079999999, 5))
    def test_float_integer_gate_exhaustive_counts(self):
        for teacher in range(271):
            for correct in range(271):
                self.assertEqual(correct / 270 >= teacher / 270 - .02, teacher - correct <= 5)
    def test_censoring_no_complete_case_rescue(self):
        rows = fixture()
        for c in rows[0]['cells']:
            if c['arm'] == 'C':
                c.update(nmse=.2, quality_pass=False)
        s = a.setting(a.validate(rows), .08, 5)
        self.assertEqual(s['coverage'], dict(both=15, C_only=0, J_only=1, neither=0))
        self.assertIsNone(s['geometric_C_over_J'])
        self.assertIsNone(s['direction_counts'])
    def test_nonmonotonic_first_pass_kept(self):
        rows = fixture()
        for c in rows[0]['cells']:
            if c['arm'] == 'C' and c['budget'] == 2048:
                c.update(nmse=.2, quality_pass=False)
        s = a.setting(a.validate(rows), .08, 5)
        self.assertEqual(s['first_passes'][0]['C'], 1536)
    def test_margins(self):
        margins = a.analyze(fixture())['selected_margins']
        self.assertTrue(all(m['correct_count_slack'] == 0 and m['correct_losses_to_fail'] == 1 for m in margins))
        self.assertTrue(all(m['nmse_slack'] == 0 for m in margins))
        self.assertTrue(all(m['tightened_allow4_budget'] is None for m in margins))
    def test_missing_seed(self):
        with self.assertRaises(ValueError): a.validate(fixture()[:-1])
    def test_duplicate_seed(self):
        rows=fixture(); rows[0]['id']=1
        with self.assertRaises(ValueError): a.validate(rows)
    def test_duplicate_cell(self):
        rows=fixture(); rows[0]['cells'][0]=copy.deepcopy(rows[0]['cells'][1])
        with self.assertRaises(ValueError): a.validate(rows)
    def test_flag_mismatch(self):
        rows=fixture(); rows[0]['cells'][0]['quality_pass']=True
        with self.assertRaises(ValueError): a.validate(rows)
    def test_invalid_nmse(self):
        for invalid in [float('nan'), float('inf'), -1, True]:
            rows=fixture(); rows[0]['cells'][0]['nmse']=invalid
            with self.assertRaises(ValueError): a.validate(rows)
    def test_invalid_count_and_n(self):
        for key, value in [('correct', 270.0), ('correct', True), ('correct', 271), ('n', 269)]:
            rows=fixture(); rows[0]['cells'][0][key]=value
            with self.assertRaises(ValueError): a.validate(rows)
    def test_invalid_reuse(self):
        rows=fixture(); rows[0]['cells'][0]['reused']=True
        with self.assertRaises(ValueError): a.validate(rows)
    def test_input_order_invariance(self):
        rows=fixture(); baseline=a.analyze(rows)
        rows.reverse()
        for row in rows: row['cells'].reverse()
        self.assertEqual(a.analyze(rows), baseline)

if __name__ == '__main__': unittest.main()
