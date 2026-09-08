from __future__ import annotations
import copy,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
V=load('publication_cohort','scripts/reproduce/core_discovery_digits/verify_confirmatory.py')
A=load('publication_audit','tools/audit_publication.py')
class CohortVerifierTests(unittest.TestCase):
    def setUp(self):
        self.expected=json.loads((ROOT/'results/core_discovery_digits/confirm_summary.json').read_text())
        self.protocol=json.loads((ROOT/'results/core_discovery_digits/PROTOCOL_LOCK.json').read_text())
        self.rows=copy.deepcopy(self.expected['per_seed_selected_budgets'])
    def test_archived_summary_self_consistent(self):self.assertEqual(V.compare_rows(self.rows,self.expected,self.protocol)[0],[])
    def test_missing_seed_rejected(self):self.assertTrue(V.compare_rows(self.rows[:-1],self.expected,self.protocol)[0])
    def test_duplicate_seed_rejected(self):self.rows[-1]=self.rows[0];self.assertTrue(V.compare_rows(self.rows,self.expected,self.protocol)[0])
    def test_unexpected_seed_rejected(self):self.rows[-1]['seed']=9999;self.assertTrue(V.compare_rows(self.rows,self.expected,self.protocol)[0])
    def test_exact_budget_mismatch_rejected(self):self.rows[0]['composed']+=256;self.assertTrue(V.compare_rows(self.rows,self.expected,self.protocol)[0])
    def test_directional_success_does_not_erase_mismatch(self):
        self.rows[2]['componentwise']=3072;self.rows[2]['composed']=1536
        errors,agg=V.compare_rows(self.rows,self.expected,self.protocol)
        self.assertTrue(errors);self.assertEqual(agg['composed_lower_count'],8)
    def test_nonfinite_rejected(self):self.rows[0]['test_acc_diff']=float('nan');self.assertTrue(V.compare_rows(self.rows,self.expected,self.protocol)[0])
    def test_test_accuracy_tolerance_not_expanded(self):self.rows[0]['test_acc_diff']+=1e-5;self.assertTrue(V.compare_rows(self.rows,self.expected,self.protocol)[0])
    def test_log_tolerance_not_expanded(self):self.rows[0]['log2_ratio']+=1e-8;self.assertTrue(V.compare_rows(self.rows,self.expected,self.protocol)[0])
    def test_bootstrap_nan_rejected(self):
        with self.assertRaises(ValueError):V.bootstrap_ci([float('nan')],20,1)
class PreservationTests(unittest.TestCase):
    def test_actual_repository_preserved(self):self.assertEqual(A.audit(ROOT)['integrity_status'],'PASS')
    def test_announcement_gate_remains_blocked(self):self.assertEqual(A.audit(ROOT,True)['integrity_status'],'FAIL')
    def test_tampered_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'x').write_bytes(b'old')
            m={'files':[{'path':'x','sha256':hashlib.sha256(b'old').hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob 3\0old').hexdigest()}]}
            self.assertEqual(A.check_preservation(root,m),[]);(root/'x').write_bytes(b'new');self.assertTrue(A.check_preservation(root,m))
    def test_escape_rejected(self):
        with self.assertRaises(ValueError):A.safe_path(ROOT,'../escape')
if __name__=='__main__':unittest.main()
