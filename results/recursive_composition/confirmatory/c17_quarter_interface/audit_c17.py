from pathlib import Path
import csv, json, numpy as np
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'seed_rows.csv').open()))
assert len(rows)==8
D=np.array([float(r['D_worst']) for r in rows]); R=np.array([float(r['R_worst']) for r in rows]); S=np.array([float(r['basis_spread']) for r in rows]); V=np.array([float(r['val_acc_diff_worst_minus_full']) for r in rows]); T=np.array([float(r['test_acc_diff_worst_minus_full']) for r in rows])
rng=np.random.default_rng(20261013); idx=rng.integers(0,8,size=(100000,8))
def ci(x): return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
calc={
 'D':(float(D.mean()),ci(D[idx].mean(1))),
 'R':(float(np.exp(np.log(R).mean())),ci(np.exp(np.log(R[idx]).mean(1)))),
 'S':(float(np.exp(np.log(S).mean())),ci(np.exp(np.log(S[idx]).mean(1)))),
 'V':(float(V.mean()),ci(V[idx].mean(1))),
 'T':(float(T.mean()),ci(T[idx].mean(1))),
}
stored=json.loads((ROOT/'RESULT.json').read_text())['confirmatory_endpoints']
pairs=[('D','P1_worst_basis_repairs_frozen','mean_D_worst'),('R','P2_worst_basis_over_full64','geometric_mean_ratio'),('S','P3_basis_sensitivity','geometric_mean_spread'),('V','validation_utility_guardrail','mean_accuracy_difference'),('T','test_safeguard','mean_accuracy_difference')]
for k,e,p in pairs:
    point,interval=calc[k]; rec=stored[e]
    assert abs(point-rec[p])<1e-12, (k,point,rec[p])
    assert np.allclose(interval,rec['bootstrap95'],rtol=0,atol=1e-12), (k,interval,rec['bootstrap95'])
assert stored['P1_worst_basis_repairs_frozen']['pass'] and stored['P2_worst_basis_over_full64']['pass'] and stored['P3_basis_sensitivity']['pass'] and stored['validation_utility_guardrail']['pass'] and stored['test_safeguard']['pass']
print('C17_AUDIT_PASS')
