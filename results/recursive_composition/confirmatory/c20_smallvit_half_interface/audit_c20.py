import csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader(open(HERE/'seed_rows.csv')))
el=[r for r in rows if r['eligible']=='True']
assert len(rows)==12 and len(el)==9
assert [int(r['seed']) for r in el]==[1510,1512,1514,1515,1517,1518,1519,1520,1521]
def a(k): return np.array([float(r[k]) for r in el])
D,R,S,V,T=map(a,['D_worst','R_worst','basis_sensitivity','validation_accuracy_difference_worst','test_accuracy_difference_worst'])
rng=np.random.default_rng(20261110); idx=rng.integers(0,len(el),size=(100000,len(el)))
def ci(x): return [float(np.quantile(x,.025)),float(np.quantile(x,.975))]
calc={
'P1':(float(D.mean()),ci(D[idx].mean(1))),
'P2':(float(np.exp(np.log(R).mean())),ci(np.exp(np.log(R)[idx].mean(1)))),
'P3':(float(np.exp(np.log(S).mean())),ci(np.exp(np.log(S)[idx].mean(1)))),
'validation_safeguard':(float(V.mean()),ci(V[idx].mean(1))),
'test_safeguard':(float(T.mean()),ci(T[idx].mean(1))),
}
res=json.load(open(HERE/'RESULT.json'))
assert res['status']=='CONFIRMATORY_PASS'
assert res['eligibility_count']==9
for k,(est,interval) in calc.items():
    e=res['endpoints'][k]
    key=[x for x in e if x.startswith('estimate_')][0]
    assert abs(e[key]-est)<1e-12,(k,e[key],est)
    assert max(abs(e['ci95'][i]-interval[i]) for i in (0,1))<1e-12,(k,e['ci95'],interval)
assert calc['P1'][1][1] < 0
assert calc['P2'][1][1] < 1.25
assert calc['P3'][1][1] < 1.15
assert calc['validation_safeguard'][1][0] > -0.03
assert calc['test_safeguard'][1][0] > -0.03
pairs=[]
for r in el:
    for k,v in r.items():
        if k.startswith('anchored_minus_sketch_nmse__') and v!='': pairs.append(float(v))
assert len(pairs)==36 and all(x<0 for x in pairs)
assert abs(float(np.mean(pairs))-res['informative_anchored_vs_sketch']['mean_nmse_difference'])<1e-12
print('C20_AUDIT_PASS')
