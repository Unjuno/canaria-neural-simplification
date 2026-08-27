import csv,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent; rows=list(csv.DictReader(open(H/'seed_rows.csv'))); el=[r for r in rows if r['eligible']=='True']
assert len(rows)==12 and len(el)==11
assert [int(r['seed']) for r in el]==[1530,1531,1533,1534,1535,1536,1537,1538,1539,1540,1541]
a=lambda k:np.array([float(r[k]) for r in el]);D,R,S,V,T=map(a,['D_worst','R_worst','basis_sensitivity','validation_accuracy_difference_worst','test_accuracy_difference_worst']);rng=np.random.default_rng(20261130);idx=rng.integers(0,len(el),size=(100000,len(el)));ci=lambda x:[float(np.quantile(x,.025)),float(np.quantile(x,.975))]
calc={'P1':(float(D.mean()),ci(D[idx].mean(1))),'P2':(float(np.exp(np.log(R).mean())),ci(np.exp(np.log(R)[idx].mean(1)))),'P3':(float(np.exp(np.log(S).mean())),ci(np.exp(np.log(S)[idx].mean(1)))),'validation_safeguard':(float(V.mean()),ci(V[idx].mean(1))),'test_safeguard':(float(T.mean()),ci(T[idx].mean(1)))}
res=json.load(open(H/'RESULT.json'))
for k,(e,interval) in calc.items():
 s=res['endpoints'][k]; key=[x for x in s if x.startswith('estimate_')][0]; assert abs(s[key]-e)<1e-12; assert np.allclose(s['ci95'],interval,rtol=0,atol=1e-12)
assert calc['P1'][1][1]<0 and calc['P2'][1][1]<1.40 and calc['P3'][1][1]<1.20 and calc['validation_safeguard'][1][0]>-0.04 and calc['test_safeguard'][1][0]>-0.04
pairs=[float(v) for r in el for k,v in r.items() if k.startswith('anchored_minus_sketch_nmse__') and v!='']; assert len(pairs)==44 and all(x<0 for x in pairs); assert abs(np.mean(pairs)-res['informative_anchored_vs_sketch']['mean_nmse_difference'])<1e-12
print('C21_AUDIT_PASS')
