import csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader(open(HERE/'seed_rows.csv')))
el=[r for r in rows if r['eligible']=='True']
assert len(rows)==12 and len(el)==10
assert [int(r['seed']) for r in el]==[1581,1582,1584,1585,1586,1587,1588,1589,1590,1591]
def a(k): return np.array([float(r[k]) for r in el])
D1,D2,D3,R,V,T=map(a,['D_frozen','D_sketch','D_generic','R_full','validation_accuracy_difference','test_accuracy_difference'])
rng=np.random.default_rng(20261330); idx=rng.integers(0,len(el),size=(100000,len(el)))
def ci(x): return [float(np.quantile(x,.025)),float(np.quantile(x,.975))]
calc={'P1':(float(D1.mean()),ci(D1[idx].mean(1))),'P2':(float(D2.mean()),ci(D2[idx].mean(1))),'P3':(float(D3.mean()),ci(D3[idx].mean(1))),'P4':(float(np.exp(np.log(R).mean())),ci(np.exp(np.log(R)[idx].mean(1)))),'validation_safeguard':(float(V.mean()),ci(V[idx].mean(1))),'test_safeguard':(float(T.mean()),ci(T[idx].mean(1)))}
res=json.load(open(HERE/'RESULT.json'))
assert res['status']=='CONFIRMATORY_PASS' and res['eligibility_count']==10
keys={'P1':'estimate_mean_D_frozen','P2':'estimate_mean_D_sketch','P3':'estimate_mean_D_generic','P4':'estimate_geomean_R_full','validation_safeguard':'estimate_mean_difference','test_safeguard':'estimate_mean_difference'}
for k,(est,interval) in calc.items():
    e=res['endpoints'][k]; assert abs(e[keys[k]]-est)<1e-12; assert max(abs(e['ci95'][i]-interval[i]) for i in (0,1))<1e-12
assert calc['P1'][1][1]<0 and calc['P2'][1][1]<0 and calc['P3'][1][1]<0
assert calc['P4'][1][1]<1.30
assert calc['validation_safeguard'][1][0]>-0.04 and calc['test_safeguard'][1][0]>-0.04
assert all(x<0 for x in D1) and all(x<0 for x in D2) and all(x<0 for x in D3)
print('C25_AUDIT_PASS')
