"""Recompute reported conditional bootstrap from public CELLS.csv; no models/data."""
import csv,json,pathlib
import numpy as np
H=pathlib.Path(__file__).resolve().parent
with (H/'CELLS.csv').open() as f:rows=list(csv.DictReader(f))
rng=np.random.default_rng(3000001000);out=[]
for tid in (0,4):
 pairs=[[next(x for x in rows if int(x['teacher'])==tid and int(x['replicate'])==r and int(x['budget'])==b) for b in (2048,3072)] for r in range(8)]
 v=np.array([np.log2(float(y['nmse'])/float(x['nmse'])) for x,y in pairs]);acc=np.array([100*(int(y['correct'])-int(x['correct']))/270 for x,y in pairs]);ix=rng.integers(0,8,(100000,8));ci=np.quantile(v[ix].mean(1),[.025,.975]);aci=np.quantile(acc[ix].mean(1),[.025,.975])
 out.append({'teacher':tid,'mean_log2_nmse3072_over2048':float(v.mean()),'ci95':ci.tolist(),'nmse_improvement_decision':'PASS' if ci[1]<0 else 'FAIL' if ci[0]>0 else 'UNCERTAIN','mean_accuracy_difference_pp':float(acc.mean()),'accuracy_ci95_pp':aci.tolist()})
print(json.dumps(out,indent=2))
