import json,math,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); P=json.loads(Path(__file__).with_name("PROTOCOL.json").read_text()); rows=[]
for s in P["seeds"]: rows.append(json.loads((root/f"seed_{s}"/"RESULT.json").read_text()))
el=[r for r in rows if r["eligible"]]
logs=[math.log(r["direct_single84"]["hold_nmse"]/r["component_pair8"]["hold_nmse"]) for r in el]
diffs=[r["direct_single84"]["val_acc"]-r["component_pair8"]["val_acc"] for r in el]
rng=np.random.default_rng(P["analysis"]["bootstrap_seed"]); n=P["analysis"]["bootstrap_resamples"]
def ci(x):
 a=np.asarray(x,float); idx=rng.integers(0,len(a),size=(n,len(a))); m=a[idx].mean(1); return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]
cl=ci(logs); cd=ci(diffs); advance=len(el)>=P["minimum_eligible"] and cl[1]<0 and cd[0]>-.03
out={"experiment":P["experiment"],"attempted":len(rows),"eligible":len(el),
"mean_nmse_ratio":float(np.exp(np.mean(logs))),"bootstrap95_nmse_ratio":[float(np.exp(cl[0])),float(np.exp(cl[1]))],
"mean_val_diff_pp":100*float(np.mean(diffs)),"bootstrap95_val_diff_pp":[100*cd[0],100*cd[1]],
"decision":"ADVANCE_MATCHED_BUDGET_DIRECT_TO_CONFIRMATION" if advance else "NO_ADVANCE_FROM_MATCHED_BUDGET_EXPLORATION","rows":rows}
print(json.dumps(out,indent=2)); (root/"DECISION.json").write_text(json.dumps(out,indent=2)+"\n")
