import json,math,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); P=json.loads(Path(__file__).with_name("PROTOCOL.json").read_text()); rows=[]
for s in P["fresh_seeds"]: rows.append(json.loads((root/f"seed_{s}"/"RESULT.json").read_text()))
el=[r for r in rows if r["eligible"]]
logs=[math.log(r["direct_single84"]["hold_nmse"]/r["component_pair8"]["hold_nmse"]) for r in el]
diffs=[r["direct_single84"]["val_acc"]-r["component_pair8"]["val_acc"] for r in el]
rng=np.random.default_rng(P["decision"]["bootstrap_seed"]); n=P["decision"]["bootstrap_resamples"]
def ci(x):
 a=np.asarray(x,float); idx=rng.integers(0,len(a),size=(n,len(a))); m=a[idx].mean(1); return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]
cl=ci(logs); cd=ci(diffs)
coverage=len(el)>=P["minimum_eligible"]
primary=coverage and cl[1]<0; safeguard=coverage and cd[0]>-.03
decision="VIT_MATCHED_BUDGET_CONFIRMATORY_V2_PASS" if primary and safeguard else ("UNCERTAIN_COVERAGE" if not coverage else "VIT_MATCHED_BUDGET_CONFIRMATORY_V2_FAIL")
out={"experiment":P["experiment"],"decision":decision,"attempted":len(rows),"eligible":len(el),"mean_nmse_ratio":float(np.exp(np.mean(logs))) if logs else None,"bootstrap95_nmse_ratio":[float(np.exp(cl[0])),float(np.exp(cl[1]))] if logs else None,"mean_val_diff_pp":100*float(np.mean(diffs)) if diffs else None,"bootstrap95_val_diff_pp":[100*cd[0],100*cd[1]] if diffs else None,"primary_pass":primary,"validation_safeguard_pass":safeguard,"rows":rows}
print(json.dumps(out,indent=2)); (root/"DECISION.json").write_text(json.dumps(out,indent=2)+"\n")
