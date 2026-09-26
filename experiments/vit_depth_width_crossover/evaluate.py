import json,math,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);P=json.loads(Path(__file__).with_name("PROTOCOL.json").read_text());rows=[json.loads((root/f"seed_{s}"/"RESULT.json").read_text()) for s in P["seeds"]];el=[r for r in rows if r["eligible"]];rng=np.random.default_rng(P["analysis"]["bootstrap_seed"]);N=P["analysis"]["bootstrap_resamples"];summary=[]
def ci(a):
 a=np.asarray(a,float);idx=rng.integers(0,len(a),size=(N,len(a)));m=a[idx].mean(1);return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]
for j,z in enumerate(P["pairs"]):
 logs=[math.log(r["comparisons"][j]["single_nmse"]/r["comparisons"][j]["pair_nmse"]) for r in el];diff=[r["comparisons"][j]["single_val_acc"]-r["comparisons"][j]["pair_val_acc"] for r in el];cl=ci(logs);cd=ci(diff);summary.append({**z,"mean_nmse_ratio":float(np.exp(np.mean(logs))),"bootstrap95_nmse_ratio":[float(np.exp(cl[0])),float(np.exp(cl[1]))],"single_lower_nmse_count":sum(x<0 for x in logs),"mean_val_diff_pp":100*float(np.mean(diff)),"bootstrap95_val_diff_pp":[100*cd[0],100*cd[1]]})
out={"experiment":P["experiment"],"attempted":len(rows),"eligible":len(el),"summary":summary,"rows":rows};print(json.dumps(out,indent=2));(root/"DECISION.json").write_text(json.dumps(out,indent=2)+"\n")
