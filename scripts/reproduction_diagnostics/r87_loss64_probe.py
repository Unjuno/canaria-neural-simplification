"""R87D4 local post-hoc numerical mitigation; never replace original science."""
import os,sys,json,hashlib,argparse
from pathlib import Path
import torch.nn.functional as F
sys.path.insert(0,str(Path(os.environ.get('CANARIA_REPO',str(Path(__file__).resolve().parents[2])) )/'scripts/reproduction_diagnostics'))
from r87_dispatch import execute
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--mode',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
assert a.seed in (1202,1205)
original=F.cross_entropy
# This is intentionally a different numeric recipe. Weight storage remains float32.
def loss64(x,y,*args,**kwargs):return original(x.to(dtype=__import__('torch').float64),y,*args,**kwargs)
F.cross_entropy=loss64
try:execute(a.seed,a.mode,a.out)
finally:F.cross_entropy=original
p=a.out/'record.json';r=json.loads(p.read_text());r['experiment']='R87D4_TEACHER_LOSS64_MITIGATION';r['evidence_class']='POST_HOC_NUMERICAL_MITIGATION_KNOWN_SEEDS';r['intervention']={'teacher_cross_entropy':'float64 input and scalar loss; autograd back to float32 logits','other_fits':'original float32 MSE','wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'historical_runner_modified':False};p.write_text(json.dumps(r,indent=2)+'\n')
