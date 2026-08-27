from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--reference',type=Path,required=True); a=ap.parse_args()
    out=np.fromfile(a.output,dtype='<f4'); ref=np.fromfile(a.reference,dtype='<f4')
    if out.size != 2*48*24 or ref.size != out.size: raise SystemExit('shape/size mismatch')
    d=out.astype(np.float64)-ref.astype(np.float64)
    max_abs=float(np.max(np.abs(d))); rel=float(np.linalg.norm(d)/(np.linalg.norm(ref.astype(np.float64))+1e-30))
    checks={'max_abs_lte_5e_5':max_abs<=5e-5,'relative_l2_lte_1e_5':rel<=1e-5}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','max_absolute_output_difference':max_abs,'relative_l2_output_difference':rel,'output_sum':float(out.sum()),'reference_sum':float(ref.sum()),'output_sha256':sha256(a.output),'checks':checks}
    print(json.dumps(report))
if __name__=='__main__': main()
