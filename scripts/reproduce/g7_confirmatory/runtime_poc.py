from __future__ import annotations
import argparse, copy, gc, json, os, statistics, subprocess, sys, time
from pathlib import Path
import psutil
import torch
import run_seed as g

def build_models(seed: int):
    tr,va,te=g.datasets()
    m=g.new_model(4,48,seed); opt=g.task_optimizer(m,g.epoch_lr(0)); snap4=state4=None
    for ep in range(12):
        g.set_lr(opt,g.epoch_lr(ep)); g.train_one_epoch(m,tr,opt,seed+50000+ep)
        if ep+1==4:
            snap4=copy.deepcopy(m); state4=g.named_opt_state(m,opt)
    large=copy.deepcopy(m)
    compact=copy.deepcopy(snap4)
    c1,_=g.fit_replacement(compact,tr,3,36,15,seed+150000); g.replace_inplace(compact,c1)
    compact,o,_=g.continue_from(compact,state4,tr,va,4,8,seed); state8=g.named_opt_state(compact,o)
    c2,_=g.fit_replacement(compact,tr,2,24,14,seed+160000); g.replace_inplace(compact,c2)
    compact,o,_=g.continue_from(compact,state8,tr,va,8,12,seed)
    return large,compact

def save_artifact(model,kind,path):
    torch.save({'state_dict':model.state_dict()},path)
    manifest={'format':'canaria-runtime-poc-v1','kind':kind,'d':24,'heads':4,'seq_len':48,
              'depth':4 if kind=='large' else 2,'mlp':48 if kind=='large' else 24,
              'compiler':kind=='compact','params':g.nparams(model)}
    mp=path.with_suffix('.json'); mp.write_text(json.dumps(manifest,indent=2))
    return path.stat().st_size+mp.stat().st_size

def materialize(path,kind):
    if kind=='large': model=g.DecoderLM(depth=4,d=24,heads=4,mlp=48)
    else:
        model=g.DecoderLM(depth=0,d=24,heads=4,mlp=24)
        model.compiler=g.GenericCompiler(depth=2,mlp=24)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(payload['state_dict']); model.eval(); return model

def probe(path,kind):
    proc=psutil.Process(os.getpid()); gc.collect(); before=proc.memory_info().rss
    t0=time.perf_counter(); model=materialize(path,kind); load_ms=(time.perf_counter()-t0)*1000; after=proc.memory_info().rss
    _,_,te=g.datasets(); batch=next(iter(torch.utils.data.DataLoader(te,batch_size=128,shuffle=False)))[0][:,:-1]
    with torch.no_grad():
        for _ in range(3): model(batch)
        t0=time.perf_counter()
        for _ in range(20): model(batch)
        infer_ms=(time.perf_counter()-t0)*1000/20
    return {'load_materialize_ms':load_ms,'rss_delta_bytes':after-before,'inference_batch128_ms':infer_ms,'test':g.tf_metrics(model,te),'params':g.nparams(model)}

def child_probe(script,path,kind):
    out=subprocess.check_output([sys.executable,str(script),'--probe',str(path),'--kind',kind],text=True)
    return json.loads(out)

def summarize(rows):
    return {
      'n':len(rows),
      'load_materialize_ms_mean':statistics.mean(r['load_materialize_ms'] for r in rows),
      'load_materialize_ms_median':statistics.median(r['load_materialize_ms'] for r in rows),
      'load_materialize_ms_stdev':statistics.stdev(r['load_materialize_ms'] for r in rows) if len(rows)>1 else 0.0,
      'rss_delta_bytes_mean':statistics.mean(r['rss_delta_bytes'] for r in rows),
      'rss_delta_bytes_median':statistics.median(r['rss_delta_bytes'] for r in rows),
      'inference_batch128_ms_mean':statistics.mean(r['inference_batch128_ms'] for r in rows),
      'inference_batch128_ms_median':statistics.median(r['inference_batch128_ms'] for r in rows),
      'inference_batch128_ms_stdev':statistics.stdev(r['inference_batch128_ms'] for r in rows) if len(rows)>1 else 0.0,
      'test_ppl':rows[0]['test']['ppl'],'params':rows[0]['params']}

def run(seed,out_dir,repeat):
    out_dir.mkdir(parents=True,exist_ok=True); large,compact=build_models(seed)
    lp=out_dir/f'large_seed{seed}.pt'; cp=out_dir/f'compact_seed{seed}.pt'
    lbytes=save_artifact(large,'large',lp); cbytes=save_artifact(compact,'compact',cp)
    script=Path(__file__).resolve(); rows={}
    for kind,path in [('large',lp),('compact',cp)]: rows[kind]=[child_probe(script,path,kind) for _ in range(repeat)]
    L=summarize(rows['large']); C=summarize(rows['compact'])
    report={'poc':'G7 compact functional runtime materialization','seed':seed,'format':'canaria-runtime-poc-v1','hardware':'CPU only',
      'representation':'torch state_dict plus JSON manifest; the compact artifact executes the learned 2-block compiler natively and does not reconstruct the original 4-block model.',
      'serialized_bytes_including_manifest':{'large':lbytes,'compact':cbytes,'reduction_fraction':1-cbytes/lbytes},
      'large':L,'compact':C,
      'derived':{'parameter_reduction_fraction':1-C['params']/L['params'],
                 'load_mean_ratio_compact_over_large':C['load_materialize_ms_mean']/L['load_materialize_ms_mean'],
                 'load_median_ratio_compact_over_large':C['load_materialize_ms_median']/L['load_materialize_ms_median'],
                 'inference_mean_ratio_compact_over_large':C['inference_batch128_ms_mean']/L['inference_batch128_ms_mean'],
                 'inference_median_ratio_compact_over_large':C['inference_batch128_ms_median']/L['inference_batch128_ms_median'],
                 'rss_mean_ratio_compact_over_large':C['rss_delta_bytes_mean']/L['rss_delta_bytes_mean'] if L['rss_delta_bytes_mean'] else None,
                 'ppl_compact_minus_large':C['test_ppl']-L['test_ppl']},
      'interpretation_boundary':'Single seed, CPU-only, small-model PoC. File/load/execution results are not evidence of universal runtime speedup; RSS delta is allocator/process dependent.'}
    (out_dir/'runtime_poc_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=4300); ap.add_argument('--out-dir',type=Path,default=Path('runtime_poc_out')); ap.add_argument('--repeat',type=int,default=5); ap.add_argument('--probe'); ap.add_argument('--kind',choices=['large','compact']); a=ap.parse_args()
    if a.probe: print(json.dumps(probe(Path(a.probe),a.kind)))
    else: run(a.seed,a.out_dir,a.repeat)
if __name__=='__main__': main()
