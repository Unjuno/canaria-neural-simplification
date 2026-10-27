#!/usr/bin/env python3
"""Portable worker; only supervise.py may start a separately authorized attempt.
Import is inert: no ML imports, configuration, archives, clock sampling or writes.
"""
import argparse,datetime,io,json,os,pathlib,platform,resource,signal,subprocess,time,traceback
import latency_contract as contract
from timing_core import ARMS,ROUNDS,ITERS,WARMUP,EvidenceWriter,ordered,timed_block,summarize,failure_census,full_census,validate_raw
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNTIME_ENV={'ATEN_CPU_CAPABILITY':'avx2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','MKL_DYNAMIC':'FALSE','OMP_DYNAMIC':'FALSE','MKL_CBWR':'COMPATIBLE'}


def configure_runtime():
    os.environ.update(RUNTIME_ENV)
    import deployment as d
    d.torch.set_num_threads(1);d.torch.set_num_interop_threads(1)
    d.torch.backends.mkldnn.enabled=False;d.torch.use_deterministic_algorithms(True)
    return d


def environment(d):
    """PRIVATE evidence only; raw host/process descriptors must not be published."""
    import sklearn,scipy
    versions={'python':platform.python_version(),'torch':d.torch.__version__,'numpy':d.np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__}
    assert versions=={'python':'3.13.5','torch':'2.10.0+cpu','numpy':'2.3.5','sklearn':'1.8.0','scipy':'1.17.0'}
    assert d.torch.backends.cpu.get_cpu_capability()=='AVX2' and d.torch.get_num_threads()==d.torch.get_num_interop_threads()==1
    assert d.torch.backends.mkldnn.enabled is False
    return {'versions':versions,'cpuinfo':pathlib.Path('/proc/cpuinfo').read_text(),'affinity':sorted(os.sched_getaffinity(0)),'loadavg':os.getloadavg(),'processes':subprocess.check_output(['ps','-eo','pid,ppid,pcpu,comm,args'],text=True),'torch_config':d.torch.__config__.show(),'mkldnn':d.torch.backends.mkldnn.enabled}


def run(args):
    contract.require(__debug__,'optimized Python is forbidden: integrity assertions required')
    contract.require(os.environ.get('CANARIA_LATENCY_SUPERVISOR')==str(os.getppid()),'sole entry point is supervise.py')
    out=pathlib.Path(args.out);registry=pathlib.Path(args.registry)
    contract.require(registry.is_dir() and not registry.is_symlink(),'claimed private registry required')
    writer=EvidenceWriter([out,registry]);census=full_census();raw=[];checks=[];failure=None;trace=None;out_created=False
    def timeout(*_):raise TimeoutError('600 second total cap or separately approved deadline')
    try:
        release=contract.read_json(args.release)
        source_hashes=contract.package_hashes(HERE)
        deadline=contract.validate_release(release,args.manifest_sha256,source_hashes,contract.git_identity(ROOT),time.time())
        request=contract.read_json(registry/'REQUEST.json')
        contract.require(request['release_sha256']==contract.sha(args.release) and request['input_manifest_sha256']==args.manifest_sha256,'claimed request binding')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(min(600,max(1,int(deadline-time.time()))))
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        out.mkdir(exist_ok=False);out_created=True
        writer.json(registry/'START.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'STARTED'})
        d=configure_runtime();np=d.np;torch=d.torch
        writer.json(out/'PRIVATE_ENV_BEFORE.json',environment(d))
        writer.json(out/'RELEASE.json',release)
        inputs=contract.load_inputs(args.manifest,args.manifest_sha256,args.input_root)
        allmods={}
        for item in inputs['models']:
            s=item['selection']
            mods=d.load_models(s,inputs['artifacts'],item['expected'],inputs['storage_root'],item['sources'])
            checks.append(d.validation(s,mods,item['validation'],inputs['csv'],item['split'],item['references']))
            allmods[s['id']]=mods
        writer.json(out/'CORRECTNESS.json',checks)
        X=np.random.default_rng(20261027).integers(0,17,size=(128,64)).astype(np.float32)/16
        buf=io.BytesIO();np.save(buf,X,allow_pickle=False);writer.write(out/'INPUT.npy',buf.getvalue())
        with torch.no_grad():
            for seed in range(16):
                mods=allmods[seed]
                for batch in ((1,128) if seed%2==0 else (128,1)):
                    x=torch.from_numpy(X[:batch]);current=[c for c in census if c['id']==seed and c['batch']==batch]
                    for c in current:c['status']='STARTED'
                    for arm in ARMS:
                        for _ in range(WARMUP):mods[arm](x)
                    for rnd in range(ROUNDS):
                        for position,arm in enumerate(ordered(seed,rnd)):
                            elapsed=timed_block(mods[arm],x);assert elapsed>0
                            row={'id':seed,'batch':batch,'round':rnd,'position':position,'arm':arm,'iterations':ITERS,'total_ns':elapsed,'ns_per_forward':elapsed/ITERS}
                            writer.write(out/'RAW.jsonl',json.dumps(row)+'\n',append=True);raw.append(row)
                    for c in current:c['status']='COMPLETE'
        validate_raw(raw)
        assert contract.package_hashes(HERE)==source_hashes
        contract.load_inputs(args.manifest,args.manifest_sha256,args.input_root)
        writer.json(out/'PRIVATE_ENV_AFTER.json',environment(d))
    except BaseException as error:
        failure=type(error).__name__;trace=traceback.format_exc();failure_census(census)
    finally:
        signal.alarm(0)
        summary=summarize(raw) if failure is None else {'status':'FAILED','error_type':failure,'paired':[]}
        writer.json(registry/'CENSUS.json',census,final=True)
        writer.json(registry/'EXIT.json',{'status':summary['status'],'error_type':failure,'raw_rounds':len(raw)},final=True)
        if out_created:
            writer.json(out/'CENSUS.json',census,final=True);writer.json(out/'SUMMARY.json',summary,final=True)
            if trace:writer.write(out/'PRIVATE_FAILURE_TRACEBACK.txt',trace,final=True)
        elif trace:writer.write(registry/'PRIVATE_FAILURE_TRACEBACK.txt',trace,final=True)
    if failure:raise RuntimeError('Private benchmark attempt failed; no retry')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','registry','release','manifest','manifest-sha256','input-root'):p.add_argument('--'+name,required=True)
    run(p.parse_args(argv))


if __name__=='__main__':main()
