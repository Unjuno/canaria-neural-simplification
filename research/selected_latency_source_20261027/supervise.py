#!/usr/bin/env python3
"""Sole entry point for a new, explicit private technical attempt. Never a historical resume.
Requires separate execution authority; packaging/source publication grants none.
"""
import argparse,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,time
import latency_contract as contract
from timing_core import EvidenceWriter,full_census
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def bounded_wait(proc,seconds,stop_at=None):
    def allowance(value):return value if stop_at is None else max(0,min(value,stop_at-time.monotonic()))
    try:return proc.wait(timeout=allowance(seconds)),False
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:return proc.wait(timeout=allowance(5)),True
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            return proc.wait(timeout=allowance(5)),True


def terminal(registry,out,rc,timed_out,elapsed,launch_error=None):
    metadata_error=None
    try:
        census=contract.read_json(registry/'CENSUS.json')
        identities={(x['id'],x['batch'],x['arm']) for x in census}
        expected={(x['id'],x['batch'],x['arm']) for x in full_census()}
        contract.require(len(census)==96 and identities==expected,'exact terminal census')
        contract.require(all(type(x['id']) is int and type(x['batch']) is int and x['status'] in ('NOT_STARTED','STARTED','COMPLETE','FAILED') for x in census),'terminal status/type')
        worker=contract.read_json(registry/'EXIT.json')
        contract.require(type(worker) is dict,'worker exit object')
    except (OSError,ValueError,TypeError,KeyError) as exc:
        metadata_error=type(exc).__name__;census=full_census();worker={}
    if timed_out:
        for cell in census:
            if cell['status']!='COMPLETE':cell['status']='ABORTED_BY_SUPERVISOR'
    complete=(rc==0 and not timed_out and launch_error is None and metadata_error is None
              and worker.get('status')=='COMPLETE' and worker.get('raw_rounds')==2880
              and all(c['status']=='COMPLETE' for c in census))
    event={'exit_code':rc,'timed_out':timed_out,'elapsed_seconds':elapsed,
           'status':'COMPLETE' if complete else 'FAILED','census':census,
           'launch_error_type':launch_error,'metadata_error_type':metadata_error}
    data=(json.dumps(event,indent=2,sort_keys=True)+'\n').encode()
    contract.require(len(data)<64*1024,'supervisor rescue reserve')
    writer=EvidenceWriter([out,registry])
    contract.require(writer.used()+len(data)<=128*1024**2,'aggregate evidence cap prevents rescue write')
    with (registry/'SUPERVISOR_EXIT.json').open('xb') as stream:stream.write(data)
    return event


def main(argv=None):
    start=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','release','manifest','manifest-sha256','input-root','control-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--input-mode',choices=('historical-v1','fresh-v1'),default='historical-v1')
    args=p.parse_args(argv)
    active_contract=contract
    if args.input_mode=='fresh-v1':
        import fresh_latency_contract as active_contract
    contract.require(__debug__,'optimized Python forbidden')
    release=active_contract.read_json(args.release)
    deadline=active_contract.validate_release(release,args.manifest_sha256,active_contract.package_hashes(HERE),contract.git_identity(ROOT),time.time())
    out,control,registry=contract.paths(args.out,args.input_root,args.control_root,ROOT,release['attempt_id'])
    contract.require(hashlib.sha256(str(control).encode()).hexdigest()==release['control_root_sha256'],'authorizer-selected control root binding')
    stop_at=min(start+600,time.monotonic()+deadline-time.time())
    locks=[];claimed=False;rc=None;timed_out=False;error=None;proc=None
    try:
        for name in ('supervisor.lock','benchmark.lock'):
            target=control/name
            contract.require(not target.is_symlink(),'lease symlink forbidden')
            lock=target.open('a+');locks.append(lock)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Recheck after locks. Never erase, reset, resume or automatically retry.
        out,control,registry=contract.paths(args.out,args.input_root,args.control_root,ROOT,release['attempt_id'])
        registry.mkdir(exist_ok=False);claimed=True
        writer=EvidenceWriter([out,registry])
        writer.json(registry/'REQUEST.json',{'schema':active_contract.RELEASE_SCHEMA,'attempt_id':release['attempt_id'],
            'release_sha256':contract.sha(args.release),'input_manifest_sha256':args.manifest_sha256})
        seconds=min(590-(time.monotonic()-start),deadline-time.time()-10)
        contract.require(seconds>0,'insufficient supervisor window')
        command=[sys.executable,str(HERE/'benchmark.py')]
        values={'out':str(out),'registry':str(registry),'release':str(pathlib.Path(args.release).resolve()),
                'manifest':str(pathlib.Path(args.manifest).resolve()),'manifest-sha256':args.manifest_sha256,
                'input-root':str(pathlib.Path(args.input_root).resolve())}
        for name,value in values.items():command.extend(['--'+name,value])
        if args.input_mode=='fresh-v1':command.extend(['--input-mode','fresh-v1'])
        proc=subprocess.Popen(command,start_new_session=True,env={**os.environ,'CANARIA_LATENCY_SUPERVISOR':str(os.getpid())})
        seconds=min(seconds,max(0,stop_at-time.monotonic()-10))
        rc,timed_out=bounded_wait(proc,seconds,stop_at=stop_at)
    except BaseException as exc:
        error=type(exc).__name__
        if proc is not None:
            # Bounded terminate/reap before releasing; an unreaped child is explicit failure.
            try:
                if proc.poll() is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:rc=proc.wait(timeout=max(0,min(5,stop_at-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        rc=proc.wait(timeout=max(0,min(5,stop_at-time.monotonic())))
            except BaseException as cleanup:
                error=error+';unreaped:'+type(cleanup).__name__
                # Explicit unrecoverable supervisor failure; terminal is never COMPLETE.
        raise
    finally:
        try:
            if claimed:event=terminal(registry,out,rc,timed_out,time.monotonic()-start,error)
        finally:
            for lock in reversed(locks):
                fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if event['status']!='COMPLETE':raise RuntimeError('Benchmark failed; permanent attempt retained; no automatic retry')


if __name__=='__main__':main()
