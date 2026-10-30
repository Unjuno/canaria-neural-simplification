#!/usr/bin/env python3
"""Modified portable storage-only entry point. Import is inert; no bundled inputs.
Selection generation and scientific execution are not authorized by this source.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import zlib
import manifest_contract as contract

HERE = Path(__file__).resolve().parent
MAX_BYTES = 128 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def summarize(rows, census):
    summary = {'cohort':16, 'complete':len(rows),
               'failures':[x for x in census if x['status'] != 'PASS'], 'ratios':{}}
    if rows:
        for key in rows[0]:
            if '_over_' in key:
                values = [r[key] for r in rows]
                summary['ratios'][key] = {
                    'geomean':math.exp(sum(map(math.log,values))/len(values)),
                    'min':min(values), 'max':max(values),
                    'smaller':sum(x<1 for x in values), 'tie':sum(x==1 for x in values),
                    'larger':sum(x>1 for x in values)}
    return summary


def measure(manifest, input_root, output, manifest_sha256):
    """Storage only. Unit tests may call this with artificial arrays, not science.
    CLI enforces historical runtime/bounds before entering this function.
    """
    contract.validate_manifest(manifest)
    contract.validate_digest(manifest_sha256)
    target = contract.check_output(output, input_root, HERE)
    # Atomic claim precedes any array reads; never reuse or erase old output.
    target.mkdir(exist_ok=False)
    import storage_codec as codec
    rows, census, artifacts, sources, mapping = [], [], [], [], []
    dump(target/'ENV.json', {'python':platform.python_version(), 'numpy':codec.np.__version__,
         'zlib_compile':zlib.ZLIB_VERSION, 'zlib_runtime':zlib.ZLIB_RUNTIME_VERSION,
         'manifest_sha256':manifest_sha256,
         'source_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))}})
    input_bytes = 0
    for selection in sorted(manifest['models'], key=lambda row:row['id']):
        sid = selection['id']
        try:
            selected_sources = [('teacher', selection['teacher']), *[(a,selection['arms'][a]) for a in ('C','J')]]
            paths = {}
            for arm, source in selected_sources:
                path = contract.source_path(input_root,source['path'])
                input_bytes += path.stat().st_size
                if input_bytes >= MAX_BYTES:
                    raise ValueError('total input byte ceiling')
                paths[arm] = path
            teacher = codec.load_source(paths['teacher'],selection['teacher']['file_sha256'],codec.shapes_teacher())
            models = {'teacher':teacher}
            sources.append({'id':sid, 'arm':'teacher', 'path':selection['teacher']['path'], 'sha256':selection['teacher']['file_sha256']})
            for arm in ('C','J'):
                source = selection['arms'][arm]
                replacement = codec.load_source(paths[arm], source['file_sha256'],codec.shapes_arm(arm,source['budget']))
                logical = codec.state_hash(replacement)
                if logical != source['state_sha256']:
                    raise ValueError('replacement logical SHA256 mismatch')
                models[arm] = codec.assemble(teacher,replacement,arm,source['budget'])
                sources.append({'id':sid, 'arm':arm, 'path':source['path'], 'sha256':source['file_sha256'], 'logical_sha256':logical})
            record = {'id':sid, 'C_budget':selection['arms']['C']['budget'], 'J_budget':selection['arms']['J']['budget']}
            for arm, arrays in models.items():
                record[arm+'_parameters'] = sum(a.size for a in arrays.values())
                record[arm+'_tensors'] = len(arrays)
                record[arm+'_payload_bytes'] = sum(a.nbytes for a in arrays.values())
                for key,array in sorted(arrays.items()):
                    mapping.append({'id':sid, 'arm':arm, 'destination':key,
                        'source_key':key.removeprefix('replacement.'),
                        'source_arm':arm if key.startswith('replacement.') else 'teacher',
                        'shape':list(array.shape), 'dtype':array.dtype.str,
                        'sha256':hashlib.sha256(array.tobytes()).hexdigest()})
                for label,compression in [('stored',codec.zipfile.ZIP_STORED),('deflated',codec.zipfile.ZIP_DEFLATED)]:
                    path = target/f'seed_{sid}_{arm}_{label}.npz'
                    size = codec.serialize(path,arrays,compression)
                    record[arm+'_'+label+'_bytes'] = size
                    artifacts.append({'file':path.name, 'bytes':size, 'sha256':sha(path)})
                    if sum(a['bytes'] for a in artifacts) >= MAX_BYTES:
                        raise ValueError('total artifact byte ceiling')
            for metric in ('parameters','stored_bytes','deflated_bytes'):
                for a,b in (('C','J'),('C','teacher'),('J','teacher')):
                    record[a+'_over_'+b+'_'+metric] = record[a+'_'+metric]/record[b+'_'+metric]
            rows.append(record)
            census.append({'id':sid,'status':'PASS'})
        except Exception as error:
            # Do not put host paths, tracebacks, or external exception text in outputs.
            census.append({'id':sid, 'status':'FAIL', 'error_type':type(error).__name__})
    summary = summarize(rows,census)
    summary['expected_models'] = 48
    summary['expected_archives'] = 96
    summary['produced_archives'] = len(artifacts)
    summary['artifact_bytes'] = sum(a['bytes'] for a in artifacts)
    summary['status'] = 'COMPLETE' if len(rows)==16 and len(artifacts)==96 else 'INCOMPLETE'
    for name,value in [('ROWS',rows),('CENSUS',census),('ARTIFACTS',artifacts),('SOURCES',sources),('TENSORS',mapping),('SUMMARY',summary)]:
        dump(target/(name+'.json'),value)
    if rows:
        with (target/'ROWS.csv').open('x',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def environment():
    if platform.python_version() != '3.13.5' or zlib.ZLIB_VERSION != '1.3.1' or zlib.ZLIB_RUNTIME_VERSION != '1.3.1':
        raise RuntimeError('historical Python/zlib versions required')
    if sys.byteorder != 'little' or sys.platform != 'linux':
        raise RuntimeError('historical little-endian Linux runtime required')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    import storage_codec as codec
    if codec.np.__version__ != '2.3.5':
        raise RuntimeError('historical NumPy 2.3.5 required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    # Existing-output rejection occurs before manifest reads/numerical imports.
    contract.check_output(args.output,args.input_root,HERE)
    manifest = contract.read_manifest(args.manifest,args.manifest_sha256)
    signal.alarm(600)
    try:
        environment()
        result = measure(manifest,args.input_root,args.output,args.manifest_sha256)
    finally:
        signal.alarm(0)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['status']=='COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
