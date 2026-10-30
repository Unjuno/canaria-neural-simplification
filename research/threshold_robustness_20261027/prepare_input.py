"""Project public saved scalars after verifying the recorded public source hash."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('provenance', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    provenance = json.loads(a.provenance.read_text())
    raw = a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance['public_source']['sha256']:
        raise ValueError('Public input hash differs from recorded source')
    source = json.loads(raw)
    fields = ('arm', 'budget', 'nmse', 'correct', 'n', 'quality_pass', 'reused')
    rows = [dict(id=r['id'], status=r['status'], teacher_correct=r['teacher_correct'],
                 cells=[{k: c[k] for k in fields} for c in r['cells']]) for r in source['all_rows']]
    out = dict(accepted_source_commit=provenance['original_source']['commit'],
               accepted_source_summary_sha256=provenance['original_source']['sha256'],
               accepted_original={k: source[k] for k in ('geometric_ratio', 'minima', 'ratio_ci95')},
               rows=rows)
    result = json.dumps(out, indent=2, allow_nan=False) + '\n'
    if hashlib.sha256(result.encode()).hexdigest() != provenance['portable_input_sha256']:
        raise ValueError('Projected scalar input differs from recorded input')
    with a.output.open('x') as stream:
        stream.write(result)


if __name__ == '__main__':
    main()
