# Repository tools

This directory contains verification utilities for repository structure, evidence status, reproducibility, and pre-announcement readiness. These tools are not themselves scientific evidence; they check that checked-in evidence and communication surfaces remain internally consistent.

## `audit_repo.py`

The scientific/evidence integrity audit checks items such as:

- required review/correction artifacts exist;
- invalidated Phase 2 evidence remains marked invalid and is not silently promoted;
- portable/reproduction runners do not depend on private `/mnt/data` paths;
- selected evidence-status and result invariants remain consistent.

Run:

```bash
python tools/audit_repo.py
```

## `audit_readiness.py`

The pre-announcement state audit checks a different class of invariant:

- `ANNOUNCEMENT_READINESS.md` and the pinned residual-MLP reproduction path exist;
- README/STATUS retain explicit announcement-state wording while Issue #13 is open;
- stale “publication complete / reviewed public baseline” state wording does not reappear in selected current surfaces;
- pinned headline reproduction dependencies remain the recorded versions unless deliberately revised;
- the historical v0.2.0 checklist remains clearly separated from the current readiness gate;
- if a pinned cohort-reproduction report is checked in, it must be a PASS and remain labeled as reproduction of existing confirmatory evidence.

Run:

```bash
python tools/audit_readiness.py
```

Both audits run in `.github/workflows/ci.yml` together with reusable-code tests and a representative core-runner smoke test.

## Adding an evidence-specific audit

For a new confirmatory phase, prefer a small deterministic audit that reconstructs the declared endpoint/statistics from immutable stored result files rather than trusting a hand-written summary alone.

Such an audit should:

1. read the locked protocol and stored result artifacts;
2. recompute decision-rule inputs and aggregate statistics where feasible;
3. fail loudly on missing seeds, changed accounting, or inconsistent summaries;
4. remain separate from the evidence-producing run itself;
5. not modify result artifacts.

One-shot workflow wrappers used to invoke an experiment, reproduction, audit, or maintenance action need not remain on the active branch after completion; the reusable audit script itself may remain when useful.