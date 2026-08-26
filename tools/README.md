# Repository tools

This directory contains verification utilities that audit repository structure, evidence status, and reproducibility invariants. These tools are not themselves scientific evidence; they check that the checked-in evidence and public surface remain internally consistent.

## `audit_repo.py`

The repository-wide integrity audit checks items such as:

- required public/review/correction artifacts exist;
- invalidated Phase 2 evidence remains marked invalid and is not silently promoted;
- public runners do not depend on private `/mnt/data` paths;
- required evidence-status invariants remain consistent.

Run from the repository root:

```bash
python tools/audit_repo.py
```

The same audit is part of `.github/workflows/ci.yml` together with reusable-code tests and a minimal public-runner smoke test.

## Adding an evidence-specific audit

For a new confirmatory phase, prefer a small deterministic audit that reconstructs the declared endpoint/statistics from immutable stored result files rather than trusting a hand-written summary alone.

Such an audit should:

1. read the locked protocol and stored result artifacts;
2. recompute decision-rule inputs and aggregate statistics where feasible;
3. fail loudly on missing seeds, changed accounting, or inconsistent summaries;
4. remain separate from the evidence-producing run itself;
5. not modify result artifacts.

One-shot workflow wrappers used to invoke an audit need not remain on the active branch after the audit has completed; the audit script itself may remain when it is useful for future verification.
