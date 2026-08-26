# Environment records

This directory preserves environment/provenance material for older research snapshots.

Current contents are under `history/v10/` and include:

- `REPRODUCIBILITY_LIMITS.md` — limitations of the recorded historical environment;
- `current_audit_environment.json` — captured audit-environment metadata;
- `requirements_inferred.txt` — inferred historical dependencies.

These files are historical evidence, not the recommended installation path for current reusable code. For current setup use repository-root `pyproject.toml` / `requirements.txt` and the relevant reproduction runner documentation.

Do not edit historical environment records to match a modern machine. If a clean-port reproduction needs different dependencies, record that environment alongside the new reproduction instead.
