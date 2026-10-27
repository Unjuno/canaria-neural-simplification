# Public export transformations

This folder is a bounded public analysis package, not a raw workspace upload.
Original artifacts are preserved separately and were not overwritten.

1. `SCALAR_INPUT.json`, `analyze_thresholds.py`, `test_analyze_thresholds.py`, and
   all five files under `results/` are byte-identical to the original analysis
   artifact. No values, thresholds, selections or censoring flags were edited.
2. `README.md` is rewritten around relative, caller-owned reproduction paths.
   `PROTOCOL.md` removes administrative coordination wording and labels itself
   a public scientific extract. `REPORT.md` replaces administrative provenance
   context with usable public-source references and the final review status.
3. `prepare_input.py` is a public-source variant, not byte-identical to the
   original preparer. It reads `SOURCE_PROVENANCE.json`, verifies the existing
   public summary SHA-256, projects only named scientific scalar fields,
   preserves the historical original-source metadata, checks the resulting
   portable input hash, and refuses to overwrite an existing output file.
4. Original source and public source hashes differ because an earlier export
   removed nonscientific metadata. All 224 projected cells and accepted reference
   minima, ratio and historical interval agree exactly. Both hashes are supplied;
   they are not represented as hashes of the same file.
5. Administrative records, execution logs, Git history/cache, model weights and
   datasets are omitted. Original local commit identifiers are provenance labels,
   not claims of accessible public commits or externally certified chronology.
6. `SOURCE_PROVENANCE.json` and this transformation record are new export files.
   `FILES_SHA256.json` is regenerated from the public files, not copied from the
   original artifact; it excludes itself to avoid a self-hash dependency.

Scalar recomputation and source/hash audits do not constitute independent model
reproduction, fresh validation, new examples, or evidence beyond the same saved
cohort. See `REPORT.md` for the scientific limits.
