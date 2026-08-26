# Schemas

This directory contains machine-readable schemas used by research metadata or event records.

Current schemas:

- `run_metadata_schema.json` — metadata structure for experiment/run provenance.
- `blind_map_event_schema.json` — schema for blinded mapping/event records used by the historical research workflow.

## Policy

Schemas describe data structure; they do not by themselves establish scientific validity or current claim status. Always read the associated protocol/result documentation.

When changing a schema:

- preserve compatibility when practical;
- document incompatible field/semantic changes rather than silently reinterpreting historical records;
- do not rewrite historical JSON solely to make it conform to a newer schema unless a documented migration is part of the evidence record.
