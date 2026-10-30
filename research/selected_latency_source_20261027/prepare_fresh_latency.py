#!/usr/bin/env python3
"""Prepare one PRIVATE fresh-input manifest from already-reviewed existing bytes.

No models, references, selections or approval are generated; no payload is copied.
Actual-input use requires separate authorization. Missing input bytes fail closed.
The output file must be absent and outside every input/source tree.
"""
from pathlib import Path
import argparse
import fresh_latency_contract as c


def describe_inputs(input_root, selection_directory, storage_directory, matched_directory,
                    grid_directory, parent_descriptor, upstream_review, csv):
    """Read-only descriptor builder, not approval or proof of upstream science."""
    def desc(rel):
        path = c.contained(input_root, rel)
        return {'path': rel, 'sha256': c.sha(path)}
    for directory in (selection_directory, storage_directory, matched_directory, grid_directory):
        c.contained(input_root, directory, directory=True)
    origin = c.read_json(c.contained(input_root, selection_directory + '/FRESH_SELECTION.json'))
    selected = c.historical.storage.validate_manifest(c.read_json(c.contained(input_root, selection_directory + '/selected_models.json')))
    return {'schema': c.SCHEMA, 'selection_directory': selection_directory,
            'selection': desc(selection_directory + '/selected_models.json'),
            'fresh_selection': desc(selection_directory + '/FRESH_SELECTION.json'),
            'pending_gate': desc(selection_directory + '/REVIEW_REQUIRED.json'),
            'parent_descriptor': desc(parent_descriptor), 'upstream_review': desc(upstream_review),
            'matched_directory': matched_directory, 'grid_directory': grid_directory,
            'grid_summary': desc(grid_directory + '/SUMMARY.json'),
            'storage': {'directory': storage_directory, 'files': {name: desc(storage_directory + '/' + name) for name in c.STORAGE_FILES}},
            'csv': desc(csv),
            'validation': c.validation_descriptors(origin, selected['models'], matched_directory, grid_directory)}


def prepare(input_root, selection_directory, storage_directory, matched_directory,
            grid_directory, parent_descriptor, upstream_review, csv, input_review,
            input_review_sha256, output):
    # No overwriting, source copying, registry, model imports, or automatic retry.
    a, _, _, _, _ = c.dependencies()
    target = a.target_path(output, (input_root, c.HERE.parent))
    manifest = describe_inputs(input_root, selection_directory, storage_directory, matched_directory,
                               grid_directory, parent_descriptor, upstream_review, csv)
    manifest['input_review'] = {'path': c.relative_path(input_review), 'sha256': c.validate_digest(input_review_sha256)}
    c.load_metadata(manifest, input_root)
    # Repeat all metadata/byte pins immediately before the exclusive output write.
    c.load_metadata(manifest, input_root)
    with target.open('xb') as stream:
        stream.write(c.encoded(manifest))
    c.require(c.read_json(target) == manifest, 'prepared manifest readback mismatch')
    return {'status': 'PREPARED_PRIVATE_INPUT_NOT_EXECUTION_AUTHORITY', 'schema': c.SCHEMA,
            'manifest_sha256': c.sha(target), 'historical_attempt': 'CLOSED_NOT_REOPENED'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input-root', 'selection-directory', 'storage-directory', 'matched-directory', 'grid-directory',
                'parent-descriptor', 'upstream-review', 'csv', 'input-review', 'input-review-sha256', 'output'):
        parser.add_argument('--' + key, required=True)
    print(c.encoded(prepare(**vars(parser.parse_args(argv)))).decode(), end='')


if __name__ == '__main__':
    main()
