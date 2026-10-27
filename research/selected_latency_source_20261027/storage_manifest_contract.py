"""New portable selection handoff contract; no model reads or numerical imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA = 'canaria.selected-whole-storage.v1'
BUDGETS = (512, 1024, 1536, 2048, 3072, 4096, 6144)
SHA = re.compile(r'[0-9a-f]{64}\Z')
RELATIVE_NPZ = re.compile(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*\.npz\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_keys(obj, keys, label):
    require(type(obj) is dict and set(obj) == set(keys), label + ': fields')


def validate_digest(value):
    require(type(value) is str and SHA.fullmatch(value) is not None, 'invalid SHA256')
    return value


def relative_npz(value):
    require(type(value) is str and RELATIVE_NPZ.fullmatch(value) is not None, 'invalid relative NPZ path')
    require(all(part not in ('.', '..') for part in value.split('/')), 'path traversal')
    require(str(PurePosixPath(value)) == value, 'noncanonical path')
    return value


def validate_manifest(manifest):
    exact_keys(manifest, ('schema', 'cohort_size', 'selection_origin_sha256', 'models'), 'manifest')
    require(manifest['schema'] == SCHEMA, 'manifest schema')
    require(type(manifest['cohort_size']) is int and manifest['cohort_size'] == 16, 'cohort size')
    validate_digest(manifest['selection_origin_sha256'])
    rows = manifest['models']
    require(type(rows) is list and len(rows) == 16, 'sixteen teachers required')
    ids, paths = [], []
    for row in rows:
        exact_keys(row, ('id', 'teacher', 'arms'), 'cohort row')
        require(type(row['id']) is int and 0 <= row['id'] < 16, 'teacher id')
        ids.append(row['id'])
        exact_keys(row['teacher'], ('path', 'file_sha256'), 'teacher source')
        exact_keys(row['arms'], ('C', 'J'), 'arm census')
        for arm, source in [('teacher', row['teacher']), *row['arms'].items()]:
            fields = ('path', 'file_sha256') if arm == 'teacher' else ('path', 'file_sha256', 'state_sha256', 'budget')
            exact_keys(source, fields, 'source')
            paths.append(relative_npz(source['path']))
            validate_digest(source['file_sha256'])
            if arm != 'teacher':
                validate_digest(source['state_sha256'])
                require(type(source['budget']) is int and source['budget'] in BUDGETS, 'nonadmitted budget')
                require(source['budget'] % (128 if arm == 'C' else 256) == 0, 'budget width')
    require(set(ids) == set(range(16)) and len(set(ids)) == 16, 'teacher id census')
    require(len(set(paths)) == 48, 'source paths must be distinct')
    return manifest


def no_duplicate_fields(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def read_manifest(path, expected_sha256):
    validate_digest(expected_sha256)
    raw = Path(path).read_bytes()
    require(len(raw) <= 1024 * 1024, 'manifest byte ceiling')
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'manifest SHA256 mismatch')
    return validate_manifest(json.loads(raw, object_pairs_hook=no_duplicate_fields))


def source_path(root, relative):
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), 'input root must be a directory')
    relative_npz(relative)
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        require(not candidate.is_symlink(), 'source symlinks forbidden')
    resolved = candidate.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), 'source must be a contained regular file')
    return resolved


def check_output(output, input_root, package_root):
    """Check only; never delete, reuse, or create a directory here."""
    output = Path(output).absolute()
    require(not output.is_symlink() and not output.exists(), 'output must be fresh and absent')
    parent = output.parent.resolve(strict=True)
    require(parent.is_dir(), 'output parent must already exist')
    target = parent / output.name
    for protected in (Path(input_root).resolve(strict=True), Path(package_root).resolve(strict=True)):
        require(not target.is_relative_to(protected) and not protected.is_relative_to(target), 'output overlaps protected tree')
    return target
