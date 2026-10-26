"""Calibration-only capability boundary; no CIFAR archive/pickle/train loader.

Production still needs a separately audited calibration-only artifact producer.
It must supply separate fit images, diagnostic images, and diagnostic labels;
original validation/test cannot be decoded even transiently by that producer.
"""
import io
import json
from pathlib import Path

import numpy as np
import torch
from research.cifar10_replacements import core as previous_core
from research.cifar10_replacements import data as previous_data
from . import protocol as p

NAMES = ('fit_images.npy', 'diagnostic_images.npy', 'diagnostic_labels.npy')


def verify_artifact_manifest(manifest, partition_record, expected_sha256):
    if p.digest(manifest) != p.require_hash(expected_sha256):
        raise ValueError('Audited calibration-only artifact manifest mismatch')
    if manifest.get('partition_sha256') != p.verify_partition(partition_record):
        raise ValueError('Calibration-only artifact partition mismatch')
    if manifest.get('scope') != 'calibration-only; original validation/test never decoded':
        raise ValueError('Calibration-only provenance assertion required')
    if set(manifest.get('payload_sha256', {})) != set(NAMES):
        raise ValueError('Only fit/diagnostic image and sealed diagnostic-label payloads allowed')
    for value in manifest['payload_sha256'].values(): p.require_hash(value)
    norm = manifest.get('normalization', {})
    if norm.get('source') != 'teachertrain only': raise ValueError('Frozen original normalization required')
    mean = np.asarray(norm.get('mean'), dtype=np.float64)
    std = np.asarray(norm.get('std'), dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid frozen normalization')
    if norm.get('sha256') != p.sha(np.asarray([mean, std], dtype='<f8').tobytes()):
        raise ValueError('Normalization hash mismatch')
    return mean, std


def read_array(reader, manifest, name, shape, dtype):
    blob = reader(name)
    if p.sha(blob) != manifest['payload_sha256'][name]: raise ValueError('Payload full-file hash mismatch')
    value = np.load(io.BytesIO(blob), allow_pickle=False)
    if type(value) is not np.ndarray or value.shape != shape or value.dtype != dtype:
        raise ValueError('Unexpected calibration-only payload shape/dtype')
    value.flags.writeable = False
    return value


class ImageAccess:
    def __init__(self, ids, images, mean, std):
        self.ids = tuple(ids)
        self._positions = {identifier: index for index, identifier in enumerate(self.ids)}
        self._images = images
        self._mean = torch.tensor(mean, dtype=torch.float32)[None,:,None,None]
        self._std = torch.tensor(std, dtype=torch.float32)[None,:,None,None]

    def image_batch(self, ids):
        ids = tuple(ids)
        if not ids or any(type(i) is not int or i not in self._positions for i in ids):
            raise PermissionError('Only this exact partition capability is accessible')
        array = self._images[[self._positions[i] for i in ids]].copy()
        return (torch.from_numpy(array).float()/255-self._mean)/self._std


class FitAccess(ImageAccess):
    """No label method, no diagnostic method, no original validation/test method."""
    pass


def open_fit_access(reader, manifest, approval, expected_approval_sha256, original_lock,
                    partition_record, source_hashes, *, execution_authorized=False, now=None):
    p.execution_gate(approval, expected_approval_sha256, original_lock, partition_record,
                     source_hashes, execution_authorized=execution_authorized, now=now)
    mean, std = verify_artifact_manifest(manifest, partition_record, approval['artifact_manifest_sha256'])
    if manifest['normalization'] != approval['evaluator_identity']['normalization']:
        raise ValueError('Normalization differs from original approved teacher identity')
    # The only payload callback reachable before release is fit_images.npy.
    images = read_array(reader, manifest, 'fit_images.npy', (4000,3,32,32), np.dtype('uint8'))
    return FitAccess(partition_record['fit_ids'], images, mean, std)


class DiagnosticVault(ImageAccess):
    def __init__(self, ids, images, mean, std, reader, manifest, directory, consumed, cohort_clock):
        super().__init__(ids, images, mean, std)
        self._reader = reader; self._manifest = manifest; self._labels_consumed = False
        self._directory = Path(directory); self._consumed = consumed; self._cohort_clock = cohort_clock

    def reveal_labels(self, predictions, expected_prediction_sha256, *, now=None):
        if self._labels_consumed: raise PermissionError('Diagnostic labels are one-look; retry denied')
        if p.digest(predictions) != p.require_hash(expected_prediction_sha256):
            raise ValueError('All predictions must be durably frozen before labels')
        if predictions.get('status') != 'all-planned-predictions-frozen' or predictions.get('diagnostic_ids') != list(self.ids):
            raise ValueError('Complete prediction/status freeze required')
        p.check_release_clock(self._consumed, self._cohort_clock, now=now)
        if predictions.get('frozen_cohort_sha256') != self._consumed['frozen_cohort_sha256']:
            raise ValueError('Prediction cohort differs from approved release')
        if (self._directory/'predictions_frozen.json').read_bytes() != p.canonical(predictions):
            raise ValueError('Prediction freeze must exist durably and match exactly')
        expected={f'{name}@{step}' for name in p.CELLS for step in p.BUDGETS}
        if set(predictions.get('teachers',{})) != {str(seed) for seed in p.SEEDS}:
            raise ValueError('Full three-teacher prediction census required')
        for teacher in predictions['teachers'].values():
            if set(teacher.get('checkpoints',{})) != expected:
                raise ValueError('Full 24-checkpoint prediction/status census required')
            if any(value.get('status') not in ('predicted','missing','failed') for value in teacher['checkpoints'].values()):
                raise ValueError('Unfrozen prediction status')
        p.write_exclusive_json(self._directory/'diagnostic_labels_opened.json',
                               dict(predictions_sha256=expected_prediction_sha256))
        self._labels_consumed = True  # failure consumes this attempt, too
        labels = read_array(self._reader, self._manifest, 'diagnostic_labels.npy', (1000,), np.dtype('int64'))
        if (labels < 0).any() or (labels >= 10).any(): raise ValueError('Invalid diagnostic class label')
        return labels


def open_diagnostic_vault(reader, manifest, partition_record, approved_manifest_sha256,
                          frozen, original_lock, source_hashes, release_approval,
                          expected_release_sha256, consumed, cohort_clock,
                          *, directory, release_authorized=False, now=None):
    frozen_sha = p.release_gate(release_approval, expected_release_sha256, frozen,
        partition_record, original_lock, source_hashes, release_authorized=release_authorized)
    if consumed.get('frozen_cohort_sha256') != frozen_sha:
        raise ValueError('Durable one-look consumed marker binding mismatch')
    p.check_release_clock(consumed, cohort_clock, now=now)
    mean, std = verify_artifact_manifest(manifest, partition_record, approved_manifest_sha256)
    directory=Path(directory)
    if json.loads((directory/'diagnostic_consumed.json').read_text()) != consumed:
        raise ValueError('Exact durable consumed marker required')
    p.write_exclusive_json(directory/'diagnostic_images_opened.json',dict(frozen_cohort_sha256=frozen_sha))
    images = read_array(reader, manifest, 'diagnostic_images.npy', (1000,3,32,32), np.dtype('uint8'))
    return DiagnosticVault(partition_record['diagnostic_ids'], images, mean, std, reader, manifest, directory, consumed, cohort_clock)


def admit_teacher(export, metrics, checkpoint_blob, identity, original_record, seed):
    """Reuse pinned quality verifier without previous_data.load_approved_data.

    Caller has already passed execution_gate before obtaining these bytes. No
    original-validation images/labels are needed: metrics are frozen metadata.
    """
    if p.sha(checkpoint_blob) != original_record.get('teacher_sha256'):
        raise ValueError('Teacher checkpoint differs from frozen original cohort')
    if identity.get('model_config') != previous_data.MODEL_CONFIG or export.get('spec') != previous_data.APPROVED_SPEC:
        raise ValueError('Unapproved teacher architecture/training recipe')
    required = dict(python='3.13.5', torch='2.10.0+cpu', numpy='2.3.5',
        source_sha256=previous_data.TEACHER_SOURCE_SHA256, threads=4, interop=1,
        deterministic=True, thread_env={key:'4' for key in p.THREAD_ENV})
    if any(export.get('environment', {}).get(k) != v for k,v in required.items()):
        raise ValueError('Unapproved teacher source/runtime')
    if export.get('checkpoint') != 'final_epoch120.ckpt':
        raise ValueError('Only original fixed-final teacher admitted')
    binding = previous_data.APPROVED_MANIFEST['preflight_sha256']['data_manifest.json']
    with torch.random.fork_rng(devices=[]):
        teacher = previous_core.admit_teacher(export, metrics, checkpoint_blob, identity, binding, seed)
    state = torch.load(io.BytesIO(checkpoint_blob.split(b'\n', 2)[2]), map_location='cpu', weights_only=True)
    if p.ids_hash(state['train']) != identity['splits']['teachertrain']['sha256']:
        raise ValueError('Actual checkpoint training membership mismatch')
    for key in ('mean', 'std'):
        wanted = torch.tensor(identity['normalization'][key], dtype=torch.float32)[None,:,None,None]
        if not torch.equal(state[key], wanted): raise ValueError('Checkpoint normalization differs')
    return teacher
