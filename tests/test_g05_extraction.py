"""#RSNA #Kaggle #Pesquisa — synthetic paired pixel/cache/duplicate safeguards."""
import ast
import copy
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from scripts.g05_duplicate_audit import descriptor, screen
import scripts.g05_pair_runtime as runtime
from scripts.freeze_weak_validation import digest
from scripts.prepare_g05_extraction import assemble, geometry_contract
from scripts.resolution_comparison import load_manifest
from scripts.v02_pilot_runtime import normalize_cache_compatible
from scripts.prepare_v04_confirmation import V03_BUILD, literal
from tests.test_g05_headers import write_header


def image():
    return np.random.default_rng(5).integers(0, 256, (3, 224, 224), dtype=np.uint8)


def record(uid, split, im, plane='Sagittal'):
    vector, phash = descriptor(im)
    return {'study': uid, 'series': uid+'series', 'split': split, 'plane': plane,
            'pixel_sha256': {'224': hashlib.sha256(im.tobytes()).hexdigest()}, 'phash': phash}, vector


def test_fixed_duplicate_audit_is_conservative_not_patient_proof():
    im = image(); changed = im.copy(); changed[1, :8, :8] ^= np.uint8(1)
    a, av = record('a', 'train', im); b, bv = record('b', 'development', im)
    c, cv = record('c', 'confirmation', changed)
    result = screen([a, b, c], [av, bv, cv])
    assert result['exact_cross_split_candidates'] and result['near_cross_split_candidates']
    assert not result['screen_passed'] and not result['patient_independence_certified']
    # Identical images within train are not a cross-split violation.
    b['split'] = 'train'
    result = screen([a, b], [av, bv])
    assert result['screen_passed'] and not result['patient_independence_certified']
    b['split'] = 'confirmation'; b['plane'] = 'Axial'
    assert screen([a, b], [av, bv])['exact_cross_split_candidates']


def test_descriptor_invalid_arrays_and_record_aliases_rejected():
    with pytest.raises(ValueError): descriptor(np.zeros((3, 224, 224), np.uint8))
    with pytest.raises(ValueError): descriptor(np.zeros((3, 336, 336), np.uint8))
    r, v = record('a', 'train', image())
    with pytest.raises(ValueError): screen([r, r], [v, v])
    with pytest.raises(ValueError): screen([r], [v*np.nan])


def test_paired_series_decodes_once_and_checks_exact224_parity(tmp_path, monkeypatch):
    import pydicom
    source = V03_BUILD.read_text()
    definition = next(ast.get_source_segment(source, n) for n in ast.parse(source).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'physical_plan')
    scope = {'np': np}; exec(definition, scope)
    monkeypatch.setattr(runtime, 'physical_plan', scope['physical_plan'], raising=False)
    monkeypatch.setattr(runtime, 'normalize_cache_compatible', normalize_cache_compatible, raising=False)
    monkeypatch.setattr(runtime, 'descriptor', descriptor, raising=False)
    helper = {'__name__': 'synthetic_pixel_helper', '__file__': str(tmp_path/'helper.py')}
    exec(literal(source, 'CACHE_SOURCE'), helper)
    folder = tmp_path/'series'; folder.mkdir()
    for i in range(5):
        path = folder/f'{4-i}.dcm'
        write_header(path, sop=f'1.2.3.4.{i+1}')
        ds = pydicom.dcmread(path)
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [0, 0, i*3]
        ds.Rows = 32; ds.Columns = 32; ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.PixelData = ((np.arange(1024).reshape(32, 32)+i) % 256).astype(np.uint8).tobytes()
        ds.save_as(path, enforce_file_format=True)
    calls = []
    decode = helper['_pixel_array']
    helper['_pixel_array'] = lambda ds: (calls.append('decoded') or decode(ds))
    row = {'StudyInstanceUID': '1.2.3', 'split': 'train'}
    series = {'series_uid': '1.2.3.4', 'plane': 'Sagittal'}
    values, record, vector = runtime.paired_series(folder, row, series, helper, None, lambda: None)
    assert len(calls) == 3 and values[224].shape == (3, 224, 224) and values[336].shape == (3, 336, 336)
    expected = {'selected': record['selected'], 'pixel_sha256': record['pixel_sha256']['224']}
    _, verified, _ = runtime.paired_series(folder, row, series, helper, expected, lambda: None)
    assert verified['baseline224_reference_available'] and np.isfinite(vector).all()
    expected['pixel_sha256'] = 'f'*64
    with pytest.raises(ValueError): runtime.paired_series(folder, row, series, helper, expected, lambda: None)


def test_cache_fingerprints_and_identity_checked_before_use(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'sha', digest, raising=False)
    p = tmp_path/'pixels.npz'
    np.savez_compressed(p, ids=['u'], images224=np.zeros((1, 3, 3, 224, 224), np.uint8),
                        images336=np.zeros((1, 3, 3, 336, 336), np.uint8), contract_hash='frozen')
    pin = {'sha256': digest(p), 'ids': ['u']}
    assert set(runtime.load_pixel_shard(p, pin, 'frozen')) == {224, 336}
    with pytest.raises(ValueError): runtime.load_pixel_shard(p, dict(pin, sha256='a'*64), 'frozen')
    with pytest.raises(ValueError): runtime.load_pixel_shard(p, dict(pin, ids=['other']), 'frozen')
    with pytest.raises(ValueError): runtime.load_pixel_shard(p, pin, 'changed')


def synthetic_full_headers(path):
    manifest = load_manifest()
    records = [{'StudyInstanceUID': r['StudyInstanceUID'], 'all_headers_consistent': True,
                'all_official_series_checked': True}
               for split in ['train', 'development', 'confirmation'] for r in manifest['splits'][split]]
    value = {'status': 'COMPLETE_FULL_HEADERS_NOT_PATIENT_SEMANTICS_CERTIFICATION',
             'v05_sha256': manifest_sha(), 'studies': records, 'cross_study_sop_collisions': []}
    path.write_text(json.dumps(value))
    return manifest


def manifest_sha():
    from scripts.resolution_comparison import V05_SHA
    return V05_SHA


def test_packed_pixel_builder_keeps_split_and_no_quality_or_gpu(tmp_path):
    headers = tmp_path/'headers.json'; synthetic_full_headers(headers)
    source, spec = assemble('pixels', headers)
    assert len(source.encode()) < 1_000_000 and len(spec['expected_geometry']) == 3000
    assert {s: sum(r['split'] == s for r in spec['rows']) for s in ['train', 'development', 'confirmation']} == {'train': 1000, 'development': 300, 'confirmation': 300}
    assert all('labels' not in row for row in spec['rows'])
    scope = {'__name__': 'synthetic_build'}
    exec(compile(source, '<g05-synthetic>', 'exec'), scope)
    assert scope['G05_SPEC'] == spec and scope['G05_SPEC']['stage'] == 'pixels'
    assert 'identity_gate(spec' in source  # GPU extraction has an explicit gate
    assert not spec['confirmation_evaluated']


def test_gpu_build_requires_patient_evidence_before_any_execution(tmp_path):
    headers = tmp_path/'headers.json'; synthetic_full_headers(headers)
    identity = tmp_path/'identity.json'; identity.write_text('{}')
    with pytest.raises(ValueError): assemble('features', headers, identity, tmp_path/'absent.json')
    with pytest.raises(ValueError): assemble('features', headers)


def test_geometry_contract_is_exact_frozen_training_not_development():
    rows = geometry_contract(); manifest = load_manifest()
    assert len(rows) == 3000
    assert {r['study'] for r in rows} == {r['StudyInstanceUID'] for r in manifest['splits']['train']}
    assert all(set(r['selected']) == {'indices', 'files', 'positions_mm', 'gaps_mm'} for r in rows)


def test_runtime_identity_failure_precedes_any_cuda_model_or_cache_access(tmp_path, monkeypatch):
    identity = tmp_path/'g05_identity.json'; identity.write_text('{}')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(Dinov2Model=None, Dinov2Config=None))
    monkeypatch.setattr(runtime, 'unique_artifact', lambda *args: identity)
    def closed(*args): raise ValueError('patient semantics missing')
    monkeypatch.setattr(runtime, 'identity_gate', closed, raising=False)
    with pytest.raises(ValueError, match='patient semantics missing'):
        runtime.extract_features(tmp_path, tmp_path, {'identity_sha256': 'not-used', 'split_identity_rows': {}})
    assert not (tmp_path/'g05_features.npz').exists()
