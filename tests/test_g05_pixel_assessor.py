"""#RSNA #Kaggle #Pesquisa — independent synthetic receipt replay, not MRI quality."""
import copy
import hashlib

import numpy as np
import pytest

from scripts.assess_g05_pixels import verify_record, candidate_keys
from scripts.g05_duplicate_audit import descriptor


def fixture():
    images = {s: np.random.default_rng(s).integers(0, 256, (3, s, s), dtype=np.uint8) for s in (224, 336)}
    row = {'StudyInstanceUID': 'study', 'split': 'train'}
    series = {'series_uid': 'series', 'plane': 'Sagittal'}
    selection = {'indices': [1, 2, 3], 'files': ['1.dcm', '2.dcm', '3.dcm'],
                 'positions_mm': [1., 2., 3.], 'gaps_mm': [1., 1.]}
    record = {'study': 'study', 'split': 'train', 'series': 'series', 'plane': 'Sagittal',
              'selected': selection, 'pixel_sha256': {str(s): hashlib.sha256(v.tobytes()).hexdigest() for s, v in images.items()},
              'baseline224_reference_available': True, 'raw_slice_sha256': ['a'*64]*3,
              'normalized_slice_sha256': ['b'*64]*3, 'phash': descriptor(images[224])[1]}
    reference = {'selected': copy.deepcopy(selection), 'pixel_sha256': record['pixel_sha256']['224']}
    return images, record, row, series, reference


def test_independent_replay_validates_arrays_not_boolean():
    args = fixture()
    assert verify_record(*args).shape == (1024,)
    args[0][336][0, 0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match='pixel hash'): verify_record(*args)


@pytest.mark.parametrize('mutation', ['hash', 'phash', 'split', 'baseline', 'selection', 'raw_hash'])
def test_independent_replay_fails_closed(mutation):
    images, r, row, series, ref = fixture()
    if mutation == 'hash': r['pixel_sha256']['224'] = 'f'*64
    if mutation == 'phash': r['phash'] ^= 1
    if mutation == 'split': r['split'] = 'confirmation'
    if mutation == 'baseline': ref['pixel_sha256'] = 'c'*64
    if mutation == 'selection': r['selected']['files'].pop()
    if mutation == 'raw_hash': r['raw_slice_sha256'][1] = 'nohash'
    with pytest.raises(ValueError): verify_record(images, r, row, series, ref)


def test_duplicate_identity_comparison_ignores_only_float_cosine():
    value = {'exact_cross_split_candidates': [{'first': 'a', 'second': 'b'}],
             'near_cross_split_candidates': [{'first': 'a', 'second': 'c', 'cosine': .996, 'hamming': 2}]}
    changed = copy.deepcopy(value); changed['near_cross_split_candidates'][0]['cosine'] += 1e-8
    assert candidate_keys(value) == candidate_keys(changed)
    changed['near_cross_split_candidates'][0]['second'] = 'd'
    assert candidate_keys(value) != candidate_keys(changed)
