"""#RSNA #Kaggle #Pesquisa — conservative cross-split image screening, not patient proof."""
import hashlib
import numpy as np
from PIL import Image
from scipy.fft import dctn

DUPLICATE_RECIPE = {'thumbnail': 'middle_channel_uint8_PIL_bilinear32_centered_L2',
                    'phash': 'DCT_8x8_DC_excluded_63bits_median',
                    'maximum_hamming': 4, 'minimum_centered_cosine': .995,
                    'exact': 'entire_selected_3channel224_sha256',
                    'policy': 'any_cross_split_candidate_blocks_until_documented_review',
                    'proves_patient_independence': False}


def descriptor(image):
    if image.shape != (3, 224, 224) or image.dtype != np.uint8:
        raise ValueError('Expected baseline224 three-channel image')
    thumb = np.asarray(Image.fromarray(image[1]).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    centered = thumb-thumb.mean(); norm = np.linalg.norm(centered)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Constant/nonfinite duplicate thumbnail')
    vector = (centered/norm).reshape(-1)
    coefficients = dctn(thumb, norm='ortho')[:8, :8].reshape(-1)[1:]
    bits = coefficients > np.median(coefficients)
    phash = sum(int(bit) << i for i, bit in enumerate(bits))
    return vector, phash


def screen(records, vectors):
    """No label/prediction use; candidates cannot be automatically excluded/reassigned."""
    v = np.asarray(vectors, dtype=np.float32)
    if (v.shape != (len(records), 1024) or not np.isfinite(v).all() or
            not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)):
        raise ValueError('Invalid normalized duplicate descriptors')
    if len({(r['study'], r['series']) for r in records}) != len(records):
        raise ValueError('Duplicate record identity')
    exact, near, seen = [], [], {}
    for i, record in enumerate(records):
        sha = record['pixel_sha256']['224']
        for j in seen.get(sha, []):
            previous = records[j]
            if previous['split'] != record['split'] and previous['study'] != record['study']:
                exact.append({'first': previous['study'], 'second': record['study'],
                              'first_series': previous['series'], 'second_series': record['series'], 'pixel_sha256': sha})
        seen.setdefault(sha, []).append(i)
    # Compare same-plane pairs in blocks; exact hashes also checked across planes.
    for plane in sorted({r['plane'] for r in records}):
        indices = [i for i, r in enumerate(records) if r['plane'] == plane]
        vv = v[indices]
        for begin in range(0, len(indices), 128):
            similarity = vv[begin:begin+128] @ vv.T
            for a, b in np.argwhere(similarity >= DUPLICATE_RECIPE['minimum_centered_cosine']):
                left, right = indices[begin+int(a)], indices[int(b)]
                if left >= right or records[left]['split'] == records[right]['split']:
                    continue
                distance = (int(records[left]['phash']) ^ int(records[right]['phash'])).bit_count()
                if distance <= DUPLICATE_RECIPE['maximum_hamming']:
                    near.append({'first': records[left]['study'], 'second': records[right]['study'],
                                 'plane': plane, 'hamming': distance, 'cosine': float(similarity[a, b])})
    return {'recipe': DUPLICATE_RECIPE, 'records': len(records),
            'exact_cross_split_candidates': exact, 'near_cross_split_candidates': near,
            'screen_completed': True, 'screen_passed': not exact and not near,
            'patient_independence_certified': False,
            'limitation': 'No candidates under this fixed image screen is not proof of distinct patients.'}
