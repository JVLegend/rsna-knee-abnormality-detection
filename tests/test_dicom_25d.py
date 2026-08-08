from __future__ import annotations

import numpy as np
import pytest

from scripts.build_dicom_25d_features import normalize_slice, sample_indices


def test_sample_indices_are_deterministic_and_repeat_for_short_series() -> None:
    assert sample_indices(10) == (2, 4, 7)
    assert sample_indices(1) == (0, 0, 0)


def test_normalize_slice_clips_outliers_and_returns_float32() -> None:
    pixels = np.arange(100, dtype=np.float32).reshape(10, 10)
    pixels[0, 0] = -1000
    pixels[-1, -1] = 1000

    normalized, low, high = normalize_slice(pixels)

    assert normalized.dtype == np.float32
    assert normalized.shape == pixels.shape
    assert low < high
    assert np.isclose(float(normalized.min()), 0.0)
    assert np.isclose(float(normalized.max()), 1.0)


def test_sample_indices_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        sample_indices(0)
    with pytest.raises(ValueError):
        sample_indices(3, (-0.1, 0.5, 1.0))
