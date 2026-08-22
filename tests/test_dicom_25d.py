from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from scripts.build_dicom_25d_features import _physical_sort_key, normalize_slice, sample_indices


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


def test_physical_sort_key_uses_ipp_projection_over_instance_number() -> None:
    orientation = [1, 0, 0, 0, 1, 0]
    entries = [
        (Path("slice_2.dcm"), SimpleNamespace(ImagePositionPatient=[0, 0, 2], ImageOrientationPatient=orientation, InstanceNumber=1)),
        (Path("slice_0.dcm"), SimpleNamespace(ImagePositionPatient=[0, 0, 0], ImageOrientationPatient=orientation, InstanceNumber=3)),
        (Path("slice_1.dcm"), SimpleNamespace(ImagePositionPatient=[0, 0, 1], ImageOrientationPatient=orientation, InstanceNumber=2)),
    ]

    ordered = sorted(entries, key=_physical_sort_key)

    assert [path.name for path, _ in ordered] == ["slice_0.dcm", "slice_1.dcm", "slice_2.dcm"]
