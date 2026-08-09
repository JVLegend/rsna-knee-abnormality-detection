from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_v2_multiview import _resolve_device, _series_index, _weak_target_arrays
from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS


def test_weak_targets_keep_uncertain_rows_out_of_fit() -> None:
    train = pd.DataFrame(
        {
            KEY_COLUMN: ["a", "b", "c", "d"],
            "Report": ["ACL tear", "ACL intact", "ACL tear", "Normal knee MRI"],
        }
    )
    for target in TARGET_COLUMNS:
        train[target] = [0.0, 1.0, np.nan, np.nan]
    teacher = pd.DataFrame({KEY_COLUMN: train[KEY_COLUMN]})
    for target in TARGET_COLUMNS:
        teacher[target] = [0.1, 0.9, 0.95, 0.1]

    result = _weak_target_arrays(train, teacher, threshold=0.85, sample_weight=0.1)

    labels, weights, included = result["ACL"]
    assert included.tolist() == [True, True, True, False]
    assert labels[:3].tolist() == [0.0, 1.0, 1.0]
    assert weights[:2].tolist() == [1.0, 1.0]
    assert weights[2] > 0
    assert weights[3] == 0


def test_series_index_groups_by_study_without_losing_series() -> None:
    series = pd.DataFrame(
        [
            {KEY_COLUMN: "a", "SeriesInstanceUID": "a-1"},
            {KEY_COLUMN: "a", "SeriesInstanceUID": "a-2"},
            {KEY_COLUMN: "b", "SeriesInstanceUID": "b-1"},
        ]
    )

    indexed = _series_index(series)

    assert set(indexed) == {"a", "b"}
    assert indexed["a"]["SeriesInstanceUID"].tolist() == ["a-1", "a-2"]


def test_device_auto_falls_back_from_legacy_cuda(monkeypatch) -> None:
    import rsna_knee_v2_multiview as module

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda: (6, 0))

    assert _resolve_device("auto") == "cpu"
