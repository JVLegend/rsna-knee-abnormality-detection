from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "rsna_knee_v3_external_labels.py"
SPEC = importlib.util.spec_from_file_location("rsna_knee_v3_external_labels", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_external_weak_labels_keep_gold_and_filter_low_confidence() -> None:
    targets = MODULE.TARGET_COLUMNS
    train = pd.DataFrame({MODULE.KEY_COLUMN: ["a", "b", "c"]})
    for target in targets:
        train[target] = np.nan
    train.loc[0, "ACL"] = 1.0

    teacher = pd.DataFrame({"ACL": [0.1, 0.9, 0.8], "ACL__confidence": [0.9, 0.9, 0.8]})
    for target in targets:
        if target != "ACL":
            teacher[target] = 0.5
            teacher[f"{target}__confidence"] = 0.0

    result = MODULE._weak_target_arrays(train, teacher, threshold=0.85, sample_weight=0.1)
    labels, weights, included = result["ACL"]

    assert included.tolist() == [True, True, False]
    assert labels.tolist() == [1.0, 1.0, 1.0]
    assert np.allclose(weights, [1.0, 0.09, 0.0])
