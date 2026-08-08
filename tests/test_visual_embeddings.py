from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_visual_embeddings import _fold_count, evaluate_embeddings
from rsna_knee_baseline.constants import TARGET_COLUMNS


def test_fold_count_rejects_single_class_fold() -> None:
    with pytest.raises(ValueError, match="Rótulo insuficiente"):
        _fold_count(np.array([0, 0, 0, 1]), requested=5)


def test_visual_evaluator_returns_macro_auc() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(12, 4)).astype(np.float32)
    frame = {target: np.array([0, 1] * 6, dtype=float) for target in TARGET_COLUMNS}
    result = evaluate_embeddings(matrix, pd.DataFrame(frame), folds=2, seed=42)
    assert result["embedding_shape"] == [12, 4]
    assert np.isfinite(result["macro_auc"])
    assert len(result["targets"]) == 12
