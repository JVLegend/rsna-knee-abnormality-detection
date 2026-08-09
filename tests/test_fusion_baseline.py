from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee_baseline.constants import TARGET_COLUMNS
from scripts.evaluate_fusion_baseline import _parse_alphas, _parse_target_alphas


def test_parse_alphas() -> None:
    assert _parse_alphas("0,0.5,1") == (0.0, 0.5, 1.0)


def test_parse_target_alphas() -> None:
    value = ",".join(f"{target}=0.4" for target in TARGET_COLUMNS)
    parsed = _parse_target_alphas(value)
    assert parsed["ACL"] == 0.4
    assert len(parsed) == 12
