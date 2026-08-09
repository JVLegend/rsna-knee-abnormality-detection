from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_fusion_baseline import _parse_alphas


def test_parse_alphas() -> None:
    assert _parse_alphas("0,0.5,1") == (0.0, 0.5, 1.0)
