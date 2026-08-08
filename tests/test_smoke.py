from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS
from rsna_knee_baseline.model import KneeReportBaseline


def _frame(rows: int) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                KEY_COLUMN: f"study-{index}",
                "PatientSex": "Male" if index % 2 else "Female",
                "Report": "ACL tear and joint effusion" if index % 2 else "Normal knee MRI",
                **{target: float(index % 2) for target in TARGET_COLUMNS},
            }
        )
    return pd.DataFrame(records)


def test_baseline_produces_valid_submission() -> None:
    train = _frame(8)
    test = _frame(3).drop(columns=TARGET_COLUMNS)
    series = pd.DataFrame(
        {
            KEY_COLUMN: [f"study-{index}" for index in range(8)],
            "SeriesInstanceUID": [f"series-{index}" for index in range(8)],
            "Fluid_Sensitive": [index % 2 for index in range(8)],
            "Fat_Suppression": [1 for _ in range(8)],
            "Anatomical_Plane": ["Sagittal", "Coronal"] * 4,
        }
    )
    test_series = series.iloc[:3].copy()
    test_series[KEY_COLUMN] = [f"study-{index}" for index in range(3)]

    submission = KneeReportBaseline().fit(train, series).predict(test, test_series)

    assert list(submission.columns) == [KEY_COLUMN, *TARGET_COLUMNS]
    assert len(submission) == len(test)
    assert submission[TARGET_COLUMNS].notna().all().all()
    assert np.isfinite(submission[TARGET_COLUMNS].to_numpy()).all()
    assert ((submission[TARGET_COLUMNS] >= 0) & (submission[TARGET_COLUMNS] <= 1)).all().all()
