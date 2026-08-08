from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS
from scripts.select_dicom_subset import select_subset


def test_subset_selects_preferred_series_and_both_classes() -> None:
    train = pd.DataFrame(
        [
            {KEY_COLUMN: "study-positive", "PatientSex": "F", "Report": "", **{target: 1 for target in TARGET_COLUMNS}},
            {KEY_COLUMN: "study-negative", "PatientSex": "M", "Report": "", **{target: 0 for target in TARGET_COLUMNS}},
        ]
    )
    series = pd.DataFrame(
        [
            {KEY_COLUMN: study, "SeriesInstanceUID": f"{study}-plain", "Fluid_Sensitive": 0, "Fat_Suppression": 0, "Anatomical_Plane": "Coronal"}
            for study in train[KEY_COLUMN]
        ]
        + [
            {KEY_COLUMN: study, "SeriesInstanceUID": f"{study}-preferred", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"}
            for study in train[KEY_COLUMN]
        ]
    )

    manifest = select_subset(train, series, per_class=1, max_studies=2)

    assert len(manifest["studies"]) == 2
    assert {entry["series_uid"] for entry in manifest["studies"]} == {"study-positive-preferred", "study-negative-preferred"}
    for target in TARGET_COLUMNS:
        assert {entry["labels"][target] for entry in manifest["studies"]} == {0, 1}
