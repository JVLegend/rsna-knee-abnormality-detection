#!/usr/bin/env python3
"""Cria manifesto dos 58 estudos oficiais com uma série preferencial por plano."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMN = "StudyInstanceUID"
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
PLANES = ("Sagittal", "Coronal", "Axial")


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def build_manifest(data_dir: Path) -> dict[str, object]:
    train = pd.read_csv(data_dir / "train.csv")
    series = pd.read_csv(data_dir / "train_series.csv")
    official = train[train[TARGETS].notna().any(axis=1)].copy()
    official[KEY_COLUMN] = official[KEY_COLUMN].astype(str)
    series[KEY_COLUMN] = series[KEY_COLUMN].astype(str)
    series = series[series[KEY_COLUMN].isin(official[KEY_COLUMN])].copy()
    series["_plane"] = series["Anatomical_Plane"].fillna("").astype(str)
    series["_fluid"] = pd.to_numeric(series.get("Fluid_Sensitive", 0), errors="coerce").fillna(0)
    series["_fat"] = pd.to_numeric(series.get("Fat_Suppression", 0), errors="coerce").fillna(0)

    entries: list[dict[str, object]] = []
    for study_uid in official[KEY_COLUMN].tolist():
        candidates = series[series[KEY_COLUMN].eq(study_uid)]
        for plane in PLANES:
            preferred = candidates[candidates["_plane"].eq(plane)].sort_values(
                ["_fluid", "_fat", "SeriesInstanceUID"],
                ascending=[False, False, True],
            )
            if preferred.empty:
                raise ValueError(f"Estudo {study_uid} sem série {plane} no train_series.csv")
            row = preferred.iloc[0]
            entries.append(
                {
                    "study_uid": study_uid,
                    "series_uid": str(row["SeriesInstanceUID"]),
                    "anatomical_plane": plane,
                    "fluid_sensitive": int(row["_fluid"]),
                    "fat_suppression": int(row["_fat"]),
                    "selection_policy": "official labels; best Fluid_Sensitive/Fat_Suppression per plane",
                }
            )
    return {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "selection_policy": {
            "purpose": "complete official gold holdout with one preferred series per plane",
            "study_count": int(official[KEY_COLUMN].nunique()),
            "series_per_study": 3,
            "planes": list(PLANES),
            "priority": "Fluid_Sensitive, Fat_Suppression, SeriesInstanceUID",
        },
        "studies": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dicom_gold_three_plane_manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(_resolve(args.data_dir))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"studies={manifest['selection_policy']['study_count']} series={len(manifest['studies'])}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
