#!/usr/bin/env python3
"""Seleciona um lote DICOM pequeno e auditável a partir dos metadados locais."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS
from rsna_knee_baseline.data import find_data_dir, load_competition_tables


PLANE_ORDER = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def _preferred_series(series: pd.DataFrame, study_uid: str) -> pd.Series:
    candidates = series[series[KEY_COLUMN].astype(str).eq(study_uid)].copy()
    if candidates.empty:
        raise ValueError(f"Nenhuma série encontrada para o estudo {study_uid}.")
    candidates["_fluid"] = pd.to_numeric(candidates.get("Fluid_Sensitive", 0), errors="coerce").fillna(0)
    candidates["_fat"] = pd.to_numeric(candidates.get("Fat_Suppression", 0), errors="coerce").fillna(0)
    candidates["_plane_order"] = candidates.get("Anatomical_Plane", "").map(PLANE_ORDER).fillna(9)
    candidates = candidates.sort_values(
        ["_fluid", "_fat", "_plane_order", "SeriesInstanceUID"],
        ascending=[False, False, True, True],
    )
    return candidates.iloc[0]


def select_subset(
    train: pd.DataFrame,
    train_series: pd.DataFrame,
    per_class: int = 1,
    max_studies: int | None = 24,
    excluded_studies: set[str] | None = None,
) -> dict[str, object]:
    if per_class < 1:
        raise ValueError("--per-class precisa ser pelo menos 1.")

    excluded_studies = excluded_studies or set()
    selected: dict[str, dict[str, object]] = {}
    for target in TARGET_COLUMNS:
        labels = pd.to_numeric(train[target], errors="coerce")
        for label in (1, 0):
            candidates = train.loc[labels.eq(label)].sort_values(KEY_COLUMN)
            added = 0
            for _, row in candidates.iterrows():
                study_uid = str(row[KEY_COLUMN])
                if study_uid in excluded_studies:
                    continue
                if study_uid not in selected:
                    selected[study_uid] = {
                        "study_uid": study_uid,
                        "selected_for": [],
                        "labels": {
                            column: int(pd.to_numeric(row[column], errors="coerce"))
                            for column in TARGET_COLUMNS
                            if pd.notna(row[column])
                        },
                    }
                if target not in selected[study_uid]["selected_for"]:
                    selected[study_uid]["selected_for"].append(target)
                added += 1
                if added >= per_class:
                    break

    entries: list[dict[str, object]] = []
    for study_uid, entry in selected.items():
        chosen = _preferred_series(train_series, study_uid)
        entry["series_uid"] = str(chosen["SeriesInstanceUID"])
        entry["fluid_sensitive"] = int(chosen["_fluid"])
        entry["fat_suppression"] = int(chosen["_fat"])
        entry["anatomical_plane"] = str(chosen.get("Anatomical_Plane", ""))
        entries.append(entry)

    if max_studies is not None:
        entries = entries[:max_studies]

    return {
        "selection_policy": {
            "per_class_per_target": per_class,
            "max_studies": max_studies,
            "excluded_studies": len(excluded_studies),
            "series_priority": "Fluid_Sensitive, Fat_Suppression, Sagittal, SeriesInstanceUID",
            "labels_are_local_only": True,
        },
        "studies": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--max-studies", type=int, default=24)
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--output", default="data/processed/dicom_subset_manifest.json")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    tables = load_competition_tables(data_dir)
    if tables["train"].empty or tables["train_series"].empty:
        raise RuntimeError("train.csv e train_series.csv são necessários.")

    excluded_studies: set[str] = set()
    if args.exclude_manifest is not None:
        exclude_path = args.exclude_manifest.expanduser()
        if not exclude_path.is_absolute():
            exclude_path = ROOT / exclude_path
        excluded_manifest = json.loads(exclude_path.read_text(encoding="utf-8"))
        excluded_studies = {str(entry["study_uid"]) for entry in excluded_manifest.get("studies", [])}

    manifest = select_subset(
        tables["train"],
        tables["train_series"],
        args.per_class,
        args.max_studies,
        excluded_studies,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"data_dir={data_dir}")
    print(f"studies={len(manifest['studies'])}")
    print(f"output={output}")
    for entry in manifest["studies"]:
        print(
            f"{entry['study_uid']}: series={entry['series_uid']} "
            f"plane={entry['anatomical_plane']} selected_for={','.join(entry['selected_for'])}"
        )


if __name__ == "__main__":
    main()
