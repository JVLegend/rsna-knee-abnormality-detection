#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — inventaria todas as séries dos 58 estudos oficiais.

O CSV da competição só informa dois flags binários de aquisição. Como eles
aparecem juntos no lote oficial, o manifesto nomeia as categorias observadas
como ``FLUID_FS`` e ``NONFLUID`` por plano. Todas as séries são preservadas;
uma etapa posterior pode escolher uma por slot ou agregar réplicas, sem
confundir ausência de uma categoria com uma série vazia.
"""

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


def _slot_name(plane: str, fluid_sensitive: int, fat_suppression: int) -> str:
    if fluid_sensitive == 1 and fat_suppression == 1:
        category = "FLUID_FS"
    elif fluid_sensitive == 0 and fat_suppression == 0:
        category = "NONFLUID"
    else:
        category = "OTHER"
    return f"{plane}_{category}"


def build_manifest(data_dir: Path) -> dict[str, object]:
    train = pd.read_csv(data_dir / "train.csv")
    series = pd.read_csv(data_dir / "train_series.csv")
    official = train[train[TARGETS].notna().all(axis=1)].copy()
    official[KEY_COLUMN] = official[KEY_COLUMN].astype(str)
    series[KEY_COLUMN] = series[KEY_COLUMN].astype(str)
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
    series = series[series[KEY_COLUMN].isin(official[KEY_COLUMN])].copy()
    series["_plane"] = series["Anatomical_Plane"].fillna("").astype(str)
    series["_fluid"] = pd.to_numeric(series["Fluid_Sensitive"], errors="coerce").fillna(0).astype(int)
    series["_fat"] = pd.to_numeric(series["Fat_Suppression"], errors="coerce").fillna(0).astype(int)
    invalid_planes = sorted(set(series["_plane"]) - set(PLANES))
    if invalid_planes:
        raise ValueError(f"Planos inesperados nos estudos oficiais: {invalid_planes}")

    entries: list[dict[str, object]] = []
    for study_uid in official[KEY_COLUMN].tolist():
        candidates = series[series[KEY_COLUMN].eq(study_uid)].sort_values(
            ["_plane", "_fluid", "_fat", "SeriesInstanceUID"],
            ascending=[True, False, False, True],
        )
        for _, row in candidates.iterrows():
            plane = str(row["_plane"])
            fluid_sensitive = int(row["_fluid"])
            fat_suppression = int(row["_fat"])
            entries.append(
                {
                    "study_uid": study_uid,
                    "series_uid": str(row["SeriesInstanceUID"]),
                    "anatomical_plane": plane,
                    "slot": _slot_name(plane, fluid_sensitive, fat_suppression),
                    "fluid_sensitive": fluid_sensitive,
                    "fat_suppression": fat_suppression,
                    "selection_policy": "all official train_series rows; deterministic UID order within observed slot",
                }
            )

    slot_counts = (
        pd.DataFrame(entries)["slot"].value_counts().sort_index().to_dict()
        if entries
        else {}
    )
    study_counts = pd.Series([entry["study_uid"] for entry in entries]).value_counts()
    return {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "selection_policy": {
            "purpose": "audit additional acquisition slots in the 58 official gold studies",
            "study_count": int(official[KEY_COLUMN].nunique()),
            "series_count": len(entries),
            "series_per_study_min": int(study_counts.min()),
            "series_per_study_max": int(study_counts.max()),
            "planes": list(PLANES),
            "observed_slot_counts": {str(key): int(value) for key, value in slot_counts.items()},
            "note": "Slot names reflect observed metadata flags; they are not clinical protocol claims.",
        },
        "studies": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dicom_gold_six_slot_manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(_resolve(args.data_dir))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy = manifest["selection_policy"]
    print(f"studies={policy['study_count']} series={policy['series_count']}")
    print(f"slot_counts={policy['observed_slot_counts']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
