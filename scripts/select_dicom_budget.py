#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — cria manifesto DICOM por orçamento aproximado."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TARGETS = (
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
)
PLANES = ("Sagittal", "Coronal", "Axial")
KEY_COLUMN = "StudyInstanceUID"


def _tie_breaker(uid: str) -> str:
    return hashlib.sha1(uid.encode("utf-8")).hexdigest()


def _excluded_studies(data_dir: Path, manifests: list[Path]) -> set[str]:
    excluded = {path.name for path in (data_dir / "train_series").iterdir() if path.is_dir()}
    for manifest in manifests:
        payload = json.loads(manifest.expanduser().read_text(encoding="utf-8"))
        excluded.update(str(entry["study_uid"]) for entry in payload.get("studies", []))
    return excluded


def _select_studies(
    teacher: pd.DataFrame,
    available: set[str],
    excluded: set[str],
    target_studies: int,
    bin_quota: int,
) -> list[str]:
    teacher = teacher.copy()
    teacher[KEY_COLUMN] = teacher[KEY_COLUMN].astype(str)
    teacher = teacher[teacher[KEY_COLUMN].isin(available) & ~teacher[KEY_COLUMN].isin(excluded)].copy()
    teacher["_tie"] = teacher[KEY_COLUMN].map(_tie_breaker)

    selected: list[str] = []
    selected_set: set[str] = set()
    for target in TARGETS:
        values = pd.to_numeric(teacher[target], errors="coerce")
        for positive, mask in ((True, values >= 0.85), (False, values <= 0.15)):
            candidates = teacher.loc[mask].copy()
            candidates["_value"] = pd.to_numeric(candidates[target], errors="coerce")
            candidates = candidates.sort_values(
                ["_value", "_tie"], ascending=[not positive, True]
            )
            added = 0
            for uid in candidates[KEY_COLUMN].astype(str):
                if uid in selected_set:
                    continue
                selected.append(uid)
                selected_set.add(uid)
                added += 1
                if len(selected) >= target_studies or added >= bin_quota:
                    break
            if len(selected) >= target_studies:
                break
        if len(selected) >= target_studies:
            break

    if len(selected) < target_studies:
        for uid in teacher.sort_values("_tie")[KEY_COLUMN].astype(str):
            if uid not in selected_set:
                selected.append(uid)
                selected_set.add(uid)
            if len(selected) >= target_studies:
                break
    return selected[:target_studies]


def _series_entries(series: pd.DataFrame, study_uids: list[str]) -> tuple[list[dict[str, object]], list[str]]:
    series = series.copy()
    series[KEY_COLUMN] = series[KEY_COLUMN].astype(str)
    plane_rank = {plane: index for index, plane in enumerate(PLANES)}
    entries: list[dict[str, object]] = []
    missing: list[str] = []
    for study_uid in study_uids:
        candidates = series[series[KEY_COLUMN].eq(study_uid)].copy()
        for plane in PLANES:
            plane_rows = candidates[candidates["Anatomical_Plane"].eq(plane)].copy()
            if plane_rows.empty:
                missing.append(f"{study_uid}:{plane}")
                continue
            plane_rows["_fluid"] = pd.to_numeric(plane_rows["Fluid_Sensitive"], errors="coerce").fillna(0)
            plane_rows["_fat"] = pd.to_numeric(plane_rows["Fat_Suppression"], errors="coerce").fillna(0)
            chosen = plane_rows.sort_values(
                ["_fluid", "_fat", "SeriesInstanceUID"], ascending=[False, False, True]
            ).iloc[0]
            entries.append(
                {
                    "study_uid": study_uid,
                    "series_uid": str(chosen["SeriesInstanceUID"]),
                    "anatomical_plane": plane,
                    "fluid_sensitive": int(chosen["Fluid_Sensitive"]),
                    "fat_suppression": int(chosen["Fat_Suppression"]),
                    "selection_policy": "one best Fluid_Sensitive/Fat_Suppression series per plane",
                }
            )
    return entries, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-studies", type=int, default=700)
    parser.add_argument("--bin-quota", type=int, default=28)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()

    if args.target_studies < 1 or args.bin_quota < 1:
        raise ValueError("target-studies e bin-quota precisam ser positivos")

    data_dir = args.data_dir.expanduser()
    train_series = pd.read_csv(data_dir / "train_series.csv")
    teacher = pd.read_csv(args.teacher.expanduser())
    available = set(train_series[KEY_COLUMN].astype(str))
    excluded = _excluded_studies(data_dir, [path.expanduser() for path in args.exclude_manifest])
    study_uids = _select_studies(teacher, available, excluded, args.target_studies, args.bin_quota)
    entries, missing = _series_entries(train_series, study_uids)

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_policy": {
            "target_bytes": 50_000_000_000,
            "target_studies": args.target_studies,
            "series_per_study": 3,
            "planes": list(PLANES),
            "bin_quota": args.bin_quota,
            "teacher": str(args.teacher.expanduser()),
            "excluded_studies": len(excluded),
            "labels_are_weak_selection_only": True,
            "note": "O tamanho final será confirmado pela API antes do download.",
        },
        "studies": entries,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")
    print(f"selected_studies={len(study_uids)}")
    print(f"selected_series={len(entries)}")
    print(f"excluded_studies={len(excluded)}")
    print(f"missing_planes={len(missing)}")
    if missing:
        print(f"missing_examples={missing[:5]}")


if __name__ == "__main__":
    main()
