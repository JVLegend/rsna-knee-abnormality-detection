#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — audita grupos de laudos duplicados.

Agrupa os laudos por hash de texto normalizado sem tentar interpretar o
conteúdo clínico. O objetivo é evitar vazamento entre treino e validação de
modelos de labels fracos e medir se os 58 estudos gold estão representados em
grupos duplicados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMN = "StudyInstanceUID"
REPORT_COLUMN = "Report"
TARGET_COLUMNS = (
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


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def normalize_report(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def report_hash(value: object) -> str:
    return hashlib.sha256(normalize_report(value).encode("utf-8")).hexdigest()


def audit(train_path: Path) -> dict[str, object]:
    frame = pd.read_csv(train_path)
    required = {KEY_COLUMN, REPORT_COLUMN, *TARGET_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Colunas ausentes em {train_path}: {missing}")

    frame = frame.copy()
    frame[KEY_COLUMN] = frame[KEY_COLUMN].astype(str)
    frame["report_hash"] = frame[REPORT_COLUMN].map(report_hash)
    grouped = frame.groupby("report_hash", sort=False)[KEY_COLUMN].agg(list)
    duplicate_groups = grouped[grouped.map(len).gt(1)]
    gold_mask = frame[list(TARGET_COLUMNS)].notna().any(axis=1)
    gold_ids = set(frame.loc[gold_mask, KEY_COLUMN])
    gold_duplicate_groups = [
        {
            "report_hash": str(report_hash_value),
            "rows": int(len(study_ids)),
            "gold_rows": int(len(gold_ids.intersection(study_ids))),
        }
        for report_hash_value, study_ids in duplicate_groups.items()
        if gold_ids.intersection(study_ids)
    ]

    rows_in_duplicate_groups = int(sum(len(study_ids) for study_ids in duplicate_groups))
    return {
        "tags": ["RSNA", "Kaggle", "Pesquisa"],
        "format": "rsna-report-hash-group-audit-v1",
        "train_path": str(train_path),
        "rows": int(len(frame)),
        "unique_normalized_reports": int(grouped.size),
        "duplicate_groups": int(len(duplicate_groups)),
        "rows_in_duplicate_groups": rows_in_duplicate_groups,
        "largest_duplicate_group": int(grouped.map(len).max()),
        "gold_rows": int(gold_mask.sum()),
        "gold_rows_in_duplicate_groups": int(
            sum(item["gold_rows"] for item in gold_duplicate_groups)
        ),
        "gold_duplicate_groups": gold_duplicate_groups,
        "normalization": "Unicode NFKC + casefold + whitespace collapse",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/report_hash_groups.json"))
    args = parser.parse_args()
    result = audit(_resolve(args.train))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"rows={result['rows']} unique_reports={result['unique_normalized_reports']} "
        f"duplicate_groups={result['duplicate_groups']} "
        f"rows_in_duplicate_groups={result['rows_in_duplicate_groups']} "
        f"gold_rows={result['gold_rows']} "
        f"gold_rows_in_duplicate_groups={result['gold_rows_in_duplicate_groups']}"
    )
    print(f"report={output}")


if __name__ == "__main__":
    main()
