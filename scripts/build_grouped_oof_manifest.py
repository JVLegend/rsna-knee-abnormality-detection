#!/usr/bin/env python3
"""Cria uma partição OOF reprodutível e agrupada por laudo.

O arquivo produzido é um contrato de validação, não uma métrica. Ele separa
os 58 estudos com os 12 rótulos oficiais completos por grupos de texto
normalizado e registra prevalências por fold. Modelos, calibradores e blends
devem consumir este manifesto para gerar previsões out-of-fold sem escolher
pesos no mesmo gold usado para diagnóstico.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_report_hash_groups import normalize_report, report_hash


KEY_COLUMN = "StudyInstanceUID"
REPORT_COLUMN = "Report"
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


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _group_key(uid: str, report: object) -> str:
    """Do not collapse every blank report into one artificial group."""
    if normalize_report(report):
        return f"report:{report_hash(report)}"
    return f"uid:{uid}"


def _strata(labels: np.ndarray, folds: int) -> tuple[np.ndarray, str]:
    """Choose a small, stable stratification target for a 58-study panel."""
    positives = labels.sum(axis=1).astype(int)
    candidates = [
        (np.minimum(positives, 4), "positive_count_clipped_4"),
        (np.minimum(positives, 3), "positive_count_clipped_3"),
        ((positives > 0).astype(int), "any_positive"),
    ]
    for values, name in candidates:
        counts = Counter(values.tolist())
        if len(counts) >= 2 and min(counts.values()) >= folds:
            return values, name
    return np.zeros(len(labels), dtype=np.int8), "unstratified_fallback"


def _folds(
    ids: list[str],
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    strata, strata_name = _strata(labels, folds)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = np.arange(len(ids))
    try:
        splits = list(splitter.split(rows, strata, groups))
    except ValueError:
        fallback = StratifiedGroupKFold(n_splits=folds, shuffle=False)
        splits = list(fallback.split(rows, np.zeros(len(rows), dtype=np.int8), groups))
        strata_name = "grouped_fallback"
    if len(splits) != folds or any(len(valid) == 0 for _train, valid in splits):
        raise RuntimeError("não foi possível criar folds OOF não vazios")
    return splits, strata_name


def build_manifest(train_path: Path, folds: int, seed: int) -> dict[str, object]:
    frame = pd.read_csv(train_path, dtype={KEY_COLUMN: str})
    required = {KEY_COLUMN, REPORT_COLUMN, *TARGETS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"colunas ausentes em {train_path}: {missing}")
    gold = frame[frame[list(TARGETS)].notna().all(axis=1)].copy()
    if len(gold) < folds:
        raise ValueError(f"gold tem {len(gold)} estudos; folds={folds} é inviável")
    ids = gold[KEY_COLUMN].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("StudyInstanceUID duplicado no gold")
    labels = gold[list(TARGETS)].to_numpy(dtype=np.int8)
    groups = np.asarray(
        [_group_key(uid, report) for uid, report in zip(ids, gold[REPORT_COLUMN])],
        dtype=object,
    )
    splits, strata_name = _folds(ids, labels, groups, folds, seed)

    fold_records: list[dict[str, object]] = []
    seen_valid: set[str] = set()
    for fold_index, (train_rows, valid_rows) in enumerate(splits):
        train_ids = [ids[index] for index in train_rows]
        valid_ids = [ids[index] for index in valid_rows]
        train_groups = set(groups[train_rows].tolist())
        valid_groups = set(groups[valid_rows].tolist())
        overlap = sorted(train_groups & valid_groups)
        if overlap:
            raise RuntimeError(f"vazamento de grupos no fold {fold_index}: {overlap[:3]}")
        if seen_valid.intersection(valid_ids):
            raise RuntimeError("um estudo apareceu em mais de uma validação")
        seen_valid.update(valid_ids)
        prevalence = {
            target: float(labels[valid_rows, target_index].mean())
            for target_index, target in enumerate(TARGETS)
        }
        fold_records.append(
            {
                "fold": fold_index,
                "train_ids": train_ids,
                "valid_ids": valid_ids,
                "train_groups": len(train_groups),
                "valid_groups": len(valid_groups),
                "valid_prevalence": prevalence,
                "valid_positive_count": int(labels[valid_rows].sum()),
            }
        )
    if seen_valid != set(ids):
        raise RuntimeError("a união dos folds de validação não cobre o gold inteiro")

    group_counts = Counter(groups.tolist())
    duplicate_groups = {key: count for key, count in group_counts.items() if count > 1}
    return {
        "tags": ["RSNA", "Kaggle", "Pesquisa"],
        "format": "rsna-grouped-oof-manifest-v1",
        "diagnostic_only": True,
        "source": str(train_path),
        "selection": "StudyInstanceUID com os 12 rótulos oficiais não nulos",
        "study_count": len(ids),
        "target_count": len(TARGETS),
        "fold_count": folds,
        "seed": seed,
        "stratification": strata_name,
        "grouping": "report normalizado (NFKC/casefold/espaços); relatório vazio usa UID",
        "duplicate_report_groups": duplicate_groups,
        "group_leakage_audit": {
            "groups": len(group_counts),
            "duplicate_groups": len(duplicate_groups),
            "max_group_size": max(group_counts.values()),
            "cross_fold_overlap": 0,
        },
        "overall_prevalence": {
            target: float(labels[:, target_index].mean())
            for target_index, target in enumerate(TARGETS)
        },
        "folds": fold_records,
        "next_step": "gerar predições OOF fold-conditioned; não ajustar blend no gold inteiro",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/grouped_oof_manifest_gold_20260905.json"),
    )
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("folds precisa ser >= 2")
    result = build_manifest(_resolve(args.train), args.folds, args.seed)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"study_count={result['study_count']} folds={result['fold_count']} "
        f"stratification={result['stratification']} "
        f"duplicate_groups={result['group_leakage_audit']['duplicate_groups']}"
    )
    print(f"report={output}")


if __name__ == "__main__":
    main()
