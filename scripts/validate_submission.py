#!/usr/bin/env python3
"""Valida o contrato do submission.csv contra o test.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]
KEY_COLUMN = "StudyInstanceUID"


def validate(test_path: Path, submission_path: Path) -> None:
    test = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)
    expected_columns = [KEY_COLUMN, *TARGET_COLUMNS]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Colunas inválidas. Esperado: {expected_columns}; recebido: {list(submission.columns)}")
    if len(submission) != len(test):
        raise ValueError(f"Número de linhas inválido: esperado {len(test)}, recebido {len(submission)}")
    expected_ids = test[KEY_COLUMN].astype(str).tolist()
    received_ids = submission[KEY_COLUMN].astype(str).tolist()
    if received_ids != expected_ids:
        raise ValueError("StudyInstanceUID não coincide com test.csv na mesma ordem.")
    if submission[KEY_COLUMN].duplicated().any():
        raise ValueError("Há StudyInstanceUID duplicado na submissão.")
    values = submission[TARGET_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Há NaN ou infinito nas probabilidades.")
    if not ((values >= 0).all() and (values <= 1).all()):
        raise ValueError("Há probabilidades fora do intervalo [0, 1].")
    print(f"OK: {submission_path} ({len(submission):,} linhas, {len(TARGET_COLUMNS)} alvos)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()
    validate(args.test, args.submission)


if __name__ == "__main__":
    main()
