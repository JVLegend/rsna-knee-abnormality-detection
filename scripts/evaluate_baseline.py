#!/usr/bin/env python3
"""Calcula validação cruzada por estudo para a v0 de texto e metadados."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS
from rsna_knee_baseline.data import find_data_dir, load_competition_tables
from rsna_knee_baseline.model import KneeReportBaseline


def _label_series(train: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        target: pd.to_numeric(train.get(target, pd.Series(np.nan, index=train.index)), errors="coerce")
        for target in TARGET_COLUMNS
    }


def _n_splits(labels: dict[str, pd.Series], requested: int) -> int:
    counts: list[int] = []
    for target in TARGET_COLUMNS:
        values = labels[target].dropna().to_numpy()
        if values.size:
            counts.extend([int((values == 0).sum()), int((values == 1).sum())])
    if not counts:
        raise ValueError("Nenhum rótulo supervisionado foi encontrado.")
    folds = min(requested, min(counts))
    if folds < 2:
        raise ValueError("São necessários pelo menos dois exemplos de cada classe para a validação.")
    return folds


def _common_splits(
    labels: dict[str, pd.Series],
    labeled_indices: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str] | None:
    """Tenta usar os mesmos folds para todos os alvos, reduzindo custo e variância."""

    anchors = sorted(
        TARGET_COLUMNS,
        key=lambda target: min(
            int((labels[target].iloc[labeled_indices] == 0).sum()),
            int((labels[target].iloc[labeled_indices] == 1).sum()),
        ),
    )
    for offset in range(100):
        anchor = anchors[0]
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + offset)
        anchor_y = labels[anchor].iloc[labeled_indices].to_numpy()
        splits = list(splitter.split(labeled_indices, anchor_y))
        valid = True
        for target in TARGET_COLUMNS:
            target_y = labels[target].iloc[labeled_indices].to_numpy()
            for train_positions, val_positions in splits:
                if np.unique(target_y[train_positions]).size < 2:
                    valid = False
                    break
                if np.unique(target_y[val_positions]).size < 2:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return splits, f"common_stratified_by_{anchor}_seed_{seed + offset}"
    return None


def _masked_training_frame(
    train: pd.DataFrame,
    labels: dict[str, pd.Series],
    train_indices: np.ndarray,
    target: str | None = None,
) -> pd.DataFrame:
    """Mantém apenas rótulos do fold de treino; ausências continuam ausências."""

    masked = train.copy()
    for column in TARGET_COLUMNS:
        masked[column] = np.nan

    targets = [target] if target is not None else TARGET_COLUMNS
    for column in targets:
        masked.loc[train_indices, column] = labels[column].iloc[train_indices].to_numpy()
    return masked


def _target_specific_splits(
    labels: dict[str, pd.Series],
    target: str,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labeled_indices = np.flatnonzero(labels[target].notna().to_numpy())
    y = labels[target].iloc[labeled_indices].to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(labeled_indices[train], labeled_indices[val]) for train, val in splitter.split(labeled_indices, y)]


def _series_for_studies(series: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if series.empty or KEY_COLUMN not in series.columns:
        return series
    study_ids = set(frame[KEY_COLUMN].astype(str))
    return series[series[KEY_COLUMN].astype(str).isin(study_ids)].copy()


def _evaluate(
    train: pd.DataFrame,
    train_series: pd.DataFrame,
    requested_folds: int,
    seed: int,
) -> dict[str, object]:
    labels = _label_series(train)
    folds = _n_splits(labels, requested_folds)
    labeled_sets = [np.flatnonzero(labels[target].notna().to_numpy()) for target in TARGET_COLUMNS]
    same_labeled_set = all(np.array_equal(labeled_sets[0], indices) for indices in labeled_sets[1:])

    oof = {target: np.full(len(train), np.nan, dtype=float) for target in TARGET_COLUMNS}
    split_mode = "target_specific"

    common = None
    if same_labeled_set:
        common = _common_splits(labels, labeled_sets[0], folds, seed)

    if common is not None:
        splits, split_mode = common
        for train_positions, val_positions in splits:
            train_indices = labeled_sets[0][train_positions]
            val_indices = labeled_sets[0][val_positions]
            fit_frame = _masked_training_frame(train, labels, train_indices)
            model = KneeReportBaseline().fit(fit_frame, train_series)
            validation_frame = train.iloc[val_indices]
            predictions = model.predict(validation_frame, _series_for_studies(train_series, validation_frame))
            for target in TARGET_COLUMNS:
                oof[target][val_indices] = predictions[target].to_numpy()
        folds_used = len(splits)
    else:
        folds_used = folds
        for target in TARGET_COLUMNS:
            for train_indices, val_indices in _target_specific_splits(labels, target, folds, seed):
                fit_frame = _masked_training_frame(train, labels, train_indices, target=target)
                model = KneeReportBaseline().fit(fit_frame, train_series)
                validation_frame = train.iloc[val_indices]
                predictions = model.predict(validation_frame, _series_for_studies(train_series, validation_frame))
                oof[target][val_indices] = predictions[target].to_numpy()

    target_results: list[dict[str, int | float | str]] = []
    aucs: list[float] = []
    for target in TARGET_COLUMNS:
        target_labels = labels[target].notna().to_numpy()
        y_true = labels[target].loc[target_labels].to_numpy(dtype=float)
        y_score = oof[target][target_labels]
        if not np.isfinite(y_score).all():
            raise RuntimeError(f"Predições OOF incompletas para {target}.")
        auc = float(roc_auc_score(y_true, y_score))
        aucs.append(auc)
        target_results.append(
            {
                "target": target,
                "labeled": int(len(y_true)),
                "positive": int((y_true == 1).sum()),
                "negative": int((y_true == 0).sum()),
                "auc": auc,
            }
        )

    return {
        "model": "v0_report_metadata",
        "seed": seed,
        "requested_folds": requested_folds,
        "folds_used": folds_used,
        "split_mode": split_mode,
        "study_level_split": True,
        "unlabeled_reports_used_for_feature_vocabulary": True,
        "macro_auc": float(np.mean(aucs)),
        "targets": target_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/v0_report_metadata_cv.json")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    tables = load_competition_tables(data_dir)
    train = tables["train"].reset_index(drop=True)
    if train.empty:
        raise RuntimeError("train.csv é necessário para a validação.")

    result = _evaluate(train, tables["train_series"], args.folds, args.seed)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"data_dir={data_dir}")
    print(f"split_mode={result['split_mode']} folds={result['folds_used']} seed={result['seed']}")
    for row in result["targets"]:
        print(f"{row['target']}: AUC={row['auc']:.6f} labeled={row['labeled']}")
    print(f"macro_auc={result['macro_auc']:.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
