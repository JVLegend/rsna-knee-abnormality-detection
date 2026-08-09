#!/usr/bin/env python3
"""Avalia uma fusão simples entre o baseline textual e embeddings visuais."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS
from rsna_knee_baseline.data import find_data_dir, load_competition_tables
from rsna_knee_baseline.model import KneeReportBaseline
from scripts.evaluate_visual_embeddings import _fold_count, _load_visual_rows, _resolve


def _series_for_studies(series: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if series.empty or KEY_COLUMN not in series.columns:
        return series
    ids = set(frame[KEY_COLUMN].astype(str))
    return series[series[KEY_COLUMN].astype(str).isin(ids)].copy()


def _parse_alphas(value: str) -> tuple[float, ...]:
    alphas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not alphas or any(alpha < 0 or alpha > 1 for alpha in alphas):
        raise ValueError("--alphas precisa conter valores entre 0 e 1.")
    return alphas


def _parse_target_alphas(value: str) -> dict[str, float]:
    """Lê pesos por alvo no formato ``ACL=0.5,Effusion=0.5``."""

    result: dict[str, float] = {}
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Peso por alvo inválido: {item!r}; use Alvo=0.4.")
        target, raw_alpha = (part.strip() for part in item.split("=", 1))
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Alvo desconhecido em --target-alphas: {target!r}.")
        alpha = float(raw_alpha)
        if not 0 <= alpha <= 1:
            raise ValueError(f"Peso visual fora de [0, 1] para {target!r}: {alpha}.")
        result[target] = alpha
    if set(result) != set(TARGET_COLUMNS):
        missing = sorted(set(TARGET_COLUMNS) - set(result))
        raise ValueError(f"--target-alphas precisa definir os 12 alvos; faltam: {missing}.")
    return result


def evaluate_fusion(
    matrix: np.ndarray,
    frame: pd.DataFrame,
    series: pd.DataFrame,
    folds: int = 5,
    seed: int = 42,
    text_c: float = 32.0,
    visual_c: float = 1.0,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    target_alphas: dict[str, float] | None = None,
) -> dict[str, object]:
    if matrix.shape[0] != len(frame):
        raise ValueError("Número de embeddings e estudos não coincide.")
    if text_c <= 0 or visual_c <= 0:
        raise ValueError("As regularizações precisam ser positivas.")
    if target_alphas is not None and set(target_alphas) != set(TARGET_COLUMNS):
        raise ValueError("target_alphas precisa definir exatamente os 12 alvos.")

    oof_text: dict[str, np.ndarray] = {}
    oof_visual: dict[str, np.ndarray] = {}
    target_meta: dict[str, dict[str, int]] = {}

    for target in TARGET_COLUMNS:
        labels = pd.to_numeric(frame.get(target), errors="coerce")
        labeled = labels.notna().to_numpy()
        indices = np.flatnonzero(labeled)
        y = labels.loc[labeled].to_numpy(dtype=np.int64)
        used_folds = _fold_count(y, folds)
        splitter = StratifiedKFold(n_splits=used_folds, shuffle=True, random_state=seed)
        text_scores = np.full(len(y), np.nan, dtype=np.float64)
        visual_scores = np.full(len(y), np.nan, dtype=np.float64)
        x = matrix[labeled]

        for train_positions, validation_positions in splitter.split(x, y):
            train_indices = indices[train_positions]
            validation_indices = indices[validation_positions]
            fit_frame = frame.iloc[train_indices].reset_index(drop=True)
            validation_frame = frame.iloc[validation_indices].reset_index(drop=True)
            text_model = KneeReportBaseline(c=text_c, use_lexicon=True, lexicon_weight=1.0)
            text_model.fit(fit_frame, _series_for_studies(series, fit_frame))
            text_scores[validation_positions] = text_model.predict(
                validation_frame, _series_for_studies(series, validation_frame)
            )[target].to_numpy(dtype=float)

            visual_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=visual_c, class_weight="balanced", max_iter=2000, solver="liblinear"),
            )
            visual_model.fit(x[train_positions], y[train_positions])
            visual_scores[validation_positions] = visual_model.predict_proba(x[validation_positions])[:, 1]

        if not np.isfinite(text_scores).all() or not np.isfinite(visual_scores).all():
            raise RuntimeError(f"Predições OOF incompletas para {target}.")
        oof_text[target] = text_scores
        oof_visual[target] = visual_scores
        target_meta[target] = {
            "labeled": int(len(y)),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "folds_used": used_folds,
        }

    models: dict[str, dict[str, object]] = {}
    for alpha in alphas:
        rows: list[dict[str, object]] = []
        aucs: list[float] = []
        for target in TARGET_COLUMNS:
            labels = pd.to_numeric(frame[target], errors="coerce")
            y = labels.dropna().to_numpy(dtype=float)
            score = (1 - alpha) * oof_text[target] + alpha * oof_visual[target]
            auc = float(roc_auc_score(y, score))
            aucs.append(auc)
            rows.append({"target": target, **target_meta[target], "auc": auc})
        key = f"alpha_{alpha:g}"
        models[key] = {"visual_weight": alpha, "text_weight": 1 - alpha, "macro_auc": float(np.mean(aucs)), "targets": rows}

    if target_alphas is not None:
        rows = []
        aucs = []
        for target in TARGET_COLUMNS:
            labels = pd.to_numeric(frame[target], errors="coerce")
            y = labels.dropna().to_numpy(dtype=float)
            alpha = target_alphas[target]
            score = (1 - alpha) * oof_text[target] + alpha * oof_visual[target]
            auc = float(roc_auc_score(y, score))
            aucs.append(auc)
            rows.append({"target": target, **target_meta[target], "visual_weight": alpha, "auc": auc})
        models["targetwise"] = {
            "visual_weights": target_alphas,
            "macro_auc": float(np.mean(aucs)),
            "targets": rows,
        }

    return {
        "model": "text_visual_probability_blend",
        "embedding_shape": list(matrix.shape),
        "study_count": int(len(frame)),
        "requested_folds": folds,
        "seed": seed,
        "text_c": text_c,
        "visual_c": visual_c,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-c", type=float, default=32.0)
    parser.add_argument("--visual-c", type=float, default=1.0)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--target-alphas", default="")
    parser.add_argument("--output", type=Path, default=Path("reports/fusion_embeddings_cv.json"))
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    tables = load_competition_tables(data_dir)
    train = tables["train"].reset_index(drop=True)
    matrix, frame, index = _load_visual_rows(_resolve(args.index), train)
    result = evaluate_fusion(
        matrix,
        frame,
        tables["train_series"],
        folds=args.folds,
        seed=args.seed,
        text_c=args.text_c,
        visual_c=args.visual_c,
        alphas=_parse_alphas(args.alphas),
        target_alphas=_parse_target_alphas(args.target_alphas) if args.target_alphas else None,
    )
    result["source_index"] = str(_resolve(args.index))
    result["weights_sha256"] = index.get("weights_sha256")
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"studies={result['study_count']} embedding_shape={result['embedding_shape']} folds={args.folds} seed={args.seed}")
    for name, values in result["models"].items():
        print(f"{name}_macro_auc={values['macro_auc']:.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
