#!/usr/bin/env python3
"""Compara labels públicos de laudos contra os 58 rótulos oficiais.

O script é deliberadamente independente do modelo visual. Ele serve para
separar o ganho de supervisão do ganho de representação e evita interpretar
``UNK`` ou ``0,5`` como um negativo verdadeiro.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float | None:
    mask = y_true.notna() & score.notna()
    if mask.sum() == 0:
        return None
    y = y_true.loc[mask].astype(float).to_numpy()
    s = score.loc[mask].astype(float).to_numpy()
    if np.unique(y).size < 2 or np.unique(s).size < 2:
        return None
    return float(roc_auc_score(y, s))


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)


def _source_frame(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fonte não encontrada: {path}")
    frame = pd.read_csv(path)
    if KEY_COLUMN not in frame.columns:
        raise ValueError(f"{name}: ausência de {KEY_COLUMN} em {path}")
    if frame[KEY_COLUMN].duplicated().any():
        raise ValueError(f"{name}: UIDs duplicados em {path}")

    frame = frame.copy()
    frame[KEY_COLUMN] = frame[KEY_COLUMN].astype(str)
    result = pd.DataFrame({KEY_COLUMN: frame[KEY_COLUMN]})
    for target in TARGET_COLUMNS:
        if target not in frame.columns:
            raise ValueError(f"{name}: ausência do alvo {target!r} em {path}")
        raw = _as_float(frame[target])
        result[f"{target}__raw"] = raw
        verdict_column = f"{target}__verdict"
        if verdict_column in frame.columns:
            verdict = frame[verdict_column].fillna("UNK").astype(str).str.upper()
            state = raw.mask(verdict.eq("UNK"), 0.5)
            result[f"{target}__state"] = state
            result[f"{target}__unk"] = verdict.eq("UNK")
        else:
            result[f"{target}__state"] = raw
            result[f"{target}__unk"] = raw.sub(0.5).abs().le(1e-9)
        confidence_column = f"{target}__conf"
        if confidence_column in frame.columns:
            result[f"{target}__conf"] = _as_float(frame[confidence_column])
    result.attrs["name"] = name
    result.attrs["path"] = str(path)
    return result


def _metric_row(
    source: str,
    target: str,
    gold: pd.Series,
    score: pd.Series,
    *,
    score_kind: str,
) -> dict[str, object]:
    valid = gold.notna() & score.notna()
    y = gold.loc[valid].astype(float)
    s = score.loc[valid].astype(float)
    non_neutral = s.sub(0.5).abs().gt(1e-9)
    predicted = s.gt(0.5)
    row: dict[str, object] = {
        "source": source,
        "score_kind": score_kind,
        "target": target,
        "n_gold": int(valid.sum()),
        "positive_gold": int((y == 1).sum()),
        "coverage_non_neutral": float(non_neutral.mean()) if len(s) else None,
        "auc": _safe_auc(y, s),
        "direction_accuracy_non_neutral": None,
        "positive_precision": None,
        "negative_precision": None,
    }
    if non_neutral.any():
        y_non = y.loc[non_neutral]
        s_non = s.loc[non_neutral]
        row["direction_accuracy_non_neutral"] = float((s_non.gt(0.5) == y_non.eq(1)).mean())
    positive = s.gt(0.5)
    negative = s.lt(0.5)
    if positive.any():
        row["positive_precision"] = float((y.loc[positive] == 1).mean())
    if negative.any():
        row["negative_precision"] = float((y.loc[negative] == 0).mean())
    return row


def _evaluate_source(
    source: str,
    gold: pd.DataFrame,
    source_frame: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    merged = gold[[KEY_COLUMN, *TARGET_COLUMNS]].merge(source_frame, on=KEY_COLUMN, how="left")
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        gold_target = merged[target]
        rows.append(_metric_row(source, target, gold_target, merged[f"{target}__raw"], score_kind="raw"))
        rows.append(_metric_row(source, target, gold_target, merged[f"{target}__state"], score_kind="state"))
    return rows, merged


def _rank_consensus(
    source_frames: dict[str, pd.DataFrame],
    *,
    state: bool,
) -> pd.DataFrame:
    """Cria consenso simples de rankings preservando os UIDs disponíveis."""

    all_frames: list[pd.DataFrame] = []
    for name, frame in source_frames.items():
        part = pd.DataFrame({KEY_COLUMN: frame[KEY_COLUMN]})
        for target in TARGET_COLUMNS:
            column = f"{target}__state" if state else f"{target}__raw"
            values = frame[column]
            ranks = values.rank(method="average", pct=True)
            part[f"{target}__{name}"] = ranks
        all_frames.append(part)

    merged = all_frames[0]
    for frame in all_frames[1:]:
        merged = merged.merge(frame, on=KEY_COLUMN, how="outer")
    result = pd.DataFrame({KEY_COLUMN: merged[KEY_COLUMN].astype(str)})
    for target in TARGET_COLUMNS:
        columns = [column for column in merged.columns if column.startswith(f"{target}__")]
        result[target] = merged[columns].mean(axis=1, skipna=True)
    return result


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Fonte deve estar no formato nome=/caminho/arquivo.csv")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Nome e caminho da fonte são obrigatórios")
    return name, Path(raw_path).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", default="reports/external_label_audit.json")
    args = parser.parse_args()

    gold = pd.read_csv(Path(args.data_dir) / "train.csv")
    if KEY_COLUMN not in gold.columns:
        raise ValueError(f"Treino sem {KEY_COLUMN}: {args.data_dir}")
    gold[KEY_COLUMN] = gold[KEY_COLUMN].astype(str)
    gold = gold[gold[TARGET_COLUMNS].notna().any(axis=1)].copy()

    source_frames: dict[str, pd.DataFrame] = {}
    metrics: list[dict[str, object]] = []
    for name, path in args.source:
        frame = _source_frame(name, path)
        source_frames[name] = frame
        rows, _ = _evaluate_source(name, gold, frame)
        metrics.extend(rows)

    for state, suffix in ((True, "state_rank"), (False, "raw_rank")):
        consensus = _rank_consensus(source_frames, state=state)
        merged = gold[[KEY_COLUMN, *TARGET_COLUMNS]].merge(consensus, on=KEY_COLUMN, how="left")
        for target in TARGET_COLUMNS:
            metrics.append(
                _metric_row(
                    f"consensus_{suffix}",
                    target,
                    merged[target + "_x"],
                    merged[target + "_y"],
                    score_kind="consensus",
                )
            )

    summary: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_rows_with_any_label": int(len(gold)),
        "gold_rows_total": int(pd.read_csv(Path(args.data_dir) / "train.csv").shape[0]),
        "sources": {name: frame.attrs for name, frame in source_frames.items()},
        "metrics": metrics,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    table = pd.DataFrame(metrics)
    macro = table.groupby(["source", "score_kind"], dropna=False)["auc"].mean().sort_values(ascending=False)
    print(f"gold_rows_with_any_label={len(gold)}")
    print("macro_auc_by_source=")
    for (source, score_kind), value in macro.items():
        print(f"  {source}/{score_kind}: {value:.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
