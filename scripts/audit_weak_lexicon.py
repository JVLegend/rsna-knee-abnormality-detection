#!/usr/bin/env python3
"""Audita sinais lexicais multilíngues sem criar pseudo-rótulos."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.data import find_data_dir, load_competition_tables
from rsna_knee_baseline.constants import TARGET_COLUMNS
from rsna_knee_baseline.lexicon import LEXICON, compile_patterns, score_report


def audit(train: pd.DataFrame) -> dict[str, object]:
    reports = train.get("Report", pd.Series("", index=train.index)).fillna("").astype(str)
    result: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        patterns = compile_patterns(LEXICON[target])
        scores = np.asarray([score_report(report, patterns) for report in reports], dtype=int)
        labels = pd.to_numeric(train[target], errors="coerce")
        labeled = labels.notna().to_numpy()
        y_true = labels.loc[labeled].to_numpy(dtype=float)
        y_score = scores[labeled]
        matched = y_score != 0
        positive_match = y_score == 1
        score_auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_score).size > 1 else None
        positives = int((y_true == 1).sum())
        result.append(
            {
                "target": target,
                "terms": list(LEXICON[target]),
                "all_report_mentions": int((scores != 0).sum()),
                "labeled": int(len(y_true)),
                "gold_positive": positives,
                "gold_negative": int((y_true == 0).sum()),
                "matched_labeled": int(matched.sum()),
                "unnegated_labeled": int((y_score == 1).sum()),
                "negated_labeled": int((y_score == -1).sum()),
                "unnegated_precision": float((y_true[positive_match] == 1).mean()) if positive_match.any() else None,
                "unnegated_recall": float((y_true[positive_match] == 1).sum() / positives) if positives else None,
                "score_auc": score_auc,
            }
        )
    valid_auc = [row["score_auc"] for row in result if row["score_auc"] is not None]
    return {
        "audit_date": date.today().isoformat(),
        "scope": "labeled_train_reports_only_for_precision_recall",
        "negation_window_characters": 90,
        "creates_pseudo_labels": False,
        "macro_score_auc": float(np.mean(valid_auc)) if valid_auc else None,
        "targets": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="reports/weak_lexicon_audit.json")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    train = load_competition_tables(data_dir)["train"]
    if train.empty:
        raise RuntimeError("train.csv é necessário para a auditoria.")
    result = audit(train)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"data_dir={data_dir}")
    for row in result["targets"]:
        precision = row["unnegated_precision"]
        recall = row["unnegated_recall"]
        auc = row["score_auc"]
        print(f"{row['target']}: mentions={row['all_report_mentions']} labeled_matches={row['matched_labeled']} precision={precision if precision is not None else 'NA'} recall={recall if recall is not None else 'NA'} auc={auc if auc is not None else 'NA'}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
