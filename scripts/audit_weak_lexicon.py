#!/usr/bin/env python3
"""Audita sinais lexicais multilíngues sem criar pseudo-rótulos."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.constants import TARGET_COLUMNS
from rsna_knee_baseline.data import find_data_dir, load_competition_tables


LEXICON: dict[str, tuple[str, ...]] = {
    "ACL": ("acl", "lca", "anterior cruciate", "ligamento cruzado anterior", "ligament croise anterieur"),
    "MCL": ("mcl", "lcm", "medial collateral", "ligamento colateral medial", "ligament collateral medial"),
    "Medial Meniscus": ("medial meniscus", "meniscus medialis", "menisco medial", "menisque medial", "menisco interno"),
    "Lateral Meniscus": ("lateral meniscus", "meniscus lateralis", "menisco lateral", "menisque lateral", "menisco externo"),
    "Medial OA": ("medial osteoarthritis", "medial arthrosis", "medial osteoarthrosis", "medial compartment", "compartimento medial", "compartiment medial"),
    "Lateral OA": ("lateral osteoarthritis", "lateral arthrosis", "lateral osteoarthrosis", "lateral compartment", "compartimento lateral", "compartiment lateral"),
    "PF OA": ("patellofemoral osteoarthritis", "patellofemoral arthrosis", "patellofemoral compartment", "patellofemoral", "femoropatellar", "femoro-patellar"),
    "Effusion": ("joint effusion", "effusion", "derrame articular", "derrame", "epanchement"),
    "Synovitis": ("synovitis", "sinovitis", "synovite"),
    "Baker's": ("baker", "popliteal cyst", "cisto popliteo", "kyste poplite"),
    "Contusion": ("bone contusion", "bone bruise", "contusion", "contusao ossea", "contusion ossea", "contusion oseuse", "bone marrow edema"),
    "Fracture": ("fracture", "fractura", "fratura"),
}

NEGATION_CUES = ("no", "not", "without", "intact", "normal", "preserved", "absent", "sin", "sem", "aucun", "aucune", "sans")


def normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().split())


def _patterns(terms: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(rf"(?<!\w){re.escape(normalize(term))}(?!\w)") for term in terms)


def score_report(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Retorna 1 para menção não negada, -1 para menção negada e 0 para ausência."""

    normalized = normalize(text)
    positive = False
    negative = False
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            context = normalized[max(0, match.start() - 90) : match.start()]
            if any(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", context) for cue in NEGATION_CUES):
                negative = True
            else:
                positive = True
    if positive:
        return 1
    if negative:
        return -1
    return 0


def audit(train: pd.DataFrame) -> dict[str, object]:
    reports = train.get("Report", pd.Series("", index=train.index)).fillna("").astype(str)
    result: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        patterns = _patterns(LEXICON[target])
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
