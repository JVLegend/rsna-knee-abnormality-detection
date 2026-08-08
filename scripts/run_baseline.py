#!/usr/bin/env python3
"""Treina a v0 e grava um CSV de submissão."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.data import find_data_dir, load_competition_tables
from rsna_knee_baseline.model import KneeReportBaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--output", default="submissions/submission_v0_report_metadata.csv")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    tables = load_competition_tables(data_dir)
    if tables["train"].empty or tables["test"].empty:
        raise RuntimeError("train.csv e test.csv são necessários para gerar a submissão.")

    model = KneeReportBaseline(c=args.c)
    model.fit(tables["train"], tables["train_series"])
    submission = model.predict(tables["test"], tables["test_series"])

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"submission={output}")
    print(f"rows={len(submission):,} columns={len(submission.columns)}")


if __name__ == "__main__":
    main()
