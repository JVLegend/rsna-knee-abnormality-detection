"""Runner Kaggle da candidata v0.2, sem depender do repositório local."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rsna_knee_v0 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("RSNA_DATA_DIR", "/kaggle/input/rsna-knee-abnormality-detection"))
    parser.add_argument("--output", default=os.environ.get("RSNA_OUTPUT", "/kaggle/working/submission.csv"))
    args = parser.parse_args()

    submission = run(Path(args.data_dir), Path(args.output), c=32, use_lexicon=True)
    print(f"submission gravada em {args.output} com {len(submission)} linhas; candidato=v0.2; C=32; lexicon=True")


if __name__ == "__main__":
    main()
