#!/usr/bin/env python3
"""Ablação controlada: labels weak suaves em dense6.

Preserva a configuração da v6 (dense6, preprocessamento integral, teacher
target-wise, B0, pooling por alvo original e blend global) e mantém a
probabilidade do teacher nos estudos weak, em vez de convertê-la imediatamente
para 0/1. É a implementação direta da hipótese de labels suaves inspirada no
pipeline do BishnoiYash.
"""

from __future__ import annotations

import os
from pathlib import Path

from rsna_knee_v4_dense_target_pool import run


def main() -> None:
    data_dir = Path(os.environ.get("RSNA_DATA_DIR", "/kaggle/input/rsna-knee-abnormality-detection"))
    output = Path(os.environ.get("RSNA_OUTPUT", "/kaggle/working/submission.csv"))
    run(
        data_dir,
        output,
        visual_weight=float(os.environ.get("RSNA_VISUAL_WEIGHT", "0.4")),
        batch_size=int(os.environ.get("RSNA_BATCH_SIZE", "32")),
        device=os.environ.get("RSNA_DEVICE", "auto"),
        targetwise=False,
        weak_visual=True,
        weak_threshold=0.85,
        weak_sample_weight=0.10,
        weak_label_mode="soft",
        external_labels_dir=None,
        slice_profile="dense6",
        view_pooling="target",
        teacher_profile="targetwise",
        fast_preprocess=False,
        target_pooling=None,
    )


if __name__ == "__main__":
    main()
