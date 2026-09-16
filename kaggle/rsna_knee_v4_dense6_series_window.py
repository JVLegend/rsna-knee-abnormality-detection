#!/usr/bin/env python3
"""Ablação controlada: janela de intensidade comum por série.

Preserva a configuração da v6 (dense6, B0, teacher target-wise, pooling por
alvo original, weak labels hard e blend global) e troca somente a normalização
1–99% independente por slice por uma janela comum às fatias usadas na série.
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
        weak_label_mode="hard",
        external_labels_dir=None,
        slice_profile="dense6",
        view_pooling="target",
        teacher_profile="targetwise",
        fast_preprocess=False,
        intensity_window="series",
        target_pooling=None,
    )


if __name__ == "__main__":
    main()
