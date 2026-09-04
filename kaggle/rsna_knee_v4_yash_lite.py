#!/usr/bin/env python3
"""Ablação article-inspired: adjacent3 + fast preprocessing + max MIL.

Mantém o teacher, o EfficientNet-B0 e o blend da v4 para isolar duas ideias
transferíveis do pipeline publicado: menos views adjacentes e max pooling de
probabilidades por view. Fine-tuning B3, janela por série e fusão contínua de
teachers ficam deliberadamente fora deste experimento.
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
        external_labels_dir=None,
        slice_profile="adjacent3",
        view_pooling="target",
        teacher_profile="targetwise",
        fast_preprocess=True,
        target_pooling="max",
    )


if __name__ == "__main__":
    main()
