"""Offline DINOv2-S/14 + MIL inference for a Kaggle kernel.

The public bundle contains the DINOv2 source, the official-looking pretrained
backbone file, and four small MIL heads.  The original ``predict.py`` did not
pass the local backbone file to ``torch.hub.load``; on Kaggle that would make
the run depend on an unavailable network.  This entrypoint wires the local
weight path explicitly and keeps an opt-in license gate for the bundle heads.

Required Kaggle inputs:

* the competition data;
* ``ericwang-dinov2-mil-bundle`` (audit its ``other`` license before use).

Set ``RSNA_DINOV2_LICENSE_ACK=1`` only after the bundle's license and the
competition rules have been reviewed.  The default is intentionally safe.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


BUNDLE_ROOT = Path(os.environ.get("BUNDLE_ROOT", "/kaggle/input/rsna-knee-dinov2-mil-bundle"))
DATA_ROOT = Path(os.environ.get("RSNA_DATA", "/kaggle/input/competitions/rsna-knee-abnormality-detection"))
WORK_ROOT = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working"))
OUTPUT_PATH = WORK_ROOT / "submission.csv"
LIMIT = int(os.environ.get("RSNA_DINOV2_LIMIT", "0"))

sys.path.insert(0, str(BUNDLE_ROOT / "src"))
sys.path.insert(0, str(BUNDLE_ROOT / "src" / "rsna_knee"))

from rsna_knee.data import LABELS, UID, load_tables  # noqa: E402
from rsna_knee.model import StudyMIL  # noqa: E402
from rsna_knee.train import EmbeddingDataset, _collate, cache_embeddings  # noqa: E402


def _logger() -> logging.Logger:
    logger = logging.getLogger("rsna_knee_dinov2_offline")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class OfflineDinoV2Encoder(nn.Module):
    """Frozen DINOv2-S/14 whose weights are loaded from the attached bundle."""

    def __init__(self, device: torch.device, bundle_root: Path):
        super().__init__()
        repo_dir = bundle_root / "dinov2_src"
        weight_path = bundle_root / "weights" / "dinov2_vits14_pretrain.pth"
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"DINOv2 source not found: {repo_dir}")
        if not weight_path.is_file():
            raise FileNotFoundError(f"DINOv2 pretrained weights not found: {weight_path}")

        self.encoder = torch.hub.load(
            str(repo_dir),
            "dinov2_vits14",
            source="local",
            weights=str(weight_path),
        )
        self.encoder.eval().to(device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.embedding_dim = 384

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            features = self.encoder(images)
        return features.float()


def _assert_license_ack() -> None:
    acknowledged = os.environ.get("RSNA_DINOV2_LICENSE_ACK", "0") == "1"
    if not acknowledged:
        raise RuntimeError(
            "DINOv2 bundle is marked 'other' by Kaggle. Review its license "
            "and competition rules, then set RSNA_DINOV2_LICENSE_ACK=1."
        )


def _complete_data_root(path: Path) -> Path:
    required = ("train.csv", "test.csv", "train_series.csv", "test_series.csv")
    candidates = [path]
    if path.is_dir():
        candidates.extend(parent for parent in path.rglob("train.csv") for parent in (parent.parent,))
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(f"complete competition data root not found below {path}")


def main() -> int:
    _assert_license_ack()
    data_root = _complete_data_root(DATA_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"bundle: {BUNDLE_ROOT}", flush=True)
    print(f"data:   {data_root}", flush=True)
    print(f"device: {device}", flush=True)

    _, _, test, _, _ = load_tables(data_root)
    test_studies = test[UID].astype(str).tolist()
    if LIMIT > 0:
        test_studies = test_studies[:LIMIT]
    print(f"test studies: {len(test_studies)}", flush=True)

    run_dir = WORK_ROOT / "run_dinov2"
    run_dir.mkdir(parents=True, exist_ok=True)
    encoder = OfflineDinoV2Encoder(device, BUNDLE_ROOT)
    cache = cache_embeddings(
        data_root,
        run_dir,
        test_studies,
        "test",
        encoder,
        _logger(),
        limit=len(test_studies),
    )
    print(f"cache: {cache}", flush=True)
    if cache["failed_studies"] or cache["studies"] != len(test_studies):
        raise RuntimeError(f"embedding cache incomplete: {cache}")

    records = json.loads((run_dir / "embeddings" / "test" / "index.json").read_text())
    records = [{**row, "path": str(run_dir / row["file"])} for row in records]
    dummy = np.zeros((len(records), len(LABELS)), dtype=np.float32)
    loader = DataLoader(
        EmbeddingDataset(records, dummy),
        batch_size=1,
        shuffle=False,
        collate_fn=_collate,
    )

    config_path = BUNDLE_ROOT / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {"folds": 4, "pooling": "attention"}
    fold_preds: list[np.ndarray] = []
    for fold in range(1, int(config.get("folds", 4)) + 1):
        checkpoint = BUNDLE_ROOT / "weights" / f"fold_{fold}.pt"
        if not checkpoint.is_file():
            print(f"missing checkpoint: {checkpoint}", flush=True)
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = StudyMIL(pooling=str(config.get("pooling", "attention"))).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        predictions = []
        with torch.inference_mode():
            for embeddings, _, mask in loader:
                logits = model(embeddings.to(device), mask.to(device))
                predictions.append(torch.sigmoid(logits).cpu().numpy()[0])
        fold_preds.append(np.stack(predictions))
        print(f"fold {fold} done", flush=True)

    if not fold_preds:
        raise RuntimeError("no DINOv2 MIL fold checkpoints loaded")
    probabilities = np.mean(np.stack(fold_preds), axis=0)
    if probabilities.shape != (len(test_studies), len(LABELS)):
        raise RuntimeError(f"unexpected prediction shape: {probabilities.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([UID, *LABELS])
        for study_uid, row in zip(test_studies, probabilities):
            writer.writerow([study_uid, *(f"{float(value):.6f}" for value in row)])
    print(f"wrote {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
