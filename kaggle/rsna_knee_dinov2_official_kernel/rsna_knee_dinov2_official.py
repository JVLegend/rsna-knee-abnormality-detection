#!/usr/bin/env python3
"""DINOv2-S oficial + pooling por alvo para o RSNA Knee.

Esta é uma sonda de uma família nova, separada da referência H-23. O backbone
vem do modelo oficial MetaResearch no Kaggle, licença Apache 2.0; não usa o
bundle de terceiros com licença ``other``. O texto e os teachers são apenas
supervisão auxiliar no treino. A inferência final usa exclusivamente DICOM,
metadados de séries e o blend textual já validado.

O arquivo ``common.py`` é uma cópia versionada dos helpers de leitura DICOM,
teacher e pooling do kernel H-23. Mantê-lo no diretório deixa o kernel
autocontido e permite comparar a troca de backbone isoladamente.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    KEY_COLUMN,
    TARGET_COLUMNS,
    TARGETWISE_VISUAL_WEIGHTS,
    _external_teacher,
    _fit_target_view_model,
    _series_index,
    _weak_target_arrays,
    _resolve_data_dir,
    study_images,
    text_predictions,
    validate_submission,
)


def _find_model_dir(requested: Path | None = None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested.expanduser())
    configured = os.environ.get("RSNA_DINOV2_MODEL_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    root = Path("/kaggle/input")
    if root.is_dir():
        candidates.extend(
            path
            for pattern in (
                "*/pytorch/*/*",
                "*/*/*",
                "*/*",
            )
            for path in root.glob(pattern)
            if path.is_dir()
        )
    candidates.extend(
        [
            Path("/kaggle/input/dinov2/pytorch/small/1"),
            Path("/kaggle/input/dinov2_official_small"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "config.json").is_file() and (
            (candidate / "pytorch_model.bin").is_file()
            or (candidate / "model.safetensors").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Modelo DINOv2 oficial não encontrado; caminhos verificados="
        + ", ".join(str(path) for path in candidates[:30])
    )


def _device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"device inválido: {requested!r}")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas não está disponível.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS solicitado, mas não está disponível.")
    return requested


class DinoEncoder:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.device_name = _device(device)
        self.device = torch.device(self.device_name)
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None or int(hidden) != 384:
            raise ValueError(f"Esperava DINOv2-S com hidden_size=384; recebi {hidden}.")

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        if not images:
            return np.empty((0, 384), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = self.processor(
                images=images[start : start + batch_size],
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = self.model(**batch).last_hidden_state[:, 0]
            else:
                hidden = self.model(**batch).last_hidden_state[:, 0]
            outputs.append(hidden.float().cpu().numpy())
        matrix = np.vstack(outputs).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise FloatingPointError("DINOv2 gerou embedding não finito.")
        return matrix


def _collect_views(
    data_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    encoder: DinoEncoder,
    *,
    slice_profile: str,
    intensity_window: str,
    batch_size: int,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], dict[str, object]]:
    series_indexes = {"train": _series_index(train_series), "test": _series_index(test_series)}
    records = [("train", row, series_indexes["train"]) for _, row in train.iterrows()]
    records.extend(("test", row, series_indexes["test"]) for _, row in test.iterrows())
    pending: list[Image.Image] = []
    owners: list[int] = []
    views: list[list[np.ndarray]] = [[] for _ in records]
    valid: list[bool] = []
    views_seen = 0

    def flush() -> None:
        nonlocal views_seen
        if not pending:
            return
        values = encoder.encode(pending, batch_size=batch_size)
        for owner, value in zip(owners, values):
            views[owner].append(value)
        views_seen += len(pending)
        pending.clear()
        owners.clear()

    for position, (split, row, series) in enumerate(records, start=1):
        images, is_valid = study_images(
            data_dir,
            split,
            str(row[KEY_COLUMN]),
            series,
            slice_profile=slice_profile,
            intensity_window=intensity_window,
        )
        for image in images:
            pending.append(Image.fromarray(np.moveaxis(image, 0, -1)))
            owners.append(position - 1)
        valid.append(is_valid)
        if len(pending) >= batch_size:
            flush()
        if position % 25 == 0 or position == len(records):
            print(
                f"dino={position}/{len(records)} valid={sum(valid)} "
                f"views={views_seen + len(pending)}",
                flush=True,
            )
    flush()
    train_count = len(train)
    train_views = views[:train_count]
    test_views = views[train_count:]
    if not all(valid[:train_count]) or not all(valid[train_count:]):
        print(
            f"warning: estudos válidos train={sum(valid[:train_count])}/{train_count} "
            f"test={sum(valid[train_count:])}/{len(test)}",
            flush=True,
        )
    return train_views, test_views, {
        "device": encoder.device_name,
        "embedding_dim": 384,
        "views_total": views_seen,
        "views_mean_train": float(np.mean([len(row) for row in train_views])) if train_views else 0.0,
        "views_mean_test": float(np.mean([len(row) for row in test_views])) if test_views else 0.0,
        "slice_profile": slice_profile,
        "intensity_window": intensity_window,
        "valid_train": int(sum(valid[:train_count])),
        "valid_test": int(sum(valid[train_count:])),
    }


def run(
    data_dir: Path,
    output: Path,
    *,
    model_dir: Path | None,
    device: str,
    batch_size: int,
    visual_weight: float,
    slice_profile: str,
    intensity_window: str,
    external_labels_dir: Path | None,
    limit_studies: int,
) -> pd.DataFrame:
    if not 0 <= visual_weight <= 1:
        raise ValueError("visual_weight precisa estar entre 0 e 1.")
    if batch_size < 1 or limit_studies < 0:
        raise ValueError("batch_size/limit_studies inválidos.")
    started = time.perf_counter()
    data_dir = _resolve_data_dir(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train_series = pd.read_csv(data_dir / "train_series.csv")
    test_series = pd.read_csv(data_dir / "test_series.csv")
    if limit_studies:
        train = train.head(limit_studies).copy()
        test = test.head(min(1, len(test))).copy()

    teacher, label_dir = _external_teacher(train, external_labels_dir, "targetwise")
    weak_targets = _weak_target_arrays(train, teacher, threshold=0.85, sample_weight=0.10)
    text = text_predictions(train, test, train_series, test_series)
    model_path = _find_model_dir(model_dir)
    encoder = DinoEncoder(model_path, device)
    train_views, test_views, meta = _collect_views(
        data_dir,
        train,
        test,
        train_series,
        test_series,
        encoder,
        slice_profile=slice_profile,
        intensity_window=intensity_window,
        batch_size=batch_size,
    )

    visual = pd.DataFrame({KEY_COLUMN: test[KEY_COLUMN].astype(str).to_numpy()})
    for target in TARGET_COLUMNS:
        labels, sample_weights, included, soft_probabilities = weak_targets[target]
        fit_mask = included & np.asarray([bool(values) for values in train_views], dtype=bool)
        visual[target] = _fit_target_view_model(
            target,
            train_views,
            test_views,
            labels,
            fit_mask,
            sample_weights,
            target_pooling=None,
            soft_probabilities=soft_probabilities,
        )

    submission = text.copy()
    for target in TARGET_COLUMNS:
        alpha = visual_weight
        submission[target] = np.clip(
            (1.0 - alpha) * text[target].to_numpy(dtype=float)
            + alpha * visual[target].to_numpy(dtype=float),
            1e-6,
            1.0 - 1e-6,
        )
    validate_submission(submission, test)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(
        f"data_dir={data_dir} model_dir={model_path} label_dir={label_dir} "
        f"slice_profile={slice_profile} intensity_window={intensity_window} "
        f"visual_weight={visual_weight} meta={meta} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(f"submission gravada em {output} com {len(submission)} linhas", flush=True)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("RSNA_DATA_DIR", "/kaggle/input/rsna-knee-abnormality-detection"))
    parser.add_argument("--model-dir", default=os.environ.get("RSNA_DINOV2_MODEL_DIR"))
    parser.add_argument("--output", default=os.environ.get("RSNA_OUTPUT", "/kaggle/working/submission.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--visual-weight", type=float, default=0.4)
    parser.add_argument("--slice-profile", choices=("quantile3", "adjacent3", "dense6", "dense9"), default="adjacent3")
    parser.add_argument("--intensity-window", choices=("slice", "series"), default="series")
    parser.add_argument("--external-labels-dir", default=os.environ.get("RSNA_EXTERNAL_LABELS_DIR"))
    parser.add_argument("--limit-studies", type=int, default=0)
    args = parser.parse_args()
    run(
        Path(args.data_dir),
        Path(args.output),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        device=args.device,
        batch_size=args.batch_size,
        visual_weight=args.visual_weight,
        slice_profile=args.slice_profile,
        intensity_window=args.intensity_window,
        external_labels_dir=Path(args.external_labels_dir) if args.external_labels_dir else None,
        limit_studies=args.limit_studies,
    )


if __name__ == "__main__":
    main()
