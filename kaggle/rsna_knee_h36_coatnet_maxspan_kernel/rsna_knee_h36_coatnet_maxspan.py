#!/usr/bin/env python3
"""H-36: CoAtNet-384 trained on the public Max-Span corpus.

This is an independent inference implementation for the RSNA Knee competition.
The model checkpoint is supplied by the CC0-1.0 dataset
dreaddevelopment/raptor-knee-maxspan. The inference contract recorded in
that dataset is:

* CoAtNet RMLP 2 RW at 384 pixels;
* five slots with 64 slices per study;
* a physical 140 mm central crop;
* dense evaluation of the 62 three-slice windows available in a 64-slice bag;
* finding-specific attention pooling followed by percentile ranks.

The code deliberately does not use private kernel outputs, RadImageNet,
DINOv3, report text at test time, or internet access.

#Kaggle #Tecnologia #Academia #JoaoVictor
"""

from __future__ import annotations

import contextlib
import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom.pixel_data_handlers.util import apply_modality_lut


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

CHECKPOINT_NAME = "raptor_ft_coatnet_v5_full_swa.pt"
EXPECTED_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
EXPECTED_RESOLUTION = 384
IMAGE_SIZE = 336
CROP_MM = 140.0
SLOT_SLICES = (
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
)
TOTAL_SLICES = sum(item[2] for item in SLOT_SLICES)
EVAL_WINDOWS = TOTAL_SLICES - 2
SPAN = (0.02, 0.98)
WINDOW_BATCHES = (16, 8, 4)
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def log(message: str) -> None:
    print(f"[H36 {time.time() - START:8.1f}s] {message}", flush=True)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _positive_float(value: object, default: float = 0.5) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) and value > 0 else default
    except (TypeError, ValueError):
        return default


def find_competition_root() -> Path:
    override = os.environ.get("RSNA_KNEE_ROOT")
    if override:
        root = Path(override)
        if (root / "test.csv").is_file():
            return root
        raise FileNotFoundError(f"RSNA_KNEE_ROOT has no test.csv: {root}")

    candidates = (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    )
    for root in candidates:
        if (root / "test.csv").is_file():
            return root

    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for parent in (input_root, input_root / "competitions", input_root / "datasets"):
            if not parent.is_dir():
                continue
            for root in parent.iterdir():
                if root.is_dir() and (root / "test.csv").is_file():
                    if (root / "test_series.csv").is_file():
                        return root
    raise FileNotFoundError("RSNA competition mount with test.csv was not found")


def find_checkpoint() -> Path:
    override = os.environ.get("RSNA_KNEE_WEIGHT")
    if override:
        path = Path(override)
        if path.is_file():
            return path
        raise FileNotFoundError(f"RSNA_KNEE_WEIGHT is not a file: {path}")

    input_root = Path("/kaggle/input")
    direct = (
        input_root / "raptor-knee-maxspan" / CHECKPOINT_NAME,
        input_root / "raptor-knee-maxspan" / "1" / CHECKPOINT_NAME,
        input_root / "datasets/dreaddevelopment/raptor-knee-maxspan" / CHECKPOINT_NAME,
        input_root / "datasets/dreaddevelopment/raptor-knee-maxspan/1" / CHECKPOINT_NAME,
    )
    for path in direct:
        if path.is_file():
            return path

    if input_root.is_dir():
        for top in sorted(input_root.iterdir()):
            if not top.is_dir() or "competition" in top.name.lower():
                continue
            for path in top.glob(f"**/{CHECKPOINT_NAME}"):
                if path.is_file():
                    return path
    raise FileNotFoundError(f"{CHECKPOINT_NAME} was not found in Kaggle inputs")


def build_backbone(arch: str) -> nn.Module:
    # CoAtNet is a hybrid: use average pooling, never a ViT token path.
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(
        token in arch for token in ("vit", "deit", "dinov2", "eva", "beit")
    )
    kwargs = {"pretrained": False, "num_classes": 0, "in_chans": 3}
    if is_vit:
        kwargs.update(global_pool="token", dynamic_img_size=True)
    else:
        kwargs.update(global_pool="avg")
    return timm.create_model(arch, **kwargs)


class CoAtNetClassifier(nn.Module):
    """Backbone plus one attention pooling distribution per finding."""

    def __init__(self, backbone: nn.Module, feature_dim: int, n_targets: int = 12):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(feature_dim)
        self.att = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, n_targets),
        )
        self.clsW = nn.Parameter(torch.zeros(n_targets, feature_dim))
        self.clsb = nn.Parameter(torch.zeros(n_targets))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        batch, windows = images.shape[:2]
        features = self.backbone(images.flatten(0, 1))
        return features.view(batch, windows, -1)

    def head(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(features)
        attention = torch.softmax(self.att(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(images))


def load_model(checkpoint_path: Path, device: torch.device) -> CoAtNetClassifier:
    log(f"loading checkpoint {checkpoint_path.name} ({checkpoint_path.stat().st_size:,} bytes)")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    arch = checkpoint.get("arch", EXPECTED_ARCH)
    resolution = int(checkpoint.get("res", EXPECTED_RESOLUTION))
    labels = checkpoint.get("lab")
    if arch != EXPECTED_ARCH:
        raise RuntimeError(f"checkpoint architecture drift: {arch!r} != {EXPECTED_ARCH!r}")
    if resolution != EXPECTED_RESOLUTION:
        raise RuntimeError(
            f"checkpoint resolution drift: {resolution} != {EXPECTED_RESOLUTION}"
        )
    if labels is not None and list(labels) != TARGETS:
        raise RuntimeError("checkpoint target order differs from the competition schema")

    backbone = build_backbone(arch)
    model = CoAtNetClassifier(backbone, feature_dim=backbone.num_features)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    log(
        "model ready: "
        f"arch={arch}, features={backbone.num_features}, resolution={resolution}, "
        f"gold_auc={checkpoint.get('gold_auc', 'n/a')}"
    )
    del checkpoint
    gc.collect()
    return model


def order_series(series_dir: Path) -> tuple[list[tuple[Path, float]], float]:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        files = sorted(path for path in series_dir.iterdir() if path.is_file())
    records: list[tuple[float, Path, float]] = []
    spacings: list[float] = []
    for position, path in enumerate(files):
        coordinate = float(position)
        spacing = 0.5
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
                row = np.asarray(iop[:3], dtype=np.float64)
                column = np.asarray(iop[3:6], dtype=np.float64)
                coordinate = float(np.dot(np.asarray(ipp[:3], dtype=np.float64), np.cross(row, column)))
            else:
                coordinate = float(getattr(header, "InstanceNumber", position) or position)
            pixel_spacing = getattr(header, "PixelSpacing", None)
            if pixel_spacing is not None and len(pixel_spacing):
                spacing = _positive_float(pixel_spacing[0])
            spacings.append(spacing)
        except Exception:
            pass
        records.append((coordinate, path, spacing))

    records.sort(key=lambda item: item[0])
    median_spacing = float(np.median(spacings)) if spacings else 0.5
    return ([(path, spacing) for _, path, spacing in records], median_spacing)


def read_pixels(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path, force=True)
    pixels = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    return pixels


def crop_resize(image: np.ndarray, pixel_spacing: float) -> np.ndarray:
    height, width = image.shape
    crop_pixels = int(round(CROP_MM / _positive_float(pixel_spacing)))
    crop_pixels = min(crop_pixels, height, width)
    if crop_pixels > 16 and crop_pixels < min(height, width):
        y0 = (height - crop_pixels) // 2
        x0 = (width - crop_pixels) // 2
        image = image[y0 : y0 + crop_pixels, x0 : x0 + crop_pixels]
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def series_directory(root: Path, study_uid: str, series_uid: str) -> Path:
    candidates = (
        root / "test_series" / study_uid / series_uid,
        root / "test_series" / series_uid,
        root / "test_images" / study_uid / series_uid,
        root / "test_dicom" / study_uid / series_uid,
        root / "test_dicoms" / study_uid / series_uid,
    )
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"missing DICOM series {study_uid}/{series_uid}")


def choose_series(
    rows: list[dict[str, object]],
    plane: str,
    fluid: int,
    used: set[str],
) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("Anatomical_Plane", "")) == plane
        and str(row.get("SeriesInstanceUID", "")) not in used
    ]
    if fluid in (0, 1):
        preferred = [
            row for row in candidates if _as_int(row.get("Fluid_Sensitive")) == fluid
        ]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def build_study(
    root: Path,
    study_uid: str,
    records_by_study: dict[str, list[dict[str, object]]],
) -> tuple[np.ndarray, np.ndarray]:
    volume = np.zeros((TOTAL_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = np.zeros(TOTAL_SLICES, dtype=np.uint8)
    rows = records_by_study.get(study_uid, [])
    used: set[str] = set()
    cursor = 0

    for plane, fluid, count in SLOT_SLICES:
        selected = choose_series(rows, plane, fluid, used)
        if selected is None:
            cursor += count
            continue

        series_uid = str(selected["SeriesInstanceUID"])
        used.add(series_uid)
        files, median_spacing = order_series(
            series_directory(root, study_uid, series_uid)
        )
        if not files:
            cursor += count
            continue

        n_files = len(files)
        if n_files == 1:
            picks = np.zeros(count, dtype=np.int64)
        else:
            start = max(0, int(n_files * SPAN[0]))
            stop = min(n_files - 1, max(start, int(n_files * SPAN[1]) - 1))
            picks = np.linspace(start, stop, count).round().astype(np.int64)

        arrays: list[np.ndarray | None] = []
        spacings: list[float] = []
        for pick in picks:
            path, spacing = files[min(int(pick), n_files - 1)]
            try:
                arrays.append(read_pixels(path))
                spacings.append(spacing)
            except Exception:
                arrays.append(None)
                spacings.append(median_spacing)

        valid = [array for array in arrays if array is not None]
        if valid:
            intensities = np.concatenate([array.ravel() for array in valid])
            low, high = np.percentile(intensities, [2.0, 98.0])
        else:
            low, high = 0.0, 1.0

        for array, spacing in zip(arrays, spacings):
            if array is not None:
                normalised = np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)
                volume[cursor] = np.rint(
                    crop_resize(normalised, spacing) * 255.0
                ).astype(np.uint8)
                mask[cursor] = 1
            cursor += 1

    return volume, mask


def evaluation_centers(mask: np.ndarray, depth: int, count: int) -> list[int]:
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    low, high = int(valid.min()), int(valid.max())
    centers = [
        center
        for center in range(low + 1, high)
        if center - 1 >= low and center + 1 <= high
    ]
    if not centers:
        centers = [max(1, min((low + high) // 2, depth - 2))]
    indices = np.linspace(0, len(centers) - 1, count).round().astype(int)
    return [centers[index] for index in indices]


def make_windows(
    volume: np.ndarray,
    mask: np.ndarray,
    count: int,
    resolution: int,
) -> torch.Tensor:
    centers = evaluation_centers(mask, volume.shape[0], count)
    windows = np.empty((len(centers), 3, resolution, resolution), dtype=np.float32)
    for index, center in enumerate(centers):
        center = max(1, min(int(center), volume.shape[0] - 2))
        triplet = (
            np.stack(
                [volume[center - 1], volume[center], volume[center + 1]], axis=0
            ).astype(np.float32)
            / 255.0
        )
        tensor = torch.from_numpy(triplet)
        if tensor.shape[-1] != resolution:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )[0]
        windows[index] = tensor.numpy()
    output = torch.from_numpy(windows)
    return (output - IMAGE_MEAN) / IMAGE_STD


def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def infer_probabilities(
    model: CoAtNetClassifier,
    windows: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Encode in safe chunks, then pool all windows jointly."""
    last_error: RuntimeError | None = None
    for batch_size in WINDOW_BATCHES:
        try:
            parts: list[torch.Tensor] = []
            for start in range(0, len(windows), batch_size):
                batch = windows[start : start + batch_size].to(
                    device, non_blocking=True
                )
                with _autocast(device):
                    features = model.backbone(batch)
                parts.append(features.float())
            features = torch.cat(parts, dim=0).unsqueeze(0)
            logits = model.head(features)
            return torch.sigmoid(logits.float())[0].cpu().numpy()
        except RuntimeError as error:
            last_error = error
            is_oom = device.type == "cuda" and "out of memory" in str(error).lower()
            if not is_oom:
                raise
            log(f"CUDA memory retry: window batch {batch_size} failed")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    assert last_error is not None
    raise last_error


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort(axis=0).argsort(axis=0).astype(np.float64)
    return order / max(1, values.shape[0] - 1)


def write_submission(
    values: np.ndarray,
    test_ids: list[str],
    sample_columns: list[str],
) -> Path:
    output = pd.DataFrame(values.astype(np.float32), columns=TARGETS)
    output.insert(0, "StudyInstanceUID", test_ids)
    output = output[sample_columns]
    if output.columns.tolist() != sample_columns:
        raise RuntimeError("submission column order drift")
    if output["StudyInstanceUID"].tolist() != test_ids:
        raise RuntimeError("submission row order drift")
    if output["StudyInstanceUID"].duplicated().any():
        raise RuntimeError("duplicate StudyInstanceUID in submission")
    numeric = output[TARGETS].to_numpy()
    if not np.isfinite(numeric).all() or numeric.min() < 0.0 or numeric.max() > 1.0:
        raise RuntimeError("submission contains invalid numeric values")

    default_path = (
        Path("/kaggle/working/submission.csv")
        if Path("/kaggle/working").is_dir()
        else Path("submission.csv")
    )
    destination = Path(os.environ.get("RSNA_KNEE_OUTPUT", str(default_path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination


def main() -> None:
    root = find_competition_root()
    checkpoint_path = find_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"root={root}")
    log(
        f"device={device}; "
        f"cuda_count={torch.cuda.device_count() if device.type == 'cuda' else 0}"
    )

    test = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(
        root / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    sample = pd.read_csv(root / "sample_submission.csv", nrows=0)
    sample_columns = sample.columns.tolist()
    expected_columns = ["StudyInstanceUID"] + TARGETS
    if sample_columns != expected_columns:
        raise RuntimeError(
            f"competition sample schema drift: {sample_columns} != {expected_columns}"
        )

    test_ids = test["StudyInstanceUID"].astype(str).tolist()
    records_by_study = {
        str(uid): group.to_dict("records")
        for uid, group in series.groupby("StudyInstanceUID", sort=False)
    }
    limit = _as_int(os.environ.get("RSNA_KNEE_LIMIT"), 0)
    if limit > 0:
        test_ids = test_ids[:limit]
        log(f"local smoke limit active: {limit} studies")

    model = load_model(checkpoint_path, device)
    raw = np.full((len(test_ids), len(TARGETS)), 0.5, dtype=np.float32)
    decoded = 0
    failures = 0
    started = time.time()
    for index, study_uid in enumerate(test_ids):
        try:
            volume, mask = build_study(root, study_uid, records_by_study)
            decoded += int(mask.any())
            windows = make_windows(
                volume,
                mask,
                count=EVAL_WINDOWS,
                resolution=EXPECTED_RESOLUTION,
            )
            raw[index] = infer_probabilities(model, windows, device)
        except Exception as error:
            failures += 1
            log(
                f"study {index + 1}/{len(test_ids)} fallback 0.5: "
                f"{type(error).__name__}: {error}"
            )
        if (index + 1) % 100 == 0 or index + 1 == len(test_ids):
            elapsed = time.time() - started
            eta = elapsed / (index + 1) * (len(test_ids) - index - 1)
            log(
                f"studies {index + 1}/{len(test_ids)}; decoded={decoded}; "
                f"failures={failures}; eta={eta / 60:.1f} min"
            )

    if failures and len(test_ids) >= 4000:
        raise RuntimeError(
            f"{failures} hidden studies failed; refusing silent partial output"
        )
    if not np.isfinite(raw).all():
        raise RuntimeError("raw model output contains NaN or infinity")

    ranked = percentile_ranks(np.clip(raw, 0.0, 1.0)).astype(np.float32)
    destination = write_submission(ranked, test_ids, sample_columns)
    log(
        f"wrote {destination}: rows={len(test_ids)}, cols={len(sample_columns)}, "
        f"decoded={decoded}, failures={failures}, "
        f"raw_range=({raw.min():.5f},{raw.max():.5f})"
    )
    print(pd.read_csv(destination).head().to_string(index=False), flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


START = time.time()

if __name__ == "__main__":
    main()
