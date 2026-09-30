"""#RSNA #Kaggle #Pesquisa — gate local do WideDense no gold oficial.

O script reproduz a leitura do H-40/H-41 no conjunto de treino anotado: cinco
slots, crop físico de 140 mm, 64 fatias, janelas de avaliação e atenção por
alvo do ``RaptorClassifier``. O modelo nunca vê os rótulos durante a
inferência. A saída é uma auditoria local de complementaridade, não um score
de leaderboard nem uma validação OOF de treinamento.

Exemplo:

    python3 scripts/evaluate_widedense_gold.py \
      --data-dir data/raw \
      --model full=data/models/raptor-knee-widedense/raptor_ft_coatnet_v4_full.pt \
      --model maxspan=data/models/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt \
      --output reports/widedense_gold_20260902.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from sklearn.metrics import roc_auc_score


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
SLOTS = [
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
]
MAX_SLICES = sum(item[2] for item in SLOTS)
IMAGE_SIZE = 336
CROP_MM = 140.0
K_EVAL = 62
SPAN = (0.02, 0.98)
NORM = "imagenet"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
cv2.setNumThreads(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_files(series_dir: Path) -> list[tuple[Path, float]]:
    rows: list[tuple[float, Path, float]] = []
    for path in series_dir.glob("*.dcm"):
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                row = np.asarray(iop[:3], dtype=float)
                col = np.asarray(iop[3:], dtype=float)
                normal = np.cross(row, col)
                position = float(np.dot(np.asarray(ipp, dtype=float), normal))
            else:
                position = float(getattr(header, "InstanceNumber", 0) or 0)
            pixel_spacing = getattr(header, "PixelSpacing", None)
            spacing = float(pixel_spacing[0]) if pixel_spacing is not None else 0.5
            rows.append((position, path, spacing))
        except Exception:
            continue
    rows.sort(key=lambda item: item[0])
    return [(path, spacing) for _, path, spacing in rows]


def read_pixels(path: Path) -> tuple[np.ndarray, float] | None:
    try:
        dicom = pydicom.dcmread(str(path), force=True)
        image = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
        if str(getattr(dicom, "PhotometricInterpretation", "")) == "MONOCHROME1":
            image = image.max() - image
        pixel_spacing = getattr(dicom, "PixelSpacing", None)
        spacing = float(pixel_spacing[0]) if pixel_spacing is not None else CROP_MM / max(image.shape)
        return image, spacing
    except Exception:
        return None


def crop_resize(image: np.ndarray, spacing: float) -> np.ndarray:
    crop_px = int(round(CROP_MM / max(spacing, 1e-3)))
    crop_px = min(crop_px, min(image.shape))
    y0 = (image.shape[0] - crop_px) // 2
    x0 = (image.shape[1] - crop_px) // 2
    crop = image[y0 : y0 + crop_px, x0 : x0 + crop_px]
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def pick_series(rows: list[dict[str, object]], plane: str, fluid: int, used: set[str]) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if row.get("Anatomical_Plane") == plane
        and str(row.get("SeriesInstanceUID")) not in used
    ]
    if fluid in (0, 1):
        preferred = [row for row in candidates if int(row.get("Fluid_Sensitive", 0) or 0) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def build_study(
    study_uid: str,
    series_by_study: dict[str, list[dict[str, object]]],
    series_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    volume = np.zeros((MAX_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    offset = 0
    used: set[str] = set()
    selected: list[dict[str, object]] = []

    for plane, fluid, count in SLOTS:
        row = pick_series(series_by_study.get(study_uid, []), plane, fluid, used)
        selected.append(
            {
                "plane": plane,
                "fluid": fluid,
                "count": count,
                "series_uid": None if row is None else str(row.get("SeriesInstanceUID")),
            }
        )
        if row is None:
            offset += count
            continue

        series_uid = str(row["SeriesInstanceUID"])
        used.add(series_uid)
        files = ordered_files(series_root / study_uid / series_uid)
        if not files:
            offset += count
            continue

        number = len(files)
        low = int(number * SPAN[0])
        high = max(low, int(number * SPAN[1]) - 1)
        picks = np.linspace(low, high, count).round().astype(int) if number > 1 else np.zeros(count, dtype=int)
        picks = np.clip(picks, 0, number - 1)
        decoded: list[tuple[np.ndarray, float] | None] = []
        for pick in picks:
            decoded.append(read_pixels(files[int(pick)][0]))
        valid = [item[0] for item in decoded if item is not None]
        if valid:
            pixels = np.concatenate([item.ravel() for item in valid])
            intensity_low, intensity_high = np.percentile(pixels, [2.0, 98.0])
        else:
            intensity_low, intensity_high = 0.0, 1.0

        for local_index, item in enumerate(decoded):
            if item is None or offset + local_index >= MAX_SLICES:
                continue
            image, spacing = item
            normalized = np.clip(
                (image - intensity_low) / (intensity_high - intensity_low + 1e-6),
                0.0,
                1.0,
            )
            volume[offset + local_index] = (crop_resize(normalized, spacing) * 255).astype(np.uint8)
        offset += count

    mask = (volume.reshape(MAX_SLICES, -1).sum(axis=1) > 0).astype(np.uint8)
    return volume, mask, {"selected": selected, "valid_slices": int(mask.sum())}


def eval_centers(mask: np.ndarray, depth: int, count: int) -> list[int]:
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    low, high = int(valid.min()), int(valid.max())
    centers = [center for center in range(low + 1, high) if center - 1 >= low and center + 1 <= high]
    if not centers:
        centers = [max(1, min((low + high) // 2, depth - 2))]
    indices = np.linspace(0, len(centers) - 1, count).round().astype(int)
    return [centers[index] for index in indices]


def eval_windows(volume: np.ndarray, mask: np.ndarray, resolution: int) -> torch.Tensor:
    centers = eval_centers(mask, volume.shape[0], K_EVAL)
    windows = np.empty((len(centers), 3, resolution, resolution), dtype=np.float32)
    for index, center in enumerate(centers):
        center = max(1, min(center, volume.shape[0] - 2))
        tri = np.stack([volume[center - 1], volume[center], volume[center + 1]], axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(tri)
        if tensor.shape[-1] != resolution:
            tensor = F.interpolate(tensor[None], size=(resolution, resolution), mode="bilinear", align_corners=False)[0]
        windows[index] = tensor.numpy()
    output = torch.from_numpy(windows)
    if NORM == "imagenet":
        output = (output - MEAN) / STD
    return output


class RaptorClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, n_targets: int = 12, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(feature_dim)
        self.att = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
        )
        self.clsW = nn.Parameter(torch.zeros(n_targets, feature_dim))
        self.clsb = nn.Parameter(torch.zeros(n_targets))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def head(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        attention = torch.softmax(self.att(normalized), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, normalized)
        return (pooled * self.clsW).sum(-1) + self.clsb


def build_backbone(architecture: str) -> nn.Module:
    hybrid = architecture.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(key in architecture for key in ("vit", "deit", "dinov2", "eva", "beit"))
    kwargs = {"pretrained": False, "num_classes": 0, "in_chans": 3}
    if is_vit:
        kwargs.update(global_pool="token", dynamic_img_size=True)
    else:
        kwargs.update(global_pool="avg")
    return timm.create_model(architecture, **kwargs)


def load_model(path: Path, device: torch.device) -> tuple[RaptorClassifier, int, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = str(checkpoint.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"))
    resolution = int(checkpoint.get("res", 384))
    backbone = build_backbone(architecture)
    model = RaptorClassifier(backbone, feature_dim=backbone.num_features, n_targets=len(TARGETS))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    metadata = {
        "architecture": architecture,
        "resolution": resolution,
        "epoch": checkpoint.get("epoch"),
        "gold_auc": checkpoint.get("gold_auc"),
        "source": checkpoint.get("src"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "checkpoint_keys": sorted(checkpoint),
    }
    del checkpoint
    gc.collect()
    return model, resolution, metadata


@torch.no_grad()
def infer(model: RaptorClassifier, windows: torch.Tensor, device: torch.device, chunk_size: int) -> np.ndarray:
    features: list[torch.Tensor] = []
    for start in range(0, len(windows), chunk_size):
        chunk = windows[start : start + chunk_size].to(device)
        features.append(model.encode_images(chunk).float().cpu())
        del chunk
    all_features = torch.cat(features, dim=0).unsqueeze(0).to(device)
    probabilities = torch.sigmoid(model.head(all_features).float())[0].cpu().numpy()
    del all_features, features
    return probabilities


def parse_models(values: list[str], root: Path) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model deve usar nome=caminho: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed.append((name, path))
    if not parsed:
        raise ValueError("informe pelo menos um --model nome=caminho")
    return parsed


def compute_metrics(predictions: dict[str, dict[str, list[float]]], gold: pd.DataFrame) -> dict[str, object]:
    frame = gold.set_index("StudyInstanceUID")
    per_model: dict[str, object] = {}
    for model_name, by_study in predictions.items():
        rows = []
        for study_uid in frame.index.astype(str):
            rows.append(by_study[study_uid])
        matrix = np.asarray(rows, dtype=float)
        aucs: dict[str, float | None] = {}
        for index, target in enumerate(TARGETS):
            y = frame[target].to_numpy(dtype=float)
            if np.unique(y).size < 2:
                aucs[target] = None
            else:
                aucs[target] = float(roc_auc_score(y, matrix[:, index]))
        valid = [value for value in aucs.values() if value is not None]
        per_model[model_name] = {
            "macro_auc": float(np.mean(valid)) if valid else None,
            "auc_by_target": aucs,
            "n_studies": int(len(matrix)),
        }
    return per_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], help="nome=caminho do checkpoint; repetir para comparar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="limita estudos para smoke; omitir para os 58 gold")
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size deve ser positivo")

    root = args.data_dir.expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    train = pd.read_csv(root / "train.csv")
    gold = train[train[TARGETS].notna().all(axis=1)].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if args.limit is not None:
        gold = gold.head(args.limit).copy()
    series = pd.read_csv(root / "train_series.csv")
    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
    gold_ids = set(gold["StudyInstanceUID"])
    series_by_study = {
        study_uid: rows.to_dict("records")
        for study_uid, rows in series[series["StudyInstanceUID"].isin(gold_ids)].groupby("StudyInstanceUID")
    }
    series_root = root / "train_series"
    missing_series = sorted(gold_ids - set(series_by_study))
    if missing_series:
        raise RuntimeError(f"gold sem metadados de série: {missing_series[:3]}")

    if args.device == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS solicitado, mas indisponível")
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[1]
    model_specs = parse_models(args.model, project_root)
    loaded: list[tuple[str, Path, RaptorClassifier, int, dict[str, object]]] = []
    for name, path in model_specs:
        print(f"loading {name}: {path} on {device}", flush=True)
        model, resolution, metadata = load_model(path, device)
        loaded.append((name, path, model, resolution, metadata))

    started = time.time()
    predictions: dict[str, dict[str, list[float]]] = {name: {} for name, *_ in loaded}
    diagnostics: dict[str, dict[str, object]] = {}
    for index, study_uid in enumerate(gold["StudyInstanceUID"]):
        volume, mask, diagnostic = build_study(study_uid, series_by_study, series_root)
        diagnostics[study_uid] = diagnostic
        for name, _path, model, resolution, _metadata in loaded:
            windows = eval_windows(volume, mask, resolution)
            predictions[name][study_uid] = infer(model, windows, device, args.chunk_size).astype(float).tolist()
            del windows
        del volume, mask
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"study {index + 1}/{len(gold)} {study_uid} valid_slices={diagnostic['valid_slices']}", flush=True)

    result = {
        "format": "rsna-widedense-gold-gate-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "data_dir": str(root),
        "gold_studies": int(len(gold)),
        "gold_series": int(len(series[series["StudyInstanceUID"].isin(gold_ids)])),
        "preprocessing": {
            "slots": SLOTS,
            "max_slices": MAX_SLICES,
            "image_size": IMAGE_SIZE,
            "crop_mm": CROP_MM,
            "span": SPAN,
            "k_eval": K_EVAL,
            "normalization": NORM,
        },
        "models": {
            name: {"path": str(path), **metadata}
            for name, path, _model, _resolution, metadata in loaded
        },
        "predictions": predictions,
        "metrics": compute_metrics(predictions, gold),
        "diagnostics": diagnostics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
