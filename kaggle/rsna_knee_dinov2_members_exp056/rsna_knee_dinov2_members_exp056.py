"""RSNA Knee — DINOv2 public members, exp056-compatible test kernel.

This standalone Kaggle entrypoint uses only the competition test DICOMs and
the two public CC0 checkpoint datasets:

* stevenleehans/rsna-knee-champ-members-only
* stevenleehans/rsna-knee-llm199-e30-members

The checkpoints persist the full DINOv2-Small encoder and a six-slot
SlotHead. The recovered public recipe uses 224 px, a 130 mm physical crop,
three adjacent slices per input and three center groups per slot. The final
candidate is the public exp056-style rank blend: 20% champ + 80% llm199e30.
No train labels, report text or private kernel output is used at inference.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F


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
SLOT_SPECS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
N_SLOT = len(SLOT_SPECS)
GROUP = 3
N_GROUP = 3
SIZE = 224
CACHE_VERSION = "h42-test-cache-v2"
CROP_MM = 130.0
WINDOW = (0.35, 0.65)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b",
    re.IGNORECASE,
)
T1_RX = re.compile(r"\bt1\b|\bt1w\b", re.IGNORECASE)
T2_RX = re.compile(r"\bt2\b|\bt2w\b", re.IGNORECASE)
PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens", re.IGNORECASE)
SEPARATORS_RX = re.compile(r"[_\-.]")
cv2.setNumThreads(1)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def find_data_root() -> Path:
    configured = os.environ.get("RSNA_DATA_ROOT")
    if configured:
        candidate = Path(configured).expanduser()
        if (candidate / "test.csv").is_file() and (candidate / "test_series.csv").is_file():
            return candidate
        raise FileNotFoundError(f"RSNA_DATA_ROOT inválido: {candidate}")
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for candidate in candidates:
        if (candidate / "test.csv").is_file() and (candidate / "test_series.csv").is_file():
            return candidate
    root = Path("/kaggle/input")
    for candidate in sorted(root.iterdir() if root.is_dir() else []):
        if (candidate / "test.csv").is_file() and (candidate / "test_series.csv").is_file():
            return candidate
    raise FileNotFoundError("competition test.csv/test_series.csv não encontrados")


def find_checkpoints(pattern: str) -> list[Path]:
    configured = os.environ.get("RSNA_CHECKPOINT_ROOT")
    if configured:
        direct = sorted(Path(configured).expanduser().rglob(pattern))
        if len(direct) == 5:
            return direct
        raise FileNotFoundError(f"RSNA_CHECKPOINT_ROOT={configured}: encontrei {len(direct)} arquivos para {pattern}")
    root = Path("/kaggle/input")
    dataset_name = {
        "champ_fold*.pt": "rsna-knee-champ-members-only",
        "llm199e30_fold*.pt": "rsna-knee-llm199-e30-members",
    }.get(pattern)
    if dataset_name is None:
        raise ValueError(f"padrão de checkpoint não suportado: {pattern}")

    # A API do Kaggle pode montar um dataset como
    # /kaggle/input/datasets/<owner>/<slug>/<version>/..., em vez de expor o
    # slug diretamente na raiz. Descobrimos apenas diretórios cujo nome
    # contém o slug e fazemos rglob somente dentro deles; nunca percorremos
    # a árvore de DICOM da competição à procura de *.pt.
    roots: set[Path] = set()
    if root.is_dir():
        frontier = [root]
        skip = {"competition", "competitions", "train_series", "test_series"}
        for depth in range(5):
            next_frontier: list[Path] = []
            for parent in frontier:
                try:
                    children = sorted(path for path in parent.iterdir() if path.is_dir())
                except OSError:
                    continue
                for child in children:
                    child_name = child.name.casefold()
                    if dataset_name in child_name:
                        roots.add(child)
                        continue
                    if child_name in skip:
                        continue
                    # Descend through the shallow API container hierarchy
                    # (root/owner/slug/version), not through image trees.
                    if depth == 0 or parent.name.casefold() == "datasets" or "dataset" in parent.name.casefold():
                        next_frontier.append(child)
            frontier = next_frontier

    unique: set[Path] = set()
    for candidate_root in sorted(roots):
        unique.update(path.resolve() for path in candidate_root.rglob(pattern))
    unique = sorted(unique)
    if len(unique) != 5:
        raise FileNotFoundError(f"checkpoint pattern={pattern}: encontrei {len(unique)} arquivos")
    return unique


def as_float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def header_value(header: pydicom.dataset.FileDataset, name: str) -> str:
    value = getattr(header, name, "")
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(item) for item in value)
    return str(value or "")


def descriptor(record: dict[str, object]) -> str:
    value = " ".join(str(record.get(field, "")) for field in ("SeriesDescription", "ProtocolName", "SequenceName"))
    return SEPARATORS_RX.sub(" ", value).lower()


def annotate(record: dict[str, object]) -> dict[str, object]:
    text = descriptor(record)
    options = {token.strip() for token in str(record.get("ScanOptions", "")).upper().split("|")}
    fatsat = bool(FATSAT_RX.search(text)) or bool(options & FATSAT_OPTS)
    tr = as_float(record.get("RepetitionTime"))
    te = as_float(record.get("EchoTime"))
    scanning = str(record.get("ScanningSequence", "")).upper()
    gre = "GR" in scanning
    t1 = bool(T1_RX.search(text))
    t2 = bool(T2_RX.search(text))
    pdw = bool(PD_RX.search(text))
    if t1 and not t2 and not pdw:
        weight = "T1"
    elif t2 and not pdw:
        weight = "T2"
    elif pdw:
        weight = "PD"
    elif gre:
        weight = "GRE"
    elif tr is not None and tr < 800:
        weight = "T1"
    elif te is not None and te > 60:
        weight = "T2"
    elif tr is not None and tr >= 800:
        weight = "PD"
    else:
        weight = "UNK"
    result = dict(record)
    result["fatsat"] = fatsat
    result["weight"] = weight
    result["fluid"] = weight in {"PD", "T2"}
    return result


def probe_series(row: dict[str, object], series_root: Path) -> dict[str, object]:
    study_uid = str(row["StudyInstanceUID"])
    series_uid = str(row["SeriesInstanceUID"])
    directory = series_root / study_uid / series_uid
    files = sorted(path.name for path in directory.glob("*.dcm"))
    record: dict[str, object] = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "plane": str(row.get("Anatomical_Plane", "")),
        "files": files,
        "n_slices": len(files),
        "pixel_spacing": None,
    }
    if not files:
        return annotate(record)
    try:
        header = pydicom.dcmread(directory / files[len(files) // 2], stop_before_pixels=True, force=True)
        for field in (
            "SeriesDescription",
            "ProtocolName",
            "SequenceName",
            "ScanOptions",
            "ScanningSequence",
            "RepetitionTime",
            "EchoTime",
        ):
            record[field] = header_value(header, field)
        spacing = getattr(header, "PixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            record["pixel_spacing"] = [float(spacing[0]), float(spacing[1])]
    except Exception as exc:
        record["probe_error"] = type(exc).__name__
    return annotate(record)


def select_slots(records: list[dict[str, object]], study_ids: list[str]) -> dict[str, dict[str, dict[str, object] | None]]:
    by_study: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_study.setdefault(str(record["study_uid"]), []).append(record)
    output: dict[str, dict[str, dict[str, object] | None]] = {}
    for study_uid in study_ids:
        candidates = by_study.get(study_uid, [])
        slots: dict[str, dict[str, object] | None] = {}
        for name, plane, fluid, fatsat in SLOT_SPECS:
            matching = [
                record
                for record in candidates
                if record.get("plane") == plane
                and bool(record.get("fluid")) == fluid
                and bool(record.get("fatsat")) == fatsat
                and int(record.get("n_slices", 0)) > 0
            ]
            matching.sort(key=lambda item: (-int(item["n_slices"]), str(item["series_uid"])))
            slots[name] = matching[0] if matching else None
        output[study_uid] = slots
    return output


def order_files(directory: Path, files: list[str]) -> list[str]:
    rows: list[tuple[float | None, float | None, str]] = []
    for name in files:
        position = None
        instance = None
        try:
            header = pydicom.dcmread(
                directory / name,
                stop_before_pixels=True,
                force=True,
                specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"],
            )
            instance = as_float(getattr(header, "InstanceNumber", None))
            ipp = np.asarray(getattr(header, "ImagePositionPatient"), dtype=float)
            iop = np.asarray(getattr(header, "ImageOrientationPatient"), dtype=float)
            if ipp.size >= 3 and iop.size >= 6 and np.isfinite(ipp[:3]).all() and np.isfinite(iop[:6]).all():
                position = float(np.dot(ipp[:3], np.cross(iop[:3], iop[3:6])))
        except Exception:
            pass
        rows.append((position, instance, name))
    if all(item[0] is not None for item in rows):
        rows.sort(key=lambda item: (float(item[0]), item[2]))
    elif all(item[1] is not None for item in rows):
        rows.sort(key=lambda item: (float(item[1]), item[2]))
    else:
        rows.sort(key=lambda item: item[2])
    return [item[2] for item in rows]


def read_pixels(path: Path) -> tuple[np.ndarray, tuple[float, float] | None]:
    dataset = pydicom.dcmread(path, force=True)
    pixels = dataset.pixel_array.astype(np.float32)
    slope = as_float(getattr(dataset, "RescaleSlope", 1.0), 1.0) or 1.0
    intercept = as_float(getattr(dataset, "RescaleIntercept", 0.0), 0.0) or 0.0
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = float(np.max(pixels)) - pixels
    spacing = getattr(dataset, "PixelSpacing", None)
    if spacing is None or len(spacing) < 2:
        return pixels, None
    return pixels, (float(spacing[0]), float(spacing[1]))


def crop_center(volume: np.ndarray, spacing: list[float] | tuple[float, float] | None) -> np.ndarray:
    if spacing is None or len(spacing) < 2:
        return volume
    rows = min(volume.shape[-2], max(1, int(round(CROP_MM / max(float(spacing[0]), 1e-3)))))
    cols = min(volume.shape[-1], max(1, int(round(CROP_MM / max(float(spacing[1]), 1e-3)))))
    y0 = max(0, (volume.shape[-2] - rows) // 2)
    x0 = max(0, (volume.shape[-1] - cols) // 2)
    return volume[..., y0 : y0 + rows, x0 : x0 + cols]


def render_group(raw: list[np.ndarray], spacings: list[tuple[float, float] | None], common: object) -> np.ndarray:
    shape = raw[0].shape
    if any(array.shape != shape for array in raw):
        raise ValueError("slab com dimensões divergentes")
    spacing = common if isinstance(common, list) else (spacings[0] if spacings else None)
    volume = crop_center(np.stack(raw), spacing)
    low, high = np.percentile(volume, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    volume = np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32)
    resized = F.interpolate(torch.from_numpy(volume[None]), size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
    return np.rint(resized.numpy() * 255.0).clip(0, 255).astype(np.uint8)


def decode_slot_diagnostic(
    record: dict[str, object], series_root: Path
) -> tuple[np.ndarray | None, str | None]:
    directory = series_root / str(record["study_uid"]) / str(record["series_uid"])
    files = order_files(directory, list(record.get("files", [])))
    if not files:
        return None, "no_files"
    if len(files) < GROUP + 2:
        return None, "too_few_slices"
    centers = [int(round(value * (len(files) - 1))) for value in np.linspace(WINDOW[0], WINDOW[1], N_GROUP)]
    groups: list[np.ndarray] = []
    for center in centers:
        indices = [center - 1, center, center + 1]
        if min(indices) < 0 or max(indices) >= len(files):
            return None, "center_out_of_bounds"
        raw: list[np.ndarray] = []
        spacings: list[tuple[float, float] | None] = []
        try:
            for index in indices:
                pixels, spacing = read_pixels(directory / files[index])
                raw.append(pixels)
                spacings.append(spacing)
            groups.append(render_group(raw, spacings, record.get("pixel_spacing")))
        except FileNotFoundError:
            return None, "missing_dicom"
        except ValueError:
            return None, "shape_or_render_error"
        except Exception as exc:
            return None, f"dicom_error:{type(exc).__name__}"
    return np.stack(groups), None


def decode_slot(record: dict[str, object], series_root: Path) -> np.ndarray | None:
    decoded, _reason = decode_slot_diagnostic(record, series_root)
    return decoded


def decode_study(
    study_uid: str,
    selected_record: dict[str, dict[str, object] | None],
    series_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    images = np.zeros((N_SLOT, N_GROUP, GROUP, SIZE, SIZE), dtype=np.uint8)
    masks = np.zeros(N_SLOT, dtype=np.uint8)
    study_diagnostics: dict[str, object] = {"filled_slots": 0, "failures": {}}
    for slot_index, (name, _plane, _fluid, _fatsat) in enumerate(SLOT_SPECS):
        record = selected_record.get(name)
        if record is None:
            study_diagnostics["failures"][name] = "no_matching_series"
            continue
        decoded, reason = decode_slot_diagnostic(record, series_root)
        if decoded is not None and decoded.shape == images[slot_index].shape:
            images[slot_index] = decoded
            masks[slot_index] = 1
            study_diagnostics["filled_slots"] = int(study_diagnostics["filled_slots"]) + 1
        else:
            study_diagnostics["failures"][name] = reason or "unexpected_shape"
    return images, masks, study_diagnostics


def build_test_images(
    study_ids: list[str],
    selected: dict[str, dict[str, dict[str, object] | None]],
    series_root: Path,
    include_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    images = np.zeros((len(study_ids), N_SLOT, N_GROUP, GROUP, SIZE, SIZE), dtype=np.uint8)
    masks = np.zeros((len(study_ids), N_SLOT), dtype=np.uint8)
    diagnostics: dict[str, object] = {}
    for study_index, study_uid in enumerate(study_ids):
        decoded_images, decoded_mask, study_diagnostics = decode_study(
            study_uid, selected[study_uid], series_root
        )
        images[study_index] = decoded_images
        masks[study_index] = decoded_mask
        diagnostics[study_uid] = study_diagnostics
        log(
            f"study {study_index + 1}/{len(study_ids)} "
            f"slots={int(masks[study_index].sum())} "
            f"failures={len(study_diagnostics['failures'])}"
        )
    if include_diagnostics:
        return images, masks, diagnostics
    return images, masks


def cache_path(cache_root: Path, study_uid: str) -> Path:
    digest = hashlib.sha256(study_uid.encode("utf-8")).hexdigest()[:24]
    return cache_root / f"study_{digest}.npz"


def load_cached_study(path: Path, study_uid: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as cached:
        cached_uid = str(cached["study_uid"].item())
        version = str(cached["cache_version"].item())
        images = np.asarray(cached["images"], dtype=np.uint8)
        masks = np.asarray(cached["masks"], dtype=np.uint8)
        diagnostic_json = str(cached["diagnostic_json"].item())
    expected_shape = (N_SLOT, N_GROUP, GROUP, SIZE, SIZE)
    if cached_uid != study_uid or version != CACHE_VERSION:
        raise ValueError(f"cache incompatível para {study_uid}")
    if images.shape != expected_shape or masks.shape != (N_SLOT,):
        raise ValueError(f"shape de cache inválido para {study_uid}: {images.shape}/{masks.shape}")
    diagnostics = json.loads(diagnostic_json)
    if not isinstance(diagnostics, dict):
        raise ValueError(f"diagnóstico de cache inválido para {study_uid}")
    return images, masks, diagnostics


def build_test_cache(
    study_ids: list[str],
    selected: dict[str, dict[str, dict[str, object] | None]],
    series_root: Path,
    cache_root: Path,
) -> tuple[list[int], dict[str, object]]:
    """Decode one study at a time and persist compact per-study arrays.

    This keeps the DICOM decode out of the model loop and bounds RAM by the
    inference batch. The cache is versioned so preprocessing changes cannot
    silently reuse stale tensors.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    coverage = np.zeros(N_SLOT, dtype=np.int64)
    diagnostics: dict[str, object] = {}
    for index, study_uid in enumerate(study_ids):
        path = cache_path(cache_root, study_uid)
        try:
            _images, masks, study_diagnostics = load_cached_study(path, study_uid)
        except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError):
            images, masks, study_diagnostics = decode_study(
                study_uid, selected[study_uid], series_root
            )
            np.savez_compressed(
                path,
                cache_version=np.asarray(CACHE_VERSION),
                study_uid=np.asarray(study_uid),
                images=images,
                masks=masks,
                diagnostic_json=np.asarray(json.dumps(study_diagnostics, ensure_ascii=False)),
            )
        coverage += masks.astype(np.int64)
        diagnostics[study_uid] = study_diagnostics
        log(
            f"study {index + 1}/{len(study_ids)} "
            f"slots={int(masks.sum())} "
            f"failures={len(study_diagnostics.get('failures', {}))}"
        )
    return coverage.astype(int).tolist(), diagnostics


def iter_cached_batches(
    study_ids: list[str], cache_root: Path, batch_size: int
):
    for start in range(0, len(study_ids), batch_size):
        stop = min(start + batch_size, len(study_ids))
        batch_images: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []
        for study_uid in study_ids[start:stop]:
            images, masks, _diagnostics = load_cached_study(cache_path(cache_root, study_uid), study_uid)
            batch_images.append(images)
            batch_masks.append(masks)
        yield start, stop, np.stack(batch_images), np.stack(batch_masks)


class EncoderWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size=518,
            patch_size=14,
            hidden_act="gelu",
            qkv_bias=True,
            layerscale_value=1.0,
            use_mask_token=True,
        )
        self.module = Dinov2Model(config)


class SlotHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(N_SLOT, 256) * 0.02)
        self.query = nn.Parameter(torch.randn(len(TARGETS), 256) * 0.02)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(256, len(TARGETS))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(features) + self.slot_emb.unsqueeze(0)
        attention = torch.einsum("bsh,oh->bos", hidden, self.query) / math.sqrt(256.0)
        attention = attention.masked_fill(~mask.bool().unsqueeze(1), -10000.0).softmax(dim=-1)
        context = self.drop(torch.einsum("bos,bsh->boh", attention, hidden))
        return (context * self.out.weight.unsqueeze(0)).sum(dim=-1) + self.out.bias.unsqueeze(0)


class Member(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = EncoderWrapper()
        self.head = SlotHead()
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slots, groups, channels, height, width = images.shape
        values = images.reshape(batch * slots * groups, channels, height, width).float().div_(255.0)
        values = (values - self.mean) / self.std
        tokens = self.encoder.module(pixel_values=values, interpolate_pos_encoding=True).last_hidden_state
        features = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
        features = features.reshape(batch, slots, groups, -1).mean(dim=2)
        return self.head(features, mask)


def infer_family(
    checkpoints: list[Path],
    images: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size precisa ser positivo")
    folds: list[np.ndarray] = []
    for path in checkpoints:
        log(f"loading {path.name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = Member()
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().to(device)
        with torch.no_grad():
            predictions: list[np.ndarray] = []
            for start in range(0, len(images), batch_size):
                stop = min(start + batch_size, len(images))
                values = torch.from_numpy(images[start:stop]).to(device)
                slot_mask = torch.from_numpy(masks[start:stop]).to(device)
                logits = model(values, slot_mask)
                predictions.append(torch.sigmoid(logits).float().cpu().numpy())
                del values, slot_mask, logits
                if start == 0 or stop == len(images) or stop % max(batch_size * 8, 1) == 0:
                    log(f"{path.name}: infer {stop}/{len(images)}")
            folds.append(np.concatenate(predictions, axis=0))
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return np.mean(np.stack(folds), axis=0)


def infer_family_cached(
    checkpoints: list[Path],
    study_ids: list[str],
    cache_root: Path,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Run a family from per-study cache without materializing all test images."""
    if batch_size < 1:
        raise ValueError("batch_size precisa ser positivo")
    folds: list[np.ndarray] = []
    for path in checkpoints:
        log(f"loading {path.name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = Member()
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().to(device)
        with torch.no_grad():
            predictions: list[np.ndarray] = []
            for start, stop, image_batch, mask_batch in iter_cached_batches(
                study_ids, cache_root, batch_size
            ):
                values = torch.from_numpy(image_batch).to(device)
                slot_mask = torch.from_numpy(mask_batch).to(device)
                logits = model(values, slot_mask)
                predictions.append(torch.sigmoid(logits).float().cpu().numpy())
                del image_batch, mask_batch, values, slot_mask, logits
                if start == 0 or stop == len(study_ids) or stop % max(batch_size * 8, 1) == 0:
                    log(f"{path.name}: infer {stop}/{len(study_ids)}")
            folds.append(np.concatenate(predictions, axis=0))
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return np.mean(np.stack(folds), axis=0)


def rank_columns(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(dtype=float)


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__.strip())
        return 0
    started = time.time()
    torch.set_grad_enabled(False)
    requested_device = os.environ.get("RSNA_DEVICE", "").strip().lower()
    if requested_device:
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    data_root = find_data_root()
    test = pd.read_csv(data_root / "test.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(data_root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    study_ids = test["StudyInstanceUID"].astype(str).tolist()
    series_root = data_root / "test_series"
    rows = series.to_dict("records")
    records = [probe_series(row, series_root) for row in rows]
    selected = select_slots(records, study_ids)
    batch_size = int(os.environ.get("RSNA_BATCH_SIZE", "8"))
    streaming = os.environ.get("RSNA_STREAMING", "1").strip().lower() not in {"0", "false", "no"}
    if streaming:
        cache_root = Path(
            os.environ.get("RSNA_CACHE_ROOT", "/kaggle/working/rsna_h42_test_cache")
        ).expanduser()
        slot_coverage, diagnostics = build_test_cache(
            study_ids, selected, series_root, cache_root
        )
        empty_studies = [
            uid for uid, item in diagnostics.items() if int(item.get("filled_slots", 0)) == 0
        ]
        if empty_studies:
            raise RuntimeError(
                f"nenhum slot DICOM decodificado para {len(empty_studies)} estudos: {empty_studies[:3]}"
            )
        log(
            f"device={device} test_studies={len(study_ids)} "
            f"coverage={slot_coverage} streaming_cache={cache_root}"
        )
        champ = infer_family_cached(
            find_checkpoints("champ_fold*.pt"), study_ids, cache_root, device, batch_size
        )
        llm199e30 = infer_family_cached(
            find_checkpoints("llm199e30_fold*.pt"), study_ids, cache_root, device, batch_size
        )
    else:
        built = build_test_images(study_ids, selected, series_root, include_diagnostics=True)
        images, masks, diagnostics = built
        slot_coverage = masks.sum(axis=0).astype(int).tolist()
        empty_studies = [uid for index, uid in enumerate(study_ids) if not masks[index].any()]
        if empty_studies:
            raise RuntimeError(
                f"nenhum slot DICOM decodificado para {len(empty_studies)} estudos: {empty_studies[:3]}"
            )
        log(f"device={device} test_studies={len(study_ids)} coverage={slot_coverage}")
        champ = infer_family(find_checkpoints("champ_fold*.pt"), images, masks, device, batch_size)
        llm199e30 = infer_family(find_checkpoints("llm199e30_fold*.pt"), images, masks, device, batch_size)
    candidate = rank_columns(0.20 * rank_columns(champ) + 0.80 * rank_columns(llm199e30))
    submission = pd.DataFrame({"StudyInstanceUID": study_ids})
    for index, target in enumerate(TARGETS):
        submission[target] = candidate[:, index]
    if list(submission.columns) != ["StudyInstanceUID", *TARGETS]:
        raise RuntimeError("schema de submissão inesperado")
    if len(submission) != len(test) or submission["StudyInstanceUID"].duplicated().any():
        raise RuntimeError("IDs de teste inválidos")
    values = submission[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0).all() and (values <= 1).all()):
        raise RuntimeError("predições inválidas")
    output = Path(os.environ.get("RSNA_OUTPUT_PATH", "/kaggle/working/submission.csv")).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    diagnostics_path = Path(
        os.environ.get("RSNA_DIAGNOSTICS_PATH", f"{output}.diagnostics.json")
    ).expanduser()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(
            {
                "study_count": len(study_ids),
                "slot_coverage": slot_coverage,
                "batch_size": batch_size,
                "streaming": streaming,
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"saved={output} shape={submission.shape} elapsed={time.time() - started:.1f}s")
    print(submission.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
