#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — reproduz o head v16 no gold oficial.

O script reproduz a receita pública DINOsaur V4 para os 58 estudos com rótulo
oficial completo: seis slots ``plano x Fat_Suppression``, crop físico de
130 mm, 16 cortes, DINOv3 ``m_f0..m_f4`` e o ``SlotAttentionHead`` v16.

Ele mede o head isolado e o blend contra o DINOv3 no holdout local. O número é
um gate de direção, não uma estimativa do leaderboard: nenhum rótulo do teste
é usado e nenhum CSV de submissão é enviado.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ("Sagittal", 1),
    ("Sagittal", 0),
    ("Coronal", 1),
    ("Coronal", 0),
    ("Axial", 1),
    ("Axial", 0),
]
N_SLOT = len(SLOTS)
N_SLICE = 16
SIZE = 336
CROP_MM = 130.0
SLICE_BAND = (0.12, 0.88)
N_SLOT_TYPES = 6
MASK_IDX = 0


def rank_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        return pd.DataFrame(array).rank(method="average", pct=True).to_numpy(np.float64)
    if array.ndim == 3:
        return np.stack([rank_columns(fold) for fold in array], axis=0)
    raise ValueError(f"rank_columns expects 2D or 3D values, got {array.shape}")


def macro_auc(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float]]:
    per_target: dict[str, float] = {}
    for index, target in enumerate(TARGETS):
        if len(np.unique(y_true[:, index])) < 2:
            per_target[target] = float("nan")
        else:
            per_target[target] = float(roc_auc_score(y_true[:, index], pred[:, index]))
    return float(np.nanmean(list(per_target.values()))), per_target


def ordered_files(series_dir: Path) -> list[Path]:
    keyed: list[tuple[int, str, Path]] = []
    for path in series_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            number = int(ds.InstanceNumber)
        except Exception:
            continue
        keyed.append((number, str(path), path))
    keyed.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in keyed]


def series_side(path: Path) -> float:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        return float(ds.ImagePositionPatient[0])
    except Exception:
        return 0.0


def read_crop(path: Path) -> np.ndarray | None:
    try:
        ds = pydicom.dcmread(str(path))
        array = ds.pixel_array.astype(np.float32)
    except Exception:
        return None

    try:
        pixel_spacing = float(ds.PixelSpacing[0])
    except Exception:
        pixel_spacing = CROP_MM / max(array.shape)

    half = int(round(CROP_MM / pixel_spacing / 2.0))
    cy, cx = array.shape[0] // 2, array.shape[1] // 2
    crop = array[
        max(0, cy - half) : min(array.shape[0], cy + half),
        max(0, cx - half) : min(array.shape[1], cx + half),
    ]
    return crop if crop.size else None


def render(path: Path, flip: bool) -> np.ndarray | None:
    crop = read_crop(path)
    if crop is None:
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    image = np.clip((crop - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    if flip:
        image = image[:, ::-1]
    return image


def build_study(
    index: int,
    study_uid: str,
    rows: list[dict[str, object]],
    series_root: Path,
) -> tuple[int, np.ndarray, np.ndarray, str]:
    output = np.zeros((N_SLOT, N_SLICE, SIZE, SIZE), dtype=np.uint8)
    mask = np.zeros(N_SLOT, dtype=np.uint8)

    for slot_index, (plane, fat_suppression) in enumerate(SLOTS):
        candidates = [
            row
            for row in rows
            if str(row.get("Anatomical_Plane")) == plane
            and int(float(row.get("Fat_Suppression", 0))) == fat_suppression
        ]
        if not candidates:
            continue

        series_uid = str(candidates[0]["SeriesInstanceUID"])
        files = ordered_files(series_root / study_uid / series_uid)
        if not files:
            continue

        flip = plane != "Sagittal" and series_side(files[0]) < 0
        start = int(round(SLICE_BAND[0] * (len(files) - 1)))
        stop = int(round(SLICE_BAND[1] * (len(files) - 1)))
        available = list(range(start, stop + 1))
        if len(available) >= N_SLICE:
            picks = [
                available[int(round(value))]
                for value in np.linspace(0, len(available) - 1, N_SLICE)
            ]
            offset = 0
        else:
            picks = available
            offset = (N_SLICE - len(picks)) // 2

        for column, picked in enumerate(picks):
            image = render(files[picked], flip)
            if image is None:
                fallback = min(len(files) - 1, picked + 1)
                image = render(files[fallback], flip)
            if image is not None:
                output[slot_index, offset + column] = (image * 255).astype(np.uint8)
        mask[slot_index] = len(picks)

    return index, output, mask, study_uid


def load_gold_images(root: Path, max_studies: int | None) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    train_path = root / "data" / "raw" / "train.csv"
    series_path = root / "data" / "raw" / "train_series.csv"
    series_root = root / "data" / "raw" / "train_series"
    train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str})
    gold = train.loc[train[TARGETS].notna().all(axis=1)].copy()
    if max_studies is not None:
        gold = gold.iloc[:max_studies].copy()
    study_ids = gold["StudyInstanceUID"].astype(str).tolist()
    labels = gold[TARGETS].to_numpy(np.float32)

    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    series = series.loc[:, ~series.columns.duplicated()]
    selected = set(study_ids)
    by_study = {
        str(uid): frame.to_dict("records")
        for uid, frame in series[series["StudyInstanceUID"].isin(selected)].groupby("StudyInstanceUID")
    }

    images = np.zeros((len(study_ids), N_SLOT, N_SLICE, SIZE, SIZE), dtype=np.uint8)
    masks = np.zeros((len(study_ids), N_SLOT), dtype=np.uint8)
    workers = max(1, min(10, (os_cpu_count() or 4)))
    print(f"gold={len(study_ids)} series_rows={len(series)} decode_workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(build_study, i, uid, by_study.get(uid, []), series_root)
            for i, uid in enumerate(study_ids)
        ]
        done = 0
        for future in as_completed(futures):
            index, array, mask, uid = future.result()
            images[index] = array
            masks[index] = mask
            done += 1
            if done == 1 or done % 8 == 0 or done == len(study_ids):
                print(f"decoded {done}/{len(study_ids)} {uid}", flush=True)

    print(
        "slot coverage:",
        dict(zip((f"{plane}_{fs}" for plane, fs in SLOTS), masks.sum(axis=0).astype(int))),
        flush=True,
    )
    return study_ids, labels, images, masks


def os_cpu_count() -> int | None:
    # Kept as a tiny wrapper so the decoder remains easy to run on Kaggle and macOS.
    import os

    return os.cpu_count()


class ViTSlotToken(nn.Module):
    """The public DINOsaur token-conditioning wrapper, reproduced verbatim."""

    def __init__(self, vit: nn.Module, n_categories: int):
        super().__init__()
        self.vit = vit
        self.tok = nn.Embedding(n_categories + 1, vit.embed_dim, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, "num_prefix_tokens", 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for block in vit.blocks:
            attention = getattr(block, "attn", None)
            if attention is not None and hasattr(attention, "num_prefix_tokens"):
                attention.num_prefix_tokens += 1

    @staticmethod
    def _maybe(module: nn.Module | None, value: torch.Tensor) -> torch.Tensor:
        return value if module is None else module(value)

    def forward_features(self, value: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        value = vit.patch_embed(value)
        positional = vit._pos_embed(value)
        rope = None
        if isinstance(positional, tuple):
            value, rope = positional
        else:
            value = positional
        value = self._maybe(getattr(vit, "patch_drop", None), value)
        value = self._maybe(getattr(vit, "norm_pre", None), value)
        n_prefix = self._orig_prefix
        slot_token = self.tok(category).unsqueeze(1)
        value = torch.cat([value[:, :n_prefix], slot_token, value[:, n_prefix:]], dim=1)
        if rope is not None:
            if getattr(vit, "rope_mixed", False):
                for index, block in enumerate(vit.blocks):
                    value = block(value, rope=rope[index])
            else:
                for block in vit.blocks:
                    value = block(value, rope=rope)
        else:
            value = vit.blocks(value)
        return vit.norm(value)


def segment_mean_max(value: torch.Tensor, study_index: torch.Tensor, batch_size: int) -> torch.Tensor:
    dimension = value.shape[1]
    counts = torch.zeros(batch_size, device=value.device, dtype=value.dtype)
    counts.index_add_(0, study_index, torch.ones(value.shape[0], device=value.device, dtype=value.dtype))
    mean = torch.zeros(batch_size, dimension, device=value.device, dtype=value.dtype)
    mean.index_add_(0, study_index, value)
    mean = mean / counts.clamp(min=1).unsqueeze(1)
    maximum = torch.full((batch_size, dimension), -10000.0, device=value.device, dtype=value.dtype)
    maximum = maximum.scatter_reduce(
        0,
        study_index.unsqueeze(1).expand(-1, dimension),
        value,
        reduce="amax",
        include_self=True,
    )
    return torch.cat([mean, maximum], dim=1)


def pad_token_slots(value: torch.Tensor, study_index: torch.Tensor, batch_size: int, norm: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    token_count, patch_count, dimension = value.shape
    counts = torch.bincount(study_index, minlength=batch_size)
    max_slots = int(counts.max().item())
    starts = torch.cumsum(counts, 0) - counts
    position = torch.arange(token_count, device=value.device) - starts[study_index]
    padded = value.new_zeros(batch_size, max_slots, patch_count, dimension)
    padded[study_index, position] = value
    keep = torch.zeros(batch_size, max_slots, dtype=torch.bool, device=value.device)
    keep[study_index, position] = True
    padded = norm(padded.reshape(batch_size, max_slots * patch_count, dimension))
    return padded, ~keep.repeat_interleave(patch_count, dim=1)


class GatedDelta(nn.Module):
    def __init__(self, dimension: int, labels: int, heads: int = 6, dropout: float = 0.2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(labels, dimension) * 0.02)
        self.kv_norm = nn.LayerNorm(dimension)
        self.attn = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(dimension)
        self.dw = nn.Parameter(torch.randn(labels, dimension) * (1.0 / dimension**0.5))
        self.db = nn.Parameter(torch.zeros(labels))
        self.gate = nn.Parameter(torch.zeros(labels))

    def delta(self, patch_tokens: torch.Tensor, study_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        key_value, padding = pad_token_slots(patch_tokens, study_index, batch_size, self.kv_norm)
        query = self.q.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query, key_value, key_value, key_padding_mask=padding, need_weights=False)
        return (self.d_norm(attended) * self.dw.unsqueeze(0)).sum(-1) + self.db


class CodexResidualPool(GatedDelta):
    def __init__(self, dimension: int, labels: int = 12, pe: int = 64, dropout: float = 0.2):
        super().__init__(dimension, labels, heads=6, dropout=dropout)
        self.base = nn.Sequential(
            nn.LayerNorm(2 * dimension + pe),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension + pe, labels),
        )

    def forward(self, tokens: torch.Tensor, slot: torch.Tensor, study_index: torch.Tensor, batch_size: int, presence: torch.Tensor) -> torch.Tensor:
        base = self.base(torch.cat([segment_mean_max(tokens[:, 0], study_index, batch_size), presence], dim=1))
        delta = self.delta(tokens[:, 1:], study_index, batch_size)
        return base + self.gate * delta


class Readout(nn.Module):
    def __init__(self, dimension: int, labels: int = 12, pe: int = 64):
        super().__init__()
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, pe, padding_idx=MASK_IDX)
        self.pool = CodexResidualPool(dimension, labels, pe=pe)
        self.drop = nn.Dropout(0.2)

    def forward(self, tokens: torch.Tensor, slot: torch.Tensor, study_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        presence_embedding = self.pres_emb(slot)
        presence = torch.zeros(batch_size, presence_embedding.shape[1], device=tokens.device, dtype=tokens.dtype)
        presence.index_add_(0, study_index, presence_embedding)
        return self.pool(tokens, slot, study_index, batch_size, presence)


class DINOStudyNet(nn.Module):
    def __init__(self, encoder: nn.Module, condition: str = "token", dimension: int = 384):
        super().__init__()
        self.enc = encoder
        self.cond = condition
        self.compress = None
        self.mixer = None
        self.tokens = True
        self.readout = Readout(dimension)

    def forward(
        self,
        images: torch.Tensor,
        slots: torch.Tensor,
        _metadata: torch.Tensor,
        study_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        tokens = self.enc.forward_features(images, slots)
        inner = getattr(self.enc, "vit", self.enc)
        original_prefix = getattr(self.enc, "_orig_prefix", getattr(inner, "num_prefix_tokens", 1))
        tokens = torch.cat([tokens[:, :1], tokens[:, original_prefix:]], dim=1)
        return self.readout(tokens, slots, study_index, batch_size)


def load_dino_model(checkpoint_path: Path, timm_module: object) -> tuple[DINOStudyNet, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["cfg"])
    encoder = timm_module.create_model(
        config["backbone"],
        pretrained=False,
        num_classes=0,
        in_chans=int(config["n_slice"]),
        img_size=int(config["img"]),
    )
    encoder = ViTSlotToken(encoder, N_SLOT_TYPES)
    model = DINOStudyNet(encoder, config.get("cond", "token"), dimension=encoder.num_features)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DINO checkpoint mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    del checkpoint
    gc.collect()
    return model.eval(), config


class V16SlotAttentionHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 384, n_target: int = 12, n_slot: int = 6):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.zeros(n_slot, hidden))
        self.query = nn.Parameter(torch.zeros(n_target, hidden))
        self.attn = nn.MultiheadAttention(hidden, num_heads=6, dropout=0.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden * 5),
            nn.Linear(hidden * 5, hidden),
            nn.GELU(),
            nn.Dropout(0.16),
        )
        self.target_weight = nn.Parameter(torch.empty(n_target, hidden))
        self.target_bias = nn.Parameter(torch.zeros(n_target))

    def forward(self, features: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        mask = slot_mask.bool()
        projected = self.proj(features.float()) + self.slot_emb.unsqueeze(0)
        projected = projected * mask.unsqueeze(-1)
        denominator = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        mean = projected.sum(dim=1, keepdim=True) / denominator
        query = self.query.unsqueeze(0).expand(len(projected), -1, -1)
        attended, _ = self.attn(query, projected, projected, key_padding_mask=~mask, need_weights=False)
        global_target = mean.expand(-1, self.query.shape[0], -1)
        fused = torch.cat(
            [attended, query, global_target, torch.abs(attended - global_target), attended * global_target],
            dim=-1,
        )
        fused = self.fuse(fused)
        return (fused * self.target_weight.unsqueeze(0)).sum(dim=-1) + self.target_bias.unsqueeze(0)


def present_inputs(
    images: np.ndarray,
    masks: np.ndarray,
    start: int,
    stop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    image_parts: list[torch.Tensor] = []
    slot_parts: list[torch.Tensor] = []
    study_parts: list[torch.Tensor] = []
    batch_size = stop - start
    for local_index in range(batch_size):
        present = np.flatnonzero(masks[start + local_index] > 0)
        if len(present) == 0:
            continue
        image_parts.append(torch.from_numpy(images[start + local_index, present]))
        slot_parts.append(torch.from_numpy(present + 1).long())
        study_parts.append(torch.full((len(present),), local_index, dtype=torch.long))
    if not image_parts:
        raise RuntimeError("gold batch has no present slots")
    values = torch.cat(image_parts).to(device).float().div_(255.0)
    slots = torch.cat(slot_parts).to(device)
    study_index = torch.cat(study_parts).to(device)
    return values, slots, study_index, batch_size


@torch.no_grad()
def infer_dino(
    model: DINOStudyNet,
    images: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_studies: int,
    extract_features: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions: list[np.ndarray] = []
    feature_output = np.zeros((len(images), N_SLOT, 1152), dtype=np.float16) if extract_features else None
    feature_mask = np.zeros((len(images), N_SLOT), dtype=np.uint8) if extract_features else None
    model = model.to(device).eval()

    for start in range(0, len(images), batch_studies):
        stop = min(start + batch_studies, len(images))
        values, slots, study_index, batch_size = present_inputs(images, masks, start, stop, device)
        metadata = torch.zeros(values.shape[0], 0, device=device)
        logits = model(values, slots, metadata, study_index, batch_size)
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())

        if extract_features:
            tokens = model.enc.forward_features(values, slots)
            prefix = int(model.enc._orig_prefix)
            cls = tokens[:, 0].float()
            patches = tokens[:, prefix:].float()
            mean = patches.mean(dim=1)
            n_patch = patches.shape[1]
            grid = int(round(math.sqrt(n_patch)))
            if grid * grid == n_patch and grid >= 3:
                patch_grid = patches.reshape(len(values), grid, grid, -1)
                radius = max(1, grid // 4)
                center = grid // 2
                focal = patch_grid[:, center - radius : center + radius + 1, center - radius : center + radius + 1].mean(dim=(1, 2))
            else:
                focal = mean
            features = torch.cat([cls, mean, focal], dim=1).cpu().numpy().astype(np.float16)
            offset = 0
            for local_index in range(batch_size):
                present = np.flatnonzero(masks[start + local_index] > 0)
                for slot in present:
                    feature_output[start + local_index, slot] = features[offset]
                    feature_mask[start + local_index, slot] = 1
                    offset += 1

        if start == 0 or stop == len(images) or stop % (batch_studies * 4) == 0:
            print(f"DINO infer {stop}/{len(images)}", flush=True)

    return np.concatenate(predictions, axis=0), feature_output


@torch.no_grad()
def infer_v16(
    head_dir: Path,
    features: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    fold_predictions: list[np.ndarray] = []
    for path in sorted(head_dir.glob("v16_slothead_f*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = V16SlotAttentionHead(
            int(checkpoint["feature_dim"]),
            hidden=int(checkpoint.get("hidden", 384)),
            n_target=int(checkpoint.get("n_target", len(TARGETS))),
            n_slot=int(checkpoint.get("n_slot", N_SLOT)),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(device).eval()
        values = torch.from_numpy(features).to(device)
        slot_mask = torch.from_numpy(masks).to(device)
        chunks = []
        for start in range(0, len(values), batch_size):
            chunks.append(torch.sigmoid(model(values[start : start + batch_size], slot_mask[start : start + batch_size])).float().cpu().numpy())
        fold_predictions.append(np.concatenate(chunks, axis=0))
        del model, checkpoint, values, slot_mask
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
        print(f"v16 head {path.name}", flush=True)
    if len(fold_predictions) < 4:
        raise RuntimeError(f"v16 expects at least four checkpoints, got {len(fold_predictions)}")
    return np.stack(fold_predictions, axis=0)


def evaluate_blends(labels: np.ndarray, base_rank: np.ndarray, specialist_rank: np.ndarray, manifest_weights: dict[str, float]) -> dict[str, object]:
    base_score, base_targets = macro_auc(labels, base_rank)
    specialist_score, specialist_targets = macro_auc(labels, specialist_rank)
    deployed = np.zeros_like(base_rank)
    for index, target in enumerate(TARGETS):
        weight = float(manifest_weights.get(target, 0.0))
        deployed[:, index] = (1.0 - weight) * base_rank[:, index] + weight * specialist_rank[:, index]
    deployed = rank_columns(deployed)
    deployed_score, deployed_targets = macro_auc(labels, deployed)

    grid: dict[str, object] = {}
    for index, target in enumerate(TARGETS):
        if len(np.unique(labels[:, index])) < 2:
            grid[target] = {"best_auc": float("nan"), "best_weight": 0.0}
            continue
        choices = []
        for weight in np.arange(0.0, 0.241, 0.02):
            mixed = (1.0 - weight) * base_rank[:, index] + weight * specialist_rank[:, index]
            choices.append((float(roc_auc_score(labels[:, index], mixed)), float(weight)))
        score, weight = max(choices)
        grid[target] = {"best_auc": score, "best_weight": weight}

    return {
        "base_dino_rank": {"macro_auc": base_score, "targets": base_targets},
        "v16_specialist_rank": {"macro_auc": specialist_score, "targets": specialist_targets},
        "manifest_blend": {
            "macro_auc": deployed_score,
            "targets": deployed_targets,
            "weights": manifest_weights,
        },
        "gold_tuned_per_target_grid": grid,
    }


def load_timm(root: Path) -> object:
    wheel = root / "data" / "models" / "mattia-knee-fold-weights" / "timm-1.0.22-py3-none-any.whl"
    if wheel.is_file():
        sys.path.insert(0, str(wheel))
    import timm

    print(f"timm={timm.__version__}", flush=True)
    return timm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-studies", type=int, default=2)
    parser.add_argument("--max-studies", type=int, default=None)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    head_dir = args.head_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device} torch={torch.__version__}", flush=True)

    cache_files = {
        "ids": output_dir / "gold_study_ids.npy",
        "labels": output_dir / "gold_labels.npy",
        "images": output_dir / "gold_images_uint8.npy",
        "masks": output_dir / "gold_slot_masks_uint8.npy",
    }
    if args.max_studies is None and all(path.is_file() for path in cache_files.values()):
        study_ids = np.load(cache_files["ids"], allow_pickle=False).astype(str).tolist()
        labels = np.load(cache_files["labels"], allow_pickle=False)
        images = np.load(cache_files["images"], mmap_mode="r", allow_pickle=False)
        masks = np.load(cache_files["masks"], mmap_mode="r", allow_pickle=False)
        print(f"loaded image cache {images.shape} from {output_dir}", flush=True)
    else:
        study_ids, labels, images, masks = load_gold_images(root, args.max_studies)
        if args.max_studies is None:
            np.save(cache_files["ids"], np.asarray(study_ids, dtype="U80"))
            np.save(cache_files["labels"], labels)
            np.save(cache_files["images"], images)
            np.save(cache_files["masks"], masks)
            print(f"saved image cache {images.shape} to {output_dir}", flush=True)
    timm_module = load_timm(root)
    checkpoint_paths = sorted((root / "data" / "models" / "mattia-knee-fold-weights").glob("m_f*.pt"))
    if len(checkpoint_paths) < 5:
        raise RuntimeError(f"expected five Mattia DINOv3 checkpoints, got {len(checkpoint_paths)}")

    dino_folds = []
    features = None
    for path in checkpoint_paths:
        model, config = load_dino_model(path, timm_module)
        prediction, fold_features = infer_dino(
            model,
            images,
            masks,
            device,
            args.batch_studies,
            extract_features=features is None,
        )
        dino_folds.append(prediction)
        if features is None:
            features = fold_features
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
        print(f"finished {path.name} cfg_img={config.get('img')} pool={config.get('pool')}", flush=True)

    if features is None:
        raise RuntimeError("DINO feature cache was not produced")
    fold_ranks = np.stack([rank_columns(prediction) for prediction in dino_folds], axis=0)
    base_rank = rank_columns(fold_ranks.mean(axis=0))
    head_folds = infer_v16(head_dir, features, masks, device)
    head_rank = rank_columns(head_folds)
    specialist_rank = rank_columns(head_rank.mean(axis=0))

    manifest_weights = {
        "ACL": 0.0,
        "MCL": 0.0,
        "Medial Meniscus": 0.22,
        "Lateral Meniscus": 0.22,
        "Medial OA": 0.0,
        "Lateral OA": 0.10,
        "PF OA": 0.10,
        "Effusion": 0.0,
        "Synovitis": 0.0,
        "Baker's": 0.10,
        "Contusion": 0.0,
        "Fracture": 0.0,
    }
    report = evaluate_blends(labels, base_rank, specialist_rank, manifest_weights)
    report.update(
        {
            "tags": ["RSNA", "Kaggle", "Pesquisa"],
            "format": "dinov3-v16-gold-reproduction-v1",
            "study_count": len(study_ids),
            "slot_coverage": dict(zip((f"{plane}_{fs}" for plane, fs in SLOTS), masks.sum(axis=0).astype(int).tolist())),
            "feature_shape": list(features.shape),
            "device": str(device),
            "dino_checkpoints": [path.name for path in checkpoint_paths],
            "dino_checkpoint_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint_paths},
            "head_checkpoints": [path.name for path in sorted(head_dir.glob("v16_slothead_f*.pt"))],
        }
    )
    np.savez_compressed(
        output_dir / "gold_dinov3_v16_predictions.npz",
        ids=np.asarray(study_ids, dtype="U80"),
        labels=labels,
        masks=masks,
        features=features,
        base_rank=base_rank,
        specialist_rank=specialist_rank,
        dino_fold_rank=fold_ranks,
        v16_fold_rank=head_rank,
    )
    (output_dir / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()
