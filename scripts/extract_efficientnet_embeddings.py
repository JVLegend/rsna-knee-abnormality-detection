#!/usr/bin/env python3
"""Extrai embeddings EfficientNet-B0 dos arrays 2.5D locais, sem rede."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


ROOT = Path(__file__).resolve().parents[1]
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def checkpoint_path(weights: EfficientNet_B0_Weights) -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder(device: str) -> tuple[torch.nn.Module, EfficientNet_B0_Weights, Path]:
    weights = EfficientNet_B0_Weights.DEFAULT
    cached = checkpoint_path(weights)
    if not cached.exists():
        raise FileNotFoundError(
            f"Pesos não encontrados no cache local: {cached}. "
            "Baixe-os previamente e execute novamente; este script não usa rede."
        )
    model = efficientnet_b0(weights=weights)
    encoder = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).to(device).eval()
    return encoder, weights, cached


def extract_embeddings(index_path: Path, output_dir: Path, batch_size: int = 8, device: str = "auto") -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("--batch-size precisa ser positivo.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index.get("records", [])
    if not records:
        raise ValueError("O índice 2.5D não contém records.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, weights, cached = load_encoder(device)
    images: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    output_records: list[dict[str, object]] = []

    def flush() -> None:
        if not images:
            return
        batch = torch.from_numpy(np.stack(images).astype(np.float32) / 255.0)
        mean = IMAGE_MEAN.to(device)
        std = IMAGE_STD.to(device)
        with torch.inference_mode():
            values = encoder(batch.to(device).sub(mean).div(std)).cpu().numpy().astype(np.float32)
        embeddings.extend(values)
        images.clear()

    for position, record in enumerate(records):
        array_path = index_path.parent / str(record["array_path"])
        with np.load(array_path) as array:
            image = np.asarray(array["image"], dtype=np.uint8)
        if image.shape != (3, int(index["size"]), int(index["size"])):
            raise ValueError(f"Shape inesperado em {array_path}: {image.shape}")
        images.append(image)
        output_records.append(
            {
                "row": position,
                "study_uid": record["study_uid"],
                "series_uid": record["series_uid"],
                "labels": record.get("labels", {}),
                "array_path": record["array_path"],
            }
        )
        if len(images) >= batch_size:
            flush()
    flush()

    matrix = np.stack(embeddings).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embeddings.npy"
    np.save(embedding_path, matrix)
    result = {
        "format": "rsna-knee-efficientnet-b0-embedding-v0",
        "source_index": str(index_path),
        "model": "torchvision.models.efficientnet_b0",
        "weights_url": weights.url,
        "weights_checkpoint": str(cached),
        "weights_sha256": sha256_file(cached),
        "device": device,
        "input_shape": [3, int(index["size"]), int(index["size"])],
        "embedding_shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "records": output_records,
        "embedding_path": embedding_path.name,
    }
    (output_dir / "index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/processed/dicom_25d_v0/index.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dicom_embeddings_efficientnet_b0"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    index_path = args.index.expanduser()
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    result = extract_embeddings(index_path, output_dir, args.batch_size, args.device)
    print(f"device={result['device']} embedding_shape={result['embedding_shape']}")
    print(f"weights_sha256={result['weights_sha256']}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
