"""Audit the public DINOv2-MIL bundle without exposing study identifiers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def audit(bundle: Path, reported_license: str = "other") -> dict[str, Any]:
    weight_path = bundle / "weights" / "dinov2_vits14_pretrain.pth"
    result: dict[str, Any] = {
        "bundle": str(bundle),
        "reported_kaggle_license": reported_license,
        "license_gate": "do_not_submit_until_clarified" if reported_license.lower() == "other" else "review_required",
        "files": {},
        "architecture": {},
        "folds": {},
    }
    for relative in (
        Path("run_config.json"),
        Path("weights/dinov2_vits14_pretrain.pth"),
        Path("dinov2_src/LICENSE"),
        Path("dinov2_src/LICENSE_CELL_DINO_CODE"),
        Path("dinov2_src/LICENSE_CELL_DINO_MODELS"),
    ):
        path = bundle / relative
        result["files"][str(relative)] = _file_record(path) if path.is_file() else None

    if weight_path.is_file():
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        result["architecture"] = {
            "state_dict_keys": len(state),
            "embedding_dim": tuple(state["cls_token"].shape)[-1] if "cls_token" in state else None,
            "pos_embed": list(state["pos_embed"].shape) if "pos_embed" in state else None,
            "patch_embed": list(state["patch_embed.proj.weight"].shape) if "patch_embed.proj.weight" in state else None,
            "expected_model": "dinov2_vits14",
        }

    for fold_path in sorted((bundle / "weights").glob("fold_*.pt")):
        state = torch.load(fold_path, map_location="cpu", weights_only=False)
        metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
        result["folds"][fold_path.name] = {
            "file": _file_record(fold_path),
            "metadata": metadata,
            "state_dict_keys": len(state.get("model", {})) if isinstance(state, dict) else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reported-license", default="other")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.bundle.expanduser().resolve(), args.reported_license)
    rendered = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
