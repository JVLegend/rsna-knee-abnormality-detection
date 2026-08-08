from __future__ import annotations

import pytest

from scripts.extract_efficientnet_embeddings import extract_embeddings


def test_embedding_extractor_rejects_empty_index(tmp_path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text('{"records": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="não contém records"):
        extract_embeddings(index_path, tmp_path / "out")
