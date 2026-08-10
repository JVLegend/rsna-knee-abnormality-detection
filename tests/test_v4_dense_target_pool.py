from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "rsna_knee_v4_dense_target_pool.py"
SPEC = importlib.util.spec_from_file_location("rsna_knee_v4_dense_target_pool", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_slice_profiles_have_expected_density() -> None:
    assert len(MODULE._sample_indices(100, "quantile3")) == 3
    assert len(MODULE._sample_indices(100, "adjacent3")) == 3
    assert len(MODULE._sample_indices(100, "dense6")) == 6
    assert len(MODULE._sample_indices(100, "dense9")) == 9


def test_dense_profiles_make_three_channel_adjacent_slabs(tmp_path, monkeypatch) -> None:
    series_dir = tmp_path / "train_series" / "study" / "series"
    series_dir.mkdir(parents=True)
    for index in range(12):
        (series_dir / f"{index:03d}.dcm").touch()

    def fake_dcmread(path, stop_before_pixels=False, force=False, **kwargs):
        index = int(Path(path).stem)
        if stop_before_pixels:
            return SimpleNamespace(InstanceNumber=index)
        return SimpleNamespace(
            pixel_array=np.full((4, 4), index, dtype=np.float32),
            RescaleSlope=1.0,
            RescaleIntercept=0.0,
            PhotometricInterpretation="MONOCHROME2",
        )

    monkeypatch.setattr(MODULE.pydicom, "dcmread", fake_dcmread)
    dense, dense_valid = MODULE._series_image(tmp_path, "train", "study", "series", size=8, slice_profile="dense6")
    adjacent, adjacent_valid = MODULE._series_image(
        tmp_path,
        "train",
        "study",
        "series",
        size=8,
        slice_profile="adjacent3",
        fast_preprocess=True,
    )
    legacy, legacy_valid = MODULE._series_image(tmp_path, "train", "study", "series", size=8, slice_profile="quantile3")

    assert dense_valid and len(dense) == 6
    assert all(image.shape == (3, 8, 8) for image in dense)
    assert adjacent_valid and len(adjacent) == 3
    assert all(image.shape == (3, 8, 8) for image in adjacent)
    assert legacy_valid and len(legacy) == 1 and legacy[0].shape == (3, 8, 8)


def test_probability_pooling_is_target_aware_and_bounded() -> None:
    values = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert MODULE._pool_probabilities(values, "mean") == 0.5
    assert MODULE._pool_probabilities(values, "topk") == 0.9
    assert MODULE._pool_probabilities(np.asarray([]), "max") == 0.5


def test_target_view_model_trains_and_pools_views() -> None:
    train_views = [
        [np.asarray([0.0, 0.0]), np.asarray([0.1, 0.0])],
        [np.asarray([0.0, 0.1]), np.asarray([0.1, 0.1])],
        [np.asarray([1.0, 1.0]), np.asarray([0.9, 1.0])],
        [np.asarray([1.0, 0.9]), np.asarray([0.9, 0.9])],
    ]
    test_views = [
        [np.asarray([0.05, 0.05]), np.asarray([0.2, 0.2])],
        [np.asarray([0.95, 0.95]), np.asarray([0.8, 0.9])],
        [],
    ]
    predictions = MODULE._fit_target_view_model(
        "Effusion",
        train_views,
        test_views,
        labels=np.asarray([0.0, 0.0, 1.0, 1.0]),
        fit_mask=np.asarray([True, True, True, True]),
        sample_weights=None,
    )
    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
    assert predictions[0] < predictions[1]
    assert predictions[2] == 0.5


def test_targetwise_teacher_uses_source_map_and_preserves_unknown(tmp_path) -> None:
    ids = ["a", "b"]

    def base_frame() -> pd.DataFrame:
        frame = pd.DataFrame({MODULE.KEY_COLUMN: ids})
        for target in MODULE.TARGET_COLUMNS:
            frame[target] = 0.5
        return frame

    v2 = base_frame()
    v4 = base_frame()
    pilkwang = base_frame()
    v2.loc[0, "Lateral Meniscus"] = 0.9
    v4.loc[0, "Lateral Meniscus"] = 0.8
    v4.loc[0, "Effusion"] = 0.7
    pilkwang.loc[0, "ACL"] = 0.9
    pilkwang.loc[0, "MCL"] = 0.1
    pilkwang.loc[0, "Fracture"] = 0.9
    pilkwang["ACL__conf"] = [0.95, 0.05]
    pilkwang["ACL__verdict"] = ["YES", "UNK"]
    pilkwang["MCL__verdict"] = ["NO", "UNK"]
    pilkwang["Fracture__verdict"] = ["YES", "UNK"]
    v2.to_csv(tmp_path / MODULE.EXTERNAL_LABEL_V2_FILENAME, index=False)
    v4.to_csv(tmp_path / MODULE.EXTERNAL_LABEL_V4_FILENAME, index=False)
    pilkwang.to_csv(tmp_path / MODULE.PILKWANG_LABEL_FILENAME, index=False)

    train = pd.DataFrame({MODULE.KEY_COLUMN: ids})
    teacher, _ = MODULE._external_teacher(train, tmp_path, profile="targetwise")

    assert teacher.loc[0, "ACL"] == 0.9
    assert teacher.loc[0, "MCL"] == 0.1
    assert teacher.loc[0, "Lateral Meniscus"] == 0.8
    assert teacher.loc[0, "ACL__confidence"] == 0.95
    assert teacher.loc[1, "ACL"] == 0.5


def test_label_file_search_finds_nested_kaggle_mount(monkeypatch, tmp_path) -> None:
    nested = tmp_path / "datasets" / "source" / "versions" / "1"
    nested.mkdir(parents=True)
    expected = nested / MODULE.EXTERNAL_LABEL_V2_FILENAME
    expected.write_text("StudyInstanceUID,ACL\nstudy,0.5\n", encoding="utf-8")

    monkeypatch.setattr(MODULE, "Path", lambda value: tmp_path if value == "/kaggle/input" else __import__("pathlib").Path(value))

    assert MODULE._find_label_file(MODULE.EXTERNAL_LABEL_V2_FILENAME) == expected
