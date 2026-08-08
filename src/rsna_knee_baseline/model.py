"""Modelo v0 baseado em laudo e metadados de séries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .constants import KEY_COLUMN, TARGET_COLUMNS
from .data import build_metadata_features, build_text_frame


@dataclass
class _ConstantPredictor:
    value: float

    def predict_proba(self, count: int) -> np.ndarray:
        return np.full(count, self.value, dtype=float)


class KneeReportBaseline:
    """Classificador por alvo para uma primeira submissão barata e auditável.

    O vocabulário é aprendido somente no conjunto de treino. Cada alvo usa
    apenas as linhas que possuem anotação para aquele alvo; linhas sem rótulo
    não são tratadas como negativas.
    """

    def __init__(self, c: float = 2.0, max_iter: int = 800) -> None:
        self.c = c
        self.max_iter = max_iter
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=120_000,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=120_000,
        )
        self.models: dict[str, LogisticRegression | _ConstantPredictor] = {}
        self.metadata_columns: list[str] = []
        self.is_fitted = False

    def _fit_features(self, frame: pd.DataFrame, series: pd.DataFrame | None) -> csr_matrix:
        text = build_text_frame(frame)
        word = self.word_vectorizer.fit_transform(text)
        char = self.char_vectorizer.fit_transform(text)
        metadata = build_metadata_features(frame, series)
        self.metadata_columns = list(metadata.columns)
        return hstack([word, char, csr_matrix(metadata.to_numpy(dtype=float))], format="csr")

    def _transform_features(self, frame: pd.DataFrame, series: pd.DataFrame | None) -> csr_matrix:
        text = build_text_frame(frame)
        word = self.word_vectorizer.transform(text)
        char = self.char_vectorizer.transform(text)
        metadata = build_metadata_features(frame, series)
        metadata = metadata.reindex(columns=self.metadata_columns, fill_value=0)
        return hstack([word, char, csr_matrix(metadata.to_numpy(dtype=float))], format="csr")

    def fit(self, train: pd.DataFrame, train_series: pd.DataFrame | None = None) -> "KneeReportBaseline":
        if KEY_COLUMN not in train.columns:
            raise ValueError(f"A tabela de treino precisa da coluna {KEY_COLUMN!r}.")

        features = self._fit_features(train, train_series)
        for target in TARGET_COLUMNS:
            if target not in train.columns:
                self.models[target] = _ConstantPredictor(0.5)
                continue

            labels = pd.to_numeric(train[target], errors="coerce")
            labeled = labels.notna().to_numpy()
            values = labels.loc[labeled].astype(float).to_numpy()
            if values.size == 0:
                self.models[target] = _ConstantPredictor(0.5)
            elif np.unique(values).size < 2:
                self.models[target] = _ConstantPredictor(float(values.mean()))
            else:
                model = LogisticRegression(
                    C=self.c,
                    class_weight="balanced",
                    max_iter=self.max_iter,
                    solver="liblinear",
                )
                model.fit(features[labeled], values)
                self.models[target] = model

        self.is_fitted = True
        return self

    def predict(self, frame: pd.DataFrame, series: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser ajustado antes da predição.")
        if KEY_COLUMN not in frame.columns:
            raise ValueError(f"A tabela de teste precisa da coluna {KEY_COLUMN!r}.")

        features = self._transform_features(frame, series)
        prediction = pd.DataFrame({KEY_COLUMN: frame[KEY_COLUMN].astype(str).values})
        for target in TARGET_COLUMNS:
            model = self.models[target]
            if isinstance(model, _ConstantPredictor):
                values = model.predict_proba(len(frame))
            else:
                values = model.predict_proba(features)[:, 1]
            prediction[target] = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
        return prediction[[KEY_COLUMN, *TARGET_COLUMNS]]
