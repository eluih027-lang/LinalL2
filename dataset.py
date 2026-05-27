from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class DataPack:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def X_train(self) -> np.ndarray:  # aliases for old imports
        return self.x_train

    @property
    def X_test(self) -> np.ndarray:
        return self.x_test


class ZNorm:
    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "ZNorm":
        x = np.asarray(values, dtype=float)
        self.mu = x.mean(axis=0)
        self.sigma = x.std(axis=0)
        self.sigma[self.sigma == 0.0] = 1.0
        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("normalizer is not fitted")
        return (np.asarray(values, dtype=float) - self.mu) / self.sigma

    def fit_apply(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).apply(values)


def build_points(size: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=size,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return x.astype(float), y.astype(int)


def stratified_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_part: float = 0.30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < test_part < 1:
        raise ValueError("test_part must be inside (0, 1)")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_ids: list[int] = []
    test_ids: list[int] = []

    for mark in np.unique(labels):
        ids = np.flatnonzero(labels == mark)
        rng.shuffle(ids)
        border = int(round(len(ids) * test_part))
        test_ids += ids[:border].tolist()
        train_ids += ids[border:].tolist()

    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    return features[train_ids], features[test_ids], labels[train_ids], labels[test_ids]


def load_prepared_data(test_size: float = 0.30, seed: int = 42) -> DataPack:
    x, y = build_points(seed=seed)
    raw_train, raw_test, y_train, y_test = stratified_split(x, y, test_size, seed)
    scaler = ZNorm()
    return DataPack(
        x_train=scaler.fit_apply(raw_train),
        x_test=scaler.apply(raw_test),
        y_train=y_train,
        y_test=y_test,
    )


DatasetSplits = DataPack
Standardizer = ZNorm
make_dataset = build_points
stratified_train_test_split = stratified_split

_default = load_prepared_data()
X_train, X_test = _default.x_train, _default.x_test
Y_train, Y_test = _default.y_train, _default.y_test
y_train, y_test = Y_train, Y_test
