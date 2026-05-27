from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

WeightMode = Literal["zero", "normal", "wide_normal", "zeros", "small_random", "large_random"]
LossMode = Literal["bce", "cross_entropy", "hinge"]


@dataclass
class FitTrace:
    train: list[float] = field(default_factory=list)
    valid: list[float] = field(default_factory=list)


@dataclass
class Perceptron:
    """Один линейный нейрон для бинарной классификации.

    По умолчанию используется сигмоидальная модель с binary cross-entropy.
    Для дополнительного задания поддержан вариант Hinge loss.
    """

    n_features: int = 2
    learning_rate: float = 0.1
    epochs: int = 100
    batch_size: int = 32
    init_mode: WeightMode = "normal"
    random_state: int | None = 42
    l2: float = 0.0
    momentum: float = 0.0
    threshold: float = 0.5
    loss_function: LossMode = "bce"
    verbose: bool = False

    def __init__(
        self,
        n_features: int = 2,
        learning_rate: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        init_mode: WeightMode = "normal",
        initialization: WeightMode | None = None,
        seed: int | None = None,
        random_state: int | None = 42,
        l2: float = 0.0,
        momentum: float = 0.0,
        threshold: float = 0.5,
        loss_function: LossMode = "bce",
        loss: LossMode | None = None,
        verbose: bool = False,
        init_scale: float = 0.01,
    ) -> None:
        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.init_mode = initialization or init_mode
        self.random_state = random_state if seed is None else seed
        self.l2 = float(l2)
        self.momentum = float(momentum)
        self.threshold = float(threshold)
        self.loss_function = loss or loss_function
        self.verbose = bool(verbose)
        self.init_scale = float(init_scale)

        aliases = {"cross_entropy": "bce"}
        self.loss_function = aliases.get(self.loss_function, self.loss_function)
        if self.loss_function not in {"bce", "hinge"}:
            raise ValueError("loss_function must be 'bce' or 'hinge'")

        self._rng = np.random.default_rng(self.random_state)
        self.theta = self._make_weights(self.init_mode)
        self.bias = 0.0
        self.trace = FitTrace()
        self._speed_theta = np.zeros(self.n_features)
        self._speed_bias = 0.0

    @property
    def weights(self) -> np.ndarray:
        return self.theta

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.theta = np.asarray(value, dtype=float).reshape(self.n_features)
        self._speed_theta = np.zeros_like(self.theta)

    @property
    def w(self) -> np.ndarray:
        return self.theta

    @w.setter
    def w(self, value: np.ndarray) -> None:
        self.weights = value

    @property
    def b(self) -> float:
        return self.bias

    @b.setter
    def b(self, value: float) -> None:
        self.bias = float(value)
        self._speed_bias = 0.0

    def _make_weights(self, mode: WeightMode) -> np.ndarray:
        aliases = {"zeros": "zero", "small_random": "normal", "large_random": "wide_normal"}
        mode = aliases.get(mode, mode)
        if mode == "zero":
            return np.zeros(self.n_features)
        if mode == "normal":
            return self._rng.normal(0.0, self.init_scale, self.n_features)
        if mode == "wide_normal":
            return self._rng.normal(0.0, 10.0, self.n_features)
        raise ValueError(f"Unknown weight initialization: {mode}")

    @staticmethod
    def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
        z = np.asarray(values, dtype=float)
        out = np.empty_like(z, dtype=float)
        right = z >= 0
        out[right] = 1.0 / (1.0 + np.exp(-z[right]))
        ez = np.exp(z[~right])
        out[~right] = ez / (1.0 + ez)
        return float(out) if np.isscalar(values) else out

    def linear_output(self, features: np.ndarray) -> np.ndarray:
        x = self._as_matrix(features)
        return x @ self.theta + self.bias

    def forward(self, features: np.ndarray) -> np.ndarray:
        return self.sigmoid(self.linear_output(features))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.loss_function == "hinge":
            return (self.linear_output(features) >= 0.0).astype(int)
        return (self.predict_proba(features) >= self.threshold).astype(int)

    def compute_loss(self, target: np.ndarray, prediction: np.ndarray) -> float:
        y = self._as_target(target)
        values = np.asarray(prediction, dtype=float).reshape(-1)
        if len(y) != len(values):
            raise ValueError("target and prediction must have the same length")

        if self.loss_function == "hinge":
            signed = self._signed_target(y)
            margin = signed * values
            base_loss = np.maximum(0.0, 1.0 - margin).mean()
        else:
            p = np.clip(values, 1e-12, 1.0 - 1e-12)
            base_loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()
        return float(base_loss + 0.5 * self.l2 * np.dot(self.theta, self.theta))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
    ) -> tuple[list[float], list[float]]:
        x = self._as_matrix(x_train)
        y = self._as_target(y_train)
        if len(x) != len(y):
            raise ValueError("x_train and y_train contain different number of rows")

        xv = None if x_valid is None else self._as_matrix(x_valid)
        yv = None if y_valid is None else self._as_target(y_valid)
        if (xv is None) != (yv is None):
            raise ValueError("validation features and labels must be passed together")

        steps = self.epochs if epochs is None else int(epochs)
        step_size = self.learning_rate if lr is None else float(lr)
        pack_size = max(1, min(self.batch_size if batch_size is None else int(batch_size), len(x)))
        self.trace = FitTrace()

        for ep in range(steps):
            order = self._rng.permutation(len(x))
            for left in range(0, len(order), pack_size):
                ids = order[left:left + pack_size]
                self._update(x[ids], y[ids], step_size)

            self.trace.train.append(self._loss_for_features(x, y))
            if xv is not None and yv is not None:
                self.trace.valid.append(self._loss_for_features(xv, yv))

            if self.verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == steps):
                msg = f"epoch {ep + 1:03d}: train={self.trace.train[-1]:.4f}"
                if self.trace.valid:
                    msg += f", valid={self.trace.valid[-1]:.4f}"
                print(msg)

        return self.trace.train, self.trace.valid

    def _loss_for_features(self, features: np.ndarray, target: np.ndarray) -> float:
        if self.loss_function == "hinge":
            return self.compute_loss(target, self.linear_output(features))
        return self.compute_loss(target, self.forward(features))

    def _update(self, x_batch: np.ndarray, y_batch: np.ndarray, lr: float) -> None:
        if self.loss_function == "hinge":
            signed = self._signed_target(y_batch)
            margin = signed * self.linear_output(x_batch)
            active = margin < 1.0
            if active.any():
                grad_theta = -(x_batch[active].T @ signed[active]) / len(x_batch)
                grad_bias = -float(signed[active].sum()) / len(x_batch)
            else:
                grad_theta = np.zeros(self.n_features)
                grad_bias = 0.0
            grad_theta = grad_theta + self.l2 * self.theta
        else:
            delta = self.forward(x_batch) - y_batch
            grad_theta = x_batch.T @ delta / len(x_batch) + self.l2 * self.theta
            grad_bias = float(delta.mean())

        self._speed_theta = self.momentum * self._speed_theta + grad_theta
        self._speed_bias = self.momentum * self._speed_bias + grad_bias
        self.theta -= lr * self._speed_theta
        self.bias -= lr * self._speed_bias

    def _as_matrix(self, value: np.ndarray) -> np.ndarray:
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features:
            raise ValueError(f"expected feature matrix with {self.n_features} columns")
        return matrix

    @staticmethod
    def _as_target(value: np.ndarray) -> np.ndarray:
        target = np.asarray(value, dtype=float).reshape(-1)
        labels = set(np.unique(target))
        if not labels <= {0.0, 1.0}:
            raise ValueError("labels must be encoded by 0 and 1")
        return target

    @staticmethod
    def _signed_target(value: np.ndarray) -> np.ndarray:
        return 2.0 * np.asarray(value, dtype=float).reshape(-1) - 1.0


BinaryPerceptron = Perceptron
Initialization = WeightMode
