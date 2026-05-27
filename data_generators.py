from __future__ import annotations

import numpy as np


def spoil_labels(labels: np.ndarray, probability: float, seed: int = 42) -> np.ndarray:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    result = np.asarray(labels, dtype=int).copy()
    mask = rng.random(len(result)) < probability
    result[mask] = 1 - result[mask]
    return result


def make_linear_clouds(
    n_samples: int = 500,
    centers: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, -2.0), (2.0, 2.0)),
    covariance: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0)),
    noise: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left_count = n_samples // 2
    right_count = n_samples - left_count
    first = rng.multivariate_normal(centers[0], covariance, left_count)
    second = rng.multivariate_normal(centers[1], covariance, right_count)
    x = np.vstack((first, second))
    y = np.r_[np.zeros(left_count, dtype=int), np.ones(right_count, dtype=int)]
    return x, spoil_labels(y, noise, seed + 1)


def make_xor(n_samples: int = 500, spread: float = 0.35, noise: float = 0.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    marks = np.array([0, 1, 1, 0], dtype=int)
    counts = np.full(4, n_samples // 4)
    counts[: n_samples % 4] += 1

    x_parts = [rng.normal(corner, spread, size=(count, 2)) for corner, count in zip(corners, counts)]
    y_parts = [np.full(count, mark, dtype=int) for mark, count in zip(marks, counts)]
    return np.vstack(x_parts), spoil_labels(np.concatenate(y_parts), noise, seed + 1)


def make_circles(
    n_samples: int = 500,
    inner_radius: float = 1.0,
    outer_radius: float = 2.5,
    noise: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inner_count = n_samples // 2
    outer_count = n_samples - inner_count

    inner_angle = rng.uniform(0, 2 * np.pi, inner_count)
    inner_rad = np.sqrt(rng.uniform(0, inner_radius**2, inner_count))
    inner = np.c_[inner_rad * np.cos(inner_angle), inner_rad * np.sin(inner_angle)]

    outer_angle = rng.uniform(0, 2 * np.pi, outer_count)
    outer_rad = np.sqrt(rng.uniform((inner_radius + 0.5) ** 2, outer_radius**2, outer_count))
    outer = np.c_[outer_rad * np.cos(outer_angle), outer_rad * np.sin(outer_angle)]

    x = np.vstack((inner, outer))
    y = np.r_[np.ones(inner_count, dtype=int), np.zeros(outer_count, dtype=int)]
    return x, spoil_labels(y, noise, seed + 1)


add_label_noise = spoil_labels
