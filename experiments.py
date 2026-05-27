from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_generators import make_circles, make_linear_clouds, make_xor
from dataset import DataPack, ZNorm, load_prepared_data, stratified_split
from metrics import extended_report
from perceptron import LossMode, Perceptron, WeightMode
from visualization import ensure_output_dir, plot_decision_boundary, plot_loss_curves, plot_roc_curve


def fit_once(
    data: DataPack,
    *,
    lr: float = 0.1,
    batch: int = 32,
    init: WeightMode = "normal",
    epochs: int = 100,
    seed: int = 42,
    momentum: float = 0.0,
    l2: float = 0.0,
    loss_function: LossMode = "bce",
) -> tuple[Perceptron, list[float], list[float], dict[str, float]]:
    model = Perceptron(
        n_features=data.x_train.shape[1],
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch,
        init_mode=init,
        random_state=seed,
        momentum=momentum,
        l2=l2,
        loss_function=loss_function,
    )
    train_curve, valid_curve = model.fit(data.x_train, data.y_train, data.x_test, data.y_test)
    score = model.predict_proba(data.x_test)
    quality = extended_report(data.y_test, model.predict(data.x_test), score)
    return model, train_curve, valid_curve, quality


def save_curves(curves: dict[str, list[float]], title: str, file_name: Path, ylabel: str = "validation loss") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for caption, curve in curves.items():
        ax.plot(curve, label=caption)
    ax.set(title=title, xlabel="epoch", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(file_name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _result_row(
    group: str,
    value: float | int | str,
    train_loss: list[float],
    valid_loss: list[float],
    quality: dict[str, float],
    model: Perceptron | None = None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "experiment": group,
        "parameter": value,
        "accuracy": quality["accuracy"],
        "precision": quality["precision"],
        "recall": quality["recall"],
        "f1": quality["f1"],
        "roc_auc": quality["roc_auc"],
        "final_train_loss": train_loss[-1],
        "final_val_loss": valid_loss[-1],
    }
    if model is not None:
        row["weight_norm"] = float(np.linalg.norm(model.weights))
    return row


def check_learning_rate(data: DataPack, folder: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    curves: dict[str, list[float]] = {}
    for lr in (0.001, 0.01, 0.5, 1.0):
        model, train_loss, valid_loss, quality = fit_once(data, lr=lr, batch=32)
        curves[f"eta={lr}"] = valid_loss
        rows.append(_result_row("learning_rate", lr, train_loss, valid_loss, quality, model))
    save_curves(curves, "Learning-rate comparison", folder / "learning_rate_experiment.png")
    return rows


def check_batch_size(data: DataPack, folder: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    curves: dict[str, list[float]] = {}
    for batch in (1, 16, 64, 256):
        model, train_loss, valid_loss, quality = fit_once(data, lr=0.1, batch=batch)
        curves[f"batch={batch}"] = valid_loss
        rows.append(_result_row("batch_size", batch, train_loss, valid_loss, quality, model))
    save_curves(curves, "Batch-size comparison", folder / "batch_size_experiment.png")
    return rows


def check_initialization(data: DataPack, folder: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    curves: dict[str, list[float]] = {}
    variants: tuple[WeightMode, ...] = ("zero", "normal", "wide_normal")
    for init in variants:
        model, train_loss, valid_loss, quality = fit_once(data, lr=0.1, batch=32, init=init)
        curves[init] = valid_loss
        rows.append(_result_row("initialization", init, train_loss, valid_loss, quality, model))
    save_curves(curves, "Weight initialization comparison", folder / "initialization_experiment.png")
    return rows


def check_loss_functions(data: DataPack, folder: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    curves: dict[str, list[float]] = {}
    settings: tuple[tuple[str, LossMode, float], ...] = (
        ("binary_cross_entropy", "bce", 0.1),
        ("hinge", "hinge", 0.01),
    )
    for caption, loss_function, lr in settings:
        model, train_loss, valid_loss, quality = fit_once(data, lr=lr, batch=32, loss_function=loss_function)
        curves[caption] = valid_loss
        rows.append(_result_row("loss_function", caption, train_loss, valid_loss, quality, model))
    save_curves(curves, "Binary cross-entropy vs Hinge loss", folder / "loss_function_experiment.png")
    return rows


def check_l2_regularization(data: DataPack, folder: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    curves: dict[str, list[float]] = {}
    for l2 in (0.0, 0.001, 0.01, 0.1):
        model, train_loss, valid_loss, quality = fit_once(data, lr=0.1, batch=32, l2=l2)
        curves[f"lambda={l2}"] = valid_loss
        rows.append(_result_row("l2_regularization", l2, train_loss, valid_loss, quality, model))
    save_curves(curves, "L2-regularization comparison", folder / "l2_regularization_experiment.png")
    return rows


def check_momentum(data: DataPack, folder: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    curves: dict[str, list[float]] = {}
    for beta in (0.0, 0.5, 0.9, 0.99):
        model, train_loss, valid_loss, quality = fit_once(data, lr=0.1, batch=32, momentum=beta)
        curves[f"beta={beta}"] = valid_loss
        rows.append(_result_row("momentum", beta, train_loss, valid_loss, quality, model))
    save_curves(curves, "SGD momentum comparison", folder / "momentum_experiment.png")
    return rows


def _make_pack_from_generator(generator: Callable[..., tuple[np.ndarray, np.ndarray]], seed: int = 42) -> DataPack:
    x, y = generator(seed=seed)
    raw_train, raw_test, y_train, y_test = stratified_split(x, y, test_part=0.30, seed=seed)
    scaler = ZNorm()
    return DataPack(scaler.fit_apply(raw_train), scaler.apply(raw_test), y_train, y_test)


def check_synthetic_generators(folder: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    settings: tuple[tuple[str, Callable[..., tuple[np.ndarray, np.ndarray]]], ...] = (
        ("linear_clouds", make_linear_clouds),
        ("xor", make_xor),
        ("circles", make_circles),
    )
    for caption, generator in settings:
        data = _make_pack_from_generator(generator)
        model, train_loss, valid_loss, quality = fit_once(data, lr=0.1, batch=32, epochs=120)
        plot_loss_curves(train_loss, valid_loss, folder / f"synthetic_{caption}_loss.png", title=f"Synthetic data: {caption}")
        plot_decision_boundary(
            model,
            data.x_train,
            data.y_train,
            data.x_test,
            data.y_test,
            folder / f"synthetic_{caption}_boundary.png",
            title=f"Decision boundary: {caption}",
        )
        rows.append(_result_row("synthetic_data", caption, train_loss, valid_loss, quality, model))
    return pd.DataFrame(rows)


def _make_folds(labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        rng.shuffle(ids)
        by_class.append(np.array_split(ids, folds))
    return [np.concatenate([part[i] for part in by_class]) for i in range(folds)]


def cross_validate(
    data: DataPack,
    learning_rates: Iterable[float],
    batch_sizes: Iterable[int],
    folds: int = 5,
    epochs: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    fold_ids = _make_folds(data.y_train, folds, seed)
    rows: list[dict[str, float | int]] = []
    all_ids = np.arange(len(data.y_train))

    for lr in learning_rates:
        for batch in batch_sizes:
            scores = []
            for fold_number, valid_ids in enumerate(fold_ids):
                train_ids = np.setdiff1d(all_ids, valid_ids, assume_unique=False)
                fold_pack = DataPack(
                    x_train=data.x_train[train_ids],
                    x_test=data.x_train[valid_ids],
                    y_train=data.y_train[train_ids],
                    y_test=data.y_train[valid_ids],
                )
                model, _, _, _ = fit_once(fold_pack, lr=lr, batch=batch, epochs=epochs, seed=seed + fold_number)
                scores.append(extended_report(fold_pack.y_test, model.predict(fold_pack.x_test), model.predict_proba(fold_pack.x_test))["accuracy"])
            rows.append(
                {
                    "learning_rate": lr,
                    "batch_size": batch,
                    "mean_accuracy": float(np.mean(scores)),
                    "std_accuracy": float(np.std(scores)),
                }
            )
    return pd.DataFrame(rows).sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])


def train_final_model_after_cv(data: DataPack, grid: pd.DataFrame, folder: Path, report_folder: Path) -> pd.DataFrame:
    best = grid.iloc[0]
    lr = float(best["learning_rate"])
    batch = int(best["batch_size"])
    model, train_loss, valid_loss, quality = fit_once(data, lr=lr, batch=batch, epochs=120, seed=123)
    plot_loss_curves(train_loss, valid_loss, folder / "final_model_loss.png", title=f"Final model after CV: eta={lr}, batch={batch}")
    plot_decision_boundary(model, data.x_train, data.y_train, data.x_test, data.y_test, folder / "final_model_boundary.png", title="Final model decision boundary")
    plot_roc_curve(data.y_test, model.predict_proba(data.x_test), folder / "final_model_roc_curve.png", title="Final model ROC curve")

    result = pd.DataFrame(
        [
            {
                "learning_rate": lr,
                "batch_size": batch,
                **quality,
                "final_train_loss": train_loss[-1],
                "final_val_loss": valid_loss[-1],
                "weight_norm": float(np.linalg.norm(model.weights)),
            }
        ]
    )
    result.to_csv(report_folder / "final_model_metrics.csv", index=False)
    return result


def main() -> None:
    plot_folder = ensure_output_dir()
    report_folder = Path("reports")
    report_folder.mkdir(exist_ok=True)
    data = load_prepared_data(test_size=0.30, seed=42)

    rows = [
        *check_learning_rate(data, plot_folder),
        *check_batch_size(data, plot_folder),
        *check_initialization(data, plot_folder),
        *check_loss_functions(data, plot_folder),
        *check_l2_regularization(data, plot_folder),
        *check_momentum(data, plot_folder),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(report_folder / "experiment_results.csv", index=False)
    print("\nExperiment results")
    print(summary.round(4))

    synthetic = check_synthetic_generators(plot_folder)
    synthetic.to_csv(report_folder / "synthetic_results.csv", index=False)
    print("\nSynthetic data results")
    print(synthetic[["parameter", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(4))

    grid = cross_validate(data, learning_rates=(0.01, 0.1, 0.5), batch_sizes=(16, 32, 64))
    grid.to_csv(report_folder / "cross_validation_results.csv", index=False)
    print("\nCross-validation results")
    print(grid.round(4))

    final_metrics = train_final_model_after_cv(data, grid, plot_folder, report_folder)
    print("\nFinal model after cross-validation")
    print(final_metrics[["learning_rate", "batch_size", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(4))


if __name__ == "__main__":
    main()
