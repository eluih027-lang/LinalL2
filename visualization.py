from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import roc_auc_score, roc_curve_points


def ensure_output_dir(folder: str | Path = "reports/figures") -> Path:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curves(
    train_loss: list[float],
    val_loss: list[float],
    save_path: str | Path | None = None,
    title: str = "Loss by epoch",
    val_label: str = "test",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_loss, label="train")
    if val_loss:
        ax.plot(val_loss, label=val_label)
    ax.set(title=title, xlabel="epoch", ylabel="loss")
    ax.grid(alpha=0.25)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_decision_boundary(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    save_path: str | Path | None = None,
    title: str = "Separating line",
) -> None:
    cloud = np.vstack((x_train, x_test))
    x_left, x_right = cloud[:, 0].min() - 0.7, cloud[:, 0].max() + 0.7
    y_bottom, y_top = cloud[:, 1].min() - 0.7, cloud[:, 1].max() + 0.7
    x_axis = np.linspace(x_left, x_right, 280)
    y_axis = np.linspace(y_bottom, y_top, 280)
    xx, yy = np.meshgrid(x_axis, y_axis)

    surface = model.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, surface, levels=25, alpha=0.22)
    ax.contour(xx, yy, surface, levels=[0.5], linewidths=2)
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker="o", edgecolors="k", label="train")
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker="x", label="test")
    ax.set(title=title, xlabel="x1, normalized", ylabel="x2, normalized")
    ax.grid(alpha=0.2)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_score, save_path: str | Path | None = None, title: str = "ROC curve") -> None:
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="random classifier")
    ax.set(title=title, xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_error_analysis(model, x_test, y_test, save_path: str | Path | None = None) -> None:
    predicted = model.predict(x_test)
    correct = predicted == np.asarray(y_test).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_test[correct, 0], x_test[correct, 1], c=y_test[correct], marker="o", edgecolors="k", label="correct")
    if (~correct).any():
        ax.scatter(
            x_test[~correct, 0],
            x_test[~correct, 1],
            c=y_test[~correct],
            marker="X",
            s=120,
            edgecolors="k",
            label="wrong",
        )
    ax.set(title="Error analysis on test set", xlabel="x1, normalized", ylabel="x2, normalized")
    ax.grid(alpha=0.25)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
