from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset import load_prepared_data
from metrics import extended_report
from perceptron import Perceptron
from visualization import ensure_output_dir, plot_decision_boundary, plot_error_analysis, plot_loss_curves, plot_roc_curve


def teach_model() -> tuple[Perceptron, list[float], list[float]]:
    data = load_prepared_data(test_size=0.30, seed=42)
    neuron = Perceptron(
        n_features=data.x_train.shape[1],
        learning_rate=0.1,
        epochs=100,
        batch_size=32,
        init_mode="normal",
        random_state=42,
        verbose=True,
    )
    train_loss, test_loss = neuron.fit(data.x_train, data.y_train, data.x_test, data.y_test)
    return neuron, train_loss, test_loss


def main() -> None:
    folder = ensure_output_dir()
    report_folder = Path("reports")
    report_folder.mkdir(exist_ok=True)
    data = load_prepared_data(test_size=0.30, seed=42)

    model = Perceptron(
        n_features=data.x_train.shape[1],
        learning_rate=0.1,
        epochs=100,
        batch_size=32,
        init_mode="normal",
        random_state=42,
        verbose=True,
    )
    train_loss, test_loss = model.fit(data.x_train, data.y_train, data.x_test, data.y_test)

    train_score = model.predict_proba(data.x_train)
    test_score = model.predict_proba(data.x_test)
    table = pd.DataFrame(
        {
            "train": extended_report(data.y_train, model.predict(data.x_train), train_score),
            "test": extended_report(data.y_test, model.predict(data.x_test), test_score),
        }
    ).T
    table.to_csv(report_folder / "baseline_metrics.csv")

    print("\nQuality")
    print(table[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4))

    plot_loss_curves(train_loss, test_loss, folder / "baseline_loss.png", title="Baseline binary cross-entropy")
    plot_decision_boundary(model, data.x_train, data.y_train, data.x_test, data.y_test, folder / "decision_boundary.png")
    plot_roc_curve(data.y_test, test_score, folder / "roc_curve.png")
    plot_error_analysis(model, data.x_test, data.y_test, folder / "error_analysis.png")


if __name__ == "__main__":
    main()
