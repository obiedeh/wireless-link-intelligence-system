"""Train and evaluate ML estimators for synthetic QPSK link conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS


MODEL_SPECS = {
    "snr_estimator": "snr_db",
    "ber_predictor": "ber",
    "channel_classifier": "channel_is_rayleigh",
    "link_quality_scorer": "link_quality_score",
}


def load_dataset(csv_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, str]]]:
    """Load a generated CSV dataset into feature and label arrays."""
    rows: list[dict[str, str]] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    x = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = {
        "snr_db": np.array([float(row["snr_db"]) for row in rows], dtype=float),
        "ber": np.array([float(row["ber"]) for row in rows], dtype=float),
        "channel_is_rayleigh": np.array([int(row["channel_is_rayleigh"]) for row in rows], dtype=int),
        "link_quality_score": np.array([float(row["link_quality_score"]) for row in rows], dtype=float),
    }
    return x, y, rows


def _make_model(name: str) -> Any:
    if name == "channel_classifier":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            random_state=11,
        )
    return RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=2,
        random_state=11,
    )


def train_models(csv_path: str | Path = "data/link_conditions.csv",
                 output_dir: str | Path = "models",
                 test_size: float = 0.25,
                 seed: int = 11) -> dict[str, Any]:
    """Train SNR, BER, channel, and quality estimators from generated data."""
    x, labels, rows = load_dataset(csv_path)
    indices = np.arange(x.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "dataset": str(csv_path),
        "samples": int(x.shape[0]),
        "test_samples": int(test_idx.size),
        "features": FEATURE_COLUMNS,
        "models": {},
    }

    predictions: dict[str, np.ndarray] = {}
    for model_name, target in MODEL_SPECS.items():
        model = _make_model(model_name)
        model.fit(x[train_idx], labels[target][train_idx])
        pred = model.predict(x[test_idx])
        predictions[model_name] = pred
        joblib.dump(model, output_dir / f"{model_name}.joblib")

        if model_name == "channel_classifier":
            metrics["models"][model_name] = {
                "target": target,
                "accuracy": float(accuracy_score(labels[target][test_idx], pred)),
            }
        else:
            metrics["models"][model_name] = {
                "target": target,
                "mae": float(mean_absolute_error(labels[target][test_idx], pred)),
                "r2": float(r2_score(labels[target][test_idx], pred)),
            }

    metrics["comparison_examples"] = _comparison_examples(rows, test_idx, labels, predictions)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _comparison_examples(rows: list[dict[str, str]],
                         test_idx: np.ndarray,
                         labels: dict[str, np.ndarray],
                         predictions: dict[str, np.ndarray],
                         limit: int = 12) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for offset, row_idx in enumerate(test_idx[:limit]):
        measured_channel = "rayleigh" if labels["channel_is_rayleigh"][row_idx] else "awgn"
        predicted_channel = "rayleigh" if int(predictions["channel_classifier"][offset]) else "awgn"
        examples.append({
            "sample_id": int(rows[row_idx]["sample_id"]),
            "channel": measured_channel,
            "predicted_channel": predicted_channel,
            "snr_db": float(labels["snr_db"][row_idx]),
            "predicted_snr_db": float(predictions["snr_estimator"][offset]),
            "measured_ber": float(labels["ber"][row_idx]),
            "predicted_ber": float(predictions["ber_predictor"][offset]),
            "quality_score": float(labels["link_quality_score"][row_idx]),
            "predicted_quality_score": float(predictions["link_quality_scorer"][offset]),
        })
    return examples


def write_comparison_report(metrics: dict[str, Any],
                            output_path: str | Path = "reports/link_estimation_report.md") -> Path:
    """Write a grounded Markdown report from training/evaluation metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snr = metrics["models"]["snr_estimator"]
    ber = metrics["models"]["ber_predictor"]
    channel = metrics["models"]["channel_classifier"]
    quality = metrics["models"]["link_quality_scorer"]

    lines = [
        "# AI-Assisted Wireless Link Estimation Report",
        "",
        "This report summarizes synthetic experiments around a classical QPSK baseline. "
        "The project estimates link conditions from simulated constellation statistics; "
        "it is not a full AI-RAN base station and not a production telecom receiver.",
        "",
        "## Dataset",
        "",
        f"- Source CSV: `{metrics['dataset']}`",
        f"- Samples: {metrics['samples']}",
        f"- Held-out test samples: {metrics['test_samples']}",
        "- Labels: SNR, measured BER, channel type, and synthetic link-quality score",
        "- Features: constellation power, I/Q moments, EVM, quadrant balance, and fading coefficient summary",
        "",
        "## Model Results",
        "",
        f"- SNR estimation MAE: {snr['mae']:.3f} dB",
        f"- SNR estimation R2: {snr['r2']:.3f}",
        f"- BER prediction MAE: {ber['mae']:.6f}",
        f"- BER prediction R2: {ber['r2']:.3f}",
        f"- AWGN vs Rayleigh classification accuracy: {channel['accuracy']:.3f}",
        f"- Link-quality scoring MAE: {quality['mae']:.3f}",
        "",
        "## Classical BER vs Predicted BER",
        "",
        "| Sample | Channel | Measured BER | Predicted BER | SNR dB | Predicted SNR dB | Predicted Channel |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in metrics["comparison_examples"]:
        lines.append(
            f"| {item['sample_id']} | {item['channel']} | {item['measured_ber']:.6f} | "
            f"{item['predicted_ber']:.6f} | {item['snr_db']:.2f} | "
            f"{item['predicted_snr_db']:.2f} | {item['predicted_channel']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The measured BER remains the classical simulator baseline.",
        "- ML predictions are estimates from synthetic features and should be validated against any real RF capture before use.",
        "- AWGN/Rayleigh classification is a controlled two-class experiment, not generalized channel recognition.",
        "- SNR error is reported on held-out synthetic samples and should not be treated as field performance.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AI-assisted QPSK link estimators.")
    parser.add_argument("--dataset", default="data/link_conditions.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--report", default="reports/link_estimation_report.md")
    args = parser.parse_args()

    metrics = train_models(csv_path=args.dataset, output_dir=args.output_dir)
    report_path = write_comparison_report(metrics, output_path=args.report)
    print(json.dumps(metrics["models"], indent=2))
    print(f"Wrote comparison report: {report_path}")


if __name__ == "__main__":
    main()
