"""Train the PyTorch SNR estimator, export FP32 + INT8 ONNX, write comparison.

End-to-end edge-deployment proof for the link-estimation work:

  link_conditions.csv  ─► PyTorch MLP (FP32)  ─► FP32 ONNX  ─► INT8 ONNX

Reports holdout MAE, file size, and CPU inference latency for each
deployment artifact. Output: reports/snr_quantization_comparison.json.

Run::

    python train_snr_torch.py --dataset data/link_conditions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

from ai_link_estimation.features import FEATURE_COLUMNS
from ai_link_estimation.snr_torch import (
    SNRTrainConfig,
    evaluate_onnx_model,
    evaluate_torch_model,
    export_fp32_onnx,
    quantize_to_int8,
    train_snr_mlp,
)


def _load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from the synthetic link-condition CSV."""
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATURE_COLUMNS] for r in rows])
    y = np.array([float(r["snr_db"]) for r in rows])
    return X, y


def _maybe_load_sklearn_baseline(
    sklearn_path: Path,
    X_test_scaled_pytorch: np.ndarray,
    y_test: np.ndarray,
    scaler,
    X_test_unscaled: np.ndarray,
) -> dict[str, float] | None:
    """If the sklearn estimator is available, evaluate it on the same holdout
    for a baseline comparison. Returns None if not present."""
    if not sklearn_path.exists():
        return None
    sk = joblib.load(sklearn_path)
    # The sklearn estimator was trained on unscaled features (it has its own
    # internal preprocessing); use the unscaled holdout for fair MAE.
    y_pred = sk.predict(X_test_unscaled)
    return {
        "mae_db": float(mean_absolute_error(y_test, y_pred)),
        "model_path": str(sklearn_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PyTorch SNR estimator → export FP32 + INT8 ONNX, compare.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/link_conditions.csv"))
    parser.add_argument("--fp32-output", type=Path, default=Path("models/onnx/snr_estimator_fp32.onnx"))
    parser.add_argument("--int8-output", type=Path, default=Path("models/onnx/snr_estimator_int8.onnx"))
    parser.add_argument("--scaler-output", type=Path, default=Path("models/snr_scaler.joblib"))
    parser.add_argument("--sklearn-baseline", type=Path, default=Path("models/snr_estimator.joblib"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/snr_quantization_comparison.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}. Run generate_dataset.py first."
        )

    print(f"Loading dataset {args.dataset} ...")
    X, y = _load_dataset(args.dataset)
    print(f"  {X.shape[0]} samples × {X.shape[1]} features")

    cfg = SNRTrainConfig(epochs=args.epochs, seed=args.seed)
    print(f"Training PyTorch MLP (epochs={cfg.epochs}, lr={cfg.lr}, seed={cfg.seed}) ...")
    model, scaler, fp32_torch_mae, (X_test_s, y_test) = train_snr_mlp(X, y, cfg)
    print(f"  PyTorch FP32 holdout MAE = {fp32_torch_mae:.4f} dB")

    # Save scaler for inference reproducibility.
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.scaler_output)

    print(f"Exporting FP32 ONNX -> {args.fp32_output}")
    export_fp32_onnx(model, args.fp32_output, input_dim=X.shape[1])

    print(f"Quantising to INT8 ONNX -> {args.int8_output}")
    quantize_to_int8(args.fp32_output, args.int8_output)

    print("Evaluating all artifacts on the same holdout ...")
    torch_metrics = evaluate_torch_model(model, X_test_s, y_test)
    fp32_metrics = evaluate_onnx_model(args.fp32_output, X_test_s, y_test)
    int8_metrics = evaluate_onnx_model(args.int8_output, X_test_s, y_test)

    # Reconstruct the unscaled holdout for fair sklearn-baseline comparison.
    # scaler.inverse_transform(X_test_s) reverses the per-feature scaling.
    X_test_unscaled = scaler.inverse_transform(X_test_s)
    sklearn_metrics = _maybe_load_sklearn_baseline(
        args.sklearn_baseline, X_test_s, y_test, scaler, X_test_unscaled
    )

    comparison = {
        "config": {
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "hidden1": cfg.hidden1,
            "hidden2": cfg.hidden2,
            "input_features": FEATURE_COLUMNS,
            "n_samples": int(X.shape[0]),
            "n_holdout": int(X_test_s.shape[0]),
        },
        "sklearn_baseline": sklearn_metrics,
        "pytorch_fp32": {
            "mae_db": torch_metrics["mae_db"],
        },
        "onnx_fp32": fp32_metrics,
        "onnx_int8": int8_metrics,
        "interpretation": (
            "Dynamic INT8 quantisation typically incurs <0.05 dB MAE drift on a "
            "small MLP regression task. The expected payoff is ~3-4× smaller "
            "model file and ~1.5-3× faster inference on CPU. Latency on Jetson "
            "AGX Thor is reported separately in reports/jetson_inference_benchmark.json."
        ),
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print("=== Comparison ===")
    if sklearn_metrics is not None:
        print(f"  sklearn baseline   | MAE = {sklearn_metrics['mae_db']:.4f} dB")
    print(f"  PyTorch FP32       | MAE = {torch_metrics['mae_db']:.4f} dB")
    print(
        f"  ONNX FP32          | MAE = {fp32_metrics['mae_db']:.4f} dB"
        f" | size = {fp32_metrics['file_size_bytes']:,} B"
        f" | latency = {fp32_metrics['latency_us_per_sample']:.2f} µs/sample"
    )
    print(
        f"  ONNX INT8 (dyn PTQ)| MAE = {int8_metrics['mae_db']:.4f} dB"
        f" | size = {int8_metrics['file_size_bytes']:,} B"
        f" | latency = {int8_metrics['latency_us_per_sample']:.2f} µs/sample"
    )
    print(f"Wrote metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
