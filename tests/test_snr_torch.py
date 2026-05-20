"""Tests for the PyTorch SNR estimator + ONNX export + INT8 quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from ai_link_estimation.snr_torch import (  # noqa: E402
    SNRTrainConfig,
    build_mlp,
    evaluate_onnx_model,
    evaluate_torch_model,
    export_fp32_onnx,
    quantize_to_int8,
    train_snr_mlp,
)


def _small_synthetic_dataset(n_samples: int = 80, n_features: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # Simple linear truth so the tiny MLP can learn quickly.
    w_true = rng.standard_normal(n_features)
    y = X @ w_true + 0.05 * rng.standard_normal(n_samples)
    return X, y


def test_build_mlp_input_output_shapes():
    model = build_mlp(input_dim=12, hidden1=32, hidden2=16)
    x = torch.zeros(3, 12, dtype=torch.float32)
    y = model(x)
    assert y.shape == (3, 1)


def test_train_snr_mlp_smoke():
    """Tiny config so the test stays fast; verify training runs and MAE is finite."""
    X, y = _small_synthetic_dataset()
    cfg = SNRTrainConfig(epochs=10, batch_size=16, lr=1e-2, hidden1=16, hidden2=8, seed=1)
    model, scaler, mae, (X_test_s, y_test) = train_snr_mlp(X, y, cfg)
    assert np.isfinite(mae)
    assert mae > 0  # not perfect
    assert hasattr(scaler, "transform")
    assert X_test_s.shape[1] == 12
    assert y_test.shape[0] == X_test_s.shape[0]


def test_export_fp32_onnx_creates_file(tmp_path: Path):
    X, y = _small_synthetic_dataset()
    cfg = SNRTrainConfig(epochs=5, batch_size=16, hidden1=8, hidden2=4, seed=2)
    model, _, _, _ = train_snr_mlp(X, y, cfg)
    out = tmp_path / "snr_fp32.onnx"
    export_fp32_onnx(model, out, input_dim=12)
    assert out.exists()
    assert out.stat().st_size > 0


def test_quantize_to_int8_creates_smaller_file_or_at_least_runs(tmp_path: Path):
    """INT8 quantisation should either shrink the file (typical) or at worst
    produce a valid output. The point of the test is that the pipeline runs
    and yields a usable ONNX file, not the size ratio (which depends on
    the model's matmul vs metadata ratio)."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    X, y = _small_synthetic_dataset()
    cfg = SNRTrainConfig(epochs=5, batch_size=16, hidden1=16, hidden2=8, seed=3)
    model, scaler, _, (X_test_s, y_test) = train_snr_mlp(X, y, cfg)
    fp32_path = tmp_path / "snr_fp32.onnx"
    int8_path = tmp_path / "snr_int8.onnx"
    export_fp32_onnx(model, fp32_path, input_dim=12)
    quantize_to_int8(fp32_path, int8_path)
    assert int8_path.exists()
    # Verify inference still works after quantisation.
    metrics = evaluate_onnx_model(int8_path, X_test_s, y_test, n_warmup=2, n_timed=20)
    assert np.isfinite(metrics["mae_db"])
    assert metrics["file_size_bytes"] > 0
    assert metrics["latency_us_per_sample"] > 0


def test_evaluate_torch_model_returns_finite_mae():
    X, y = _small_synthetic_dataset()
    cfg = SNRTrainConfig(epochs=5, batch_size=16, hidden1=16, hidden2=8, seed=4)
    model, _, _, (X_test_s, y_test) = train_snr_mlp(X, y, cfg)
    out = evaluate_torch_model(model, X_test_s, y_test)
    assert np.isfinite(out["mae_db"])
