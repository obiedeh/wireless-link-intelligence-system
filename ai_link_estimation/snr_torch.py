"""PyTorch MLP for SNR estimation from constellation statistics.

This module replaces the sklearn `snr_estimator.joblib` with a tiny PyTorch
model that is both ONNX-exportable and INT8-quantizable for edge inference.
The point is the deployment-path proof (FP32 ONNX → INT8 ONNX → measured
latency on Jetson), not a heavy DL flex.

Architecture: 3-layer MLP (12 → 32 → 16 → 1) with ReLU. ~600 parameters.
Tiny by ML standards, well-suited to the tabular 12-feature input.

Modules and utilities:
- ``SNREstimatorMLP``: the model.
- ``train_snr_mlp``: training loop with deterministic seed.
- ``export_fp32_onnx`` / ``quantize_to_int8``: deployment artifacts.
- ``evaluate_model``: MAE / file-size / inference-latency comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# 12 input features that match the existing dataset schema in
# ai_link_estimation.features.FEATURE_COLUMNS. Imported lazily to avoid
# circular import at module load.
def _feature_columns() -> list[str]:
    from ai_link_estimation.features import FEATURE_COLUMNS
    return list(FEATURE_COLUMNS)


@dataclass
class SNRTrainConfig:
    epochs: int = 400
    batch_size: int = 32
    lr: float = 3e-3
    hidden1: int = 64
    hidden2: int = 32
    seed: int = 42
    test_size: float = 0.2


@dataclass
class SNRModelResult:
    """All artifacts produced by training + export + quantization."""

    model: object  # torch.nn.Module
    fp32_onnx_path: Path
    int8_onnx_path: Path
    metrics: dict
    config: SNRTrainConfig


def build_mlp(input_dim: int, hidden1: int, hidden2: int):
    """Construct the SNR estimator MLP. Lazy import keeps the module usable
    without torch installed (for the rest of the package)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
    )


def train_snr_mlp(
    X: np.ndarray,
    y: np.ndarray,
    config: SNRTrainConfig | None = None,
):
    """Train the MLP. Returns (model, scaler, holdout_mae_fp32).

    Inputs are standardised per-feature; the scaler is returned so inference
    can reproduce the standardisation.
    """
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    config = config or SNRTrainConfig()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.seed
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    torch.manual_seed(config.seed)
    model = build_mlp(X_train_s.shape[1], config.hidden1, config.hidden2)

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()
    Xt = torch.tensor(X_train_s, dtype=torch.float32)
    Yt = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    n = Xt.shape[0]
    for epoch in range(config.epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(config.seed + epoch))
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            out = model(Xt[idx])
            loss = loss_fn(out, Yt[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()

    # Evaluate FP32 holdout MAE.
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy().ravel()
    holdout_mae = float(np.mean(np.abs(y_pred - y_test)))
    return model, scaler, holdout_mae, (X_test_s, y_test)


def export_fp32_onnx(model, output_path: Path, input_dim: int = 12) -> Path:
    """Export the MLP to a FP32 ONNX file.

    Uses the legacy TorchScript exporter (``dynamo=False``) because the
    Torch 2.12+ dynamo exporter emits Unicode emojis (✅/⚠️) on success that
    crash on Windows cp1252 consoles. The legacy exporter is byte-clean and
    sufficient for this graph (no control flow, no custom ops).
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["snr_db"],
        dynamic_axes={"features": {0: "batch"}, "snr_db": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return output_path


def quantize_to_int8(fp32_path: Path, int8_path: Path) -> Path:
    """Post-training dynamic quantisation of a FP32 ONNX model to INT8.

    Uses ONNX Runtime's dynamic quantizer. For a small MLP on tabular input,
    dynamic PTQ produces near-identical accuracy to QAT at a fraction of the
    engineering cost — and it's the standard production path for sklearn-style
    edge inference. QAT is the heavier hammer reserved for vision/audio CNNs
    where the activation range shifts substantially under quantisation.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return int8_path


def evaluate_onnx_model(
    onnx_path: Path,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    n_warmup: int = 10,
    n_timed: int = 1000,
) -> dict[str, float]:
    """Compute MAE, file size, and per-sample inference latency for an ONNX model."""
    import time

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    X = X_test_scaled.astype(np.float32)

    # MAE
    y_pred = sess.run(None, {input_name: X})[0].ravel()
    mae = float(np.mean(np.abs(y_pred - y_test)))

    # Latency: time per-sample inference.
    sample = X[:1]
    for _ in range(n_warmup):
        sess.run(None, {input_name: sample})
    start = time.perf_counter()
    for _ in range(n_timed):
        sess.run(None, {input_name: sample})
    elapsed = time.perf_counter() - start
    latency_us = (elapsed / n_timed) * 1e6

    return {
        "mae_db": mae,
        "file_size_bytes": int(Path(onnx_path).stat().st_size),
        "latency_us_per_sample": latency_us,
    }


def evaluate_torch_model(model, X_test_scaled: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute MAE for the PyTorch FP32 model directly (no ONNX)."""
    import torch

    model.eval()
    with torch.no_grad():
        y_pred = (
            model(torch.tensor(X_test_scaled.astype(np.float32))).numpy().ravel()
        )
    return {
        "mae_db": float(np.mean(np.abs(y_pred - y_test))),
    }
