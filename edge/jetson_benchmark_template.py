"""Jetson ONNX Runtime benchmark template for link-estimation models.

Run after exporting ONNX models:

    python export_onnx.py
    python edge/jetson_benchmark_template.py --model models/onnx/snr_estimator.onnx

This measures inference overhead for tabular link-estimation features. It does
not benchmark a production receiver or a full AI-RAN base station workload.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def benchmark(model_path: str | Path, runs: int = 1000, batch_size: int = 1) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime on the target Jetson before benchmarking.") from exc

    from ai_link_estimation.features import FEATURE_COLUMNS

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(batch_size, len(FEATURE_COLUMNS)).astype(np.float32)

    for _ in range(25):
        session.run(None, {input_name: x})

    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: x})
    elapsed = time.perf_counter() - start
    return {
        "runs": float(runs),
        "batch_size": float(batch_size),
        "total_seconds": elapsed,
        "mean_latency_ms": (elapsed / runs) * 1000.0,
        "inferences_per_second": runs * batch_size / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an exported ONNX estimator on Jetson.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    result = benchmark(args.model, runs=args.runs, batch_size=args.batch_size)
    for key, value in result.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
