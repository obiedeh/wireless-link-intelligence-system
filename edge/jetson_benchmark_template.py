"""Jetson ONNX Runtime latency benchmark for the SNR estimator.

Designed to run on a Jetson Orin / AGX Orin / AGX Thor with ``onnxruntime-gpu``
installed. Falls back to CPU on any host where ONNX Runtime is available.
Benchmarks both the FP32 and INT8 exports head-to-head and reports
p50 / p95 / p99 latency plus throughput. Output: ``reports/jetson_inference_benchmark.json``.

Run on the Jetson::

    pip install onnxruntime-gpu  # or onnxruntime if not GPU
    python edge/jetson_benchmark_template.py \\
        --fp32-model models/onnx/snr_estimator_fp32.onnx \\
        --int8-model models/onnx/snr_estimator_int8.onnx \\
        --runs 5000 --warmup 200 \\
        --output reports/jetson_inference_benchmark.json

The script is intentionally self-contained — it has only one runtime
dependency beyond NumPy (``onnxruntime`` / ``onnxruntime-gpu``) so it ports
cleanly onto a fresh Jetson that doesn't have the full repo virtualenv.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def _detect_runtime_info() -> dict[str, str]:
    """Best-effort device info — runs on any platform, gracefully degrades."""
    info: dict[str, str] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
    }
    # Jetson-specific: read the device tree model if present.
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            # /proc/device-tree/model is null-terminated.
            info["jetson_model"] = model_path.read_text(encoding="utf-8").strip("\x00").strip()
        except OSError:
            pass
    # JetPack version (best effort — installed file path varies by JetPack version).
    jetpack_candidates = [
        Path("/etc/nv_tegra_release"),
        Path("/etc/nvpmodel.conf"),
    ]
    for cand in jetpack_candidates:
        if cand.exists():
            try:
                info["jetpack_marker_file"] = str(cand)
                info["jetpack_marker_excerpt"] = cand.read_text(encoding="utf-8", errors="ignore").splitlines()[0][:120]
            except OSError:
                pass
            break
    return info


def _select_providers() -> list[str]:
    """Return the ONNX Runtime provider list in priority order."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    desired = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in desired if p in available]


def _latency_stats_us(latencies_seconds: list[float]) -> dict[str, float]:
    """Compute p50/p95/p99/mean/std from a list of per-run latencies (seconds)."""
    arr = np.asarray(latencies_seconds)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99]) * 1e6
    return {
        "p50_us": float(p50),
        "p95_us": float(p95),
        "p99_us": float(p99),
        "mean_us": float(arr.mean() * 1e6),
        "std_us": float(arr.std() * 1e6),
    }


def benchmark(
    model_path: str | Path,
    runs: int,
    warmup: int,
    batch_size: int,
    input_dim: int,
) -> dict:
    """Time per-inference latency on a single fixed input.

    Records every individual run's latency, then reports p50 / p95 / p99
    rather than just the mean — that's the right summary for real-time
    inference budgets, which are bound by the tail, not the mean.
    """
    import onnxruntime as ort

    providers = _select_providers()
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(
        str(model_path), sess_options=sess_opts, providers=providers
    )
    input_name = sess.get_inputs()[0].name
    x = np.random.default_rng(seed=0).standard_normal(
        (batch_size, input_dim)
    ).astype(np.float32)

    # Warm-up — first inferences include kernel autotuning on GPU providers.
    for _ in range(warmup):
        sess.run(None, {input_name: x})

    # Timed runs.
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        latencies.append(time.perf_counter() - t0)

    throughput = (runs * batch_size) / sum(latencies)
    stats = _latency_stats_us(latencies)

    return {
        "model_path": str(model_path),
        "providers_used": providers,
        "runs": runs,
        "warmup": warmup,
        "batch_size": batch_size,
        "input_dim": input_dim,
        **stats,
        "throughput_infs_per_sec": float(throughput),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jetson ONNX Runtime latency benchmark for the SNR estimator."
    )
    parser.add_argument("--fp32-model", type=Path, default=Path("models/onnx/snr_estimator_fp32.onnx"))
    parser.add_argument("--int8-model", type=Path, default=Path("models/onnx/snr_estimator_int8.onnx"))
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=12, help="Number of input features.")
    parser.add_argument("--output", type=Path, default=Path("reports/jetson_inference_benchmark.json"))
    args = parser.parse_args()

    results: dict = {
        "device_info": _detect_runtime_info(),
        "models": {},
    }

    for label, path in [("fp32", args.fp32_model), ("int8", args.int8_model)]:
        if not path.exists():
            print(f"[WARN] {label} model not found at {path} — skipping.")
            continue
        print(f"--- Benchmarking {label.upper()} model: {path} ---")
        bench = benchmark(
            model_path=path,
            runs=args.runs,
            warmup=args.warmup,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
        )
        results["models"][label] = bench
        print(
            f"  providers: {bench['providers_used']}\n"
            f"  p50 = {bench['p50_us']:.2f} µs · p95 = {bench['p95_us']:.2f} µs · "
            f"p99 = {bench['p99_us']:.2f} µs\n"
            f"  throughput = {bench['throughput_infs_per_sec']:,.0f} inferences/sec"
        )

    if results["models"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote benchmark results: {args.output}")
    else:
        print("No models benchmarked — nothing written.")


if __name__ == "__main__":
    main()
