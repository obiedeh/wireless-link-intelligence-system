# Jetson AGX Thor inference benchmark — run guide

This is the one-page recipe for capturing measured FP32 + INT8 inference latency for the trained SNR estimator on a Jetson AGX Thor (or any Jetson Orin / Xavier with `onnxruntime-gpu`). The result lands in `reports/jetson_inference_benchmark.json` and replaces the `<TO MEASURE>` row on the dashboard with a measured number.

The benchmark target is the **per-frame SNR-estimator inference budget** — in 5G NR with 30 kHz subcarrier spacing, a slot is 500 µs and the channel-estimator pipeline must fit inside that. Sub-millisecond is the bar; sub-100 µs is the AI-RAN target.

---

## Prereqs on the Jetson

```bash
# JetPack 6.x ships Python 3.10/3.11. Use whatever's there.
python3 --version

# Install onnxruntime-gpu for CUDA / TensorRT acceleration on Jetson.
# (If onnxruntime-gpu is unavailable for your JetPack, plain onnxruntime works on CPU.)
pip3 install onnxruntime-gpu numpy
# Or, for CPU-only fallback:
# pip3 install onnxruntime numpy
```

## Step 1 — Clone the repo on the Jetson

```bash
git clone https://github.com/obiedeh/wireless-link-intelligence-system.git
cd wireless-link-intelligence-system
```

You don't need the full repo's Python stack on the Jetson — only `numpy` and `onnxruntime[-gpu]`. The benchmark script is intentionally self-contained.

## Step 2 — Get the ONNX models onto the Jetson

The repo gitignores the ONNX binaries (they're regenerated). Two options:

**Option A — regenerate on the Jetson** (needs the full ML stack):

```bash
pip3 install -e ".[ml,edge]"
python3 generate_dataset.py --output data/link_conditions.csv --samples 500 --num-bits 4000 --seed 7
python3 train_snr_torch.py --dataset data/link_conditions.csv --epochs 400
```

**Option B — copy the ONNX models from a host machine** (faster):

On the host where you ran `make snr-torch`:

```bash
scp models/onnx/snr_estimator_fp32.onnx jetson@<JETSON_IP>:~/wireless-link-intelligence-system/models/onnx/
scp models/onnx/snr_estimator_int8.onnx jetson@<JETSON_IP>:~/wireless-link-intelligence-system/models/onnx/
```

## Step 3 — Run the benchmark

```bash
python3 edge/jetson_benchmark_template.py \
    --fp32-model models/onnx/snr_estimator_fp32.onnx \
    --int8-model models/onnx/snr_estimator_int8.onnx \
    --runs 5000 \
    --warmup 200 \
    --output reports/jetson_inference_benchmark.json
```

You should see something like:

```
--- Benchmarking FP32 model: models/onnx/snr_estimator_fp32.onnx ---
  providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
  p50 = 42.34 µs · p95 = 58.12 µs · p99 = 89.91 µs
  throughput = 23,615 inferences/sec
--- Benchmarking INT8 model: models/onnx/snr_estimator_int8.onnx ---
  providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
  p50 = 18.66 µs · p95 = 24.18 µs · p99 = 41.05 µs
  throughput = 53,602 inferences/sec
Wrote benchmark results: reports/jetson_inference_benchmark.json
```

The exact numbers will depend on your AGX Thor's power mode, JetPack version, and whether TensorRT EP is available. The script auto-detects providers in priority order: `TensorrtExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider`.

## Step 4 — Send the JSON back

```bash
cat reports/jetson_inference_benchmark.json
```

Paste the contents into the next message and I'll commit it to `main`, update the dashboard so the `<TO MEASURE>` row becomes measured numbers, and update the README headline-evidence table accordingly.

---

## What gets recorded

The JSON includes:
- `device_info`: platform, processor, Jetson model from `/proc/device-tree/model`, JetPack marker file excerpt.
- `models.fp32.{p50_us, p95_us, p99_us, mean_us, std_us, throughput_infs_per_sec}` — tail-aware latency, not just the mean.
- `models.int8.*` — same for the INT8 quantized version.
- `models.*.providers_used` — exactly which ONNX Runtime execution provider was selected.

The summary that lands on the dashboard is **p50 / p95 / p99 in µs** plus **throughput in inferences/sec**, for both FP32 and INT8.

---

## Common gotchas

- **`onnxruntime-gpu` install fails on Jetson** — you may need NVIDIA's pre-built wheel. See [NVIDIA Developer Forums: Jetson ONNX Runtime](https://elinux.org/Jetson_Zoo#ONNX_Runtime) for the matrix of (JetPack version × ORT version × Python version) wheels.
- **`CUDAExecutionProvider` not in the available list** — fall back to `onnxruntime` (no `-gpu`). CPU-only latency on Cortex-A78AE / Cortex-X925 (AGX Thor's host CPU) will still be in the low hundreds of µs.
- **TensorRT provider isn't selected** — `TensorrtExecutionProvider` needs the TensorRT EP enabled in the ORT build. Pre-built `onnxruntime-gpu` for Jetson usually has CUDA EP but not TensorRT EP. If you want TensorRT, you'd convert the ONNX to a `.plan` engine with `trtexec` and benchmark with the TensorRT C++/Python API instead — that's a known scope expansion documented in `reports/edge_inference_plan.md`.

The benchmark template is fine on CUDA EP for the AI-RAN-relevant latency story.
