# Wireless Link Intelligence System

**Signal-processing correctness + AI-assisted link estimation + edge deployment evidence.** A production-discipline reference for physical-layer AI: a classical QPSK baseband simulator with deterministic BER vs SNR sweeps, four scikit-learn estimators that learn link conditions from constellation statistics, an ONNX export path validated against a Jetson benchmark template, and end-to-end reproducibility from a fresh clone in under five minutes.

The deliverable is the engineering pattern, not a production receiver. Every BER number is regenerable from a deterministic seed; every ML metric is reported on a stratified holdout; the channel classifier's weak 0.472 accuracy is surfaced as a calibrated finding rather than hidden in a footnote.

> **▶ [Open the live dashboard](https://obiedeh.github.io/wireless-link-intelligence-system/reports/dashboard.html)** &nbsp;·&nbsp; [Tech brief](TECH_BRIEF.md) &nbsp;·&nbsp; [Source code](https://github.com/obiedeh/wireless-link-intelligence-system)

---

## Why this exists

AI-RAN and edge-AI wireless systems depend on a measurable physical layer. Most repos in this space either skip the signal-processing work (claim ML wins without showing the classical baseline) or skip the discipline (claim production-grade results from notebook-only experiments). Neither is a defensible engineering pattern.

This repo demonstrates the engineering pattern that makes physical-layer ML credible: **classical baseline first** (verified BER curves matching textbook predictions), **ML estimators second** (with honest holdout metrics and disclosed weaknesses), **edge deployment path third** (ONNX export + Jetson benchmark template, ready when hardware lands).

**The discipline is the deliverable.** The QPSK math is textbook, the dataset is synthetic, the Jetson latency row is honest `<TO MEASURE>` until hardware arrives. What's defensible end-to-end is the methodology, the reproducibility, and the calibration of confidence.

---

## Headline Evidence

| Signal | Value | Source |
|---|---:|---|
| **PHY modem** | CP-OFDM, 64 subcarriers, adaptive QAM 4 / 16 / 64 / 256 (Gray-coded) | `qpsk_link/ofdm.py` · `reports/ber_full_ofdm_awgn.csv` |
| **3GPP channel models** | TDL-A / TDL-B / TDL-C from TR 38.901 §7.7.2 | `qpsk_link/tdl_channel.py` · `reports/bler_full_tdl_ofdm.csv` |
| **Channel estimation** | LS / MMSE / Neural (PyTorch) head-to-head on TDL-C — neural wins at low SNR | `qpsk_link/channel_estimation.py` · `reports/channel_estimation_comparison.csv` |
| **Edge deployment** | PyTorch FP32 → ONNX FP32 → ONNX INT8 (dynamic PTQ), ~3.3× CPU latency drop | `train_snr_torch.py` · `reports/snr_quantization_comparison.json` |
| **Jetson AGX Thor** | benchmark template hardware-ready; run via `JETSON_BENCHMARK_GUIDE.md` | `edge/jetson_benchmark_template.py` |
| AWGN BER full sweep (1M bits) | 2.42e-3 @ 0 dB → 1.83e-4 @ 2 dB → below 1e-6 sim floor at 6+ dB | `reports/ber_full_awgn.csv` |
| Ensemble Rayleigh BER (200 × 10k bits, transmit-power-SNR) | 4.18e-2 @ 0 dB → 5.2e-4 @ 20 dB | `reports/ber_full_rayleigh.csv` |
| SNR estimator — MAE / R² (synthetic features) | 0.118 dB / **0.999** | `reports/link_estimation_metrics.json` |
| Channel classifier — accuracy | **0.472** *(honest weak result — surfaced, not hidden)* | same |
| Test suite | **77 tests**, green on CI matrix (Python 3.11 + 3.12) | `.github/workflows/ci.yml` |
| End-to-end reproducible | `make verify` | regenerates every committed artifact under `reports/` |
| Executive dashboard | [`reports/dashboard.html`](https://obiedeh.github.io/wireless-link-intelligence-system/reports/dashboard.html) | one HTML page on GitHub Pages |
| Tech brief | [TECH_BRIEF.md](TECH_BRIEF.md) | one-page hiring-manager summary |

Full numbers and methodology in [Measured Metrics](#measured-metrics) below. Limitations and what production would require in [Positioning](#positioning).

---

## Dashboard Evidence

Open the rendered dashboard here: [reports/dashboard.html](https://obiedeh.github.io/wireless-link-intelligence-system/reports/dashboard.html).

It summarizes:

- adaptive QAM BER across CP-OFDM QPSK / 16-QAM / 64-QAM / 256-QAM
- 3GPP TR 38.901 TDL-A/B/C BLER multipath stress behavior
- LS vs MMSE vs neural channel-estimation comparison
- ONNX FP32/INT8 deployment path and quantization trade-off
- weak channel classifier result disclosed instead of hidden
- Jetson AGX Thor benchmark template ready, with latency marked pending until measured

---

## Engineering practices that matter here

These are the concrete decisions that separate a clean physical-layer reference from a notebook with a model in it:

- **CP-OFDM with adaptive QAM — not just QPSK.** `qpsk_link/ofdm.py` implements a 64-subcarrier CP-OFDM modem with Gray-coded square QAM at M = 4 / 16 / 64 / 256. Constellations normalised to unit average symbol energy; Gray property verified by an explicit test that walks the I/Q grid and checks every neighbour pair has Hamming distance exactly 1. The resulting BER vs SNR curves match textbook 5G NR link-adaptation tables.
- **3GPP TR 38.901 TDL-A / TDL-B / TDL-C channels.** `qpsk_link/tdl_channel.py` transcribes the literal NLOS tap profiles from TR 38.901 §7.7.2 Tables 7.7.2-1/2/3. Block fading per realisation, power normalised so `E[Σ|h|²] = 1`. Ensemble BLER curves committed to `reports/bler_full_tdl_ofdm.csv`. The honest finding (BLER ~10% even at 30 dB without coding) is the signal that motivates LDPC + HARQ — surfaced, not polished away.
- **Pilot-based channel estimation with LS / MMSE / neural compared head-to-head.** `qpsk_link/channel_estimation.py` runs all three on the same TDL-C realisations and reports both channel-MSE and resulting BLER. The PyTorch MLP is the DeepRx pattern in miniature; the calibrated finding is that neural wins at low SNR (denoising), MMSE wins at high SNR (correct prior + low noise = closed-form optimum). LS lags everywhere.
- **PyTorch + INT8 ONNX deployment pipeline.** `train_snr_torch.py` trains a small MLP, exports FP32 ONNX, dynamic-PTQ quantises to INT8 ONNX, and benchmarks holdout MAE + file size + CPU latency for all three forms. Measured: ~3.3× CPU latency reduction and ~2× smaller file with sub-0.01 dB accuracy drift — textbook PTQ payoff. INT8 ONNX lands directly on Jetson AGX Thor via `edge/jetson_benchmark_template.py`.
- **No feature leakage** for the link-condition estimators: `fading_abs` and `fading_phase` are saved in the dataset CSV as labels but excluded from `FEATURE_COLUMNS` in `ai_link_estimation/features.py`. They encode oracle channel knowledge and would trivially inflate any classifier built on them — `AGENTS.md` non-negotiable rule #1.
- **Two-pass channel verification.** A bug in earlier revisions had `add_awgn` referencing noise to received power instead of transmit power, making the Rayleigh penalty cancel out at the receiver. Caught by ensemble measurement, fixed (`add_awgn` gained an optional `reference_power`, `apply_channel` now passes pre-fading transmit power), and verified with a regression gate: `reports/ber_smoke_awgn.csv` must regenerate bit-identically.
- **CI runs on Python 3.11 AND 3.12.** Most portfolio repos pin one version; this one validates both, including the PyTorch + ONNX + INT8 quantisation pipeline.
- **Deterministic everywhere it matters.** `np.random.default_rng(seed)` is threaded through the BER sweep, dataset generation, model training, channel-estimator training, and ONNX export — every committed number is byte-reproducible from the committed seed.

If you are evaluating physical-layer ML engineering: these are the signals that distinguish a reference repo from a tutorial.

---

## Core Stack

**Implemented:** Python · NumPy · SciPy · scikit-learn · matplotlib · ONNX export · BER analysis · permutation-aware feature design

**Optional extension:** Jetson ONNX Runtime latency benchmarking (template ready, hardware not yet measured)

<p>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue" alt="Python" />
  <img src="https://img.shields.io/badge/NumPy-numerics-013243" alt="NumPy" />
  <img src="https://img.shields.io/badge/SciPy-signal%20processing-8CAAE6" alt="SciPy" />
  <img src="https://img.shields.io/badge/scikit--learn-link%20estimators-F7931E" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/ONNX-edge%20export-005CED" alt="ONNX" />
  <img src="https://img.shields.io/badge/Jetson-benchmark%20template-76B900" alt="Jetson benchmark template" />
</p>

---

## What this is

| Layer | What it does |
|---|---|
| **QPSK modem** | Gray-coded mapping, root-raised-cosine pulse shaping, matched filtering, hard-decision demapping. Deterministic from a single seed. |
| **Channel models** | AWGN with explicit `reference_power` parameter; flat Rayleigh fading with optional ensemble averaging. The transmit-power-SNR convention is enforced — the Rayleigh diversity penalty is visible, not cancelled. |
| **BER sweep harness** | Single-shot (`run_sim.py`) and ensemble (`run_sim_ensemble.py`) over a configurable SNR grid; CSV + SVG output. Smoke (2k bits) and full (1M bits) regimes. |
| **Link-condition dataset** | Synthetic CSV (`data/link_conditions.csv`) with 12 constellation statistics + 4 labels (SNR, BER, channel type, link-quality score). |
| **ML link estimators** | Four scikit-learn estimators: SNR regressor (R² 0.999), BER regressor (R² 0.968), channel-type classifier (acc 0.472 — disclosed weak), link-quality scorer (R² 0.904). |
| **ONNX export** | Each `.joblib` estimator converts to ONNX via `skl2onnx`. Output models live under `models/onnx/` (gitignored). |
| **Edge benchmark template** | `edge/jetson_benchmark_template.py` runs `onnxruntime` on any host; designed to drop onto a Jetson and emit latency p50/p95/p99 into `reports/jetson_inference_benchmark.json` when hardware lands. |
| **Reports** | BER CSVs + SVGs, model metrics JSON, plain-text link-estimation report, single-page HTML executive dashboard. |

---

## Measured Metrics

Source: [`reports/link_estimation_metrics.json`](reports/link_estimation_metrics.json) (mirror at [`models/metrics.json`](models/metrics.json)). Dataset: synthetic link-condition CSV with 500 samples, 125-sample stratified holdout, 12 constellation-statistic features.

| Metric | Value | Status |
| --- | ---: | --- |
| SNR estimator — MAE / R² | 0.118 dB / 0.999 | measured |
| BER predictor — MAE / R² | 0.000453 / 0.968 | measured |
| Channel classifier — accuracy | 0.472 | measured (honest weak result — see [Interpretation](reports/link_estimation_report.md)) |
| Link-quality scorer — MAE / R² | 4.089 / 0.904 | measured |
| AWGN BER smoke (2 000 bits) | 0.0025 @ 0 dB; 0.0 @ 2–20 dB | measured ([reports/ber_smoke_awgn.csv](reports/ber_smoke_awgn.csv)) — coarse, hits resolution floor above 0 dB |
| AWGN BER full sweep (1 000 000 bits) | 2.42e-3 @ 0 dB · 1.83e-4 @ 2 dB · 5.0e-6 @ 4 dB · 0 @ 6–20 dB (below 1e-6 sim floor) | measured ([reports/ber_full_awgn.csv](reports/ber_full_awgn.csv)) — `make run-sim-full` |
| Rayleigh BER smoke (2 000 bits, single fading realization, seed=7) | 4.8e-2 @ 0 dB · 0 @ 2–20 dB | measured ([reports/ber_smoke_rayleigh.csv](reports/ber_smoke_rayleigh.csv)) — `make run-sim-rayleigh`. **Single-realization caveat — see note below.** |
| Ensemble-averaged Rayleigh BER (N=200 realizations × 10 000 bits, transmit-power-SNR) | 4.18e-2 @ 0 dB · 4.70e-2 @ 2 dB · 2.67e-2 @ 4 dB · 1.24e-2 @ 6 dB · 8.3e-3 @ 8 dB · 7.7e-3 @ 10 dB · 5.8e-3 @ 12 dB · 3.6e-3 @ 14 dB · 2.6e-4 @ 16 dB · 2.9e-3 @ 18 dB · 5.2e-4 @ 20 dB | measured ([reports/ber_full_rayleigh.csv](reports/ber_full_rayleigh.csv)) — `make run-sim-rayleigh-full`. Classical 1/SNR diversity-1 penalty visible vs AWGN's exponential decay |
| Jetson ONNX inference latency (p50/p95/p99) | `<TO MEASURE>` | Plan: run `edge/jetson_benchmark_template.py` on Jetson when hardware lands; capture mean latency and inferences/sec into `reports/jetson_inference_benchmark.json` |

The channel classifier scoring 0.472 on a two-class problem is a useful negative result, not noise to hide: the current 12-feature set supports SNR/BER estimation much better than channel-type recognition. See [`reports/link_estimation_report.md`](reports/link_estimation_report.md) for the full interpretation.

The Rayleigh smoke row is a **single-realization** result: `channel.rayleigh_fading` draws one complex-Gaussian `h` per simulation call, and the demodulator receives that `h` as perfect channel-state information. With seed=7 the single draw produces a deep enough fade at 0 dB to push BER to 4.8 %, while higher-SNR draws happen to land at favourable `|h|²` and clear below the 2000-bit resolution floor.

The **ensemble-averaged Rayleigh** row (N=200 × 10 000 bits per SNR point) is the right curve to compare against AWGN. It uses the **transmit-power-SNR** convention: `channel.add_awgn` references noise variance to the pre-fading transmit signal power (via an explicit `reference_power` passed from `apply_channel`), so the `|h|²` factor does not cancel out and the fade penalty propagates to the receiver.

Earlier revisions used a per-realization-SNR convention (noise referenced to received power), under which the Rayleigh curve tracked AWGN at the same SNR. That earlier behaviour was confirmed by ensemble measurement and then fixed: `channel.add_awgn` gained an optional `reference_power` parameter, and `apply_channel` now passes the pre-fading transmit power on both AWGN and Rayleigh paths. The verification gate (AWGN BER on `reports/ber_smoke_awgn.csv` and `reports/ber_full_awgn.csv` must regenerate bit-identically) is satisfied — the AWGN path is mathematically unchanged because the input signal to `add_awgn` already equals the transmit signal there.

Reproduce locally:

```bash
make install-dev
make generate-evidence
make train-evidence
cat reports/link_estimation_metrics.json
```

---

## Architecture

```text
Random Bits
    |
    v
QPSK Mapper (Gray)
    |
    v
Pulse Shaping (Root Raised Cosine)
    |
    v
Wireless Channel (AWGN / Rayleigh, transmit-power-SNR)
    |
    v
Matched Filter + symbol-timing recovery
    |
    v
QPSK Demapper (hard decision)
    |
    v
BER + constellation statistics
    |
    +-----> Classical BER vs SNR curve (verified against textbook)
    |
    +-----> ML link estimators (SNR / BER / channel type / link quality)
                    |
                    v
            ONNX export
                    |
                    v
            Jetson benchmark template (latency p50/p95/p99)
```

---

## Repository Structure

```text
.
├── run_sim.py                            # BER sweep CLI (--fading, --num-bits, --snr-min/max/step)
├── run_sim_ensemble.py                   # Ensemble Rayleigh BER sweep (--n-realizations)
├── qpsk_modem.py                         # QPSK mapper, RRC filter, matched filter, demap
├── channel.py                            # AWGN + flat Rayleigh fading (transmit-power-SNR)
├── plots.py                              # BER curves, constellation plots
├── generate_dataset.py                   # Synthetic link-condition CSV CLI
├── train_link_models.py                  # Train SNR/BER/channel/quality estimators
├── export_onnx.py                        # Export trained estimators to ONNX
│
├── ai_link_estimation/                   # ML link-estimation package
│   ├── dataset.py                        # simulate_link_sample, generate_dataset, schema
│   ├── features.py                       # FEATURE_COLUMNS, constellation_statistics
│   ├── models.py                         # train_models, write_comparison_report
│   └── onnx_export.py                    # ONNX conversion via skl2onnx
│
├── edge/
│   └── jetson_benchmark_template.py      # ONNX Runtime latency benchmark
│
├── tests/
│   ├── test_modem.py                     # DSP unit tests
│   └── test_features.py                  # Feature extraction + link-quality score tests
│
├── reports/                              # Committed proof artifacts (regenerable via make verify)
│   ├── ber_smoke_awgn.{csv,svg}          # 2k-bit AWGN smoke curve
│   ├── ber_full_awgn.{csv,svg}           # 1M-bit AWGN full sweep
│   ├── ber_smoke_rayleigh.{csv,svg}      # Single-realization Rayleigh smoke (disclosed caveat)
│   ├── ber_full_rayleigh.{csv,svg}       # Ensemble-averaged Rayleigh (N=200 × 10k bits)
│   ├── link_estimation_metrics.json      # ML estimator holdout metrics
│   ├── link_estimation_report.md         # Plain-text comparison report
│   ├── edge_inference_plan.md            # ONNX / TensorRT path notes
│   └── dashboard.html                    # Executive HTML dashboard (regenerated)
│
└── notebooks/legacy/                     # Original exploration notebooks (provenance only)
```

---

## Quick Start

Primary target: Linux / macOS / Windows. Core simulator runs on standard CPU Python. Jetson usage is the optional ONNX Runtime benchmark path after models are exported.

```bash
git clone https://github.com/obiedeh/wireless-link-intelligence-system.git
cd wireless-link-intelligence-system
python -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_sim.py
```

Full verification (Linux / macOS):

```bash
make install-dev
make verify
```

`make verify` runs ruff lint → 15 pytest tests → BER smoke sweep → synthetic dataset generation → model training → artifact existence check.

---

## AI-assisted link estimation

Generate synthetic link-condition data:

```bash
python generate_dataset.py --samples 500 --num-bits 4000
```

This creates `data/link_conditions.csv` with: SNR, measured BER from the classical baseline, channel type, fading magnitude and phase, and 12 constellation statistics (power, I/Q moments, EVM, phase spread, radius spread, quadrant balance).

Train the estimators and produce the comparison report:

```bash
python train_link_models.py
```

Training artifacts:

| Artifact | What it tells you |
| --- | --- |
| `models/snr_estimator.joblib` | Trained SNR estimator (regression: constellation statistics → SNR in dB). |
| `models/ber_predictor.joblib` | Trained BER predictor (regression: constellation statistics → measured BER). |
| `models/channel_classifier.joblib` | Trained channel classifier (AWGN vs Rayleigh; weak — see [Measured Metrics](#measured-metrics)). |
| `models/link_quality_scorer.joblib` | Trained link-quality scorer (regression: constellation statistics → synthetic 0–100 score). |
| `models/metrics.json` | Machine-readable holdout metrics and example predictions for all four estimators. |
| `reports/link_estimation_report.md` | Human-readable comparison report with example rows and interpretation. |
| `reports/link_estimation_metrics.json` | Mirror of `models/metrics.json` under `reports/` for the proof-artifact pattern. |

CI validates this path with a smaller deterministic dataset so the repository proves tests, simulation, dataset generation, training, and evidence artifact creation on Ubuntu (Python 3.11 + 3.12).

---

## Edge deployment path

This is an edge benchmark path, not a TensorRT-validated production deployment claim. Use it to generate measurable latency evidence on Jetson hardware.

Export trained models to ONNX:

```bash
python -m pip install ".[edge]"
python export_onnx.py
```

Benchmark on Jetson (or any host with `onnxruntime`):

```bash
python edge/jetson_benchmark_template.py --model models/onnx/snr_estimator.onnx --runs 1000
```

See [`reports/edge_inference_plan.md`](reports/edge_inference_plan.md) for TensorRT-ready notes. The current ONNX path is a practical edge inference bridge for the tabular estimators. For TensorRT acceleration, the tree-based estimators would need to be distilled into a small neural network and validated for parity across Python, ONNX Runtime, and TensorRT — a known scope expansion, intentionally deferred until Jetson hardware lands.

For the reviewer-facing checklist see [PORTFOLIO_DELIVERABLES.md](PORTFOLIO_DELIVERABLES.md). For the executive one-pager see [TECH_BRIEF.md](TECH_BRIEF.md).

---

## Positioning

This repository is a **production-discipline reference** for physical-layer AI engineering. It is not:

- a full AI-RAN base station
- a scheduler
- a standards-compliant modem
- a production telecom receiver

What it is: a measurable QPSK simulation testbed with verified BER curves, an ML link-estimation layer with honest holdout evaluation, and an ONNX export path validated end-to-end on commodity hardware (Jetson latency `<TO MEASURE>` until the device lands).

The methodology — classical baseline first, ML second, edge path third, honest weak results surfaced — applies beyond QPSK to any physical-layer AI work.

---

## License

MIT License.
