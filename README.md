# QPSK Wireless Link Simulator

Classical QPSK wireless link simulation with an AI-assisted link-estimation extension for edge communications experiments.

This repository implements a modular Python simulation of a baseband QPSK digital communication link. The classical simulator remains the baseline, and the added ML workflow estimates link conditions from synthetic constellation statistics.

The goal is not to present QPSK as a novel technique or to claim a production telecom receiver.

The goal is to build a reproducible signal-processing testbed for understanding wireless link behavior under noise, fading, and edge AI-assisted link estimation.

---

## Why This Matters

AI-RAN and edge AI systems depend on reliable wireless links, predictable channel behavior, and accurate interpretation of physical-layer performance.

Before adding AI to the RAN stack, the underlying communication system must be measurable:

- modulation behavior
- channel impairments
- fading effects
- BER vs SNR performance
- constellation distortion
- receiver recovery behavior

This repository provides that foundational wireless simulation layer.

---

## Current Capabilities

- random bit generation
- Gray-coded QPSK modulation
- Root Raised Cosine pulse shaping
- AWGN channel modeling
- Rayleigh fading channel modeling
- matched filtering
- symbol timing path
- hard-decision demodulation
- BER vs SNR simulation
- constellation visualization
- synthetic link-condition dataset generation
- ML-based SNR estimation
- ML-based BER prediction
- AWGN vs Rayleigh channel classification
- synthetic link-quality scoring
- ONNX export path and Jetson benchmark template

---

## Measured Metrics

Source: [`reports/link_estimation_metrics.json`](reports/link_estimation_metrics.json) (mirror at [`models/metrics.json`](models/metrics.json)). Dataset: synthetic link-condition CSV with 500 samples, 125-sample stratified holdout, 12 constellation-statistic features.

| Metric | Value | Status |
| --- | ---: | --- |
| SNR estimator — MAE / R² | 0.118 dB / 0.999 | measured |
| BER predictor — MAE / R² | 0.000453 / 0.968 | measured |
| Channel classifier — accuracy | 0.472 | measured (honest weak result — see [Interpretation](reports/link_estimation_report.md)) |
| Link-quality scorer — MAE / R² | 4.089 / 0.904 | measured |
| AWGN BER smoke (2000 bits) | 0.0025 @ 0 dB; 0.0 @ 2–20 dB | measured ([reports/ber_smoke_awgn.csv](reports/ber_smoke_awgn.csv)) — coarse, hits resolution floor above 0 dB |
| AWGN BER full sweep (1e6 bits) | 2.42e-3 @ 0 dB · 1.83e-4 @ 2 dB · 5.0e-6 @ 4 dB · 0 @ 6–20 dB (below 1e-6 sim floor) | measured ([reports/ber_full_awgn.csv](reports/ber_full_awgn.csv)) — run via `make run-sim-full` |
| Rayleigh BER smoke (2000 bits, single fading realization, seed=7) | 2.5e-3 @ 0 dB · 0 @ 2–20 dB | measured ([reports/ber_smoke_rayleigh.csv](reports/ber_smoke_rayleigh.csv)) — run via `make run-sim-rayleigh`. **Single-realization caveat — see note below** |
| Ensemble-averaged Rayleigh BER curve | `<TO MEASURE>` | Plan: extend the BER sweep to average over N independent fading realizations per SNR point (e.g., N=200 with `--num-bits 10000` each). The current single-realization smoke under-represents fade-driven BER because perfect CSI demodulation lets a lucky `|h|²` mask average performance. |
| Jetson ONNX inference latency (p50/p95/p99) | `<TO MEASURE>` | Plan: run `edge/jetson_benchmark_template.py` on Jetson when hardware lands; capture mean latency and inferences/sec into `reports/jetson_inference_benchmark.json`. |

The channel classifier scoring 0.472 on a two-class problem is a useful negative result, not noise to hide: the current 12-feature set supports SNR/BER estimation much better than channel-type recognition. See [`reports/link_estimation_report.md`](reports/link_estimation_report.md) for the full interpretation.

The Rayleigh smoke row is a **single-realization** result: `channel.rayleigh_fading` draws one complex-Gaussian `h` per simulation call, and the QPSK demodulator receives that `h` as perfect channel-state information. With seed=7, the SNR 0 dB call drew an `h` that produces measurable error; subsequent SNR points drew `|h|²` values where perfect-CSI compensation pushes BER below the 2000-bit resolution floor. This is honest evidence that the pipeline runs end-to-end, but it is **not** an ensemble-averaged Rayleigh curve — the ensemble-averaged row above is the right comparison against the AWGN sweep and is explicitly tagged `<TO MEASURE>`.

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
QPSK Mapper
    |
    v
Pulse Shaping
(Root Raised Cosine)
    |
    v
Wireless Channel
(AWGN / Rayleigh)
    |
    v
Matched Filter + Recovery
    |
    v
QPSK Demapper
    |
    v
BER / Constellation Analysis
```

---

## Repository Structure

```text
.
├── run_sim.py                    # Classical BER simulation entry point
├── qpsk_modem.py                 # Modulation, demodulation, and RRC filtering
├── channel.py                    # AWGN and Rayleigh fading models
├── plots.py                      # BER and constellation visualization helpers
├── ai_link_estimation/           # Dataset, features, training, and ONNX export
├── generate_dataset.py           # Synthetic link-condition dataset CLI
├── train_link_models.py          # Train SNR/BER/channel/quality models
├── export_onnx.py                # Export trained estimators to ONNX
├── edge/jetson_benchmark_template.py
├── reports/link_estimation_report.md
├── reports/edge_inference_plan.md
├── requirements.txt
└── README.md
```

---

## Quick Start

Primary target: Linux. The core simulator runs on standard CPU Python. Jetson usage is limited to the optional ONNX Runtime benchmark path after models are exported.

```bash
git clone https://github.com/obiedeh/qpsk-wireless-link-simulator.git
cd qpsk-wireless-link-simulator
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_sim.py
```

For the full Linux verification path:

```bash
make install-dev
make verify
```

Windows PowerShell:

```powershell
git clone https://github.com/obiedeh/qpsk-wireless-link-simulator.git
cd qpsk-wireless-link-simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_sim.py
```

---

## AI-Assisted Link Estimation Workflow

Generate synthetic link-condition data:

```bash
python generate_dataset.py --samples 500 --num-bits 4000
```

This creates `data/link_conditions.csv` with:

- SNR
- measured BER from the classical QPSK baseline
- channel type
- fading coefficient magnitude and phase
- constellation statistics such as power, I/Q moments, EVM, phase spread, radius spread, and quadrant balance

Train the estimators and produce the comparison report:

```bash
python train_link_models.py
```

The training step writes the following artifacts:

| Artifact | What it tells you |
| --- | --- |
| `models/snr_estimator.joblib` | Trained SNR estimator (regression: constellation statistics → SNR in dB). |
| `models/ber_predictor.joblib` | Trained BER predictor (regression: constellation statistics → measured BER). |
| `models/channel_classifier.joblib` | Trained channel classifier (AWGN vs Rayleigh; currently weak — see [Measured Metrics](#measured-metrics)). |
| `models/link_quality_scorer.joblib` | Trained link-quality scorer (regression: constellation statistics → synthetic 0–100 quality score). |
| `models/metrics.json` | Machine-readable holdout metrics and example predictions for all four estimators. |
| `reports/link_estimation_report.md` | Human-readable comparison report with example rows and interpretation. |
| `reports/link_estimation_metrics.json` | Mirror of `models/metrics.json` under `reports/` for the proof-artifact pattern. |

CI validates this path with a smaller deterministic dataset so the repository proves tests, simulation, dataset generation, training, and evidence artifact creation on Ubuntu.

The report includes:

- classical measured BER vs predicted BER
- AWGN vs Rayleigh classification accuracy
- SNR estimation error
- link-quality scoring error

---

## Edge Deployment Path

This is an edge benchmark path, not a claim of TensorRT-validated production deployment. Use it to generate measurable latency evidence on Jetson hardware.

Export trained models to ONNX:

```bash
python -m pip install ".[edge]"
python export_onnx.py
```

Benchmark an exported estimator on Jetson:

```bash
python edge/jetson_benchmark_template.py --model models/onnx/snr_estimator.onnx --runs 1000
```

See `reports/edge_inference_plan.md` for TensorRT-ready notes. The current ONNX path is a practical edge inference bridge for tabular estimators. For TensorRT acceleration, replace or distill the tree models into a small neural network and validate parity across Python, ONNX Runtime, and TensorRT.

For the reviewer-facing deliverables checklist, see [PORTFOLIO_DELIVERABLES.md](PORTFOLIO_DELIVERABLES.md).

---

## Example Outputs

Expected simulation outputs include:

- BER vs SNR curve for AWGN and Rayleigh channels
- constellation plots under different channel conditions
- printed simulation summary
- optional committed smoke artifacts: `reports/ber_smoke_awgn.csv` and `reports/ber_smoke_awgn.svg`

Typical behavior:

- AWGN BER improves sharply as SNR increases
- Rayleigh fading shows a significant performance penalty
- low-SNR constellations spread and overlap
- high-SNR constellations converge toward the ideal four-symbol pattern

---

## Engineering Extensions

### Wireless Systems

- carrier frequency offset
- timing synchronization
- AGC
- phase correction
- Rician fading
- frequency-selective multipath
- OFDM
- MIMO

### AI-Assisted Link Estimation Experiments

- ML-based channel estimation
- BER prediction from constellation statistics
- AWGN vs Rayleigh classification
- SNR estimation
- lightweight link-quality scoring
- optional future neural equalization
- optional future learned demodulation
- channel-aware inference experiments
- synthetic PHY-layer telemetry generation

### Edge AI Integration

- lightweight inference benchmarks
- signal-processing acceleration experiments
- Jetson-based simulation or receiver workflows
- telemetry export for AI-RAN analytics

---

## Positioning

This repository supports a broader engineering focus around:

- edge AI-assisted wireless link estimation
- wireless systems
- edge AI infrastructure
- physical-layer intelligence
- signal-processing foundations

It is not a full AI-RAN base station, not a scheduler, not a standards-compliant modem, and not a production telecom receiver.

---

## License

MIT License.
