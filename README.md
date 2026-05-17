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

The training step writes:

- `models/snr_estimator.joblib`
- `models/ber_predictor.joblib`
- `models/channel_classifier.joblib`
- `models/link_quality_scorer.joblib`
- `models/metrics.json`
- `reports/link_estimation_report.md`

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

---

## Example Outputs

Expected simulation outputs include:

- BER vs SNR curve for AWGN and Rayleigh channels
- constellation plots under different channel conditions
- printed simulation summary

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
