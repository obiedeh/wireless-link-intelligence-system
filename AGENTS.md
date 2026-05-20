# AGENTS.md — Coding Agent Guide

**Wireless Link Intelligence System** — operating instructions for AI coding agents.
Read this before making any changes.

## Project Purpose

A production-discipline reference for physical-layer AI: a classical QPSK baseband
simulator (BER vs SNR, constellation analysis, AWGN + Rayleigh channels) layered with
four scikit-learn link estimators that learn from constellation statistics, plus an
ONNX export path and Jetson benchmark template. End-to-end reproducible from a fresh
clone.

**This is not a production telecom receiver, not a full AI-RAN base station, not a
standards-compliant modem.** It is a measurable simulation testbed and edge-inference
preparation repo where the engineering pattern — classical baseline first, ML second,
edge path third, honest weak results surfaced — is the deliverable.

## Source of Truth

`.py` files are canonical. Notebooks (`*.ipynb`) are exploratory originals — read-only.
Never edit `.ipynb` files for algorithmic changes.

## Repository Structure

```
qpsk_link/                          # DSP package
  modem.py                          # QPSK mapper, RRC filter, matched filter, demod
  channel.py                        # AWGN + flat Rayleigh fading (transmit-power-SNR)
  plots.py                          # BER curves, constellation plots
ai_link_estimation/                 # ML link-estimation package
  features.py                       # FEATURE_COLUMNS, constellation_statistics, link_quality_score
  dataset.py                        # simulate_link_sample, generate_dataset, CSV schema
  models.py                         # train_models, write_comparison_report
  onnx_export.py                    # export_models
run_sim.py                          # CLI: BER sweep (--fading, --num-bits, --snr-min/max/step)
run_sim_ensemble.py                 # CLI: ensemble-averaged Rayleigh BER sweep
generate_dataset.py                 # CLI: synthetic link-condition CSV generation
train_link_models.py                # CLI: train SNR/BER/channel/quality estimators
export_onnx.py                      # CLI: export trained models to ONNX
edge/jetson_benchmark_template.py   # ONNX Runtime latency benchmark for Jetson
tests/
  test_modem.py                     # Unit tests: DSP functions
  test_features.py                  # Unit tests: ML feature extraction and quality score
notebooks/legacy/                   # Original exploration notebooks (provenance only)
```

## Non-Negotiable Rules

1. **No feature leakage.** `fading_abs` and `fading_phase` are saved to the CSV as labels but
   must never appear in `FEATURE_COLUMNS`. They encode oracle channel knowledge and produce
   trivially inflated classifier accuracy.

2. **No magic numbers without a comment.** The `link_quality_score` formula (0.55, 0.45, 0.08)
   is a synthetic target — document any changes with why.

3. **Tests must stay green.** Run before committing:
   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
   ```

4. **Do not commit generated artifacts.** `data/*.csv`, `models/*.joblib`, `models/metrics.json`,
   `models/onnx/`, `__pycache__/`, `*.nbconvert.ipynb` are all gitignored.

5. **Keep the DSP math correct.** The RRC filter has three cases (t=0, |t|=1/4β, general).
   The matched filter assumes perfect timing (`offset = len(h_rrc)//2`). Do not simplify these
   without verifying BER performance is preserved.

## Running the Full Pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_sim.py                        # AWGN BER sweep
python run_sim.py --fading               # Rayleigh BER sweep
python generate_dataset.py --samples 500
python train_link_models.py
python export_onnx.py                    # requires: pip install skl2onnx onnx onnxruntime
python edge/jetson_benchmark_template.py --model models/onnx/snr_estimator.onnx
```

## Adding New Features

- New DSP algorithms → `qpsk_link/modem.py` or `qpsk_link/channel.py`, with unit tests in `tests/test_modem.py`
- New ML features → update `FEATURE_COLUMNS` in `features.py` AND regenerate the dataset
- New channel models → update `apply_channel` signature stays `(tx_signal, snr_db, fading) -> (rx_signal, h)`
- New estimators → add to `MODEL_SPECS` in `models.py`

## Dependencies

Runtime: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `joblib`
Notebook: `jupyter`
Edge/ONNX: `skl2onnx`, `onnx`, `onnxruntime`
Dev/Test: `pytest`
