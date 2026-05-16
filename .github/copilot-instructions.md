# Quick Guide for AI Coding Agents

Purpose: Help an AI agent be immediately productive in this repository.

## Source of Truth

**`.py` files are canonical.** The `.ipynb` notebooks (`channel.ipynb`, `plots.ipynb`, `qpsk_modem.ipynb`) are exploratory originals kept for reference — do not edit them. All algorithmic changes go into the `.py` modules.

## Architecture

- `qpsk_modem.py` — Gray-coded QPSK mapper/demapper, RRC pulse shaping, matched filter, mod/demod chain
- `channel.py` — AWGN and flat Rayleigh fading (`apply_channel(tx_signal, snr_db, fading) -> (rx_signal, h)`)
- `plots.py` — constellation and BER visualization helpers
- `run_sim.py` — classical BER sweep CLI (`--fading`, `--num-bits`, `--snr-min/max/step`)
- `ai_link_estimation/` — ML layer: dataset generation, feature extraction, model training, ONNX export
- `edge/jetson_benchmark_template.py` — ONNX Runtime latency benchmark for Jetson

## Data Flow

```
bits -> bits_to_qpsk -> qpsk_modulate (RRC + upsample)
     -> apply_channel (AWGN / Rayleigh)
     -> recover_qpsk_symbols (matched filter + downsample)
     -> qpsk_to_bits -> BER
```

## Key Implementation Notes

- Gray mapping: `{00: 1+1j, 01: -1+1j, 11: -1-1j, 10: 1-1j}`, normalized by `1/sqrt(2)`
- Default RRC: `num_taps = 8*sps+1`, `beta=0.35`, `sps=8`
- RRC energy normalized: `h /= sqrt(sum(h**2))`
- Matched filter timing: `offset = len(h_rrc)//2`, sample every `sps` (perfect sync assumed)
- Flat-fading equalization: scalar division by `channel_coef` in `qpsk_demodulate`

## ML Feature Policy

`FEATURE_COLUMNS` in `ai_link_estimation/features.py` lists the 12 observable Rx features used
for ML training. `fading_abs` and `fading_phase` are saved to the CSV as labels but deliberately
excluded from `FEATURE_COLUMNS` — using oracle channel coefficient as a classifier input is leakage.

## AI Link Estimation Workflow

```bash
python generate_dataset.py --samples 500 --num-bits 4000  # writes data/link_conditions.csv
python train_link_models.py                               # writes models/ and reports/
python export_onnx.py                                     # writes models/onnx/
python edge/jetson_benchmark_template.py --model models/onnx/snr_estimator.onnx
```

## Tests

```bash
pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

## Developer Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_sim.py                        # AWGN BER sweep
python run_sim.py --fading               # Rayleigh BER sweep
```
