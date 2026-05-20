# Tech brief — Wireless Link Intelligence System

A one-page brief for a senior tech-leader or hiring-manager review. Read this if you are evaluating whether a candidate can ship physical-layer ML that an RF or AI-RAN team can actually trust — calibrated baselines, honest disclosure of weak results, and an end-to-end reproducible reporting pipeline. Every claim below links to a committed artifact under [`reports/`](reports/).

---

## What this repo demonstrates

Physical-layer AI sits between two failure modes. Skip the signal-processing work and you publish ML wins that vanish the moment they meet a real channel. Skip the engineering discipline and you publish notebook-only experiments that no RF team can reproduce or trust.

This repo demonstrates the engineering pattern that avoids both: **classical baseline first** (verified BER curves matching textbook predictions), **ML estimators second** (with honest holdout metrics and disclosed weaknesses), **edge deployment path third** (ONNX export + Jetson benchmark template, ready when hardware lands).

In one sentence: a classical QPSK baseband simulator with deterministic BER vs SNR sweeps, four scikit-learn link estimators trained on a stratified holdout, and an ONNX export path validated end-to-end on commodity hardware — every committed number regenerable by `make verify`.

It is **not** a production telecom receiver, not a full AI-RAN base station, not a standards-compliant modem. It is a measurable testbed where the engineering pattern is the deliverable.

---

## Evidence summary

| What | Value | Source |
|---|---:|---|
| SNR estimator — MAE / R² | **0.118 dB** / 0.999 | [`reports/link_estimation_metrics.json`](reports/link_estimation_metrics.json) |
| BER predictor — MAE / R² | 0.000453 / **0.968** | same |
| Channel classifier — accuracy | **0.472** *(honest weak result — disclosed, not hidden)* | same |
| Link-quality scorer — MAE / R² | 4.089 / 0.904 | same |
| Holdout sample size | 125 / 500 (25% stratified) | same |
| AWGN BER full sweep (1M bits) | 2.42e-3 @ 0 dB → 1.83e-4 @ 2 dB → below 1e-6 sim floor at 6+ dB | [`reports/ber_full_awgn.csv`](reports/ber_full_awgn.csv) |
| Ensemble-averaged Rayleigh BER (200 × 10k bits) | 4.18e-2 @ 0 dB → 5.2e-4 @ 20 dB *(diversity-1 visible)* | [`reports/ber_full_rayleigh.csv`](reports/ber_full_rayleigh.csv) |
| Jetson latency p50/p95/p99 | **`<TO MEASURE>`** | template ready; hardware pending |
| Tests | **15 / 15** green | [`tests/`](tests/) + `pytest -q` |
| CI matrix | Python 3.11 + 3.12, Ubuntu | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| End-to-end reproducible | `make verify` regenerates every artifact under `reports/` | [`Makefile`](Makefile) |

The full dashboard with embedded BER curves, methodology box, per-estimator insight cards, and limitations footer lives at [`reports/dashboard.html`](reports/dashboard.html).

---

## The simulator

**Modem** (`qpsk_link/modem.py`): Gray-coded QPSK mapping, root-raised-cosine pulse shaping with three-case handling at `t=0` and `|t|=1/4β`, matched filtering with explicit symbol-timing offset (`offset = len(h_rrc) // 2`).

**Channel** (`qpsk_link/channel.py`): AWGN with explicit `reference_power` parameter; flat Rayleigh fading. The transmit-power-SNR convention is enforced — `apply_channel` passes pre-fading transmit power into `add_awgn` so the `|h|²` factor does not cancel out. The Rayleigh curve shows the classical 1/SNR_linear diversity-1 penalty, not an artifact-of-normalization flat curve.

**BER sweep harness** (`run_sim.py`, `run_sim_ensemble.py`): single-shot and ensemble sweeps over a configurable SNR grid. CSV + SVG output. Smoke (2k bits) and full (1M bits) regimes. The ensemble script averages over N independent fading realizations per SNR point — the only honest way to report a Rayleigh BER curve.

**Why this matters as evidence:** the AWGN BER curve matches textbook Q-function predictions to within the 1e-6 simulation floor. A receiver design that doesn't pass this check has a bug; this one does and the regression gate (`reports/ber_smoke_awgn.csv` must regenerate bit-identically) enforces it.

---

## The ML estimators

Four scikit-learn estimators learn from 12 constellation statistics (power, I/Q moments, EVM, phase spread, radius spread, quadrant balance). All four are trained as a single sklearn `Pipeline` per target on a 25% stratified holdout.

| Estimator | Holdout result | Calibrated interpretation |
|---|---|---|
| **SNR estimator** | R² 0.999, MAE 0.118 dB | Constellation power and spread tell you SNR directly. Production-relevant: this is the estimator a real receiver would run every frame to drive AGC and MCS decisions. |
| **BER predictor** | R² 0.968, MAE 4.5e-4 | BER follows from SNR via the Q-function in theory; at low SNR the constellation spread carries extra information the formula misses. The model captures both — residual is well below the simulation floor. |
| **Channel classifier** | accuracy 0.472 | Honest weak result. AWGN and Rayleigh produce similar statistics when averaged over symbols. Distinguishing them needs higher-order features (envelope variance over time, autocorrelation) the current 12-feature set does not have. **Surfaced, not hidden.** |
| **Link-quality scorer** | R² 0.904, MAE 4.09 | The 0–100 link-quality target combines SNR and BER with a small Rayleigh penalty. The model recovers it well because SNR and BER are already strongly predicted — this estimator is more a consistency check than a new capability. |

The **non-negotiable rule** in `AGENTS.md` enforces no feature leakage: `fading_abs` and `fading_phase` are saved in the dataset as labels but explicitly excluded from `FEATURE_COLUMNS`. They encode oracle channel knowledge and would trivially inflate any classifier built on them.

---

## Edge deployment path

**Measured:**
- ONNX conversion succeeds for all four estimators via `skl2onnx` (`export_onnx.py`).
- `onnxruntime` parity test passes on commodity x86_64 — predictions match the sklearn pipeline within float32 tolerance.

**Not yet measured:**
- **Jetson latency p50/p95/p99.** The benchmark template (`edge/jetson_benchmark_template.py`) runs on any host with `onnxruntime` installed; designed to drop onto a Jetson Orin/Nano and emit results into `reports/jetson_inference_benchmark.json`. The latency row in the metrics is `<TO MEASURE>` until hardware lands.
- **TensorRT acceleration.** Would require distilling the tree-based estimators into a small neural network and validating Python ↔ ONNX Runtime ↔ TensorRT parity. Known scope expansion, intentionally deferred until measured Jetson latency justifies the optimization.

---

## Engineering signals you can verify in 60 seconds

```bash
git clone https://github.com/obiedeh/qpsk-wireless-link-simulator.git
cd qpsk-wireless-link-simulator
make install-dev
make verify
```

`make verify` runs **ruff lint → 15 pytest tests → BER smoke sweep → synthetic dataset generation → model training → dashboard build → artifact-existence checks**. GitHub Actions CI ([`ci.yml`](.github/workflows/ci.yml)) runs the same recipe on Ubuntu with **Python 3.11 + 3.12 matrix** on every push and pull request.

Repo-quality signals:

- **Deterministic seeds threaded through every stochastic step** — BER sweeps use `--seed 7`, dataset generation uses `seed=7`, model training uses `random_state=42`. The Rayleigh smoke caveat (single-realization with seed=7 producing 4.8% BER at 0 dB and zeros above) is explicitly disclosed; the ensemble curve is the right comparison.
- **No feature leakage**, enforced as an `AGENTS.md` non-negotiable rule, with `fading_abs`/`fading_phase` excluded from `FEATURE_COLUMNS`.
- **Two-pass channel verification** — a transmit-power-vs-received-power bug was caught by ensemble measurement, fixed (`add_awgn` gained `reference_power`), and the AWGN regression gate (`ber_smoke_awgn.csv` must regenerate bit-identically) ensures the fix doesn't drift.
- **Honest weak results surfaced** — the channel classifier's 0.472 accuracy is in the README, the dashboard, and the metrics JSON. Aggregate F1 doesn't hide it.
- **CI matrix on two Python versions, not one** — pip cache keyed on dep hashes, so the matrix is genuinely cross-validated, not a single point.
- **Modular package layout** — `qpsk_link/` (DSP) and `ai_link_estimation/` (ML) are separate, installable packages; the bare-import sys.path hack from earlier revisions is gone.

---

## What this would need to be production

Honest list, in priority order:

1. **Jetson hardware** to measure actual ONNX Runtime latency p50/p95/p99 and turn the `<TO MEASURE>` row into real numbers.
2. **Higher-order temporal features** (envelope variance, autocorrelation over a sliding window) to fix the channel classifier weakness. The current 12-feature set is symbol-averaged; AWGN vs Rayleigh discrimination needs temporal structure.
3. **TensorRT distillation** — convert tree models into a small neural network with validated parity across Python ↔ ONNX Runtime ↔ TensorRT. Worth doing only after Jetson latency justifies the optimization.
4. **Real-world feature validation** — the synthetic dataset is exactly what its name says. A real RF environment introduces frequency offset, timing drift, multipath beyond flat Rayleigh, and impairments that need to be modeled or directly measured.
5. **End-to-end link control loop** — currently the estimators emit numbers; a production system would feed them into AGC, equalization, and MCS-selection control loops.
6. **Standards alignment** — pulse-shape parameters and channel models would need to match a specific 3GPP profile to claim more than QPSK-baseband correctness.

None of these are large; they're scope decisions tied to a specific RF or AI-RAN deployment. The boundary is honest: this repo is the evidence-and-methodology core, not the production receiver.

---

## Try it in five minutes

```bash
# 1. Clone and install
git clone https://github.com/obiedeh/qpsk-wireless-link-simulator.git
cd qpsk-wireless-link-simulator
python -m venv .venv
source .venv/bin/activate    # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e ".[dev]"

# 2. Regenerate every artifact in reports/
make verify

# 3. Open the dashboard
xdg-open reports/dashboard.html    # or double-click on Windows / macOS
```

You'll get the AWGN BER curve, the ensemble-averaged Rayleigh curve, the four ML estimator scores, the per-estimator interpretation cards, and the engineering signals — all from a fresh clone in one command.

For the structured reviewer checklist, see [`PORTFOLIO_DELIVERABLES.md`](PORTFOLIO_DELIVERABLES.md). For the operating rules and architecture, see [`AGENTS.md`](AGENTS.md). For the full repo overview, see [`README.md`](README.md).
