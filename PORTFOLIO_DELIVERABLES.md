# Portfolio Deliverables

This repository is scoped to a reproducible wireless engineering testbed: classical QPSK simulation plus a small AI-assisted link-estimation workflow.

## One-Command Checks

```bash
make install-dev
make verify
```

CI validates linting, tests, BER simulation, synthetic dataset generation, model training, and evidence artifact creation on Ubuntu.

## Proof Artifacts

| Artifact | Purpose |
| --- | --- |
| `reports/ber_smoke_awgn.csv` | Deterministic BER smoke sweep values |
| `reports/ber_smoke_awgn.svg` | Visual BER vs SNR proof artifact |
| `reports/link_estimation_report.md` | Human-readable model comparison report |
| `reports/link_estimation_metrics.json` | Machine-readable estimator metrics and examples |
| `reports/edge_inference_plan.md` | ONNX and Jetson benchmark path notes |

Generated training data and model binaries are intentionally not committed. They are reproducible through `make verify`.

## Current Evidence

- Classical QPSK path runs with deterministic BER smoke output.
- Synthetic link-condition dataset generation works from the CLI.
- SNR and BER estimators train and report held-out metrics.
- The channel classifier is weak in the current run; the report calls this out as a negative result instead of hiding it.

## Credibility Boundary

This is not a production modem, AI-RAN base station, scheduler, or standards-compliant receiver. It is a measurable simulation and edge-inference preparation repo.

Jetson evidence remains limited to the ONNX benchmark template until real device latency artifacts are committed.
