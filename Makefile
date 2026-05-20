.PHONY: install install-dev test lint run-sim generate-evidence train-evidence dashboard validate-evidence verify

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.venv:
	python3 -m venv .venv

install: .venv
	$(PIP) install -e .

install-dev: .venv
	$(PIP) install -e ".[dev]"

test: .venv
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

lint: .venv
	$(PYTHON) -m ruff check .

run-sim: .venv
	$(PYTHON) run_sim.py --num-bits 2000 --seed 7 --output-csv reports/ber_smoke_awgn.csv --output-plot reports/ber_smoke_awgn.svg

run-sim-full: .venv
	$(PYTHON) run_sim.py --num-bits 1000000 --seed 7 --output-csv reports/ber_full_awgn.csv --output-plot reports/ber_full_awgn.svg

run-sim-rayleigh: .venv
	$(PYTHON) run_sim.py --fading --num-bits 2000 --seed 7 --output-csv reports/ber_smoke_rayleigh.csv --output-plot reports/ber_smoke_rayleigh.svg

run-sim-rayleigh-full: .venv
	$(PYTHON) run_sim_ensemble.py --fading --n-realizations 200 --num-bits 10000 --seed 7 --output-csv reports/ber_full_rayleigh.csv --output-plot reports/ber_full_rayleigh.svg

run-sim-ofdm: .venv
	$(PYTHON) run_sim_ofdm.py --num-bits 50000 --seed 7 --output-csv reports/ber_full_ofdm_awgn.csv --output-plot reports/ber_full_ofdm_awgn.svg

generate-evidence: .venv
	$(PYTHON) generate_dataset.py --output data/link_conditions.csv --samples 120 --num-bits 1200 --seed 7

train-evidence: .venv
	$(PYTHON) train_link_models.py --dataset data/link_conditions.csv --output-dir models --report reports/link_estimation_report.md --metrics-report reports/link_estimation_metrics.json

dashboard: .venv
	$(PYTHON) build_dashboard.py --output-dir reports

validate-evidence:
	test -s reports/ber_smoke_awgn.csv
	test -s reports/ber_smoke_awgn.svg
	test -s reports/link_estimation_report.md
	test -s reports/link_estimation_metrics.json
	test -s reports/dashboard.html
	test -s reports/ber_full_ofdm_awgn.csv
	test -s reports/ber_full_ofdm_awgn.svg
	test -s models/metrics.json

verify: lint test run-sim run-sim-ofdm generate-evidence train-evidence dashboard validate-evidence
