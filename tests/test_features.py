import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path so bare imports (channel, qpsk_modem) resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_link_estimation.dataset import CSV_COLUMNS, simulate_link_sample
from ai_link_estimation.features import (
    FEATURE_COLUMNS,
    constellation_statistics,
    link_quality_score,
)


def _make_symbols(n=100, snr_linear=100.0, rng=None):
    rng = rng or np.random.default_rng(0)
    tx = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j] * (n // 4), dtype=complex) / np.sqrt(2)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2 * snr_linear)
    return tx, tx + noise


def test_constellation_statistics_returns_all_feature_columns():
    tx, rx = _make_symbols()
    stats = constellation_statistics(tx, rx, channel_coef=1.0 + 0j)
    for col in FEATURE_COLUMNS:
        assert col in stats, f"Missing feature: {col}"


def test_constellation_statistics_no_leakage_features():
    """fading_abs and fading_phase must not be in FEATURE_COLUMNS."""
    assert "fading_abs" not in FEATURE_COLUMNS
    assert "fading_phase" not in FEATURE_COLUMNS


def test_constellation_statistics_empty_raises():
    with pytest.raises(ValueError):
        constellation_statistics(np.array([]), np.array([1 + 1j]), 1.0)


def test_link_quality_score_bounded():
    for snr in np.linspace(-10, 30, 10):
        for ber in [0.0, 0.1, 0.5]:
            for ch in ["awgn", "rayleigh"]:
                score = link_quality_score(snr, ber, ch)
                assert 0.0 <= score <= 100.0


def test_link_quality_score_awgn_beats_rayleigh():
    awgn = link_quality_score(10.0, 0.01, "awgn")
    rayleigh = link_quality_score(10.0, 0.01, "rayleigh")
    assert awgn > rayleigh


def test_simulate_link_sample_complete_row():
    np.random.seed(0)
    row = simulate_link_sample(sample_id=0, num_bits=400)
    for col in CSV_COLUMNS:
        assert col in row, f"Missing column in row: {col}"


def test_simulate_link_sample_ber_in_range():
    np.random.seed(1)
    row = simulate_link_sample(sample_id=1, num_bits=400, snr_db=15.0, channel_type="awgn")
    assert 0.0 <= row["ber"] <= 1.0
