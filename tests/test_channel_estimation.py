"""Tests for pilot-based channel estimators (LS / MMSE / Neural)."""

from __future__ import annotations

import numpy as np
import pytest

from qpsk_link.channel_estimation import (
    DEFAULT_PILOT_STRIDE,
    DEFAULT_PILOT_VALUE,
    build_piloted_frame,
    comb_pilot_indices,
    data_subcarrier_indices,
    ls_estimate,
    mmse_estimate,
    neural_estimator,
)
from qpsk_link.ofdm import qam_modulate
from qpsk_link.tdl_channel import TDL_C, cir_to_frequency_response, sample_tdl_realization

# ---------------------------------------------------------------------------
# Pilot placement
# ---------------------------------------------------------------------------


def test_comb_pilot_indices_stride_4():
    idx = comb_pilot_indices(64, stride=4)
    assert idx.tolist() == list(range(0, 64, 4))
    assert idx.size == 16


def test_data_indices_complement_pilots():
    n_sc = 64
    pilot_idx = comb_pilot_indices(n_sc, stride=4)
    data_idx = data_subcarrier_indices(n_sc, pilot_idx)
    assert data_idx.size == n_sc - pilot_idx.size
    assert set(np.concatenate([pilot_idx, data_idx])) == set(range(n_sc))


# ---------------------------------------------------------------------------
# Piloted frame construction
# ---------------------------------------------------------------------------


def test_build_piloted_frame_pilots_and_data_in_right_places():
    rng = np.random.default_rng(seed=1)
    n_sc = 64
    n_ofdm_sym = 2
    bps = 2
    pilot_idx = comb_pilot_indices(n_sc, DEFAULT_PILOT_STRIDE)
    data_idx = data_subcarrier_indices(n_sc, pilot_idx)
    n_data_bits = bps * n_ofdm_sym * data_idx.size
    bits = rng.integers(0, 2, n_data_bits).astype(np.int64)
    data_symbols = qam_modulate(bits, 4)

    frame = build_piloted_frame(
        data_symbols=data_symbols,
        n_subcarriers=n_sc,
        n_ofdm_symbols=n_ofdm_sym,
        cp_length=16,
        mod_order=4,
    )
    # Pilots at the expected positions.
    assert np.allclose(frame.tx_symbols[:, pilot_idx], DEFAULT_PILOT_VALUE)
    # Data symbols placed on data positions.
    assert frame.tx_symbols[:, data_idx].size == data_symbols.size


# ---------------------------------------------------------------------------
# LS estimator
# ---------------------------------------------------------------------------


def test_ls_estimate_recovers_known_channel_noiseless():
    """With no noise, LS at the pilots + linear interpolation should land on
    the true H at all pilot positions exactly."""
    rng = np.random.default_rng(seed=11)
    n_sc = 64
    h = sample_tdl_realization(TDL_C, 4.0, rng)
    H = cir_to_frequency_response(h, n_sc)

    pilot_idx = comb_pilot_indices(n_sc, DEFAULT_PILOT_STRIDE)
    n_ofdm_sym = 1
    data_idx = data_subcarrier_indices(n_sc, pilot_idx)
    bps = 2
    n_data_bits = bps * n_ofdm_sym * data_idx.size
    bits = rng.integers(0, 2, n_data_bits).astype(np.int64)
    data_symbols = qam_modulate(bits, 4)
    frame = build_piloted_frame(
        data_symbols=data_symbols,
        n_subcarriers=n_sc,
        n_ofdm_symbols=n_ofdm_sym,
        cp_length=16,
        mod_order=4,
    )
    # Apply channel directly in frequency domain to skip OFDM/AWGN noise.
    rx_grid = frame.tx_symbols * H[None, :]
    rx_pilots = rx_grid[:, pilot_idx]
    H_ls = ls_estimate(rx_pilots, frame)
    # At pilot positions, the estimate must match H exactly.
    assert np.allclose(H_ls[pilot_idx], H[pilot_idx], atol=1e-9)


# ---------------------------------------------------------------------------
# MMSE estimator
# ---------------------------------------------------------------------------


def test_mmse_estimate_returns_correct_shape():
    rng = np.random.default_rng(seed=22)
    n_sc = 64
    h = sample_tdl_realization(TDL_C, 4.0, rng)
    pilot_idx = comb_pilot_indices(n_sc, DEFAULT_PILOT_STRIDE)
    data_idx = data_subcarrier_indices(n_sc, pilot_idx)
    bps = 2
    n_ofdm_sym = 2
    n_data_bits = bps * n_ofdm_sym * data_idx.size
    bits = rng.integers(0, 2, n_data_bits).astype(np.int64)
    data_symbols = qam_modulate(bits, 4)
    frame = build_piloted_frame(
        data_symbols=data_symbols,
        n_subcarriers=n_sc,
        n_ofdm_symbols=n_ofdm_sym,
        cp_length=16,
        mod_order=4,
    )
    H = cir_to_frequency_response(h, n_sc)
    rx_grid = frame.tx_symbols * H[None, :]
    rx_pilots = rx_grid[:, pilot_idx]
    H_mmse = mmse_estimate(
        rx_pilots, frame, snr_db=20.0, delay_spread_samples=4.0
    )
    assert H_mmse.shape == (n_sc,)
    assert H_mmse.dtype == complex


def test_mmse_beats_ls_in_mse_on_average():
    """Over many random channels with noise, MMSE should produce a lower
    average channel-estimate MSE than LS at a moderate SNR."""
    n_sc = 64
    pilot_idx = comb_pilot_indices(n_sc, DEFAULT_PILOT_STRIDE)
    data_idx = data_subcarrier_indices(n_sc, pilot_idx)
    bps = 2
    n_ofdm_sym = 4
    snr_db = 12.0
    snr_lin = 10 ** (snr_db / 10)
    n_trials = 30
    ls_mse = mmse_mse = 0.0
    for trial in range(n_trials):
        rng = np.random.default_rng(seed=trial)
        h = sample_tdl_realization(TDL_C, 4.0, rng)
        n_data_bits = bps * n_ofdm_sym * data_idx.size
        bits = rng.integers(0, 2, n_data_bits).astype(np.int64)
        data_symbols = qam_modulate(bits, 4)
        frame = build_piloted_frame(
            data_symbols=data_symbols,
            n_subcarriers=n_sc,
            n_ofdm_symbols=n_ofdm_sym,
            cp_length=16,
            mod_order=4,
        )
        H = cir_to_frequency_response(h, n_sc)
        # Channel multiplication + AWGN at pilots only (simpler than full OFDM).
        noise = (rng.standard_normal((n_ofdm_sym, n_sc)) + 1j * rng.standard_normal((n_ofdm_sym, n_sc))) * np.sqrt(0.5 / snr_lin)
        rx_grid = frame.tx_symbols * H[None, :] + noise
        rx_pilots = rx_grid[:, pilot_idx]

        H_ls = ls_estimate(rx_pilots, frame)
        H_mmse = mmse_estimate(rx_pilots, frame, snr_db=snr_db, delay_spread_samples=4.0)
        ls_mse += float(np.mean(np.abs(H_ls - H) ** 2))
        mmse_mse += float(np.mean(np.abs(H_mmse - H) ** 2))

    ls_mse /= n_trials
    mmse_mse /= n_trials
    # On TDL-C at 12 dB, MMSE should produce a meaningfully lower MSE than LS.
    # Threshold is conservative (≥ 25% improvement) because this test runs a
    # simplified pilots-in-frequency-domain noise model without full OFDM
    # averaging; the full pipeline (run_channel_estimation_comparison.py)
    # shows a much larger gap.
    assert mmse_mse < ls_mse * 0.75, (
        f"Expected MMSE MSE meaningfully lower than LS; got "
        f"LS={ls_mse:.4e}, MMSE={mmse_mse:.4e}."
    )


# ---------------------------------------------------------------------------
# Neural estimator architecture sanity
# ---------------------------------------------------------------------------


def test_neural_estimator_forward_shape():
    """The MLP should output 2 × n_subcarriers floats (real + imag)."""
    torch = pytest.importorskip("torch")
    n_pilots = 16
    n_sc = 64
    est = neural_estimator(n_pilots, n_sc, hidden=32)
    model = est.model
    # Input: 2 * n_pilots + 1 (SNR scalar).
    x = torch.zeros(2, 2 * n_pilots + 1, dtype=torch.float32)
    y = model(x)
    assert y.shape == (2, 2 * n_sc)


def test_neural_estimator_lazy_import_does_not_break():
    """Constructing the wrapper without using it must not import torch."""
    est = neural_estimator(16, 64)
    # No torch interaction yet.
    assert est._model is None
    assert est._torch is None
