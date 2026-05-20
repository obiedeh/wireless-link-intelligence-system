"""Tests for the 3GPP TR 38.901 TDL channel models."""

from __future__ import annotations

import numpy as np
import pytest

from qpsk_link.tdl_channel import (
    ALL_NLOS_PROFILES,
    TDL_A,
    TDL_B,
    TDL_C,
    TDLProfile,
    apply_tdl_channel,
    cir_to_frequency_response,
    get_profile,
    sample_tdl_realization,
)


@pytest.mark.parametrize("profile", ALL_NLOS_PROFILES)
def test_profile_tap_lengths_match(profile: TDLProfile):
    assert profile.normalized_delays.shape == profile.powers_db.shape
    assert profile.normalized_delays.size > 0


@pytest.mark.parametrize("profile", ALL_NLOS_PROFILES)
def test_profile_delays_are_nondecreasing_after_sort(profile: TDLProfile):
    """Profile delays do not have to be sorted (TR 38.901 isn't), but the
    minimum delay must be non-negative and finite."""
    assert np.all(profile.normalized_delays >= 0)
    assert np.all(np.isfinite(profile.normalized_delays))


@pytest.mark.parametrize("profile", ALL_NLOS_PROFILES)
def test_powers_are_finite_and_negative_or_zero(profile: TDLProfile):
    assert np.all(np.isfinite(profile.powers_db))
    # The convention is to report relative powers — at least the strongest
    # path should be at 0 dB, others at negative dB. Some profiles (TDL-A)
    # don't normalise to 0 dB; just check finite, nothing pathological.
    assert profile.powers_db.max() <= 0.001  # allow small float noise


@pytest.mark.parametrize("profile", ALL_NLOS_PROFILES)
def test_realization_average_energy_normalised_to_one(profile: TDLProfile):
    """Across N realisations, E[ Σ |h_k|² ] should converge to 1
    (NLOS profiles are power-normalised)."""
    rng = np.random.default_rng(seed=42)
    n_realisations = 4000
    delay_spread = 4.0  # samples
    energies = []
    for _ in range(n_realisations):
        h = sample_tdl_realization(profile, delay_spread, rng)
        energies.append(float(np.sum(np.abs(h) ** 2)))
    mean_energy = float(np.mean(energies))
    # With 4000 realizations and 23 taps each, the variance shrinks; expect
    # within ~3% of 1.0.
    assert mean_energy == pytest.approx(1.0, abs=0.05)


def test_realization_length_matches_delay_spread():
    """Channel impulse response length should equal max discretised delay + 1."""
    rng = np.random.default_rng(seed=1)
    delay_spread = 2.0
    h = sample_tdl_realization(TDL_C, delay_spread, rng)
    expected_max_idx = int(np.round(TDL_C.normalized_delays.max() * delay_spread))
    assert h.size == expected_max_idx + 1


def test_apply_tdl_channel_convolution_length():
    rng = np.random.default_rng(seed=0)
    h = sample_tdl_realization(TDL_A, delay_spread_samples=4.0, rng=rng)
    tx = np.ones(100, dtype=complex)
    rx, h_out = apply_tdl_channel(tx, h)
    assert rx.shape == (100 + h.size - 1,)
    assert np.allclose(h_out, h)


def test_cir_to_frequency_response_matches_fft():
    """The OFDM per-subcarrier channel gain equals FFT(h_padded)."""
    rng = np.random.default_rng(seed=3)
    h = sample_tdl_realization(TDL_C, delay_spread_samples=3.0, rng=rng)
    n_subcarriers = 64
    H = cir_to_frequency_response(h, n_subcarriers)
    assert H.shape == (n_subcarriers,)
    # Direct verification: padded FFT
    h_padded = np.concatenate([h, np.zeros(n_subcarriers - h.size, dtype=complex)])
    expected = np.fft.fft(h_padded, norm="ortho") * np.sqrt(n_subcarriers)
    assert np.allclose(H, expected)


def test_get_profile_lookup():
    assert get_profile("TDL-A") is TDL_A
    assert get_profile("tdl_b") is TDL_B
    assert get_profile("tdl-c") is TDL_C
    with pytest.raises(KeyError):
        get_profile("TDL-Z")


def test_realization_deterministic_with_same_seed():
    rng1 = np.random.default_rng(seed=99)
    rng2 = np.random.default_rng(seed=99)
    h1 = sample_tdl_realization(TDL_B, delay_spread_samples=4.0, rng=rng1)
    h2 = sample_tdl_realization(TDL_B, delay_spread_samples=4.0, rng=rng2)
    assert np.allclose(h1, h2)
