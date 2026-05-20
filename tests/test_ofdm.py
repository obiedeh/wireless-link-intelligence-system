"""Tests for the CP-OFDM modem with adaptive Gray-coded QAM."""

from __future__ import annotations

import numpy as np
import pytest

from qpsk_link.channel import add_awgn
from qpsk_link.ofdm import (
    SUPPORTED_MOD_ORDERS,
    constellation,
    decode_ofdm_frame,
    make_ofdm_frame,
    ofdm_demodulate,
    ofdm_modulate,
    qam_demodulate,
    qam_modulate,
)

# ---------------------------------------------------------------------------
# Constellation sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod_order", SUPPORTED_MOD_ORDERS)
def test_constellation_has_correct_cardinality(mod_order: int):
    pts = constellation(mod_order)
    assert pts.shape == (mod_order,)
    # All points should be distinct.
    assert len({(p.real, p.imag) for p in pts}) == mod_order


@pytest.mark.parametrize("mod_order", SUPPORTED_MOD_ORDERS)
def test_constellation_average_energy_is_one(mod_order: int):
    pts = constellation(mod_order)
    avg_energy = float(np.mean(np.abs(pts) ** 2))
    assert avg_energy == pytest.approx(1.0, rel=1e-9)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@pytest.mark.parametrize("mod_order", SUPPORTED_MOD_ORDERS)
def test_constellation_is_gray_coded(mod_order: int):
    """Standard Gray-coded square QAM property: any two constellation points
    that are *adjacent in the I/Q grid* (one PAM step apart on exactly one
    axis, same on the other) must differ by exactly one bit in their bit
    pattern. This is what gives Gray coding its near-neighbour BER property —
    a single-symbol error at the demodulator translates to one bit flip.
    """
    pts = constellation(mod_order)
    L = int(np.sqrt(mod_order))
    norm_factor = float(np.sqrt((2.0 / 3.0) * (mod_order - 1)))
    expected_step = 2.0 / norm_factor

    # Build position → bit-pattern map by inverting the constellation.
    # For each natural-order (i_pos, q_pos), find which bit pattern produced it.
    levels = np.arange(-(L - 1), L, 2, dtype=float) / norm_factor
    pos_to_bits: dict[tuple[int, int], int] = {}
    for sym_idx, p in enumerate(pts):
        i_pos = int(np.argmin(np.abs(p.real - levels)))
        q_pos = int(np.argmin(np.abs(p.imag - levels)))
        pos_to_bits[(i_pos, q_pos)] = sym_idx

    # For each grid neighbour pair, check single-bit Hamming distance.
    for (i_pos, q_pos), bits_a in pos_to_bits.items():
        for di, dq in [(1, 0), (0, 1)]:
            ni, nq = i_pos + di, q_pos + dq
            if 0 <= ni < L and 0 <= nq < L:
                bits_b = pos_to_bits[(ni, nq)]
                # The two grid neighbours should differ by exactly one bit.
                assert _hamming(bits_a, bits_b) == 1, (
                    f"M={mod_order}: grid neighbours ({i_pos},{q_pos}) and "
                    f"({ni},{nq}) have Hamming distance "
                    f"{_hamming(bits_a, bits_b)} (expected 1)."
                )
                # And by exactly one PAM step on the moving axis, zero on the other.
                diff = pts[bits_a] - pts[bits_b]
                d_i = abs(diff.real)
                d_q = abs(diff.imag)
                assert d_i + d_q == pytest.approx(expected_step, abs=1e-9)


# ---------------------------------------------------------------------------
# QAM round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod_order", SUPPORTED_MOD_ORDERS)
def test_qam_roundtrip_no_channel(mod_order: int):
    rng = np.random.default_rng(seed=11)
    bps = int(np.log2(mod_order))
    n_bits = bps * 2048
    bits = rng.integers(0, 2, n_bits).astype(np.int64)
    symbols = qam_modulate(bits, mod_order)
    bits_hat = qam_demodulate(symbols, mod_order)
    assert bits_hat.shape == bits.shape
    assert np.array_equal(bits_hat, bits)


@pytest.mark.parametrize("mod_order", SUPPORTED_MOD_ORDERS)
def test_qam_roundtrip_high_snr(mod_order: int):
    """At 30 dB SNR, BER must be near zero for all supported QAM orders."""
    rng = np.random.default_rng(seed=mod_order + 1)
    bps = int(np.log2(mod_order))
    n_bits = bps * 4096
    bits = rng.integers(0, 2, n_bits).astype(np.int64)
    symbols = qam_modulate(bits, mod_order)
    rx_symbols = add_awgn(symbols, snr_db=30.0)
    bits_hat = qam_demodulate(rx_symbols, mod_order)
    ber = float(np.mean(bits != bits_hat))
    # 256QAM at 30 dB is the worst of these; allow up to 1e-3.
    assert ber < 1e-3, f"M={mod_order}: BER {ber:.2e} unexpectedly high at 30 dB."


# ---------------------------------------------------------------------------
# OFDM round-trip
# ---------------------------------------------------------------------------


def test_ofdm_modulate_demodulate_roundtrip_identity():
    rng = np.random.default_rng(seed=2)
    n_subcarriers = 64
    cp_length = 16
    n_ofdm_symbols = 5
    n_freq = n_subcarriers * n_ofdm_symbols
    # Random complex symbols with unit average energy.
    symbols = (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)) / np.sqrt(2)
    tx_signal = ofdm_modulate(symbols, n_subcarriers, cp_length)
    expected_length = n_ofdm_symbols * (n_subcarriers + cp_length)
    assert tx_signal.shape == (expected_length,)
    rx_symbols = ofdm_demodulate(tx_signal, n_subcarriers, cp_length)
    assert rx_symbols.shape == symbols.shape
    assert np.allclose(rx_symbols, symbols, atol=1e-9)


def test_ifft_preserves_total_energy_per_block_parseval():
    """Parseval's theorem: the unitary IFFT preserves total energy over an
    OFDM symbol. Drop the CP (which duplicates samples) and verify."""
    rng = np.random.default_rng(seed=3)
    n_subcarriers = 64
    cp_length = 16
    symbols = (
        rng.standard_normal(n_subcarriers) + 1j * rng.standard_normal(n_subcarriers)
    ) / np.sqrt(2)
    tx = ofdm_modulate(symbols, n_subcarriers, cp_length)
    # The unique time-domain samples are after the CP.
    x_no_cp = tx[cp_length:]
    e_time = float(np.sum(np.abs(x_no_cp) ** 2))
    e_freq = float(np.sum(np.abs(symbols) ** 2))
    assert e_time == pytest.approx(e_freq, rel=1e-9)


def test_ofdm_modulate_rejects_misaligned_input():
    with pytest.raises(ValueError):
        ofdm_modulate(np.zeros(63, dtype=complex), n_subcarriers=64, cp_length=16)


def test_ofdm_demodulate_rejects_misaligned_input():
    with pytest.raises(ValueError):
        ofdm_demodulate(np.zeros(79, dtype=complex), n_subcarriers=64, cp_length=16)


# ---------------------------------------------------------------------------
# End-to-end frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod_order", SUPPORTED_MOD_ORDERS)
def test_end_to_end_frame_noiseless(mod_order: int):
    rng = np.random.default_rng(seed=mod_order)
    n_subcarriers = 64
    cp_length = 16
    bps = int(np.log2(mod_order))
    # Pick a bit length that doesn't align — exercises the padding logic.
    n_bits = bps * n_subcarriers * 3 + 7
    bits = rng.integers(0, 2, n_bits).astype(np.int64)
    tx_signal, info = make_ofdm_frame(bits, n_subcarriers, cp_length, mod_order)
    bits_hat = decode_ofdm_frame(tx_signal, info)
    assert bits_hat.shape == bits.shape
    assert np.array_equal(bits_hat, bits)


@pytest.mark.parametrize("mod_order", SUPPORTED_MOD_ORDERS)
def test_end_to_end_frame_high_snr_low_ber(mod_order: int):
    """At 30 dB AWGN, end-to-end OFDM frame BER must be small for every QAM order."""
    rng = np.random.default_rng(seed=mod_order + 99)
    n_subcarriers = 64
    cp_length = 16
    bps = int(np.log2(mod_order))
    n_bits = bps * n_subcarriers * 16
    bits = rng.integers(0, 2, n_bits).astype(np.int64)
    tx_signal, info = make_ofdm_frame(bits, n_subcarriers, cp_length, mod_order)
    rx_signal = add_awgn(tx_signal, snr_db=30.0)
    bits_hat = decode_ofdm_frame(rx_signal, info)
    ber = float(np.mean(bits != bits_hat))
    assert ber < 1e-3, f"M={mod_order}: BER {ber:.2e} unexpectedly high at 30 dB."
