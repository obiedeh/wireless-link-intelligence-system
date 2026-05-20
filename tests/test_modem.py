import numpy as np
import pytest

from qpsk_link.modem import (
    bits_to_qpsk,
    qpsk_demodulate,
    qpsk_modulate,
    qpsk_to_bits,
    rrc_filter,
)


def test_bits_to_qpsk_round_trip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 1000)
    syms = bits_to_qpsk(bits)
    bits_hat = qpsk_to_bits(syms)
    assert np.array_equal(bits, bits_hat)


def test_bits_to_qpsk_requires_even():
    with pytest.raises(ValueError, match="even"):
        bits_to_qpsk(np.array([0, 1, 0]))


def test_bits_to_qpsk_unit_average_power():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 2000)
    syms = bits_to_qpsk(bits)
    assert abs(np.mean(np.abs(syms) ** 2) - 1.0) < 0.01


def test_rrc_filter_energy_normalized():
    h = rrc_filter(65, 0.35, 8)
    assert abs(np.sum(h ** 2) - 1.0) < 1e-10


def test_rrc_filter_rejects_even_taps():
    with pytest.raises(ValueError, match="odd"):
        rrc_filter(64, 0.35, 8)


def test_rrc_filter_symmetric():
    h = rrc_filter(65, 0.35, 8)
    assert np.allclose(h, h[::-1])


def test_qpsk_modulate_demodulate_zero_noise():
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, 1000)
    tx_signal, syms, h_rrc = qpsk_modulate(bits, sps=8, beta=0.35)
    bits_hat = qpsk_demodulate(tx_signal, h_rrc, sps=8, num_syms=len(syms))
    bits_hat = bits_hat[: len(bits)]
    assert np.array_equal(bits, bits_hat)


def test_qpsk_demodulate_high_snr_low_ber():
    from qpsk_link.channel import apply_channel

    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 4000)
    tx_signal, syms, h_rrc = qpsk_modulate(bits, sps=8, beta=0.35)
    rx_signal, h = apply_channel(tx_signal, snr_db=20.0, fading=False)
    bits_hat = qpsk_demodulate(rx_signal, h_rrc, sps=8, num_syms=len(syms), channel_coef=h)
    bits_hat = bits_hat[: len(bits)]
    ber = np.mean(bits != bits_hat)
    assert ber < 0.01
