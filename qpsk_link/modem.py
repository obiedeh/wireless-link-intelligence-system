"""qpsk_modem.py – Modem & pulse shaping."""

import numpy as np
from scipy.signal import upfirdn


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    """
    Map bits (0/1) to normalized QPSK symbols using Gray coding.
    Input length must be even.
    """
    bits = np.asarray(bits).astype(int)
    if bits.size % 2 != 0:
        raise ValueError("Number of bits must be even for QPSK mapping.")
    pairs = bits.reshape(-1, 2)
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    syms = np.array([mapping[tuple(b)] for b in pairs], dtype=complex)
    return syms / np.sqrt(2)  # normalize to unit average power

def qpsk_to_bits(symbols: np.ndarray) -> np.ndarray:
    """
    Hard-decision QPSK demapper (normalized constellation).
    Returns an array of 0/1 bits.
    """
    symbols = np.asarray(symbols)
    bits_out = []
    for s in symbols:
        if s.real >= 0 and s.imag >= 0:
            bits_out.extend([0, 0])
        elif s.real < 0 and s.imag >= 0:
            bits_out.extend([0, 1])
        elif s.real < 0 and s.imag < 0:
            bits_out.extend([1, 1])
        else:
            bits_out.extend([1, 0])
    return np.array(bits_out, dtype=int)

def rrc_filter(num_taps: int, beta: float, sps: int) -> np.ndarray:
    """
    Generate a Root Raised Cosine (RRC) filter impulse response.

    num_taps: number of taps (odd is typical)
    beta: roll-off factor (0..1)
    sps: samples per symbol
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps should be odd for symmetric RRC filter.")
    # Center at index num_taps//2 so t[num_taps//2]==0 and the filter is symmetric.
    # Using arange(num_taps)-center avoids the floor-division sign ambiguity of
    # np.arange(-num_taps//2, ...) where Python evaluates (-65)//2 == -33, not -32.
    t = (np.arange(num_taps) - num_taps // 2) / sps

    with np.errstate(divide="ignore", invalid="ignore"):
        num = (np.sin(np.pi * t * (1 - beta)) +
               4 * beta * t * np.cos(np.pi * t * (1 + beta)))
        den = np.pi * t * (1 - (4 * beta * t) ** 2)
        h = np.where(den != 0, num / den, 0.0)

    # t == 0
    h[num_taps // 2] = 1.0 - beta + 4 * beta / np.pi

    # |t| == 1 / (4 * beta): singularity in denominator when beta != 0
    if beta != 0:
        special_val = (beta / np.sqrt(2)) * (
            ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
            ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        )
        h[np.isclose(np.abs(t), 1.0 / (4 * beta))] = special_val

    h /= np.sqrt(np.sum(h ** 2))
    return h

def qpsk_modulate(bits: np.ndarray, sps: int = 8, beta: float = 0.35):
    """
    Full baseband QPSK modulation chain:
      bits -> symbols -> pulse shaping (RRC) -> upsampled baseband signal.

    Returns:
      tx_signal: shaped complex baseband signal
      syms: QPSK symbol sequence
      h_rrc: RRC filter used
    """
    bits = np.asarray(bits).astype(int)
    if bits.size % 2 != 0:
        bits = bits[:-1]  # drop last bit to keep it simple

    syms = bits_to_qpsk(bits)
    num_taps = 8 * sps + 1
    h_rrc = rrc_filter(num_taps, beta, sps)
    tx_signal = upfirdn(h_rrc, syms, up=sps)
    return tx_signal, syms, h_rrc

def recover_qpsk_symbols(rx_signal: np.ndarray,
                         h_rrc: np.ndarray,
                         sps: int,
                         num_syms: int,
                         channel_coef: complex = 1.0) -> np.ndarray:
    """
    Matched filter + downsample + scalar equalization.

    Returns recovered QPSK symbols before hard-decision demapping.
    """
    # Matched filter (time-reversed conjugate)
    rx_filt = np.convolve(rx_signal, h_rrc[::-1].conj(), mode="same")

    # Symbol timing: assume perfect alignment at center
    offset = len(h_rrc) // 2
    rx_syms = rx_filt[offset::sps][:num_syms]

    # Equalize flat fading (scalar divide)
    rx_syms_eq = rx_syms / channel_coef
    return rx_syms_eq

def qpsk_demodulate(rx_signal: np.ndarray,
                    h_rrc: np.ndarray,
                    sps: int,
                    num_syms: int,
                    channel_coef: complex = 1.0) -> np.ndarray:
    """
    Matched filter + downsample + equalize + hard-decision demap.

    Returns recovered bits.
    """
    rx_syms_eq = recover_qpsk_symbols(
        rx_signal=rx_signal,
        h_rrc=h_rrc,
        sps=sps,
        num_syms=num_syms,
        channel_coef=channel_coef,
    )
    bits_hat = qpsk_to_bits(rx_syms_eq)
    return bits_hat
