"""CP-OFDM modem with adaptive Gray-coded square QAM (4 / 16 / 64 / 256).

Design notes
------------
- All FFTs are unitary (1/√N scaling on IFFT, no extra factor on FFT) so that
  symbol power is preserved through IFFT/FFT pair and SNR computations match
  the frequency-domain perspective.
- QAM constellations are square (L × L with L = √M, M ∈ {4, 16, 64, 256}).
  Each per-dimension PAM uses standard Gray-coded mapping so adjacent
  amplitude levels differ by one bit.
- Constellations are normalized to unit average symbol energy: scale by
  1/√((2/3)(M−1)).
- The data flow:

      bits ──split log2(M) per symbol──► (I_bits, Q_bits)
                                              │
                                              ▼
                              Gray-decode → I_level, Q_level
                                              │
                                              ▼
                              symbol = (I + jQ) / norm
                                              │
                                              ▼
                              IFFT(N_sc) + CP → time-domain OFDM symbol

This module is pure NumPy. PyTorch enters only in
``ai_link_estimation`` and ``channel_estimation``.
"""

from __future__ import annotations

from functools import cache

import numpy as np

# Supported modulation orders for adaptive QAM.
SUPPORTED_MOD_ORDERS = (4, 16, 64, 256)


# ---------------------------------------------------------------------------
# Gray-coded PAM helpers
# ---------------------------------------------------------------------------


def _gray_code(i: int) -> int:
    """Standard reflected binary Gray code: G(i) = i ^ (i >> 1)."""
    return i ^ (i >> 1)


@cache
def _pam_gray_tables(L: int) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Build Gray-coded PAM lookup tables for an L-level PAM (L ∈ {2,4,8,16}).

    Returns
    -------
    levels:
        PAM amplitudes in natural ascending order: [-(L-1), -(L-3), ..., (L-1)].
    bits_to_level_idx:
        Length-L array where ``bits_to_level_idx[g]`` is the natural-order
        index (0..L-1) of the level whose Gray code is ``g``.
    level_idx_to_bits:
        Inverse mapping: index in natural order → Gray-coded bit pattern.
    """
    k = int(np.log2(L))
    if 1 << k != L:
        raise ValueError(f"L must be a power of 2 (got {L}).")
    levels = np.arange(-(L - 1), L, 2, dtype=float)
    bits_to_level_idx = np.zeros(L, dtype=np.int64)
    level_idx_to_bits: dict[int, int] = {}
    for i in range(L):
        g = _gray_code(i)
        bits_to_level_idx[g] = i
        level_idx_to_bits[i] = g
    return levels, bits_to_level_idx, level_idx_to_bits


def _qam_normalization(mod_order: int) -> float:
    """Normalization factor so the constellation has unit average symbol energy.

    For square M-QAM with L = √M PAM levels per dimension, the average symbol
    energy without normalization is 2 · (L² − 1) / 3 = 2(M − 1)/3 — so divide
    each symbol by √((2/3)(M − 1)).
    """
    return float(np.sqrt((2.0 / 3.0) * (mod_order - 1)))


def constellation(mod_order: int) -> np.ndarray:
    """Return the full M-QAM constellation as a complex array of length M.

    Symbols are normalized to unit average energy. Index ordering is the
    Gray-coded bit pattern interpreted as ``(I_bits << k_per_dim) | Q_bits``.
    """
    if mod_order not in SUPPORTED_MOD_ORDERS:
        raise ValueError(f"mod_order must be in {SUPPORTED_MOD_ORDERS}, got {mod_order}.")
    L = int(np.sqrt(mod_order))
    levels, _, level_idx_to_bits = _pam_gray_tables(L)
    norm = _qam_normalization(mod_order)
    k = int(np.log2(L))
    points = np.zeros(mod_order, dtype=complex)
    for i_idx in range(L):
        for q_idx in range(L):
            g_i = level_idx_to_bits[i_idx]
            g_q = level_idx_to_bits[q_idx]
            sym_index = (g_i << k) | g_q
            points[sym_index] = (levels[i_idx] + 1j * levels[q_idx]) / norm
    return points


# ---------------------------------------------------------------------------
# QAM modulation / demodulation
# ---------------------------------------------------------------------------


def qam_modulate(bits: np.ndarray, mod_order: int) -> np.ndarray:
    """Map a 1-D ``bits`` array to complex QAM symbols.

    Length of ``bits`` must be a multiple of ``log2(mod_order)``.
    """
    if mod_order not in SUPPORTED_MOD_ORDERS:
        raise ValueError(f"mod_order must be in {SUPPORTED_MOD_ORDERS}, got {mod_order}.")
    bits = np.asarray(bits, dtype=np.int64).ravel()
    bps = int(np.log2(mod_order))  # bits per symbol
    if bits.size % bps != 0:
        raise ValueError(
            f"len(bits)={bits.size} is not a multiple of log2({mod_order})={bps}."
        )

    L = int(np.sqrt(mod_order))
    k = bps // 2  # bits per dimension
    levels, bits_to_level_idx, _ = _pam_gray_tables(L)
    norm = _qam_normalization(mod_order)

    n_symbols = bits.size // bps
    groups = bits.reshape(n_symbols, bps)
    # I bits are the high half, Q bits are the low half.
    i_bits = groups[:, :k]
    q_bits = groups[:, k:]

    # Pack bit groups into integers (MSB first).
    powers = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
    i_codes = (i_bits * powers).sum(axis=1)
    q_codes = (q_bits * powers).sum(axis=1)

    i_levels = levels[bits_to_level_idx[i_codes]]
    q_levels = levels[bits_to_level_idx[q_codes]]

    return (i_levels + 1j * q_levels) / norm


def qam_demodulate(symbols: np.ndarray, mod_order: int) -> np.ndarray:
    """Hard-decision QAM demap. Returns the bit array."""
    if mod_order not in SUPPORTED_MOD_ORDERS:
        raise ValueError(f"mod_order must be in {SUPPORTED_MOD_ORDERS}, got {mod_order}.")
    symbols = np.asarray(symbols, dtype=complex).ravel()
    L = int(np.sqrt(mod_order))
    bps = int(np.log2(mod_order))
    k = bps // 2
    levels, _, level_idx_to_bits = _pam_gray_tables(L)
    norm = _qam_normalization(mod_order)

    # Rescale to natural PAM amplitudes, then snap to nearest level.
    i_amp = (symbols.real * norm)
    q_amp = (symbols.imag * norm)
    i_idx = np.clip(np.round((i_amp + (L - 1)) / 2.0).astype(np.int64), 0, L - 1)
    q_idx = np.clip(np.round((q_amp + (L - 1)) / 2.0).astype(np.int64), 0, L - 1)

    # Look up Gray-coded bit patterns for each dimension.
    idx_to_bits = np.array([level_idx_to_bits[i] for i in range(L)], dtype=np.int64)
    i_bits_int = idx_to_bits[i_idx]
    q_bits_int = idx_to_bits[q_idx]

    # Unpack ints into k-bit MSB-first arrays.
    powers = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
    i_bits = ((i_bits_int[:, None] // powers) & 1)
    q_bits = ((q_bits_int[:, None] // powers) & 1)
    out = np.concatenate([i_bits, q_bits], axis=1)
    return out.ravel().astype(np.int64)


# ---------------------------------------------------------------------------
# OFDM modulation / demodulation
# ---------------------------------------------------------------------------


def ofdm_modulate(
    symbols: np.ndarray,
    n_subcarriers: int,
    cp_length: int,
) -> np.ndarray:
    """Apply IFFT + cyclic prefix to a sequence of frequency-domain symbols.

    Parameters
    ----------
    symbols:
        1-D array of length ``n_ofdm_symbols * n_subcarriers``. Each block of
        ``n_subcarriers`` is one OFDM symbol's frequency-domain data.
    n_subcarriers:
        FFT size (must equal the per-OFDM-symbol block length).
    cp_length:
        Cyclic prefix length in samples (must be ≥ 0).

    Returns
    -------
    Time-domain signal of shape ``(n_ofdm_symbols * (n_subcarriers + cp_length),)``.
    """
    symbols = np.asarray(symbols, dtype=complex).ravel()
    if symbols.size % n_subcarriers != 0:
        raise ValueError(
            f"len(symbols)={symbols.size} must be a multiple of n_subcarriers={n_subcarriers}."
        )
    if cp_length < 0:
        raise ValueError("cp_length must be non-negative.")
    n_symbols = symbols.size // n_subcarriers
    blocks = symbols.reshape(n_symbols, n_subcarriers)

    # Unitary IFFT (norm='ortho') so power is preserved across the transform.
    time_blocks = np.fft.ifft(blocks, axis=1, norm="ortho")
    if cp_length > 0:
        cp = time_blocks[:, -cp_length:]
        time_blocks = np.concatenate([cp, time_blocks], axis=1)
    return time_blocks.ravel()


def ofdm_demodulate(
    signal: np.ndarray,
    n_subcarriers: int,
    cp_length: int,
) -> np.ndarray:
    """Inverse of :func:`ofdm_modulate`. Returns the frequency-domain symbols."""
    signal = np.asarray(signal, dtype=complex).ravel()
    symbol_length = n_subcarriers + cp_length
    if signal.size % symbol_length != 0:
        raise ValueError(
            f"len(signal)={signal.size} must be a multiple of "
            f"(n_subcarriers + cp_length)={symbol_length}."
        )
    n_symbols = signal.size // symbol_length
    blocks = signal.reshape(n_symbols, symbol_length)
    if cp_length > 0:
        blocks = blocks[:, cp_length:]
    freq_blocks = np.fft.fft(blocks, axis=1, norm="ortho")
    return freq_blocks.ravel()


# ---------------------------------------------------------------------------
# End-to-end frame helpers (TX / RX of an OFDM frame with a chosen QAM order)
# ---------------------------------------------------------------------------


def make_ofdm_frame(
    bits: np.ndarray,
    n_subcarriers: int,
    cp_length: int,
    mod_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode an arbitrary-length bit array into an OFDM frame.

    Pads bits with zeros to fit an integer number of OFDM symbols.

    Returns
    -------
    tx_signal:
        Time-domain transmit samples.
    info:
        Dict-like array with ``n_bits``, ``n_padded_bits``, ``n_ofdm_symbols``,
        and the QAM symbols themselves (for evaluation/regression tests).
    """
    bps = int(np.log2(mod_order))
    bits_per_ofdm_symbol = n_subcarriers * bps

    bits = np.asarray(bits, dtype=np.int64).ravel()
    n_bits = bits.size
    n_ofdm_symbols = int(np.ceil(n_bits / bits_per_ofdm_symbol))
    n_padded = n_ofdm_symbols * bits_per_ofdm_symbol - n_bits
    if n_padded > 0:
        bits = np.concatenate([bits, np.zeros(n_padded, dtype=np.int64)])

    symbols = qam_modulate(bits, mod_order)
    tx_signal = ofdm_modulate(symbols, n_subcarriers, cp_length)

    info = {
        "n_bits": n_bits,
        "n_padded_bits": n_padded,
        "n_ofdm_symbols": n_ofdm_symbols,
        "n_subcarriers": n_subcarriers,
        "cp_length": cp_length,
        "mod_order": mod_order,
        "tx_symbols": symbols,
    }
    return tx_signal, info


def decode_ofdm_frame(
    rx_signal: np.ndarray,
    info: dict,
) -> np.ndarray:
    """Decode a received OFDM frame back into bits.

    ``info`` must come from :func:`make_ofdm_frame`. Returns only the original
    (unpadded) bits.
    """
    n_subcarriers = info["n_subcarriers"]
    cp_length = info["cp_length"]
    mod_order = info["mod_order"]
    n_bits = info["n_bits"]

    rx_symbols = ofdm_demodulate(rx_signal, n_subcarriers, cp_length)
    rx_bits = qam_demodulate(rx_symbols, mod_order)
    return rx_bits[:n_bits]
