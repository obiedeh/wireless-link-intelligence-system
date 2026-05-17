"""Feature extraction for QPSK link-condition estimation."""

from __future__ import annotations

import numpy as np

FEATURE_COLUMNS = [
    "rx_power_mean",
    "rx_power_std",
    "real_mean",
    "imag_mean",
    "real_std",
    "imag_std",
    "iq_covariance",
    "radius_mean",
    "radius_std",
    "phase_std",
    "evm_rms",
    "quadrant_balance",
]


def constellation_statistics(tx_symbols: np.ndarray,
                             rx_symbols: np.ndarray,
                             channel_coef: complex) -> dict[str, float]:
    """Return numeric constellation statistics suitable for tabular ML."""
    tx_symbols = np.asarray(tx_symbols, dtype=complex)
    rx_symbols = np.asarray(rx_symbols, dtype=complex)
    if tx_symbols.size == 0 or rx_symbols.size == 0:
        raise ValueError("tx_symbols and rx_symbols must be non-empty")

    n = min(tx_symbols.size, rx_symbols.size)
    tx_symbols = tx_symbols[:n]
    rx_symbols = rx_symbols[:n]

    power = np.abs(rx_symbols) ** 2
    radius = np.abs(rx_symbols)
    phase = np.unwrap(np.angle(rx_symbols))
    errors = rx_symbols - tx_symbols
    evm_rms = np.sqrt(np.mean(np.abs(errors) ** 2))

    quadrants = np.array([
        np.mean((rx_symbols.real >= 0) & (rx_symbols.imag >= 0)),
        np.mean((rx_symbols.real < 0) & (rx_symbols.imag >= 0)),
        np.mean((rx_symbols.real < 0) & (rx_symbols.imag < 0)),
        np.mean((rx_symbols.real >= 0) & (rx_symbols.imag < 0)),
    ])

    return {
        "rx_power_mean": float(np.mean(power)),
        "rx_power_std": float(np.std(power)),
        "real_mean": float(np.mean(rx_symbols.real)),
        "imag_mean": float(np.mean(rx_symbols.imag)),
        "real_std": float(np.std(rx_symbols.real)),
        "imag_std": float(np.std(rx_symbols.imag)),
        "iq_covariance": float(np.cov(rx_symbols.real, rx_symbols.imag)[0, 1]),
        "radius_mean": float(np.mean(radius)),
        "radius_std": float(np.std(radius)),
        "phase_std": float(np.std(phase)),
        "evm_rms": float(evm_rms),
        "quadrant_balance": float(np.std(quadrants)),
        "fading_abs": float(np.abs(channel_coef)),
        "fading_phase": float(np.angle(channel_coef)),
    }


def link_quality_score(snr_db: float, ber: float, channel_type: str) -> float:
    """
    Compute a bounded engineering score from simulated link labels.

    This is a synthetic quality target for AI-assisted monitoring, not a
    standards-defined telecom KPI.
    """
    snr_component = np.clip((snr_db + 2.0) / 24.0, 0.0, 1.0)
    ber_component = 1.0 - np.clip(ber / 0.5, 0.0, 1.0)
    fading_penalty = 0.08 if channel_type == "rayleigh" else 0.0
    score = 100.0 * (0.55 * snr_component + 0.45 * ber_component - fading_penalty)
    return float(np.clip(score, 0.0, 100.0))
