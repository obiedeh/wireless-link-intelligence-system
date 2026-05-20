"""3GPP TR 38.901 Tapped-Delay-Line (TDL) channel models.

Implements the NLOS profiles TDL-A, TDL-B, TDL-C from
3GPP TR 38.901 v17.x §7.7.2 (Table 7.7.2-1, -2, -3). These are the standard
link-level channel profiles used in 5G-NR research and AI-RAN literature
— every neural-receiver paper benchmarks against at least one of them.

Design choices
--------------
- **Block fading per realization.** Each `sample_tdl_realization` call draws
  one independent set of complex-Gaussian tap gains. The channel is held
  constant across the OFDM frame that the resulting impulse response is
  applied to. Doppler evolution within a frame is omitted on purpose —
  for link-level BLER vs SNR curves the standard 38.901 evaluation is
  block fading, ensemble-averaged across realizations.
- **Discrete-time sampling.** Profile delays are quantised to the nearest
  sample at the simulator sample rate; if multiple profile taps land in
  the same discrete bin, their complex gains add. This is the standard
  approximation used in toy link-level sims.
- **Power normalisation.** NLOS profiles are normalised so the total
  channel energy is 1 (the average path-loss-free output power equals
  input power). The Rayleigh fade is the variation around that mean.

References
----------
- 3GPP TR 38.901: "Study on channel model for frequencies from 0.5 to 100 GHz"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 3GPP TR 38.901 §7.7.2 — Tapped-Delay-Line profiles (NLOS subset)
#
# Values transcribed from TR 38.901 v17.0.0 Tables 7.7.2-1, 7.7.2-2, 7.7.2-3.
# Delays are normalised to the delay spread; powers in dB.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TDLProfile:
    """A 3GPP TR 38.901 tapped-delay-line channel profile."""

    name: str
    normalized_delays: np.ndarray  # length N_taps, dimensionless (τ_k / τ_DS)
    powers_db: np.ndarray  # length N_taps, dB

    @property
    def n_taps(self) -> int:
        return int(self.normalized_delays.size)


# Table 7.7.2-1
TDL_A = TDLProfile(
    name="TDL-A",
    normalized_delays=np.array([
        0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375,
        0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242,
        2.1718, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579,
        4.5695, 4.7966, 5.0066, 5.3043, 9.6586,
    ]),
    powers_db=np.array([
        -13.4,  0.0, -2.2, -4.0, -6.0, -8.2,
         -9.9, -10.5, -7.5, -15.9, -6.6, -16.7,
        -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
        -18.3, -18.9, -16.6, -19.9, -29.7,
    ]),
)

# Table 7.7.2-2
TDL_B = TDLProfile(
    name="TDL-B",
    normalized_delays=np.array([
        0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986,
        0.3752, 0.5055, 0.3681, 0.3697, 0.5700, 0.5283,
        1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294,
        3.0219, 3.6187, 4.1067, 4.2790, 4.7834,
    ]),
    powers_db=np.array([
         0.0, -2.2, -4.0,  -3.2, -9.8, -1.2,
        -3.4, -5.2, -7.6,  -3.0, -8.9, -9.0,
        -4.8, -5.7, -7.5,  -1.9, -7.6, -12.2,
        -9.8, -11.4, -14.9, -9.2, -11.3,
    ]),
)

# Table 7.7.2-3 — the most-cited 5G link-level test profile
TDL_C = TDLProfile(
    name="TDL-C",
    normalized_delays=np.array([
        0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366,
        0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336,
        1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003,
        5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523,
    ]),
    powers_db=np.array([
        -4.4, -1.2,  -3.5, -5.2, -2.5,   0.0,
        -2.2, -3.9,  -7.4, -7.1, -10.7, -11.1,
        -5.1, -6.8,  -8.7, -13.2, -13.9, -13.9,
        -15.8, -17.1, -16.0, -15.7, -21.6, -22.8,
    ]),
)

ALL_NLOS_PROFILES: tuple[TDLProfile, ...] = (TDL_A, TDL_B, TDL_C)


# ---------------------------------------------------------------------------
# Sampling a TDL impulse response
# ---------------------------------------------------------------------------


def sample_tdl_realization(
    profile: TDLProfile,
    delay_spread_samples: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one realization of the TDL channel impulse response.

    Parameters
    ----------
    profile:
        One of TDL_A / TDL_B / TDL_C.
    delay_spread_samples:
        The delay spread expressed in *simulator samples*. Each profile tap k
        is placed at the discrete-time index round(τ_k_normalized × τ_DS_samples).
        Pick this so the resulting CIR fits within the OFDM cyclic prefix
        (otherwise inter-symbol interference occurs and the curve degrades).
    rng:
        NumPy random generator (for reproducibility).

    Returns
    -------
    h:
        Complex impulse response (length = max discrete delay + 1). Total
        power is normalised so ``sum(|h|² · E[|tap_gain|²])`` = 1 in expectation
        over realisations — i.e. the *expected* channel energy is 1.
    """
    if delay_spread_samples <= 0:
        raise ValueError("delay_spread_samples must be positive.")

    powers_lin = 10.0 ** (profile.powers_db / 10.0)
    powers_lin = powers_lin / powers_lin.sum()  # normalise to total power 1

    # Profile taps as complex Gaussians with variance = profile power.
    n_profile_taps = profile.n_taps
    tap_gains = (
        rng.standard_normal(n_profile_taps) + 1j * rng.standard_normal(n_profile_taps)
    ) / np.sqrt(2)
    tap_gains = tap_gains * np.sqrt(powers_lin)

    # Quantise delays to the nearest sample index.
    delay_samples_continuous = profile.normalized_delays * delay_spread_samples
    delay_idx = np.round(delay_samples_continuous).astype(np.int64)
    delay_idx = np.clip(delay_idx, 0, None)

    # Accumulate taps that land in the same discrete bin.
    cir_length = int(delay_idx.max()) + 1
    h = np.zeros(cir_length, dtype=complex)
    for idx, gain in zip(delay_idx, tap_gains, strict=False):
        h[idx] += gain
    return h


# ---------------------------------------------------------------------------
# Applying a TDL realization to a signal
# ---------------------------------------------------------------------------


def apply_tdl_channel(
    tx_signal: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve the transmit signal with a discrete-time impulse response.

    Returns the received signal (length len(tx) + len(h) - 1) and the
    channel impulse response that was applied (passed through for downstream
    channel-estimation comparison).
    """
    tx_signal = np.asarray(tx_signal, dtype=complex)
    h = np.asarray(h, dtype=complex)
    rx = np.convolve(tx_signal, h, mode="full")
    return rx, h


def cir_to_frequency_response(h: np.ndarray, n_subcarriers: int) -> np.ndarray:
    """Convert a discrete-time CIR to its OFDM frequency response.

    The result is what an idealised OFDM channel estimator would see: the
    per-subcarrier complex channel gain.
    """
    h = np.asarray(h, dtype=complex)
    if h.size < n_subcarriers:
        h_padded = np.concatenate([h, np.zeros(n_subcarriers - h.size, dtype=complex)])
    else:
        h_padded = h[:n_subcarriers]
    return np.fft.fft(h_padded, norm="ortho") * np.sqrt(n_subcarriers)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_profile(name: str) -> TDLProfile:
    """Look up a profile by name (case-insensitive). Raises KeyError otherwise."""
    name_upper = name.upper().replace("_", "-")
    if name_upper == "TDL-A":
        return TDL_A
    if name_upper == "TDL-B":
        return TDL_B
    if name_upper == "TDL-C":
        return TDL_C
    raise KeyError(f"Unknown TDL profile: {name}. Use one of TDL-A, TDL-B, TDL-C.")
