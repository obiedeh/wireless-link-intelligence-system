"""Pilot-based OFDM channel estimation: LS, MMSE, and a small neural estimator.

This module gives the receiver the realistic problem that ``run_sim_tdl.py``
deliberately ducked: it doesn't know the channel impulse response. Instead it
sees a noisy received OFDM frame with **known pilot symbols** inserted at
prearranged subcarrier positions, and has to estimate the channel response
across the full bandwidth from those pilots alone.

Three estimators are implemented for honest comparison:

1. **LS** — closed-form least-squares at pilot positions, linear interpolation
   across data subcarriers. The textbook baseline; cheap; noisy at low SNR.
2. **MMSE** — uses a frequency-domain channel autocorrelation derived from an
   assumed exponential delay profile and the known noise variance. Better
   than LS at low SNR; needs `R_HH` and `σ²`.
3. **Neural** — a tiny PyTorch MLP that maps (received pilots, SNR) → full
   channel response. Trained on synthetic TDL realizations. Demonstrates the
   AI-PHY pattern (DeepRx-style channel estimation) end-to-end.

The intent is to surface the *trade-offs* with measured numbers, not to
claim the neural estimator beats MMSE. On a small dataset and a 3-layer MLP
it usually doesn't — and that's a perfectly honest result to publish.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Pilot placement
# ---------------------------------------------------------------------------

DEFAULT_PILOT_STRIDE = 4
DEFAULT_PILOT_VALUE = 1.0 + 0.0j


def comb_pilot_indices(n_subcarriers: int, stride: int = DEFAULT_PILOT_STRIDE) -> np.ndarray:
    """Pilot subcarrier indices for a comb (every-Nth) pilot pattern."""
    return np.arange(0, n_subcarriers, stride, dtype=np.int64)


def data_subcarrier_indices(n_subcarriers: int, pilot_indices: np.ndarray) -> np.ndarray:
    """Subcarrier indices that are NOT pilots — used for data."""
    return np.setdiff1d(np.arange(n_subcarriers, dtype=np.int64), pilot_indices)


# ---------------------------------------------------------------------------
# Frame construction with pilots
# ---------------------------------------------------------------------------


@dataclass
class PilotedFrame:
    """Bundle of everything an estimator + equalizer needs about a piloted frame."""

    tx_symbols: np.ndarray            # (n_ofdm_symbols, n_subcarriers) complex grid
    data_indices: np.ndarray          # subcarrier indices carrying data
    pilot_indices: np.ndarray         # subcarrier indices carrying pilots
    pilot_value: complex
    n_subcarriers: int
    n_ofdm_symbols: int
    cp_length: int
    mod_order: int


def build_piloted_frame(
    data_symbols: np.ndarray,
    n_subcarriers: int,
    n_ofdm_symbols: int,
    cp_length: int,
    mod_order: int,
    pilot_stride: int = DEFAULT_PILOT_STRIDE,
    pilot_value: complex = DEFAULT_PILOT_VALUE,
) -> PilotedFrame:
    """Place data QAM symbols on the non-pilot subcarriers; pilots on the others."""
    pilot_idx = comb_pilot_indices(n_subcarriers, pilot_stride)
    data_idx = data_subcarrier_indices(n_subcarriers, pilot_idx)
    expected = n_ofdm_symbols * data_idx.size
    if data_symbols.size != expected:
        raise ValueError(
            f"data_symbols.size={data_symbols.size} but expected {expected} "
            f"(n_ofdm_symbols={n_ofdm_symbols} × n_data_subcarriers={data_idx.size})."
        )
    grid = np.zeros((n_ofdm_symbols, n_subcarriers), dtype=complex)
    grid[:, pilot_idx] = pilot_value
    grid[:, data_idx] = data_symbols.reshape(n_ofdm_symbols, data_idx.size)
    return PilotedFrame(
        tx_symbols=grid,
        data_indices=data_idx,
        pilot_indices=pilot_idx,
        pilot_value=pilot_value,
        n_subcarriers=n_subcarriers,
        n_ofdm_symbols=n_ofdm_symbols,
        cp_length=cp_length,
        mod_order=mod_order,
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _ls_pilot_estimate(rx_pilot_grid: np.ndarray, frame: PilotedFrame) -> np.ndarray:
    """Average received pilots over OFDM symbols and divide by the known pilot value.

    Returns a length-n_pilots complex array — the LS estimate at each pilot
    subcarrier position. Used by both ls_estimate (which then interpolates to
    all subcarriers) and mmse_estimate (which feeds into the MMSE filter).
    """
    return np.mean(rx_pilot_grid / frame.pilot_value, axis=0)


def ls_estimate(
    rx_pilot_grid: np.ndarray,
    frame: PilotedFrame,
) -> np.ndarray:
    """Least-squares channel estimate at pilot positions, linearly interpolated
    across the rest. Returns a length-``n_subcarriers`` complex array (per-symbol
    averaging if the frame has more than one OFDM symbol)."""
    if rx_pilot_grid.shape != (frame.n_ofdm_symbols, frame.pilot_indices.size):
        raise ValueError(
            f"rx_pilot_grid shape {rx_pilot_grid.shape} != "
            f"({frame.n_ofdm_symbols}, {frame.pilot_indices.size})."
        )
    h_pilot = _ls_pilot_estimate(rx_pilot_grid, frame)

    # Linear interpolation in real and imag parts separately.
    sc = np.arange(frame.n_subcarriers)
    real = np.interp(sc, frame.pilot_indices, h_pilot.real)
    imag = np.interp(sc, frame.pilot_indices, h_pilot.imag)
    return real + 1j * imag


def _channel_correlation_matrix(
    pilot_indices: np.ndarray,
    target_indices: np.ndarray,
    n_subcarriers: int,
    delay_spread_samples: float,
) -> np.ndarray:
    """Frequency-domain channel autocorrelation between two index sets.

    Assumes an exponential power-delay profile with decay constant equal to
    ``delay_spread_samples``. The standard MMSE prior in textbook OFDM
    channel-estimation chapters.
    """
    # Cap the assumed channel length at a reasonable multiple of the delay spread.
    n_taps = max(2, int(np.ceil(delay_spread_samples * 4)))
    powers = np.exp(-np.arange(n_taps) / max(delay_spread_samples, 1e-6))
    powers = powers / powers.sum()

    delta = target_indices[:, None] - pilot_indices[None, :]  # (T, P)
    tap_idx = np.arange(n_taps)[None, None, :]                # (1, 1, L)
    exponent = -1j * 2.0 * np.pi * delta[..., None] * tap_idx / n_subcarriers
    return (powers * np.exp(exponent)).sum(axis=-1)


@lru_cache(maxsize=32)
def _channel_correlation_matrix_cached(
    pilot_key: tuple[int, ...],
    target_key: tuple[int, ...],
    n_subcarriers: int,
    delay_spread_samples: float,
) -> np.ndarray:
    """Cached wrapper around ``_channel_correlation_matrix``.

    Accepts index arrays as tuples (hashable) so ``lru_cache`` can key on them.
    In a typical SNR sweep the pilot geometry and delay spread are fixed across
    all realisations, so the cache hits immediately after the first frame.
    """
    return _channel_correlation_matrix(
        np.asarray(pilot_key, dtype=np.int64),
        np.asarray(target_key, dtype=np.int64),
        n_subcarriers,
        delay_spread_samples,
    )


def mmse_estimate(
    rx_pilot_grid: np.ndarray,
    frame: PilotedFrame,
    snr_db: float,
    delay_spread_samples: float,
) -> np.ndarray:
    """MMSE channel estimate using an exponential-delay-profile prior."""
    if rx_pilot_grid.shape != (frame.n_ofdm_symbols, frame.pilot_indices.size):
        raise ValueError("rx_pilot_grid has wrong shape for this frame.")
    h_pilot_ls = _ls_pilot_estimate(rx_pilot_grid, frame)

    sc = np.arange(frame.n_subcarriers)
    pilot_idx = frame.pilot_indices
    n_sc = frame.n_subcarriers

    # R_pp and R_hp depend only on frame geometry and channel prior — not on
    # the received signal — so they are the same for every realization in a
    # sweep. The cached wrapper avoids recomputing them on every frame.
    pilot_key = tuple(pilot_idx.tolist())
    R_pp = _channel_correlation_matrix_cached(
        pilot_key, pilot_key, n_sc, delay_spread_samples
    )
    R_hp = _channel_correlation_matrix_cached(
        pilot_key, tuple(sc.tolist()), n_sc, delay_spread_samples
    )

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / snr_linear / (abs(frame.pilot_value) ** 2)

    # solve is ~2× faster and more numerically stable than explicit inv.
    w = np.linalg.solve(
        R_pp + noise_var * np.eye(pilot_idx.size, dtype=complex), h_pilot_ls
    )
    return R_hp @ w


# ---------------------------------------------------------------------------
# Neural channel estimator — small PyTorch MLP
#
# Imported lazily so consumers that only need LS/MMSE don't pay the torch
# import cost or fail when torch isn't installed.
# ---------------------------------------------------------------------------


class _LazyTorchEstimator:
    """Wrapper that constructs a torch model only on first use.

    Why this matters: importing torch is ~1 s; users running only the LS/MMSE
    comparison should not be charged that. Also lets the module import cleanly
    on machines without torch installed.
    """

    def __init__(self, n_pilots: int, n_subcarriers: int, hidden: int = 128):
        self.n_pilots = n_pilots
        self.n_subcarriers = n_subcarriers
        self.hidden = hidden
        self._model = None
        self._torch = None

    def _build(self):
        if self._model is not None:
            return
        import torch
        from torch import nn

        self._torch = torch
        input_dim = 2 * self.n_pilots + 1
        output_dim = 2 * self.n_subcarriers
        self._model = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, output_dim),
        )

    @property
    def model(self):
        self._build()
        return self._model

    @property
    def torch(self):
        self._build()
        return self._torch

    def estimate(
        self,
        rx_pilot_grid: np.ndarray,
        snr_db: float,
    ) -> np.ndarray:
        """Run inference on one frame. Returns length-n_subcarriers complex array."""
        self._build()
        h_pilot = np.mean(rx_pilot_grid, axis=0)  # average over OFDM symbols
        x = np.concatenate([h_pilot.real, h_pilot.imag, [snr_db]])
        x_t = self.torch.tensor(x, dtype=self.torch.float32).unsqueeze(0)
        self.model.eval()
        with self.torch.no_grad():
            out = self.model(x_t).numpy().ravel()
        n = self.n_subcarriers
        return out[:n] + 1j * out[n:]


def neural_estimator(n_pilots: int, n_subcarriers: int, hidden: int = 128) -> _LazyTorchEstimator:
    """Factory for the neural estimator. Returns an object exposing ``.model``,
    ``.estimate(rx_pilots, snr_db)``, and ``.torch`` for training."""
    return _LazyTorchEstimator(n_pilots, n_subcarriers, hidden=hidden)
