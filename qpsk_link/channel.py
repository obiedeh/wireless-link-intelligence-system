import numpy as np


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    reference_power: float | None = None,
) -> np.ndarray:
    """
    Add complex AWGN to signal for a given SNR (dB).

    ``reference_power`` controls which power the SNR is computed against:

    - ``None`` (default, backwards-compatible) — SNR is referenced to the average
      power of the input ``signal`` itself. This is the right convention for a
      no-fading AWGN channel where the signal entering ``add_awgn`` is the
      transmit signal.
    - explicit float — SNR is referenced to this externally supplied power. Use
      this from ``apply_channel`` when fading has scaled ``signal`` by ``h``, so
      the SNR remains transmit-power-referenced and ``|h|²`` does not cancel out.
    """
    signal = np.asarray(signal, dtype=complex)
    if reference_power is None:
        reference_power = float(np.mean(np.abs(signal) ** 2))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = reference_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise

def rayleigh_fading(signal: np.ndarray):
    """
    Flat Rayleigh fading channel (single-tap).
    Returns (faded_signal, channel_coef).
    """
    h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    return h * signal, h

def apply_channel(tx_signal: np.ndarray,
                  snr_db: float,
                  fading: bool = False):
    """
    Wrapper to apply optional Rayleigh fading + AWGN.

    SNR is referenced to the transmit signal power on both paths, so the
    Rayleigh BER curve reflects the classical diversity penalty rather than
    being normalized out by ``|h|²``.

    Signature expected by `main.ipynb`:
        apply_channel(tx_signal, snr_db=..., fading=False) -> (rx_signal, h)
    """
    tx_signal = np.asarray(tx_signal, dtype=complex)
    tx_power = float(np.mean(np.abs(tx_signal) ** 2))

    if fading:
        ch_out, h = rayleigh_fading(tx_signal)
    else:
        ch_out, h = tx_signal, 1.0

    rx = add_awgn(ch_out, snr_db, reference_power=tx_power)
    return rx, h
