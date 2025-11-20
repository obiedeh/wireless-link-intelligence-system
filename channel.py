import numpy as np

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add complex AWGN to signal for a given SNR (dB).
    SNR defined with respect to average signal power.
    """
    signal = np.asarray(signal, dtype=complex)
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
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

    Signature expected by `main.ipynb`:
        apply_channel(tx_signal, snr_db=..., fading=False) -> (rx_signal, h)
    """
    if fading:
        ch_out, h = rayleigh_fading(tx_signal)
    else:
        ch_out, h = tx_signal, 1.0

    rx = add_awgn(ch_out, snr_db)
    return rx, h
