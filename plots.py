import numpy as np
import matplotlib.pyplot as plt

def plot_constellation(tx_syms, rx_syms=None, title="QPSK Constellation"):
    tx_syms = np.asarray(tx_syms)
    plt.figure()
    plt.scatter(tx_syms.real, tx_syms.imag, marker="o", alpha=0.4, label="Tx")
    if rx_syms is not None:
        rx_syms = np.asarray(rx_syms)
        plt.scatter(rx_syms.real, rx_syms.imag, marker="x", alpha=0.4, label="Rx")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ber(snr_db_list, ber_awgn, ber_rayleigh=None):
    snr_db_list = np.array(snr_db_list)
    ber_awgn = np.array(ber_awgn)
    plt.figure()
    plt.semilogy(snr_db_list, ber_awgn, marker="o", label="AWGN")
    if ber_rayleigh is not None:
        ber_rayleigh = np.array(ber_rayleigh)
        plt.semilogy(snr_db_list, ber_rayleigh, marker="s", label="Rayleigh")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("QPSK BER Performance")
    plt.legend()
    plt.tight_layout()
    plt.show()
