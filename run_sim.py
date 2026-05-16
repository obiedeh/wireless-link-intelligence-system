import argparse

import numpy as np

from channel import apply_channel
from qpsk_modem import qpsk_demodulate, qpsk_modulate


def simulate_qpsk_link(num_bits=10000, sps=8, beta=0.35, snr_db=10.0, fading=False):
    bits_tx = np.random.randint(0, 2, num_bits)
    tx_signal, syms_tx, h_rrc = qpsk_modulate(bits_tx, sps=sps, beta=beta)
    rx_signal, h = apply_channel(tx_signal, snr_db=snr_db, fading=fading)
    num_syms = len(syms_tx)
    bits_rx = qpsk_demodulate(rx_signal, h_rrc, sps, num_syms, channel_coef=h)
    bits_rx = bits_rx[:len(bits_tx)]
    ber = np.sum(bits_tx != bits_rx) / len(bits_tx)
    return float(ber)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QPSK BER sweep over SNR range.")
    parser.add_argument("--fading", action="store_true", help="Enable Rayleigh fading channel.")
    parser.add_argument("--num-bits", type=int, default=5000)
    parser.add_argument("--snr-min", type=int, default=0)
    parser.add_argument("--snr-max", type=int, default=20)
    parser.add_argument("--snr-step", type=int, default=2)
    args = parser.parse_args()

    sps = 8
    beta = 0.35
    snr_range = range(args.snr_min, args.snr_max + 1, args.snr_step)
    label = "Rayleigh" if args.fading else "AWGN"

    print(f"=== Quick {label} BER sweep (num_bits={args.num_bits}) ===")
    for snr in snr_range:
        ber = simulate_qpsk_link(
            num_bits=args.num_bits, sps=sps, beta=beta, snr_db=snr, fading=args.fading
        )
        print(f"SNR = {snr:2d} dB | BER = {ber:.4e}")
