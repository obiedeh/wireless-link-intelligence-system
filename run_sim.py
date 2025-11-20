import numpy as np
from qpsk_modem import qpsk_modulate, qpsk_demodulate
from channel import apply_channel

def simulate_qpsk_link(num_bits=10000, sps=8, beta=0.35, snr_db=10.0, fading=False):
    bits_tx = np.random.randint(0, 2, num_bits)
    tx_signal, syms_tx, h_rrc = qpsk_modulate(bits_tx, sps=sps, beta=beta)
    rx_signal, h = apply_channel(tx_signal, snr_db=snr_db, fading=fading)
    num_syms = len(syms_tx)
    bits_rx = qpsk_demodulate(rx_signal, h_rrc, sps, num_syms, channel_coef=h)
    bits_rx = bits_rx[:len(bits_tx)]
    bit_errors = np.sum(bits_tx != bits_rx)
    ber = bit_errors / len(bits_tx)
    return ber

if __name__ == '__main__':
    sps = 8
    beta = 0.35
    snr_range = list(range(0, 21, 2))

    print('=== Quick AWGN BER sweep (num_bits=5000) ===')
    for snr in snr_range:
        ber = simulate_qpsk_link(num_bits=5000, sps=sps, beta=beta, snr_db=snr, fading=False)
        print(f'SNR = {snr:2d} dB | BER = {ber:.4e}')

    print('\n=== Example at 10 dB (AWGN) ===')
    ber_example = simulate_qpsk_link(num_bits=4000, sps=sps, beta=beta, snr_db=10, fading=False)
    print(f'Example BER at 10 dB (AWGN): {ber_example:.4e}')
