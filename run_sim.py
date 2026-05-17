import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from channel import apply_channel
from qpsk_modem import qpsk_demodulate, qpsk_modulate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def simulate_qpsk_link(num_bits=10000, sps=8, beta=0.35, snr_db=10.0, fading=False):
    bits_tx = np.random.randint(0, 2, num_bits)
    tx_signal, syms_tx, h_rrc = qpsk_modulate(bits_tx, sps=sps, beta=beta)
    rx_signal, h = apply_channel(tx_signal, snr_db=snr_db, fading=fading)
    num_syms = len(syms_tx)
    bits_rx = qpsk_demodulate(rx_signal, h_rrc, sps, num_syms, channel_coef=h)
    bits_rx = bits_rx[:len(bits_tx)]
    ber = np.sum(bits_tx != bits_rx) / len(bits_tx)
    return float(ber)


def write_ber_csv(rows: list[dict[str, float | int | str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["channel", "snr_db", "ber", "num_bits"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_ber_plot(rows: list[dict[str, float | int | str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = sorted({str(row["channel"]) for row in rows})

    plt.figure(figsize=(7, 4.5))
    for channel in channels:
        channel_rows = [row for row in rows if row["channel"] == channel]
        snr = [float(row["snr_db"]) for row in channel_rows]
        ber = [max(float(row["ber"]), 1e-6) for row in channel_rows]
        plt.semilogy(snr, ber, marker="o", label=channel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("QPSK BER Smoke Sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="QPSK BER sweep over SNR range.")
    parser.add_argument("--fading", action="store_true", help="Enable Rayleigh fading channel.")
    parser.add_argument("--num-bits", type=int, default=5000)
    parser.add_argument("--snr-min", type=int, default=0)
    parser.add_argument("--snr-max", type=int, default=20)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    sps = 8
    beta = 0.35
    snr_range = range(args.snr_min, args.snr_max + 1, args.snr_step)
    label = "Rayleigh" if args.fading else "AWGN"
    rows: list[dict[str, float | int | str]] = []

    print(f"=== Quick {label} BER sweep (num_bits={args.num_bits}) ===")
    for snr in snr_range:
        ber = simulate_qpsk_link(
            num_bits=args.num_bits, sps=sps, beta=beta, snr_db=snr, fading=args.fading
        )
        rows.append({"channel": label, "snr_db": snr, "ber": ber, "num_bits": args.num_bits})
        print(f"SNR = {snr:2d} dB | BER = {ber:.4e}")

    if args.output_csv is not None:
        output_csv = write_ber_csv(rows, args.output_csv)
        print(f"Wrote BER CSV: {output_csv}")
    if args.output_plot is not None:
        output_plot = write_ber_plot(rows, args.output_plot)
        print(f"Wrote BER plot: {output_plot}")


if __name__ == "__main__":
    main()
