"""OFDM + adaptive QAM BER sweep.

Generates BER vs SNR curves for CP-OFDM under AWGN for each supported
modulation order (QPSK / 16-QAM / 64-QAM / 256-QAM). Output: a single CSV
with all four curves and a single SVG with all four overlaid for the
adaptive-modulation decision conversation.

Run::

    python run_sim_ofdm.py --num-bits 200000 --seed 7 \\
        --output-csv reports/ber_full_ofdm_awgn.csv \\
        --output-plot reports/ber_full_ofdm_awgn.svg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from qpsk_link.channel import add_awgn
from qpsk_link.ofdm import (
    SUPPORTED_MOD_ORDERS,
    decode_ofdm_frame,
    make_ofdm_frame,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Pretty names for plot legends — matches what a telecom engineer would write.
_PRETTY = {
    4: "OFDM-QPSK",
    16: "OFDM-16QAM",
    64: "OFDM-64QAM",
    256: "OFDM-256QAM",
}


def simulate_ofdm_ber(
    num_bits: int,
    mod_order: int,
    snr_db: float,
    n_subcarriers: int = 64,
    cp_length: int = 16,
    seed: int | None = None,
) -> float:
    """End-to-end OFDM AWGN BER for a given modulation order and SNR."""
    rng = np.random.default_rng(seed)
    bps = int(np.log2(mod_order))
    # Round bits up to fill an integer number of OFDM symbols cleanly.
    bits_per_ofdm_symbol = n_subcarriers * bps
    n_ofdm_symbols = max(1, num_bits // bits_per_ofdm_symbol)
    n_bits = n_ofdm_symbols * bits_per_ofdm_symbol
    bits = rng.integers(0, 2, n_bits).astype(np.int64)

    tx_signal, info = make_ofdm_frame(bits, n_subcarriers, cp_length, mod_order)
    rx_signal = add_awgn(tx_signal, snr_db=snr_db)
    bits_hat = decode_ofdm_frame(rx_signal, info)
    return float(np.mean(bits != bits_hat))


def main() -> None:
    parser = argparse.ArgumentParser(description="OFDM + adaptive QAM BER sweep over SNR.")
    parser.add_argument("--num-bits", type=int, default=20000, help="Target bit count per (SNR, modulation) point.")
    parser.add_argument("--snr-min", type=int, default=0)
    parser.add_argument("--snr-max", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--n-subcarriers", type=int, default=64)
    parser.add_argument("--cp-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    rows: list[dict[str, float | int | str]] = []

    print(f"=== OFDM AWGN BER sweep ({args.n_subcarriers} subcarriers, CP={args.cp_length}) ===")
    for mod_order in SUPPORTED_MOD_ORDERS:
        print(f"--- {_PRETTY[mod_order]} ---")
        for i, snr in enumerate(snr_range):
            # Stride the seed by (mod_order, snr_index) so each point is
            # deterministic but distinct across the sweep.
            point_seed = args.seed * 1000 + mod_order * 100 + i
            ber = simulate_ofdm_ber(
                num_bits=args.num_bits,
                mod_order=mod_order,
                snr_db=float(snr),
                n_subcarriers=args.n_subcarriers,
                cp_length=args.cp_length,
                seed=point_seed,
            )
            rows.append(
                {
                    "modulation": _PRETTY[mod_order],
                    "mod_order": mod_order,
                    "bits_per_symbol": int(np.log2(mod_order)),
                    "snr_db": snr,
                    "ber": ber,
                    "num_bits": args.num_bits,
                }
            )
            print(f"  SNR = {snr:3d} dB | BER = {ber:.4e}")

    # ---- CSV ----
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["modulation", "mod_order", "bits_per_symbol", "snr_db", "ber", "num_bits"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote BER CSV: {args.output_csv}")

    # ---- Plot ----
    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.5, 5))
        markers = {4: "o", 16: "s", 64: "^", 256: "D"}
        for mod_order in SUPPORTED_MOD_ORDERS:
            mod_rows = [r for r in rows if r["mod_order"] == mod_order]
            snr = [float(r["snr_db"]) for r in mod_rows]
            ber = [max(float(r["ber"]), 1e-6) for r in mod_rows]
            plt.semilogy(
                snr, ber,
                marker=markers[mod_order],
                linewidth=1.4,
                label=_PRETTY[mod_order],
            )
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate (BER)")
        plt.title("CP-OFDM Adaptive QAM — AWGN BER vs SNR")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(args.output_plot)
        plt.close()
        print(f"Wrote BER plot: {args.output_plot}")


if __name__ == "__main__":
    main()
