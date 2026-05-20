"""OFDM-QPSK BLER vs SNR sweep across 3GPP TR 38.901 TDL channel profiles.

Generates per-profile (TDL-A, TDL-B, TDL-C) BLER and BER curves under
ensemble-averaged Rayleigh fading with multi-tap delay spread. Uses
*perfect channel-state-information* equalization at the receiver — this is
the canonical "best case" reference curve. The Upgrade #3 work on
pilot-based channel estimation will show how much that ideal curve degrades
when CSI is estimated, not known.

Run::

    python run_sim_tdl.py --num-bits 10000 --n-realizations 100 --seed 7 \\
        --output-csv reports/bler_full_tdl_ofdm.csv \\
        --output-plot reports/bler_full_tdl_ofdm.svg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from qpsk_link.channel import add_awgn
from qpsk_link.ofdm import (
    make_ofdm_frame,
    ofdm_demodulate,
    qam_demodulate,
)
from qpsk_link.tdl_channel import (
    ALL_NLOS_PROFILES,
    TDLProfile,
    apply_tdl_channel,
    cir_to_frequency_response,
    sample_tdl_realization,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _equalize_perfect_csi(
    rx_signal: np.ndarray,
    info: dict,
    H: np.ndarray,
) -> np.ndarray:
    """Per-subcarrier zero-forcing equalization with perfect CSI.

    Drops the convolution tail beyond the OFDM frame length, demodulates,
    divides each subcarrier by its known channel gain, returns decoded bits.
    """
    n_subcarriers = info["n_subcarriers"]
    cp_length = info["cp_length"]
    mod_order = info["mod_order"]
    n_bits = info["n_bits"]
    n_ofdm_symbols = info["n_ofdm_symbols"]

    frame_length = n_ofdm_symbols * (n_subcarriers + cp_length)
    rx_signal = rx_signal[:frame_length]

    rx_freq = ofdm_demodulate(rx_signal, n_subcarriers, cp_length).reshape(
        n_ofdm_symbols, n_subcarriers
    )
    # Per-symbol per-subcarrier zero-forcing.
    equalized = rx_freq / H[None, :]
    bits_hat = qam_demodulate(equalized.ravel(), mod_order)
    return bits_hat[:n_bits]


def simulate_tdl_frame(
    num_bits: int,
    profile: TDLProfile,
    snr_db: float,
    n_subcarriers: int,
    cp_length: int,
    mod_order: int,
    delay_spread_samples: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    """One realisation: draw channel, transmit a frame, return (BER, frame_in_error)."""
    bps = int(np.log2(mod_order))
    bits_per_ofdm_symbol = n_subcarriers * bps
    n_ofdm_symbols = max(1, num_bits // bits_per_ofdm_symbol)
    n_bits = n_ofdm_symbols * bits_per_ofdm_symbol
    bits = rng.integers(0, 2, n_bits).astype(np.int64)

    tx_signal, info = make_ofdm_frame(bits, n_subcarriers, cp_length, mod_order)
    h = sample_tdl_realization(profile, delay_spread_samples, rng)
    if h.size > cp_length:
        # Should not happen in normal use; truncate to avoid inter-symbol interference.
        h = h[: cp_length + 1]
    rx_signal, _ = apply_tdl_channel(tx_signal, h)
    rx_signal_noisy = add_awgn(rx_signal, snr_db=snr_db)

    H = cir_to_frequency_response(h, n_subcarriers)
    bits_hat = _equalize_perfect_csi(rx_signal_noisy, info, H)
    ber = float(np.mean(bits != bits_hat))
    frame_in_error = bool(np.any(bits != bits_hat))
    return ber, frame_in_error


def ensemble_sweep(
    profile: TDLProfile,
    snr_range: list[int],
    num_bits: int,
    n_realizations: int,
    n_subcarriers: int,
    cp_length: int,
    mod_order: int,
    delay_spread_samples: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Ensemble-averaged BER + BLER for one profile across the SNR sweep."""
    rows: list[dict[str, float | int | str]] = []
    print(f"--- {profile.name} (delay spread = {delay_spread_samples} samples) ---")
    for snr in snr_range:
        bers = []
        errors = 0
        # Stride seed so each (profile, snr) point is deterministic but distinct.
        rng = np.random.default_rng(seed + abs(hash((profile.name, snr))) % (1 << 31))
        for _ in range(n_realizations):
            ber, in_error = simulate_tdl_frame(
                num_bits=num_bits,
                profile=profile,
                snr_db=float(snr),
                n_subcarriers=n_subcarriers,
                cp_length=cp_length,
                mod_order=mod_order,
                delay_spread_samples=delay_spread_samples,
                rng=rng,
            )
            bers.append(ber)
            errors += int(in_error)
        avg_ber = float(np.mean(bers))
        bler = errors / n_realizations
        rows.append(
            {
                "profile": profile.name,
                "snr_db": snr,
                "ber": avg_ber,
                "bler": bler,
                "n_realizations": n_realizations,
                "num_bits": num_bits,
                "delay_spread_samples": delay_spread_samples,
            }
        )
        print(f"  SNR = {snr:3d} dB | BER = {avg_ber:.4e} | BLER = {bler:.4f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OFDM-QPSK BLER vs SNR sweep across 3GPP TR 38.901 TDL profiles "
                    "(perfect-CSI receiver).",
    )
    parser.add_argument("--num-bits", type=int, default=4096)
    parser.add_argument("--n-realizations", type=int, default=80)
    parser.add_argument("--snr-min", type=int, default=0)
    parser.add_argument("--snr-max", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=3)
    parser.add_argument("--n-subcarriers", type=int, default=64)
    parser.add_argument("--cp-length", type=int, default=16)
    parser.add_argument("--mod-order", type=int, default=4, choices=[4, 16, 64, 256])
    parser.add_argument("--delay-spread-samples", type=float, default=4.0,
                        help="Delay spread expressed in simulator samples. "
                             "Must result in CIR length <= cp_length.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    all_rows: list[dict[str, float | int | str]] = []

    print(
        f"=== TDL ensemble BLER sweep "
        f"(N={args.n_realizations} realisations, mod_order={args.mod_order}) ==="
    )
    for profile in ALL_NLOS_PROFILES:
        all_rows.extend(
            ensemble_sweep(
                profile=profile,
                snr_range=snr_range,
                num_bits=args.num_bits,
                n_realizations=args.n_realizations,
                n_subcarriers=args.n_subcarriers,
                cp_length=args.cp_length,
                mod_order=args.mod_order,
                delay_spread_samples=args.delay_spread_samples,
                seed=args.seed,
            )
        )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "profile", "snr_db", "ber", "bler", "n_realizations",
                    "num_bits", "delay_spread_samples",
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote BLER CSV: {args.output_csv}")

    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        markers = {"TDL-A": "o", "TDL-B": "s", "TDL-C": "^"}
        plt.figure(figsize=(7.5, 5))
        for profile in ALL_NLOS_PROFILES:
            rows = [r for r in all_rows if r["profile"] == profile.name]
            snr = [float(r["snr_db"]) for r in rows]
            bler = [max(float(r["bler"]), 1e-4) for r in rows]
            plt.semilogy(
                snr, bler,
                marker=markers[profile.name],
                linewidth=1.5,
                label=profile.name,
            )
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xlabel("SNR (dB)")
        plt.ylabel("BLER (frame error rate)")
        plt.title("OFDM-QPSK BLER vs SNR — 3GPP TR 38.901 TDL profiles, perfect-CSI RX")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(args.output_plot)
        plt.close()
        print(f"Wrote BLER plot: {args.output_plot}")


if __name__ == "__main__":
    main()
