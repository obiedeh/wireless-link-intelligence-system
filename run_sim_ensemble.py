"""Ensemble-averaged BER sweep.

The single-realization smoke (``run_sim.py --fading``) draws one complex-Gaussian
``h`` per call, and perfect-CSI demodulation can make a lucky ``|h|²`` mask
fade-driven BER. This script averages over N independent fading realizations
per SNR point so the resulting curve reflects the average channel rather than
one good or bad draw.

Run::

    python run_sim_ensemble.py --n-realizations 200 --num-bits 10000 \
        --seed 7 --output-csv reports/ber_full_rayleigh.csv \
        --output-plot reports/ber_full_rayleigh.svg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from run_sim import simulate_qpsk_link, write_ber_plot

matplotlib.use("Agg")


def ensemble_sweep(
    n_realizations: int,
    num_bits: int,
    snr_range: range,
    fading: bool,
    seed: int | None,
) -> list[dict[str, float | int | str]]:
    """Run ``n_realizations`` independent simulations per SNR and return the average BER per SNR."""
    if seed is not None:
        np.random.seed(seed)

    label = "Rayleigh" if fading else "AWGN"
    rows: list[dict[str, float | int | str]] = []

    print(
        f"=== Ensemble {label} BER sweep "
        f"(n_realizations={n_realizations}, num_bits={num_bits}) ==="
    )
    for snr in snr_range:
        bers = [
            simulate_qpsk_link(num_bits=num_bits, snr_db=snr, fading=fading)
            for _ in range(n_realizations)
        ]
        avg_ber = sum(bers) / n_realizations
        rows.append(
            {
                "channel": label,
                "snr_db": snr,
                "ber": avg_ber,
                "num_bits": num_bits,
                "n_realizations": n_realizations,
            }
        )
        print(f"SNR = {snr:2d} dB | avg BER over {n_realizations} realizations = {avg_ber:.4e}")

    return rows


def write_ensemble_csv(
    rows: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["channel", "snr_db", "ber", "num_bits", "n_realizations"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble-averaged QPSK BER sweep over SNR range.",
    )
    parser.add_argument("--fading", action="store_true", default=True, help="Enable Rayleigh fading (default true).")
    parser.add_argument("--no-fading", dest="fading", action="store_false", help="Disable fading (AWGN ensemble).")
    parser.add_argument("--n-realizations", type=int, default=200)
    parser.add_argument("--num-bits", type=int, default=10000)
    parser.add_argument("--snr-min", type=int, default=0)
    parser.add_argument("--snr-max", type=int, default=20)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    snr_range = range(args.snr_min, args.snr_max + 1, args.snr_step)

    rows = ensemble_sweep(
        n_realizations=args.n_realizations,
        num_bits=args.num_bits,
        snr_range=snr_range,
        fading=args.fading,
        seed=args.seed,
    )

    if args.output_csv is not None:
        output_csv = write_ensemble_csv(rows, args.output_csv)
        print(f"Wrote ensemble CSV: {output_csv}")
    if args.output_plot is not None:
        output_plot = write_ber_plot(rows, args.output_plot)
        print(f"Wrote ensemble plot: {output_plot}")


if __name__ == "__main__":
    main()
