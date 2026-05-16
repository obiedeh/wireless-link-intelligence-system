"""Synthetic dataset generation for AI-assisted QPSK link estimation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from channel import apply_channel
from qpsk_modem import qpsk_demodulate, qpsk_modulate, recover_qpsk_symbols

from .features import FEATURE_COLUMNS, constellation_statistics, link_quality_score


LABEL_COLUMNS = [
    "snr_db",
    "ber",
    "channel_type",
    "channel_is_rayleigh",
    "link_quality_score",
    "fading_abs",
    "fading_phase",
]

CSV_COLUMNS = ["sample_id", *LABEL_COLUMNS, *FEATURE_COLUMNS]


def simulate_link_sample(sample_id: int,
                         num_bits: int = 4000,
                         sps: int = 8,
                         beta: float = 0.35,
                         snr_db: float | None = None,
                         channel_type: str | None = None) -> dict[str, float | str | int]:
    """Run one classical QPSK simulation and return labels plus features."""
    if snr_db is None:
        snr_db = float(np.random.uniform(-4.0, 18.0))
    if channel_type is None:
        channel_type = str(np.random.choice(["awgn", "rayleigh"]))
    if channel_type not in {"awgn", "rayleigh"}:
        raise ValueError("channel_type must be 'awgn' or 'rayleigh'")

    bits_tx = np.random.randint(0, 2, num_bits)
    tx_signal, syms_tx, h_rrc = qpsk_modulate(bits_tx, sps=sps, beta=beta)
    rx_signal, channel_coef = apply_channel(
        tx_signal,
        snr_db=snr_db,
        fading=(channel_type == "rayleigh"),
    )
    bits_rx = qpsk_demodulate(
        rx_signal,
        h_rrc,
        sps,
        len(syms_tx),
        channel_coef=channel_coef,
    )[:len(bits_tx)]
    ber = float(np.mean(bits_tx != bits_rx))
    rx_symbols = recover_qpsk_symbols(
        rx_signal,
        h_rrc,
        sps,
        len(syms_tx),
        channel_coef=channel_coef,
    )

    row: dict[str, float | str | int] = {
        "sample_id": sample_id,
        "snr_db": float(snr_db),
        "ber": ber,
        "channel_type": channel_type,
        "channel_is_rayleigh": int(channel_type == "rayleigh"),
        "link_quality_score": link_quality_score(snr_db, ber, channel_type),
    }
    row.update(constellation_statistics(syms_tx, rx_symbols, channel_coef))
    return row


def generate_dataset(output_path: str | Path = "data/link_conditions.csv",
                     samples: int = 500,
                     num_bits: int = 4000,
                     seed: int = 7) -> Path:
    """Generate a CSV dataset of synthetic QPSK link conditions."""
    np.random.seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for sample_id in range(samples):
            writer.writerow(simulate_link_sample(sample_id, num_bits=num_bits))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic QPSK link-condition data.")
    parser.add_argument("--output", default="data/link_conditions.csv")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--num-bits", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_path = generate_dataset(
        output_path=args.output,
        samples=args.samples,
        num_bits=args.num_bits,
        seed=args.seed,
    )
    print(f"Wrote synthetic link dataset: {output_path}")


if __name__ == "__main__":
    main()
