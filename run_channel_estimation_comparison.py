"""Compare LS / MMSE / neural pilot-based channel estimators on TDL-C.

Trains a small MLP on synthetic TDL-C realizations, then sweeps SNR and
evaluates all three estimators on the same channels: channel-estimate MSE
and resulting BLER under per-subcarrier zero-forcing equalization.

Run::

    python run_channel_estimation_comparison.py --seed 7 \\
        --output-csv reports/channel_estimation_comparison.csv \\
        --output-plot reports/channel_estimation_comparison.svg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from qpsk_link.channel import add_awgn
from qpsk_link.channel_estimation import (
    DEFAULT_PILOT_STRIDE,
    DEFAULT_PILOT_VALUE,
    build_piloted_frame,
    comb_pilot_indices,
    data_subcarrier_indices,
    ls_estimate,
    mmse_estimate,
    neural_estimator,
)
from qpsk_link.ofdm import (
    ofdm_demodulate,
    ofdm_modulate,
    qam_demodulate,
    qam_modulate,
)
from qpsk_link.tdl_channel import (
    TDL_C,
    apply_tdl_channel,
    cir_to_frequency_response,
    sample_tdl_realization,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Frame TX/RX helper
# ---------------------------------------------------------------------------


def transmit_frame(
    bits: np.ndarray,
    n_subcarriers: int,
    n_ofdm_symbols: int,
    cp_length: int,
    mod_order: int,
    h: np.ndarray,
    snr_db: float,
):
    """Build piloted frame, send through CIR + AWGN, return (rx_freq_grid, frame, true_H)."""
    data_symbols = qam_modulate(bits, mod_order)
    frame = build_piloted_frame(
        data_symbols=data_symbols,
        n_subcarriers=n_subcarriers,
        n_ofdm_symbols=n_ofdm_symbols,
        cp_length=cp_length,
        mod_order=mod_order,
        pilot_stride=DEFAULT_PILOT_STRIDE,
        pilot_value=DEFAULT_PILOT_VALUE,
    )
    tx_signal = ofdm_modulate(frame.tx_symbols.ravel(), n_subcarriers, cp_length)
    rx_signal, _ = apply_tdl_channel(tx_signal, h)
    rx_signal = add_awgn(rx_signal, snr_db=snr_db)
    rx_signal = rx_signal[: n_ofdm_symbols * (n_subcarriers + cp_length)]
    rx_freq = ofdm_demodulate(rx_signal, n_subcarriers, cp_length).reshape(
        n_ofdm_symbols, n_subcarriers
    )
    true_H = cir_to_frequency_response(h, n_subcarriers)
    return rx_freq, frame, true_H


def equalize_and_decode(
    rx_freq_grid: np.ndarray,
    frame,
    H_hat: np.ndarray,
    n_bits: int,
) -> np.ndarray:
    """ZF equalize the data subcarriers using H_hat; demap to bits."""
    data_idx = frame.data_indices
    H_data = H_hat[data_idx]
    data_rx = rx_freq_grid[:, data_idx] / H_data[None, :]
    bits_hat = qam_demodulate(data_rx.ravel(), frame.mod_order)
    return bits_hat[:n_bits]


# ---------------------------------------------------------------------------
# Training the neural estimator
# ---------------------------------------------------------------------------


def train_neural_estimator(
    n_train_examples: int,
    n_subcarriers: int,
    n_ofdm_symbols: int,
    cp_length: int,
    delay_spread_samples: float,
    snr_range_train: tuple[float, float],
    epochs: int,
    seed: int,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Train the neural channel estimator on synthetic TDL-C examples."""
    rng = np.random.default_rng(seed)
    pilot_idx = comb_pilot_indices(n_subcarriers, DEFAULT_PILOT_STRIDE)
    data_idx = data_subcarrier_indices(n_subcarriers, pilot_idx)
    bps = 2  # QPSK for training (cheapest; estimator is mod-agnostic)
    n_data_bits = bps * n_ofdm_symbols * data_idx.size

    estimator = neural_estimator(pilot_idx.size, n_subcarriers, hidden=128)
    torch = estimator.torch

    # Generate training set.
    X_list, Y_list, snr_list = [], [], []
    for _ in range(n_train_examples):
        snr_db = float(rng.uniform(*snr_range_train))
        bits = rng.integers(0, 2, n_data_bits).astype(np.int64)
        h = sample_tdl_realization(TDL_C, delay_spread_samples, rng)
        if h.size > cp_length:
            h = h[: cp_length + 1]
        rx_freq, frame, true_H = transmit_frame(
            bits=bits,
            n_subcarriers=n_subcarriers,
            n_ofdm_symbols=n_ofdm_symbols,
            cp_length=cp_length,
            mod_order=4,
            h=h,
            snr_db=snr_db,
        )
        rx_pilots = rx_freq[:, frame.pilot_indices]
        h_pilot = np.mean(rx_pilots, axis=0)
        x = np.concatenate([h_pilot.real, h_pilot.imag, [snr_db]])
        y = np.concatenate([true_H.real, true_H.imag])
        X_list.append(x)
        Y_list.append(y)
        snr_list.append(snr_db)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y_list), dtype=torch.float32)

    model = estimator.model
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    n = X.shape[0]
    for epoch in range(epochs):
        # Shuffle deterministically per epoch.
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            out = model(X[idx])
            loss = loss_fn(out, Y[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()) * idx.size(0)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch+1:3d}/{epochs} | train MSE = {epoch_loss / n:.6f}")
    return estimator


# ---------------------------------------------------------------------------
# Comparison sweep
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LS / MMSE / neural channel-estimator comparison on TDL-C.",
    )
    parser.add_argument("--n-subcarriers", type=int, default=64)
    parser.add_argument("--cp-length", type=int, default=16)
    parser.add_argument("--n-ofdm-symbols", type=int, default=4)
    parser.add_argument("--delay-spread-samples", type=float, default=4.0)
    parser.add_argument("--n-realizations", type=int, default=80)
    parser.add_argument("--snr-min", type=int, default=0)
    parser.add_argument("--snr-max", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=3)
    parser.add_argument("--mod-order", type=int, default=4, choices=[4, 16, 64, 256])
    parser.add_argument("--n-train-examples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--snr-train-min", type=float, default=-5.0)
    parser.add_argument("--snr-train-max", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    print("=== Training neural channel estimator on synthetic TDL-C ===")
    estimator = train_neural_estimator(
        n_train_examples=args.n_train_examples,
        n_subcarriers=args.n_subcarriers,
        n_ofdm_symbols=args.n_ofdm_symbols,
        cp_length=args.cp_length,
        delay_spread_samples=args.delay_spread_samples,
        snr_range_train=(args.snr_train_min, args.snr_train_max),
        epochs=args.epochs,
        seed=args.seed,
    )

    print("=== Sweeping SNR with all three estimators ===")
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    bps = int(np.log2(args.mod_order))
    pilot_idx = comb_pilot_indices(args.n_subcarriers, DEFAULT_PILOT_STRIDE)
    data_idx = data_subcarrier_indices(args.n_subcarriers, pilot_idx)
    n_data_bits = bps * args.n_ofdm_symbols * data_idx.size

    estimators = ["LS", "MMSE", "Neural"]
    rows: list[dict[str, float | int | str]] = []

    for snr_db in snr_range:
        # One RNG per (snr, run) so that all three estimators see the *same*
        # channel realisations and noise — apples-to-apples comparison.
        mse_acc = {name: 0.0 for name in estimators}
        bler_acc = {name: 0.0 for name in estimators}
        n_blocks = args.n_realizations
        for run in range(n_blocks):
            seed = args.seed * 10_000 + snr_db * 100 + run
            rng = np.random.default_rng(seed)
            bits = rng.integers(0, 2, n_data_bits).astype(np.int64)
            h = sample_tdl_realization(TDL_C, args.delay_spread_samples, rng)
            if h.size > args.cp_length:
                h = h[: args.cp_length + 1]
            rx_freq, frame, true_H = transmit_frame(
                bits=bits,
                n_subcarriers=args.n_subcarriers,
                n_ofdm_symbols=args.n_ofdm_symbols,
                cp_length=args.cp_length,
                mod_order=args.mod_order,
                h=h,
                snr_db=float(snr_db),
            )
            rx_pilots = rx_freq[:, frame.pilot_indices]

            # LS
            H_ls = ls_estimate(rx_pilots, frame)
            mse_acc["LS"] += float(np.mean(np.abs(H_ls - true_H) ** 2))
            bits_ls = equalize_and_decode(rx_freq, frame, H_ls, n_data_bits)
            bler_acc["LS"] += float(np.any(bits != bits_ls))

            # MMSE
            H_mmse = mmse_estimate(
                rx_pilots, frame, float(snr_db), args.delay_spread_samples
            )
            mse_acc["MMSE"] += float(np.mean(np.abs(H_mmse - true_H) ** 2))
            bits_mmse = equalize_and_decode(rx_freq, frame, H_mmse, n_data_bits)
            bler_acc["MMSE"] += float(np.any(bits != bits_mmse))

            # Neural
            H_nn = estimator.estimate(rx_pilots, float(snr_db))
            mse_acc["Neural"] += float(np.mean(np.abs(H_nn - true_H) ** 2))
            bits_nn = equalize_and_decode(rx_freq, frame, H_nn, n_data_bits)
            bler_acc["Neural"] += float(np.any(bits != bits_nn))

        print(f"--- SNR = {snr_db} dB ---")
        for name in estimators:
            mse = mse_acc[name] / n_blocks
            bler = bler_acc[name] / n_blocks
            rows.append(
                {
                    "estimator": name,
                    "snr_db": snr_db,
                    "mse_h": mse,
                    "bler": bler,
                    "n_realizations": n_blocks,
                }
            )
            print(f"  {name:6s} | MSE(H) = {mse:.4e} | BLER = {bler:.4f}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["estimator", "snr_db", "mse_h", "bler", "n_realizations"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")

    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax_mse, ax_bler) = plt.subplots(1, 2, figsize=(13, 4.8))
        markers = {"LS": "o", "MMSE": "s", "Neural": "^"}
        colors = {"LS": "#a33d1f", "MMSE": "#294c7a", "Neural": "#136f63"}
        for name in estimators:
            est_rows = [r for r in rows if r["estimator"] == name]
            snr = [float(r["snr_db"]) for r in est_rows]
            mse = [max(float(r["mse_h"]), 1e-6) for r in est_rows]
            bler = [max(float(r["bler"]), 1e-3) for r in est_rows]
            ax_mse.semilogy(snr, mse, marker=markers[name], color=colors[name], linewidth=1.5, label=name)
            ax_bler.semilogy(snr, bler, marker=markers[name], color=colors[name], linewidth=1.5, label=name)
        for ax, title, ylabel in [
            (ax_mse, "Channel-estimate MSE", "MSE(Ĥ, H)"),
            (ax_bler, "Frame error rate", "BLER"),
        ]:
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best")
        plt.suptitle("Pilot-based channel estimation on TDL-C — LS vs MMSE vs Neural")
        plt.tight_layout()
        plt.savefig(args.output_plot)
        plt.close()
        print(f"Wrote plot: {args.output_plot}")


if __name__ == "__main__":
    main()
