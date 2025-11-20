# QPSK-Wireless-Link-Simulator

A modular Python project that simulates a baseband **QPSK digital communication system** including:

* Bit generation
* QPSK modulation (Gray-coded)
* Root Raised Cosine (RRC) pulse shaping
* AWGN and Rayleigh fading channels
* Matched filtering + symbol timing
* Hard-decision demodulation
* BER performance evaluation
* Constellation visualization

This repository is built to be **clean, modular, extendable**, and easy to integrate with additional wireless communication blocks.

---

## Project Structure

```
QPSK-Wireless-Link-Simulator/
├── main.py                  # Entry point: runs BER sims, plots
├── qpsk_modem.py            # Modulation/demodulation/RRC filter
├── channel.py               # AWGN + Rayleigh fading models
├── plots.py                 # Constellation + BER plotting helpers
├── requirements.txt         # Python dependencies
└── README.md                # Documentation (this file)
```

---

## Features

✔ Fully functional QPSK modem (mod/demod)
✔ RRC pulse shaping + matched filter
✔ AWGN channel
✔ Rayleigh fading channel
✔ Symbol recovery and equalization
✔ BER vs SNR simulation
✔ Constellation plotting
✔ Easily extendable to OFDM, MIMO, CFO, and coding

---

## Theory Summary

### QPSK Symbol Mapping (Gray Code)

Two bits map to one complex symbol:

| Bits | Symbol                      |
| ---- | --------------------------- |
| 00   | ( \frac{1 + j}{\sqrt{2}} )  |
| 01   | ( \frac{-1 + j}{\sqrt{2}} ) |
| 11   | ( \frac{-1 - j}{\sqrt{2}} ) |
| 10   | ( \frac{1 - j}{\sqrt{2}} )  |

---

### Root Raised Cosine Filter

The RRC filter is used for:

* Bandwidth control
* Inter-symbol interference reduction
* Optimal matched filtering

Filter parameters:

* Roll-off factor ( \beta = 0.35 )
* Samples per symbol (SPS) = 8
* Tap count = ( 8 \times \text{SPS} + 1 )

---

### Channel Models

#### 1. Additive White Gaussian Noise (AWGN)

Noise added based on target SNR:
[
\text{SNR} = 10 \log_{10} \left( \frac{P_s}{P_n} \right)
]

#### 2. Rayleigh Fading

Flat-fading coefficient:
[
h = \frac{X + jY}{\sqrt{2}}, \quad X, Y \sim \mathcal{N}(0,1)
]

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/obiedeh/QPSK-Wireless-Link-Simulator.git
cd QPSK-Wireless-Link-Simulator
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Simulator

### Run the BER simulation + plots

```
python main.py
```

This will:

* Generate BER vs SNR curves (AWGN & Rayleigh)
* Plot constellations
* Print performance summary

---

## Example Outputs

### BER Curve

* AWGN curve drops ~10⁻³ at 10 dB
* Rayleigh curve is much worse (fading penalty)

### Constellation

* Perfect 4-point square at high SNR
* Scatter expands at low SNR

---

## Extending the Project

Here are suggested enhancements:

### 🔧 PHY Layer Enhancements

* Carrier frequency offset (CFO)
* Symbol timing synchronization
* Automatic Gain Control (AGC)
* Phase-locked loop (PLL) for phase correction

### RF/Wireless Extensions

* Rician fading
* Frequency-selective multipath
* OFDM modulation/demodulation
* MIMO (2×2 Alamouti)

### AI/ML Extensions

* ML-based channel prediction
* Neural equalization
* Learned demodulators

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to modify.

Recommended additions:

* unit tests
* benchmarking scripts
* Jupyter demos

---

## License

This project is open-source under the MIT License.

---

## GitHub CLI Workflow (Update Your Repository)

Below are the essential **Git + GitHub CLI commands** you will use to update this project.

---

### 1. Pull Latest Changes (Always Do This First)

```
git pull --rebase origin main
```

---

### 2. After Editing Code — Stage Changes

```
git add .
```

Or stage a single file:

```
git add main.py
```

---

### 3. Commit Your Changes

```
git commit -m "Describe your update here"
```

---

### 4. Push Changes to GitHub

```
git push origin main
```

---

### 5. If You Created New Code Locally and Repo Was Not Connected

Add GitHub remote:

```
git remote add origin git@github.com:obiedeh/QPSK-Wireless-Link-Simulator.git
```

Rename branch to main:

```
git branch -M main
```

Push initial code:

```
git push -u origin main
```

---

### 6. Fix Common Error: Uncommitted Work Stopping Pull

```
git add .
git commit -m "Save work"
git pull --rebase origin main
```

---

## Contact

**Author:** Obinna Edeh
**GitHub:** [https://github.com/obiedeh](https://github.com/obiedeh)

For enhancements or requests, feel free to open an issue in the repository.

---

### If you find this useful, star the repo on GitHub!
