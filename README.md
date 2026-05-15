# QPSK Wireless Link Simulator

Wireless link simulation foundation for AI-RAN, edge communications, and physical-layer experimentation.

This repository implements a modular Python simulation of a baseband QPSK digital communication link. It is designed as a clean wireless systems foundation that can later support AI-native RAN experiments, learned receivers, channel-aware inference, and edge communication studies.

The goal is not to present QPSK as a novel technique.

The goal is to build a reproducible signal-processing testbed for understanding wireless link behavior under noise, fading, and future AI-assisted receiver workflows.

---

## Why This Matters

AI-RAN and edge AI systems depend on reliable wireless links, predictable channel behavior, and accurate interpretation of physical-layer performance.

Before adding AI to the RAN stack, the underlying communication system must be measurable:

- modulation behavior
- channel impairments
- fading effects
- BER vs SNR performance
- constellation distortion
- receiver recovery behavior

This repository provides that foundational wireless simulation layer.

---

## Current Capabilities

- random bit generation
- Gray-coded QPSK modulation
- Root Raised Cosine pulse shaping
- AWGN channel modeling
- Rayleigh fading channel modeling
- matched filtering
- symbol timing path
- hard-decision demodulation
- BER vs SNR simulation
- constellation visualization

---

## Architecture

```text
Random Bits
    |
    v
QPSK Mapper
    |
    v
Pulse Shaping
(Root Raised Cosine)
    |
    v
Wireless Channel
(AWGN / Rayleigh)
    |
    v
Matched Filter + Recovery
    |
    v
QPSK Demapper
    |
    v
BER / Constellation Analysis
```

---

## Repository Structure

```text
.
├── main.py              # Entry point for BER simulation and plots
├── qpsk_modem.py        # Modulation, demodulation, and RRC filtering
├── channel.py           # AWGN and Rayleigh fading models
├── plots.py             # BER and constellation visualization helpers
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/obiedeh/qpsk-wireless-link-simulator.git
cd qpsk-wireless-link-simulator
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Windows PowerShell:

```powershell
git clone https://github.com/obiedeh/qpsk-wireless-link-simulator.git
cd qpsk-wireless-link-simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

---

## Example Outputs

Expected simulation outputs include:

- BER vs SNR curve for AWGN and Rayleigh channels
- constellation plots under different channel conditions
- printed simulation summary

Typical behavior:

- AWGN BER improves sharply as SNR increases
- Rayleigh fading shows a significant performance penalty
- low-SNR constellations spread and overlap
- high-SNR constellations converge toward the ideal four-symbol pattern

---

## Engineering Extensions

### Wireless Systems

- carrier frequency offset
- timing synchronization
- AGC
- phase correction
- Rician fading
- frequency-selective multipath
- OFDM
- MIMO

### AI-RAN Experiments

- ML-based channel estimation
- neural equalization
- learned demodulation
- channel-aware inference experiments
- synthetic PHY-layer telemetry generation

### Edge AI Integration

- lightweight inference benchmarks
- signal-processing acceleration experiments
- Jetson-based simulation or receiver workflows
- telemetry export for AI-RAN analytics

---

## Positioning

This repository supports a broader engineering focus around:

- AI-RAN
- wireless systems
- edge AI infrastructure
- physical-layer intelligence
- signal-processing foundations
- future 6G and AI-native network experimentation

---

## License

MIT License.
