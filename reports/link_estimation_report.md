# AI-Assisted Wireless Link Estimation Report

This report summarizes synthetic experiments around a classical QPSK baseline. The project estimates link conditions from simulated constellation statistics; it is not a full AI-RAN base station and not a production telecom receiver.

## Dataset

- Source CSV: `data/link_conditions.csv`
- Samples: 500
- Held-out test samples: 125
- Labels: SNR, measured BER, channel type, and synthetic link-quality score
- Features: constellation power, I/Q moments, EVM, quadrant balance, and fading coefficient summary

## Model Results

- SNR estimation MAE: 2.356 dB
- SNR estimation R2: 0.687
- SNR estimation MAE by channel: AWGN 1.332 dB, Rayleigh 3.300 dB
- BER prediction MAE: 0.003591
- BER prediction R2: 0.916
- AWGN vs Rayleigh classification accuracy: 0.552 (majority-class rate 0.520)
- Link-quality scoring MAE: 5.832

## Classical BER vs Predicted BER

| Sample | Channel | Measured BER | Predicted BER | SNR dB | Predicted SNR dB | Predicted Channel |
|---:|---|---:|---:|---:|---:|---|
| 247 | rayleigh | 0.000000 | 0.000075 | 14.54 | 6.45 | awgn |
| 239 | awgn | 0.000000 | 0.000000 | 4.26 | 4.85 | rayleigh |
| 70 | rayleigh | 0.039250 | 0.045220 | 2.77 | 0.55 | rayleigh |
| 136 | rayleigh | 0.000000 | 0.000984 | 14.95 | 3.06 | rayleigh |
| 387 | awgn | 0.000000 | 0.000000 | 14.70 | 13.41 | rayleigh |
| 348 | awgn | 0.000000 | 0.000000 | 9.58 | 7.61 | rayleigh |
| 83 | awgn | 0.000000 | 0.000000 | 12.29 | 11.83 | awgn |
| 234 | rayleigh | 0.000000 | 0.000000 | 15.62 | 13.37 | rayleigh |
| 456 | rayleigh | 0.000000 | 0.000000 | 12.45 | 14.43 | awgn |
| 315 | rayleigh | 0.000000 | 0.000000 | 11.77 | 10.79 | rayleigh |
| 357 | awgn | 0.000000 | 0.000000 | 15.27 | 14.72 | awgn |
| 440 | rayleigh | 0.000000 | 0.000000 | 13.81 | 6.60 | awgn |

## Interpretation

- The measured BER remains the classical simulator baseline.
- ML predictions are estimates from synthetic features and should be validated against any real RF capture before use.
- AWGN/Rayleigh classification is a controlled two-class experiment, not generalized channel recognition.
- The channel classifier is weak in this run, close to the majority-class rate. Treat that as a useful negative result: the current feature set carries little channel-type signal.
- SNR error is larger on Rayleigh links than on AWGN links: fading scales the constellation, and symbol-averaged statistics cannot fully separate a deep fade from low SNR.
- SNR error is reported on held-out synthetic samples and should not be treated as field performance.
