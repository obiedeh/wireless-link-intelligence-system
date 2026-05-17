# AI-Assisted Wireless Link Estimation Report

This report summarizes synthetic experiments around a classical QPSK baseline. The project estimates link conditions from simulated constellation statistics; it is not a full AI-RAN base station and not a production telecom receiver.

## Dataset

- Source CSV: `data/link_conditions.csv`
- Samples: 500
- Held-out test samples: 125
- Labels: SNR, measured BER, channel type, and synthetic link-quality score
- Features: constellation power, I/Q moments, EVM, quadrant balance, and fading coefficient summary

## Model Results

- SNR estimation MAE: 0.118 dB
- SNR estimation R2: 0.999
- BER prediction MAE: 0.000453
- BER prediction R2: 0.968
- AWGN vs Rayleigh classification accuracy: 0.472
- Link-quality scoring MAE: 4.089

## Classical BER vs Predicted BER

| Sample | Channel | Measured BER | Predicted BER | SNR dB | Predicted SNR dB | Predicted Channel |
|---:|---|---:|---:|---:|---:|---|
| 247 | rayleigh | 0.000000 | 0.000000 | 6.14 | 6.13 | awgn |
| 239 | awgn | 0.000000 | 0.000000 | 5.87 | 5.99 | awgn |
| 70 | rayleigh | 0.000000 | 0.000000 | 17.60 | 17.83 | rayleigh |
| 136 | awgn | 0.000000 | 0.000000 | 14.38 | 14.56 | awgn |
| 387 | rayleigh | 0.000000 | 0.000000 | 13.94 | 14.10 | rayleigh |
| 348 | awgn | 0.000000 | 0.000000 | 3.22 | 3.52 | awgn |
| 83 | awgn | 0.022500 | 0.019222 | -2.93 | -2.73 | rayleigh |
| 234 | rayleigh | 0.001500 | 0.000548 | 1.00 | 0.93 | awgn |
| 456 | awgn | 0.000000 | 0.000000 | 7.67 | 7.80 | rayleigh |
| 315 | awgn | 0.000000 | 0.000000 | 5.65 | 5.91 | rayleigh |
| 357 | rayleigh | 0.000000 | 0.000000 | 11.86 | 11.91 | rayleigh |
| 440 | rayleigh | 0.000000 | 0.000000 | 9.05 | 9.33 | rayleigh |

## Interpretation

- The measured BER remains the classical simulator baseline.
- ML predictions are estimates from synthetic features and should be validated against any real RF capture before use.
- AWGN/Rayleigh classification is a controlled two-class experiment, not generalized channel recognition.
- The channel classifier is weak in this run. Treat that as a useful negative result: the current feature set supports SNR/BER estimation better than channel-type recognition.
- SNR error is reported on held-out synthetic samples and should not be treated as field performance.
