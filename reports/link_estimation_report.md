# AI-Assisted Wireless Link Estimation Report

This report summarizes synthetic experiments around a classical QPSK baseline. The project estimates link conditions from simulated constellation statistics; it is not a full AI-RAN base station and not a production telecom receiver.

## Dataset

- Source CSV: `data/link_conditions.csv`
- Samples: 250
- Held-out test samples: 63
- Labels: SNR, measured BER, channel type, and synthetic link-quality score
- Features: constellation power, I/Q moments, EVM, quadrant balance, and fading coefficient summary

## Model Results

- SNR estimation MAE: 0.215 dB
- SNR estimation R2: 0.998
- BER prediction MAE: 0.000915
- BER prediction R2: 0.907
- AWGN vs Rayleigh classification accuracy: 1.000
- Link-quality scoring MAE: 3.118

## Classical BER vs Predicted BER

| Sample | Channel | Measured BER | Predicted BER | SNR dB | Predicted SNR dB | Predicted Channel |
|---:|---|---:|---:|---:|---:|---|
| 98 | awgn | 0.000000 | 0.000000 | 7.21 | 7.56 | awgn |
| 121 | rayleigh | 0.000000 | 0.000000 | 16.01 | 16.05 | rayleigh |
| 26 | awgn | 0.000000 | 0.000053 | 3.07 | 2.64 | awgn |
| 169 | rayleigh | 0.000000 | 0.000000 | 13.83 | 13.99 | rayleigh |
| 210 | awgn | 0.000000 | 0.000000 | 14.00 | 14.18 | awgn |
| 127 | rayleigh | 0.000000 | 0.000009 | 8.77 | 8.92 | rayleigh |
| 135 | rayleigh | 0.000000 | 0.000000 | 15.86 | 15.65 | rayleigh |
| 196 | rayleigh | 0.021667 | 0.030693 | -3.26 | -3.20 | rayleigh |
| 69 | awgn | 0.000000 | 0.000000 | 11.15 | 11.49 | awgn |
| 203 | awgn | 0.000000 | 0.000000 | 5.26 | 5.58 | awgn |
| 22 | rayleigh | 0.000000 | 0.000000 | 7.97 | 7.80 | rayleigh |
| 157 | rayleigh | 0.002500 | 0.006892 | -1.11 | -1.09 | rayleigh |

## Interpretation

- The measured BER remains the classical simulator baseline.
- ML predictions are estimates from synthetic features and should be validated against any real RF capture before use.
- AWGN/Rayleigh classification is a controlled two-class experiment, not generalized channel recognition.
- SNR error is reported on held-out synthetic samples and should not be treated as field performance.
