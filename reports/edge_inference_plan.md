# Edge Inference Plan

This project supports an edge deployment path for AI-assisted wireless link estimation. The target workload is lightweight tabular inference over QPSK constellation statistics, not a production telecom receiver.

## Export Path

1. Generate synthetic link-condition data:

   ```bash
   python generate_dataset.py --samples 500 --num-bits 4000
   ```

2. Train the estimators:

   ```bash
   python train_link_models.py
   ```

3. Export trained models to ONNX:

   ```bash
   python -m pip install skl2onnx onnx onnxruntime
   python export_onnx.py
   ```

The exported files are written to `models/onnx/`.

## TensorRT-Ready Notes

- The ONNX models are tree-based scikit-learn estimators, which are useful for a first edge inference path.
- TensorRT support for tree ensembles is not the same as support for neural networks. For TensorRT deployment, replace or distill these estimators into a small MLP and export that network to ONNX.
- Keep the feature contract fixed: the runtime input vector must follow `ai_link_estimation.features.FEATURE_COLUMNS`.
- Validate numerical parity between Python, ONNX Runtime, and any TensorRT engine before reporting edge performance.

## Jetson Benchmark Template

After ONNX export, copy the repo to the Jetson or run in place:

```bash
python edge/jetson_benchmark_template.py --model models/onnx/snr_estimator.onnx --runs 1000
```

Record:

- Jetson model and power mode
- Python, ONNX Runtime, CUDA, and TensorRT versions
- Mean latency in milliseconds
- Inferences per second
- Batch size and feature count

## Grounded Scope

This is an edge AI-assisted wireless link estimation workflow. It estimates link conditions from simulated QPSK data and can be benchmarked on edge hardware. It is not a full AI-RAN base station, scheduler, production modem, or standards-compliant telecom receiver.
