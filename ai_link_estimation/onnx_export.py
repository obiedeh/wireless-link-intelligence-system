"""Export trained scikit-learn estimators to ONNX when skl2onnx is installed."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .features import FEATURE_COLUMNS


def export_models(model_dir: str | Path = "models",
                  output_dir: str | Path = "models/onnx") -> list[Path]:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires skl2onnx. Install optional edge dependencies with "
            "`python -m pip install skl2onnx onnx onnxruntime`."
        ) from exc

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    initial_types = [("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]

    for model_path in sorted(model_dir.glob("*.joblib")):
        model = joblib.load(model_path)
        onnx_model = convert_sklearn(model, initial_types=initial_types)
        output_path = output_dir / f"{model_path.stem}.onnx"
        output_path.write_bytes(onnx_model.SerializeToString())
        exported.append(output_path)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained link-estimation models to ONNX.")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="models/onnx")
    args = parser.parse_args()

    for path in export_models(args.model_dir, args.output_dir):
        print(f"Exported {path}")


if __name__ == "__main__":
    main()
