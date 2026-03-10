import os
import subprocess
import re
import csv
import argparse
import logging
import sys
from typing import Tuple, Optional

import onnx
from onnxsim import simplify


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
SUPPORTED_PREDICTORS = [
    "cortexA76cpu_tflite21",
    "adreno640gpu_tflite21",
    "adreno630gpu_tflite21",
    "myriadvpu_openvino2019r2",
]

def prepare_output_csv(output_csv: str):
    """Create fresh CSV file and write header."""

    if os.path.exists(output_csv):
        logging.info(f"Existing CSV found. Overwriting: {output_csv}")
        os.remove(output_csv)

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["Model", "Predictor", "Latency (ms)", "FPS"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def simplify_onnx_files(onnx_dir: str) -> None:
    """Simplify all ONNX models in a directory."""
    for filename in os.listdir(onnx_dir):
        if filename.endswith(".onnx") and not filename.endswith("_simplified.onnx"):
            input_path = os.path.join(onnx_dir, filename)
            output_path = os.path.join(
                onnx_dir, filename.replace(".onnx", "_simplified.onnx")
            )

            logging.info(f"Simplifying {filename}")

            try:
                model = onnx.load(input_path)
                model_simplified, check = simplify(model)

                if check:
                    onnx.save(model_simplified, output_path)
                    logging.info(f"Saved simplified model: {output_path}")
                else:
                    logging.warning(f"Simplification check failed for {filename}")

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")


def get_latency_and_fps(
    model_path: str, predictor: str, version: str
) -> Tuple[Optional[float], Optional[float]]:
    """Run nn-Meter latency prediction."""

    result = subprocess.run(
        [
            "nn-meter",
            "predict",
            "--predictor",
            predictor,
            "--predictor-version",
            version,
            "--onnx",
            model_path,
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout
    match = re.search(r"Predict latency: ([\d.]+)", output)

    if match:
        latency = float(match.group(1))
        fps = 1000 / latency
        return latency, fps

    logging.warning(f"Could not extract latency for {model_path}")
    return None, None


def benchmark_onnx_runtime(
    onnx_dir: str, predictor: str, version: str, output_csv: str
) -> None:
    """Benchmark simplified ONNX models and save results."""

    results = []

    logging.info("Benchmarking simplified ONNX models...")

    for filename in os.listdir(onnx_dir):
        if filename.endswith("_simplified.onnx"):
            model_path = os.path.join(onnx_dir, filename)

            latency, fps = get_latency_and_fps(model_path, predictor, version)

            if latency is not None:
                logging.info(f"{filename}: {latency:.2f} ms | {fps:.2f} FPS")
                results.append(
                    {
                        "Model": filename,
                        "Predictor": predictor,
                        "Latency (ms)": round(latency, 2),
                        "FPS": round(fps, 2),
                    }
                )
            else:
                results.append(
                    {"Model": filename, "Predictor": predictor, "Latency (ms)": "Error", "FPS": "Error"}
                )
    file_exists = os.path.exists(output_csv)

    with open(output_csv, "a", newline="") as csvfile:
        fieldnames = ["Model", "Predictor", "Latency (ms)", "FPS"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerows(results)

    logging.info(f"Results saved to {output_csv}")


def validate_directories(onnx_dir: str, output_dir: str):
    """Ensure required directories exist."""

    if not os.path.exists(onnx_dir):
        print("\n❌ ONNX folder not found.")
        print(f"Please create a folder named '{onnx_dir}' and place exported ONNX models inside it.")
        print("Example structure:")
        print(f"""
                nn-Meter/
                 ├── main.py
                 ├── {onnx_dir}/
                 │    ├── model1.onnx
                 │    └── model2.onnx
                 └── {output_dir}/
                """)
        sys.exit(1)

    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)


def main():

    parser = argparse.ArgumentParser(description="ONNX latency benchmarking tool")

    parser.add_argument("--onnx-dir", default="onnx")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--predictor", default="all",
                        help="Predictor name or 'all'")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--output-file", default="latency_results.csv")

    args = parser.parse_args()

    validate_directories(args.onnx_dir, args.output_dir)
    output_csv = os.path.join(args.output_dir, args.output_file)

    prepare_output_csv(output_csv)

    simplify_onnx_files(args.onnx_dir)

    # Determine predictors to run
    if args.predictor.lower() == "all":
        predictors = SUPPORTED_PREDICTORS
        logging.info("Running benchmark for ALL predictors")
    else:
        predictors = [args.predictor]
        logging.info(f"Running benchmark for predictor: {args.predictor}")

    # Run benchmarks
    for predictor in predictors:
        logging.info(f"\n===== Running predictor: {predictor} =====")

        benchmark_onnx_runtime(
            args.onnx_dir,
            predictor,
            args.version,
            output_csv,
        )

if __name__ == "__main__":
    main()