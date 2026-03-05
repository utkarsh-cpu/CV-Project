"""Run experiment matrix and aggregate evaluation metrics for reproducible reporting."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _build_command(exp, defaults, metrics_root: Path):
    exp_name = exp["name"]
    exp_metrics_dir = metrics_root / exp_name

    cmd = [sys.executable, "main.py"]
    cmd.extend(["--dataset", defaults["dataset"]])
    cmd.extend(["--dataset-dir", defaults["dataset_dir"]])
    cmd.extend(["--seed", str(defaults["seed"])])
    cmd.extend(["--metrics-dir", str(exp_metrics_dir)])

    if defaults.get("cross_dataset", False):
        cmd.append("--cross-dataset")

    cmd.extend(exp.get("args", []))
    return cmd, exp_metrics_dir


def _collect_result(exp_name: str, exp_metrics_dir: Path):
    summary_path = exp_metrics_dir / "cross_dataset_summary.json"
    if summary_path.exists():
        return _read_json(summary_path)

    # Fallback to single dataset metric files.
    collected = {}
    for metric_file in exp_metrics_dir.glob("metrics_*.json"):
        dataset_name = metric_file.stem.replace("metrics_", "")
        collected[dataset_name] = _read_json(metric_file)
    return collected


def _write_aggregate(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "experiment_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / "experiment_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "dataset", "miou", "f1", "precision", "recall", "ap50", "ap75", "ap_5095"],
        )
        writer.writeheader()
        for exp_name, per_dataset in results.items():
            for dataset_name, metrics in per_dataset.items():
                writer.writerow(
                    {
                        "experiment": exp_name,
                        "dataset": dataset_name,
                        "miou": metrics.get("miou", 0.0),
                        "f1": metrics.get("f1", 0.0),
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0),
                        "ap50": metrics.get("ap50", 0.0),
                        "ap75": metrics.get("ap75", 0.0),
                        "ap_5095": metrics.get("ap_5095", 0.0),
                    }
                )

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Run experiment matrix for baseline/ablation reporting")
    parser.add_argument("--matrix", default="scripts/experiment_matrix.json", help="Path to experiment matrix JSON")
    parser.add_argument("--output-dir", default="./outputs/experiments", help="Directory for experiment outputs")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    matrix = _read_json(matrix_path)
    defaults = matrix["defaults"]
    experiments = matrix["experiments"]

    output_dir = Path(args.output_dir)
    metrics_root = output_dir / "runs"

    all_results = {}
    for exp in experiments:
        exp_name = exp["name"]
        cmd, exp_metrics_dir = _build_command(exp, defaults, metrics_root)

        print(f"\\n[RUN] {exp_name}")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[FAIL] {exp_name} (code={proc.returncode})")
            print(proc.stdout)
            print(proc.stderr)
            continue

        result_payload = _collect_result(exp_name, exp_metrics_dir)
        all_results[exp_name] = result_payload
        print(f"[OK] {exp_name}")

    json_path, csv_path = _write_aggregate(all_results, output_dir)
    print(f"\\nSaved aggregate JSON: {json_path}")
    print(f"Saved aggregate CSV:  {csv_path}")


if __name__ == "__main__":
    main()
