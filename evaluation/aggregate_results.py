import argparse
import json
import os
import sys


def load_metrics(results_dir: str, dataset: str):
    path = os.path.join(results_dir, f"{dataset}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics", {})


def format_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def infer_model_name(results_dir: str) -> str:
    return os.path.basename(os.path.normpath(results_dir))


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results across models")
    parser.add_argument(
        "--results-dirs",
        type=str,
        required=True,
        help="Comma-separated result directories, one per model",
    )
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated dataset list to aggregate")
    parser.add_argument(
        "--model-names",
        type=str,
        default=None,
        help="Optional comma-separated model names for display; defaults to basename of each results dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: first results dir parent / summary_table.json)",
    )
    args = parser.parse_args()

    results_dirs = [d.strip() for d in args.results_dirs.split(",") if d.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not results_dirs:
        print("Error: no result directories provided", file=sys.stderr)
        sys.exit(1)

    for results_dir in results_dirs:
        if not os.path.isdir(results_dir):
            print(f"Error: {results_dir} is not a directory", file=sys.stderr)
            sys.exit(1)

    if args.model_names:
        model_names = [m.strip() for m in args.model_names.split(",")]
        if len(model_names) != len(results_dirs):
            print("Error: --model-names must match --results-dirs length", file=sys.stderr)
            sys.exit(1)
    else:
        model_names = [infer_model_name(d) for d in results_dirs]

    rows = []
    for model_name, results_dir in zip(model_names, results_dirs):
        row = {
            "model": model_name,
            "results_dir": results_dir,
            "datasets": {},
            "avg_accuracy": None,
        }
        acc_values = []
        for dataset in datasets:
            metrics = load_metrics(results_dir, dataset)
            acc = None
            if metrics and isinstance(metrics.get("accuracy"), (int, float)):
                acc = float(metrics["accuracy"])
                acc_values.append(acc)
            row["datasets"][dataset] = acc
        row["avg_accuracy"] = sum(acc_values) / len(acc_values) if acc_values else None
        rows.append(row)

    print()
    col_width = max(18, max(len(name) for name in model_names) + 2)
    print("=" * (col_width + 16 * (len(datasets) + 1)))
    print(f"{'Model':<{col_width}}", end="")
    for dataset in datasets:
        print(f"{dataset:>16}", end="")
    print(f"{'Avg':>16}")
    print("-" * (col_width + 16 * (len(datasets) + 1)))

    for row in rows:
        print(f"{row['model']:<{col_width}}", end="")
        for dataset in datasets:
            print(f"{format_pct(row['datasets'][dataset]):>16}", end="")
        print(f"{format_pct(row['avg_accuracy']):>16}")

    print("=" * (col_width + 16 * (len(datasets) + 1)))

    summary = {
        "datasets": datasets,
        "rows": rows,
    }

    default_output_dir = os.path.dirname(os.path.normpath(results_dirs[0])) or "."
    output_path = args.output or os.path.join(default_output_dir, "summary_table.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
