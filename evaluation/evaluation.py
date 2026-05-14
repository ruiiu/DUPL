import argparse
import json
import os

from utils.data_loaders import (
    load_geo3k_dataset,
    load_wemath_dataset,
    load_mathvista_dataset,
    load_mathverse_dataset,
    load_mathvision_dataset,
    load_hallubench_dataset,
    load_chartqa_dataset,
    load_logicvista_dataset,
    load_r1_onevision_bench,
    load_mmk12_dataset,
    load_mmstar_dataset,
    load_visualpuzzles_dataset,
    load_realworldqa_dataset,
    load_mmmu_pro_dataset,
)
from utils.processing import (
    prepare_metadata,
    process_outputs,
    process_outputs_pass_k,
    calculate_metrics,
    detect_pass_k_from_outputs
)
from utils.async_judge import (
    process_outputs_async,
    process_outputs_pass_k_async,
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "evaluation/vertex.json"


ALL_DATASETS = [
    "geo3k",
    "wemath",
    "mathvista",
    "mathverse",
    "mathvision",
    "hallubench",
    "chartqa",
    "logicvista",
    "r1_onevision_bench",
    "mmk12",
    "mmstar",
    "visualpuzzles",
    "realworldqa",
    "mmmu_pro",
]

DATASET_LOADERS = {
    "geo3k": load_geo3k_dataset,
    "wemath": load_wemath_dataset,
    "mathvista": load_mathvista_dataset,
    "mathverse": load_mathverse_dataset,
    "mathvision": load_mathvision_dataset,
    "hallubench": load_hallubench_dataset,
    "chartqa": load_chartqa_dataset,
    "logicvista": load_logicvista_dataset,
    "r1_onevision_bench": load_r1_onevision_bench,
    "mmk12": load_mmk12_dataset,
    "mmstar": load_mmstar_dataset,
    "visualpuzzles": load_visualpuzzles_dataset,
    "realworldqa": load_realworldqa_dataset,
    "mmmu_pro": load_mmmu_pro_dataset,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")

    # Model and runtime parameters
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--eval-threads", type=int, default=128, help="Number of threads for evaluation")
    parser.add_argument("--async-eval", action="store_true", default=True, help="Use async LLM judge (much faster)")
    parser.add_argument("--no-async-eval", dest="async_eval", action="store_false", help="Use sync threaded judge")
    parser.add_argument("--max-concurrent", type=int, default=256, help="Max concurrent async judge calls")

    # Pass@k evaluation parameters
    parser.add_argument("--pass-k-values", type=str, default="1,8", help="Comma-separated list of k values for pass@k evaluation (default: 1,8)")

    # Dataset selection
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision,mmk12 or 'all'")

    # Dataset-specific paths
    parser.add_argument("--data-path", type=str, default="dummy", help="")

    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=4194304)

    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate even if eval results already exist")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which datasets to evaluate
    datasets_to_eval = [ds.strip() for ds in args.datasets.split(",")] if args.datasets != "all" else ALL_DATASETS

    # Dictionary to store all samples
    all_samples = {}

    # Load datasets based on selection
    for dataset_name in datasets_to_eval:
        if dataset_name not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {', '.join(ALL_DATASETS)}")
        all_samples[dataset_name] = DATASET_LOADERS[dataset_name](args.data_path)
        print(f"Loaded {len(all_samples[dataset_name])} samples from {dataset_name}")

    if not all_samples:
        print("No datasets loaded. Please check the paths and dataset names.")
        return

    all_metrics = []

    # Parse pass@k values
    k_values = [int(k.strip()) for k in args.pass_k_values.split(",")]

    for dataset_name, samples in all_samples.items():
        eval_output_path = os.path.join(args.output_dir, f"{dataset_name}.json")
        if not args.overwrite and os.path.exists(eval_output_path):
            print(f"[Resume] {dataset_name}.json already exists, skipping. Use --overwrite to re-run.")
            with open(eval_output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if "metrics" in existing:
                all_metrics.append({dataset_name: existing["metrics"]})
            continue

        metadata = prepare_metadata(dataset_name, samples, args)
        if not metadata:
            print(f"No valid metadata for {dataset_name}, skipping.")
            continue

        generated_path = os.path.join(args.output_dir, f"{dataset_name}_outputs.json")
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"Missing generated outputs: {generated_path}")

        with open(generated_path, 'r', encoding='utf-8') as f:
            outputs = json.load(f)

        if len(outputs) != len(metadata):
            raise ValueError(
                f"Output/metadata size mismatch for {dataset_name}: {len(outputs)} outputs vs {len(metadata)} metadata"
            )

        # Detect if this is pass@k evaluation or regular pass@1
        detected_k = detect_pass_k_from_outputs(outputs)

        if detected_k > 1:
            # Use pass@k evaluation
            print(f"Detected pass@{detected_k} outputs, using pass@k evaluation...")
            if args.async_eval:
                print("Using async judge pipeline...")
                eval_results = process_outputs_pass_k_async(
                    outputs, metadata, k_values, args.max_concurrent
                )
            else:
                eval_results = process_outputs_pass_k(outputs, metadata, args.eval_threads, k_values)

            results = eval_results["results"]
            pass_at_k_metrics = eval_results["pass_at_k_metrics"]

            # Combine traditional metrics with pass@k metrics
            traditional_metrics = calculate_metrics(results)
            metrics = {**traditional_metrics, **pass_at_k_metrics}

            output_dict = {
                "results": results,
                "problem_results": eval_results["problem_results"],  # All samples for debugging
                "metrics": metrics,
                "total_problems": eval_results["total_problems"],
                "detected_k": detected_k,
                "config": vars(args)
            }

            print(f"{dataset_name.upper()} Results (Pass@K):")
            print(f"  Total problems: {eval_results['total_problems']}")
            print(f"  Samples per problem: {detected_k}")
            for k in k_values:
                if f"pass@{k}" in pass_at_k_metrics:
                    print(f"  Pass@{k}: {pass_at_k_metrics[f'pass@{k}']:.4f}")

        else:
            # Use traditional pass@1 evaluation
            print(f"Detected pass@1 outputs, using traditional evaluation...")
            if args.async_eval:
                print("Using async judge pipeline...")
                results = process_outputs_async(
                    outputs, metadata, args.max_concurrent
                )
            else:
                results = process_outputs(outputs, metadata, args.eval_threads)
            metrics = calculate_metrics(results)

            output_dict = {
                "results": results,
                "metrics": metrics,
                "config": vars(args)
            }

            print(f"{dataset_name.upper()} Results:")
            print(f"  Total samples: {len(results)}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")

        # Print sub-accuracies if available
        if 'sub_accuracies' in metrics:
            print("  Task/Category Accuracies:")
            for task, acc in metrics['sub_accuracies'].items():
                print(f"    {task}: {acc:.4f}")
        print()

        output_path = os.path.join(args.output_dir, f"{dataset_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

        all_metrics.append({dataset_name: metrics})
    
    with open(os.path.join(args.output_dir, "all_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
