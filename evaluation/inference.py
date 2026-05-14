import argparse
import json
import os

import torch
from vllm import LLM, SamplingParams

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
from utils.processing import prepare_prompts


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
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=4194304)
    parser.add_argument("--max-model-len", type=int, default=12288, help="Maximum total context length (prompt + generation). Should be >= max_tokens + typical_prompt_length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="Gpu memory utilization")
    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")

    # Pass@k parameters
    parser.add_argument("--k", type=int, default=1, help="Number of samples to generate per problem for pass@k evaluation (default: 1 for pass@1)")
    parser.add_argument("--pass-k-temperature", type=float, default=0.7, help="Temperature for pass@k sampling when k > 1 (default: 0.7)")

    # Dataset selection
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision,mmk12 or 'all'")

    # Dataset-specific paths
    parser.add_argument("--data-path", type=str, default="dummy", help="")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384, help="Maximum number of batched tokens for inference")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences for vLLM scheduling")
    parser.add_argument("--prompt-workers", type=int, default=8, help="Threads used to load/resize images and build prompts")
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager execution. Slower, but sometimes useful for debugging")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", help="Disable vLLM custom all-reduce")
    parser.add_argument("--pretty-json", action="store_true", help="Pretty-print output JSON files")
    parser.add_argument("--overwrite", action="store_true", help="Re-run inference even if output file already exists")

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

    # Initialize model
    print(f"Initializing model from {args.model}")

    llm = LLM(
        model=args.model,
        skip_tokenizer_init=False,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        disable_mm_preprocessor_cache=False,
        limit_mm_per_prompt = {"image": 1, "video": 0},  # Limit to 1 image per prompt for efficiency
    )

    # Configure sampling parameters based on pass@k settings
    if args.k > 1:
        # For pass@k with k > 1, use higher temperature and generate multiple samples
        sampling_params = SamplingParams(
            temperature=args.pass_k_temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.k,  # Generate k samples per prompt
        )
        print(f"Pass@k mode: generating {args.k} samples per problem with temperature={args.pass_k_temperature}")
    else:
        # For pass@1, use original deterministic settings
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )   
        print(f"Pass@1 mode: generating 1 sample per problem with temperature={args.temperature}")

    for dataset_name, samples in all_samples.items():
        output_path = os.path.join(args.output_dir, f"{dataset_name}_outputs.json")
        if not args.overwrite and os.path.exists(output_path):
            print(f"[Resume] {dataset_name}_outputs.json already exists, skipping. Use --overwrite to re-run.")
            continue

        prompts, _ = prepare_prompts(dataset_name, samples, args)
        if not prompts:
            print(f"No valid prompts for {dataset_name}, skipping.")
            continue

        outputs = llm.generate(prompts, sampling_params)

        ### save outputs for offline evaluation
        output_data = []
        for output in outputs:
            output_dict = {
                "prompt": output.prompt,
                "generated_text": [o.text for o in output.outputs],
                "logprobs": [o.logprobs for o in output.outputs] if output.outputs[0].logprobs else None,
                "finish_reason": [o.finish_reason for o in output.outputs],
                "k": args.k  # Store pass@k info for evaluation
            }
            output_data.append(output_dict)

        output_path = os.path.join(args.output_dir, f"{dataset_name}_outputs.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2 if args.pretty_json else None)

        total_samples = len(outputs) * args.k
        print(f"Generated {len(outputs)} problems × {args.k} samples = {total_samples} total outputs for {dataset_name}")


if __name__ == "__main__":
    main()
