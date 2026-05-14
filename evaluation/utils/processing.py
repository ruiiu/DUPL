import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from mathruler.grader import extract_boxed_content, grade_answer
from utils.optimized_qwen_local import llm_eval_score_retry as llm_eval_score


LLM_JUDGE_DATASETS = {
    "mathvista",
    "mathverse",
    "mathvision",
    "wemath",
    "chartqa",
    "logicvista",
    "r1_onevision_bench",
    "mmk12",
    "math12k",
    "aime24",
    "math500",
    "gsm8k",
    "gpqa",
    "olympiadbench",
    "mmstar",
    "visualpuzzles",
    "realworldqa",
    "mmmu_pro",
}


def _normalize_inline_image(image: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    """Normalize an in-memory image to RGB and resize within the configured bounds."""
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    if image.mode != "RGB":
        image = image.convert("RGB")

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    return image


def _build_prompt_and_metadata(dataset_name: str, item: Dict, args, include_image: bool) -> Optional[Tuple[Optional[Dict], Dict]]:
    """Build prompt payload and metadata with identical filtering/order semantics."""
    image = None
    image_path = item.get("image_path")
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            return None
        if include_image:
            image = load_image(image_path, args.min_pixels, args.max_pixels)
            if image is None:
                return None
    elif isinstance(image_path, Image.Image):
        if include_image:
            image = _normalize_inline_image(image_path, args.min_pixels, args.max_pixels)
    elif image_path is not None or include_image:
        return None

    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{args.system_prompt} <|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    prompt = None
    if include_image:
        prompt = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        }

    metadata = {
        "dataset": dataset_name,
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "prompt": prompt_text,
        **{k: v for k, v in item.items() if k not in ["image_path", "image", "dataset", "id", "question", "answer"]}
    }
    return prompt, metadata

def load_image(image_path: str, min_pixels: int, max_pixels: int) -> Image.Image:
    """Load and preprocess an image"""
    try:
        # image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path)
        if image.mode == "P":
            image = image.convert("RGBA")  # first handle palette+transparency
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")   # discard alpha channel
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_prompts(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []

    prompt_workers = max(getattr(args, "prompt_workers", 1), 1)
    if prompt_workers == 1:
        prepared_items = (
            _build_prompt_and_metadata(dataset_name, item, args, include_image=True)
            for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts")
        )
    else:
        with ThreadPoolExecutor(max_workers=prompt_workers) as executor:
            prepared_items = tqdm(
                executor.map(lambda item: _build_prompt_and_metadata(dataset_name, item, args, include_image=True), samples),
                total=len(samples),
                desc=f"Preparing {dataset_name} prompts",
            )

            for prepared in prepared_items:
                if prepared is None:
                    continue
                prompt, meta = prepared
                prompts.append(prompt)
                metadata.append(meta)

            return prompts, metadata

    for prepared in prepared_items:
        if prepared is None:
            continue
        prompt, meta = prepared
        prompts.append(prompt)
        metadata.append(meta)

    return prompts, metadata


def prepare_metadata(dataset_name: str, samples: List[Dict], args) -> List[Dict]:
    """Prepare evaluation metadata without loading or resizing images."""
    metadata = []

    for item in tqdm(samples, desc=f"Preparing {dataset_name} metadata"):
        prepared = _build_prompt_and_metadata(dataset_name, item, args, include_image=False)
        if prepared is None:
            continue
        _, meta = prepared
        metadata.append(meta)

    return metadata


def evaluate_prediction(prediction: str, answer: str, dataset: str, question: str = "") -> float:
    """Evaluate a prediction against the ground truth"""
    if dataset == "geo3k":
        extracted_answer = extract_boxed_content(prediction)
        return 1.0 if grade_answer(extracted_answer, answer) else 0.0

    elif dataset in LLM_JUDGE_DATASETS:
        for attempt in range(3):
            try:
                return llm_eval_score(question, prediction, answer, dataset)
            except Exception:
                if attempt < 2:
                    import time
                    time.sleep(2)
                else:
                    raise

    elif dataset == "hallubench":
        extracted_answer = extract_boxed_content(prediction)
        return 1.0 if answer.lower() in extracted_answer.lower() else 0.0

    else:
        extracted_answer = extract_boxed_content(prediction)
        return 1.0 if extracted_answer == answer else 0.0

def process_outputs(outputs, metadata, max_workers: int) -> List[Dict]:
    """Process model outputs and calculate metrics (legacy function for pass@1)"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for i, output in enumerate(outputs):
            prediction = output["generated_text"][0].strip()
            meta = metadata[i]
            dataset = meta["dataset"]
            if "question_for_eval" in meta:
                question = meta["question_for_eval"]
            else:
                question = meta["question"]

            future = executor.submit(
                evaluate_prediction,
                prediction,
                meta["answer"],
                dataset,
                question
            )
            futures.append((future, i, prediction, meta))

        for future, i, prediction, meta in tqdm(futures, desc="Evaluating predictions"):
            try:
                accuracy = future.result()
            except Exception as e:
                print(f"Error evaluating prediction {i}: {str(e)}")
                continue

            result = {
                "id": meta["id"],
                "question": meta["question"],
                "answer": meta["answer"],
                "prediction": prediction,
                "accuracy": accuracy,
                "correct": accuracy > 0,
                **{k: v for k, v in meta.items() if k not in ["dataset", "id", "question", "answer"]}
            }

            results.append(result)

    return results

def detect_pass_k_from_outputs(outputs) -> int:
    """Detect the number of samples (k) per problem from outputs structure"""
    if not outputs:
        return 1

    # Check if outputs have k field (number of samples)
    if "k" in outputs[0]:
        return outputs[0]["k"]

    # Check the number of generated texts per output (this is k)
    if "generated_text" in outputs[0]:
        return len(outputs[0]["generated_text"])

    return 1

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    if not results:
        return {"accuracy": 0.0}
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    metrics = {"accuracy": accuracy}
    
    # Calculate task-specific accuracies if available
    if any("task" in r for r in results):
        task_results = {}
        for r in results:
            if "task" in r:
                task = r["task"]
                if task not in task_results:
                    task_results[task] = []
                task_results[task].append(r["correct"])
        
        task_accuracies = {task: sum(results) / len(results) for task, results in task_results.items()}
        metrics["sub_accuracies"] = task_accuracies
    
    # Calculate problem version accuracies if available
    if any("problem_version" in r for r in results):
        version_results = {}
        for r in results:
            if "problem_version" in r:
                version = r["problem_version"]
                if version not in version_results:
                    version_results[version] = []
                version_results[version].append(r["correct"])
        
        version_accuracies = {version: sum(results) / len(results) for version, results in version_results.items()}
        metrics["sub_accuracies"] = version_accuracies
    
    # Calculate subject accuracies if available
    if any("subject" in r for r in results):
        subject_results = {}
        for r in results:
            if "subject" in r:
                subject = r["subject"]
                if subject not in subject_results:
                    subject_results[subject] = []
                subject_results[subject].append(r["correct"])
        
        subject_accuracies = {subject: sum(results) / len(results) for subject, results in subject_results.items()}
        metrics["sub_accuracies"] = subject_accuracies
    
    return metrics


def process_outputs_pass_k(outputs, metadata, max_workers: int, k_values: List[int] = None) -> Dict:
    """
    Process model outputs for pass@k evaluation with multiple k values.

    For each problem, generate k completions and mark as correct if ≥1 completion passes the checker.
    Pass@k = average fraction of problems solved.

    Args:
        outputs: List of model outputs, each containing exactly k generated_text samples
        metadata: List of metadata for each problem
        max_workers: Number of worker threads
        k_values: List of k values to calculate pass@k for (default: [1, 4, 8])

    Returns:
        Dictionary containing results and pass@k metrics
    """
    if k_values is None:
        k_values = [1, 4, 8]

    if not outputs:
        print("No outputs found")
        return {"results": [], "problem_results": {}, "pass_at_k_metrics": {}, "total_problems": 0}

    # Get the number of samples per problem (should be k)
    samples_per_problem = len(outputs[0]["generated_text"])


    # Filter k_values to only include those <= available samples
    valid_k_values = [k for k in k_values if k <= samples_per_problem]
    if not valid_k_values:
        print(f"Warning: No valid k values (requested: {k_values}, available samples: {samples_per_problem})")
        return {"results": [], "problem_results": {}, "pass_at_k_metrics": {}, "total_problems": 0}
    
    print(f"Will calculate pass@k for k = {valid_k_values}")
    k_values = valid_k_values

    # Group results by problem
    problem_results = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for i, output in enumerate(outputs):
            meta = metadata[i]
            dataset = meta["dataset"]
            if "question_for_eval" in meta:
                question = meta["question_for_eval"]
            else:
                question = meta["question"]

            # Process all generated samples for this problem
            for j, prediction in enumerate(output["generated_text"]):
                future = executor.submit(
                    evaluate_prediction,
                    prediction.strip(),
                    meta["answer"],
                    dataset,
                    question
                )
                futures.append((future, i, j, prediction.strip(), meta))

        for future, problem_idx, sample_idx, prediction, meta in tqdm(futures, desc="Evaluating predictions"):
            try:
                accuracy = future.result()
            except Exception as e:
                print(f"Error evaluating prediction {problem_idx}-{sample_idx}: {str(e)}")
                continue

            result = {
                "problem_idx": problem_idx,
                "sample_idx": sample_idx,
                "id": meta["id"],
                "question": meta["question"],
                "answer": meta["answer"],
                "prediction": prediction,
                "accuracy": accuracy,
                "correct": accuracy > 0,
                **{k: v for k, v in meta.items() if k not in ["dataset", "id", "question", "answer"]}
            }

            problem_results[problem_idx].append(result)

    # Calculate pass@k metrics for each k value
    pass_at_k_metrics = {}
    all_results = []

    for k in k_values:
        # For each problem, check if at least one of the first k samples is correct
        problem_correct_count = 0
        total_problems = len(problem_results)

        for problem_idx, problem_samples in problem_results.items():
            # Check if at least one of the first k samples is correct
            first_k_samples = problem_samples[:k]
            has_correct = any(sample["correct"] for sample in first_k_samples)
            if has_correct:
                problem_correct_count += 1

        # Calculate pass@k as the fraction of problems solved
        if total_problems > 0:
            pass_at_k_metrics[f"pass@{k}"] = problem_correct_count / total_problems
        else:
            pass_at_k_metrics[f"pass@{k}"] = 0.0

        # For k=1, also store individual results for compatibility
        if k == 1:
            for problem_idx, problem_samples in problem_results.items():
                representative_result = problem_samples[0].copy()
                representative_result["pass_at_1"] = pass_at_k_metrics["pass@1"]
                all_results.append(representative_result)

        print(f"Pass@{k}: {pass_at_k_metrics[f'pass@{k}']:.4f} ({problem_correct_count}/{total_problems} problems solved)")

    return {
        "results": all_results,
        "problem_results": dict(problem_results),  # All samples for all problems
        "pass_at_k_metrics": pass_at_k_metrics,
        "total_problems": len(problem_results)
    }
