"""
Async LLM judge client for fast batch evaluation.

Replaces synchronous per-sample HTTP calls with a two-phase async pipeline:
  Phase 1 -- Extract:  Send ALL extract prompts concurrently -> collect answers.
  Phase 2 -- Score:    Send ALL score prompts concurrently -> collect 0/1 judgements.

On a typical vLLM judge server this gives 5-20x speedup over the threaded approach
because the server can batch hundreds of requests internally.
"""

import asyncio
import os
import re
from typing import List, Dict, Optional

import aiohttp
from tqdm import tqdm

from utils.model_parser_qwen import (
    build_extract_prompt,
    build_mathverse_extract_prompt,
    build_wemath_extract_prompt,
    build_chartqa_extract_prompt,
    build_logicvista_extract_prompt,
    build_r1_onevision_extract_prompt,
    build_mmk12_extract_prompt,
    build_score_prompt,
    build_chartqa_score_prompt,
    build_logicvista_score_prompt,
    build_r1_onevision_score_prompt,
    build_mmk12_score_prompt,
)
from mathruler.grader import extract_boxed_content, grade_answer


_DEFAULT_URL = "http://28.7.194.41:8000/v1/chat/completions"
_DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


async def _call_once(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> str:
    async with semaphore:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with session.post(api_url, json=data) as resp:
            resp.raise_for_status()
            result = await resp.json()
            return result["choices"][0]["message"]["content"]


async def _call_with_retry(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
    retries: int = 5,
) -> Optional[str]:
    """Returns response string on success, None on total failure."""
    for attempt in range(retries):
        try:
            return await _call_once(
                session, semaphore, api_url, model, prompt, temperature, max_tokens
            )
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(0.3 * (1.5 ** attempt))
            else:
                print(f"Judge call failed after {retries} attempts: {e}")
                return None


async def _batch_call(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_url: str,
    model: str,
    prompts: List[str],
    desc: str = "Judge calls",
) -> List[Optional[str]]:
    """Fire all prompts concurrently, show progress, return ordered results."""
    results: List[Optional[str]] = [None] * len(prompts)
    pbar = tqdm(total=len(prompts), desc=desc)

    async def _run(idx: int, prompt: str):
        r = await _call_with_retry(session, semaphore, api_url, model, prompt)
        results[idx] = r
        pbar.update(1)

    await asyncio.gather(*[_run(i, p) for i, p in enumerate(prompts)])
    pbar.close()
    return results


NO_JUDGE_DATASETS = {"geo3k", "hallubench"}

# Multiple-choice letter match: exact letter match after extraction is sufficient
EXTRACT_ONLY_DATASETS = {"wemath", "mmstar", "mmmu_pro"}

# LLM extract + LLM score: avoids fragile string matching (e.g. "0.5" vs "1/2")
EXTRACT_AND_SCORE_DATASETS = {
    "mathvista", "mathvision", "mathverse",
    "chartqa", "logicvista", "r1_onevision_bench", "mmk12",
    "math12k", "aime24", "math500", "gsm8k", "gpqa", "olympiadbench",
    "visualpuzzles", "realworldqa",
}


def _build_extract(dataset: str, prediction: str, question: str) -> str:
    ds = dataset.lower()
    if ds == "mathverse":
        return build_mathverse_extract_prompt(prediction)
    elif ds == "wemath":
        return build_wemath_extract_prompt(prediction, question)
    elif ds == "chartqa":
        return build_chartqa_extract_prompt(prediction)
    elif ds == "logicvista":
        return build_logicvista_extract_prompt(prediction, question)
    elif ds == "r1_onevision_bench":
        return build_r1_onevision_extract_prompt(prediction)
    elif ds == "mmk12":
        return build_mmk12_extract_prompt(prediction)
    else:
        return build_extract_prompt(prediction, question)


def _build_score(dataset: str, question: str, extracted: str, answer: str) -> str:
    ds = dataset.lower()
    if ds == "chartqa":
        return build_chartqa_score_prompt(question, extracted, answer)
    elif ds == "logicvista":
        return build_logicvista_score_prompt(question, extracted, answer)
    elif ds == "r1_onevision_bench":
        return build_r1_onevision_score_prompt(question, extracted, answer)
    elif ds == "mmk12":
        return build_mmk12_score_prompt(question, extracted, answer)
    else:
        return build_score_prompt(question, extracted, answer)


def _score_from_extract(dataset: str, extracted: str, answer: str) -> Optional[float]:
    """For extract-only datasets, compute score from the extracted answer.
    Returns None if extracted is None (LLM call failed)."""
    if extracted is None:
        return None
    ds = dataset.lower()
    if ds in ("wemath", "mmstar", "mmmu_pro"):
        ex = extracted.strip().upper()
        if re.match(r'^[A-Z]$', ex):
            return 1.0 if ex == answer.strip().upper() else 0.0
        return 0.0
    return 0.0


def _score_no_judge(dataset: str, prediction: str, answer: str) -> float:
    ds = dataset.lower()
    if ds == "geo3k":
        ext = extract_boxed_content(prediction)
        return 1.0 if grade_answer(ext, answer) else 0.0
    elif ds == "hallubench":
        ext = extract_boxed_content(prediction)
        return 1.0 if answer.lower() in ext.lower() else 0.0
    return 0.0


def _parse_judgement(response: Optional[str]) -> Optional[float]:
    """Returns 0.0 or 1.0 for valid responses, None for failed calls."""
    if response is None:
        return None
    text = response.strip()
    if text in ("0", "1"):
        return float(text)
    return 0.0


def _make_result(meta: Dict, prediction: str, accuracy: float) -> Dict:
    return {
        "id": meta["id"],
        "question": meta["question"],
        "answer": meta["answer"],
        "prediction": prediction,
        "accuracy": accuracy,
        "correct": accuracy > 0,
        **{k: v for k, v in meta.items()
           if k not in ("dataset", "id", "question", "answer")},
    }


async def evaluate_dataset_async(
    outputs: List[Dict],
    metadata: List[Dict],
    max_concurrent: int = 256,
    api_url: str = None,
    model: str = None,
) -> List[Dict]:
    api_url = api_url or os.getenv("QWEN_LOCAL_URL", _DEFAULT_URL)
    model = model or os.getenv("MODEL", _DEFAULT_MODEL)
    dataset = metadata[0]["dataset"] if metadata else ""
    ds = dataset.lower()

    semaphore = asyncio.Semaphore(max_concurrent)
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)

    results = []

    if ds in NO_JUDGE_DATASETS:
        for i, output in enumerate(tqdm(outputs, desc=f"Eval {dataset}")):
            prediction = output["generated_text"][0].strip()
            meta = metadata[i]
            accuracy = _score_no_judge(ds, prediction, meta["answer"])
            results.append(_make_result(meta, prediction, accuracy))
        return results

    # Phase 1: Extract
    extract_prompts = []
    for i, output in enumerate(outputs):
        prediction = output["generated_text"][0].strip()
        meta = metadata[i]
        question = meta.get("question_for_eval", meta["question"])
        extract_prompts.append(_build_extract(ds, prediction, question))

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, trust_env=False
    ) as session:
        print(f"Phase 1/2: Extracting answers ({len(extract_prompts)} calls)...")
        extracted_answers = await _batch_call(
            session, semaphore, api_url, model, extract_prompts,
            desc=f"Extract ({dataset})",
        )

        if ds in EXTRACT_ONLY_DATASETS:
            for i, ext_ans in enumerate(extracted_answers):
                meta = metadata[i]
                prediction = outputs[i]["generated_text"][0].strip()
                accuracy = _score_from_extract(ds, ext_ans, meta["answer"])
                if accuracy is not None:
                    results.append(_make_result(meta, prediction, accuracy))
            return results

        # Phase 2: Score
        score_prompts = []
        score_indices = []
        for i, ext_ans in enumerate(extracted_answers):
            if ext_ans is None:
                continue
            meta = metadata[i]
            question = meta.get("question_for_eval", meta["question"])
            processed_ext = ext_ans.strip()
            if ds in ("logicvista", "r1_onevision_bench",
                       "math12k", "aime24", "math500", "gsm8k", "gpqa", "olympiadbench",
                       "visualpuzzles", "realworldqa"):
                processed_ext = processed_ext.upper()
            score_prompts.append(_build_score(ds, question, processed_ext, meta["answer"]))
            score_indices.append(i)

        print(f"Phase 2/2: Scoring ({len(score_prompts)} calls)...")
        score_responses = await _batch_call(
            session, semaphore, api_url, model, score_prompts,
            desc=f"Score ({dataset})",
        )

    for j, score_resp in enumerate(score_responses):
        idx = score_indices[j]
        meta = metadata[idx]
        prediction = outputs[idx]["generated_text"][0].strip()
        accuracy = _parse_judgement(score_resp)
        if accuracy is not None:
            results.append(_make_result(meta, prediction, accuracy))

    n_skipped = len(outputs) - len(results)
    if n_skipped > 0:
        print(f"Warning: {n_skipped}/{len(outputs)} samples skipped due to judge failures")

    return results


async def evaluate_dataset_pass_k_async(
    outputs: List[Dict],
    metadata: List[Dict],
    k_values: List[int],
    max_concurrent: int = 256,
    api_url: str = None,
    model: str = None,
) -> Dict:
    from collections import defaultdict

    api_url = api_url or os.getenv("QWEN_LOCAL_URL", _DEFAULT_URL)
    model = model or os.getenv("MODEL", _DEFAULT_MODEL)
    dataset = metadata[0]["dataset"] if metadata else ""
    ds = dataset.lower()

    samples_per_problem = len(outputs[0]["generated_text"])
    valid_k_values = [k for k in k_values if k <= samples_per_problem]
    if not valid_k_values:
        return {"results": [], "problem_results": {}, "pass_at_k_metrics": {}, "total_problems": 0}
    k_values = valid_k_values

    semaphore = asyncio.Semaphore(max_concurrent)
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)

    flat_items = []
    for problem_idx, output in enumerate(outputs):
        meta = metadata[problem_idx]
        question = meta.get("question_for_eval", meta["question"])
        for sample_idx, pred in enumerate(output["generated_text"]):
            flat_items.append({
                "problem_idx": problem_idx,
                "sample_idx": sample_idx,
                "prediction": pred.strip(),
                "question": question,
                "answer": meta["answer"],
                "meta": meta,
            })

    if ds in NO_JUDGE_DATASETS:
        all_scores = [
            _score_no_judge(ds, it["prediction"], it["answer"])
            for it in flat_items
        ]
    else:
        extract_prompts = [
            _build_extract(ds, it["prediction"], it["question"]) for it in flat_items
        ]

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector, trust_env=False
        ) as session:
            print(f"Pass@k Phase 1/2: Extracting ({len(extract_prompts)} calls)...")
            extracted = await _batch_call(
                session, semaphore, api_url, model, extract_prompts,
                desc=f"Extract ({dataset} pass@k)",
            )

            if ds in EXTRACT_ONLY_DATASETS:
                all_scores = [
                    _score_from_extract(ds, ext, it["answer"])
                    for ext, it in zip(extracted, flat_items)
                ]
            else:
                score_prompts = []
                score_map = []
                for i, (ext, it) in enumerate(zip(extracted, flat_items)):
                    if ext is None:
                        continue
                    processed_ext = ext.strip()
                    if ds in ("logicvista", "r1_onevision_bench",
                              "math12k", "aime24", "math500", "gsm8k", "gpqa", "olympiadbench",
                              "visualpuzzles", "realworldqa"):
                        processed_ext = processed_ext.upper()
                    score_prompts.append(_build_score(ds, it["question"], processed_ext, it["answer"]))
                    score_map.append(i)

                print(f"Pass@k Phase 2/2: Scoring ({len(score_prompts)} calls)...")
                score_responses = await _batch_call(
                    session, semaphore, api_url, model, score_prompts,
                    desc=f"Score ({dataset} pass@k)",
                )

                all_scores = [None] * len(flat_items)
                for j, resp in enumerate(score_responses):
                    all_scores[score_map[j]] = _parse_judgement(resp)

    problem_results = defaultdict(list)
    for item, score in zip(flat_items, all_scores):
        if score is None:
            continue
        meta = item["meta"]
        result = {
            "problem_idx": item["problem_idx"],
            "sample_idx": item["sample_idx"],
            "id": meta["id"],
            "question": meta["question"],
            "answer": meta["answer"],
            "prediction": item["prediction"],
            "accuracy": score,
            "correct": score > 0,
            **{k: v for k, v in meta.items()
               if k not in ("dataset", "id", "question", "answer")},
        }
        problem_results[item["problem_idx"]].append(result)

    pass_at_k_metrics = {}
    all_results = []
    for k in k_values:
        correct = sum(
            1 for samples in problem_results.values()
            if any(s["correct"] for s in samples[:k])
        )
        total = len(problem_results)
        pass_at_k_metrics[f"pass@{k}"] = correct / total if total else 0.0
        print(f"Pass@{k}: {pass_at_k_metrics[f'pass@{k}']:.4f} ({correct}/{total})")

        if k == 1:
            for samples in problem_results.values():
                rep = samples[0].copy()
                rep["pass_at_1"] = pass_at_k_metrics["pass@1"]
                all_results.append(rep)

    return {
        "results": all_results,
        "problem_results": dict(problem_results),
        "pass_at_k_metrics": pass_at_k_metrics,
        "total_problems": len(problem_results),
    }


def process_outputs_async(
    outputs: List[Dict],
    metadata: List[Dict],
    max_concurrent: int = 256,
    api_url: str = None,
) -> List[Dict]:
    """Drop-in replacement for process_outputs using async pipeline."""
    return asyncio.run(
        evaluate_dataset_async(outputs, metadata, max_concurrent, api_url)
    )


def process_outputs_pass_k_async(
    outputs: List[Dict],
    metadata: List[Dict],
    k_values: List[int],
    max_concurrent: int = 256,
    api_url: str = None,
) -> Dict:
    """Drop-in replacement for process_outputs_pass_k using async pipeline."""
    return asyncio.run(
        evaluate_dataset_pass_k_async(outputs, metadata, k_values, max_concurrent, api_url)
    )
