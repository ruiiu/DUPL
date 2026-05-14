import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset

def load_geo3k_dataset(data_path: str) -> List[Dict]:
    """Load Geo3K dataset from Hugging Face"""
    dataset_raw = load_dataset("lmms-lab/LLaVA-OneVision-Data", "geo3k", split="train")
    
    dataset = []
    for item in tqdm(dataset_raw, desc="Loading Geo3K data from Hugging Face"):
        # Extract conversation data
        conversations = item["conversations"]
        question = conversations[0]["value"]
        answer = conversations[1]["value"]

        dataset.append({
            "id": item["id"],
            "image_path": item["image"],  # This will be a PIL Image object
            "question": question,
            "answer": answer,
            "dataset": "geo3k"
        })
    
    return dataset

def load_wemath_dataset(data_path: str) -> List[Dict]:
    """Load WeMath dataset from Hugging Face"""
    dataset_raw = load_dataset("We-Math/We-Math", split="testmini")
    
    dataset = []
    for item in tqdm(dataset_raw, desc="Loading WeMath data from Hugging Face"):
        dataset.append({
            "id": item["ID"] + "@" + item["key"],
            "image_path": item["image_path"],
            "question": f"{item['question']}\n\nOptions: {item['option']}",
            "answer": item["answer"],
            "dataset": "wemath"
        })
    
    return dataset

def load_mathvista_dataset(data_path: str) -> List[Dict]:
    """Load MathVista dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("AI4Math/MathVista", split="testmini")
    
    dataset = []
    mapping = {
        "0": "A", "1": "B", "2": "C", "3": "D",
        "4": "E", "5": "F", "6": "G", "7": "H"
    }
    
    for item in dataset_raw:
        if item["question_type"] == "multi_choice":
            idx = item["choices"].index(item["answer"])
            answer = mapping[str(idx)]
        else:
            answer = item["answer"]
        
        dataset.append({
            "id": item.get("pid", ""),
            "image_path": item["decoded_image"],  # Use decoded_image from HF dataset
            "question": item["query"],
            "answer": answer,
            "task": item["metadata"]["task"],
            "dataset": "mathvista"
        })
    
    return dataset

def load_mathverse_dataset(data_path: str) -> List[Dict]:
    """Load MathVerse dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")
    
    dataset = []
    for item in dataset_raw:
        dataset.append({
            "id": item.get("sample_index", ""),
            "image_path": item["image"],  # Use image from HF dataset
            "question": item["query_cot"],
            "question_for_eval": item["question_for_eval"],
            "answer": item["answer"],
            "problem_version": item["problem_version"],
            "dataset": "mathverse"
        })
    
    return dataset

def load_mathvision_dataset(data_path: str) -> List[Dict]:
    """Load MathVision dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("MathLLMs/MathVision", split="test")
    
    dataset = []
    for item in dataset_raw:
        # Determine question type based on options
        options = item.get("options", [])
        is_multiple_choice = len(options) > 0
        
        dataset.append({
            "id": item.get("id", ""),
            "image_path": item["decoded_image"],  # Use decoded_image from HF dataset
            "question": item["question"],
            "answer": item["answer"],
            "options": options,  # Include options for evaluation
            "is_multiple_choice": is_multiple_choice,
            "subject": item.get("subject", "unknown"),
            "level": item.get("level", 1),
            "dataset": "mathvision"
        })
    
    return dataset

def load_hallubench_dataset(data_path: str) -> List[Dict]:
    """Load Hallubench dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("lmms-lab/HallusionBench", split="image")
    
    dataset = []
    for item in dataset_raw:
        if not item["filename"]:
            continue
        
        if "?" in item["question"]:
            question = item["question"].split("?")[:-1][0]
        else:
            question = item["question"]
        question += "? You final answer can only be yes or no."
        gt_answer = "yes" if int(item["gt_answer"]) == 1 else "no"
        sid, fid, qid = item["set_id"], item["figure_id"], item["question_id"]
        dataset.append({
            "id": f"{sid}_{fid}_{qid}",
            "image_path": item["image"],  # Hugging Face provides the image directly
            "question": question,
            "question_for_eval": question,
            "answer": gt_answer,
            "problem_version": item["subcategory"],
            "dataset": "hallubench"
        })
    
    return dataset

def load_chartqa_dataset(data_path: str) -> List[Dict]:
    """Load ChartQA dataset from Hugging Face"""
    # Use HuggingFaceM4/ChartQA dataset directly
    dataset_raw = load_dataset("HuggingFaceM4/ChartQA", split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        # Extract data from HuggingFaceM4/ChartQA format
        question = item["query"]
        answer = item["label"][0] if item["label"] else ""  # Get first (and usually only) label
        
        dataset.append({
            "id": f"chartqa_{idx}",
            "image_path": item["image"],  # This will be a PIL Image object
            "question": question,
            "answer": answer,
            "problem_version": "chartqa",
            "dataset": "chartqa"
        })
    
    return dataset

def load_logicvista_dataset(data_path: str) -> List[Dict]:
    """Load LogicVista dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("lscpku/LogicVista", split="test")

    dataset = []
    for item in dataset_raw:
        dataset.append({
            "id": item["id"],
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "logicvista"
        })
    
    return dataset


def load_r1_onevision_bench(data_path: str) -> List[Dict]:
    """Load R1-Onevision-Bench dataset from Hugging Face"""
    # Use Hugging Face dataset directly (only has 'train' split)
    dataset_raw = load_dataset("Fancy-MLLM/R1-Onevision-Bench", split="train")

    dataset = []
    for item in tqdm(dataset_raw, desc="Loading R1-Onevision-Bench from HuggingFace"):
        # Decode base64 image string to PIL Image
        import base64
        from io import BytesIO
        
        try:
            image_data = base64.b64decode(item["image"])
            image = Image.open(BytesIO(image_data))
            
            dataset.append({
                "id": item["index"],
                "image_path": image,  # Store PIL Image directly
                "question": item["question"].split("Question: ")[-1].strip(),
                "answer": item["answer"],
                "level": item["level"],
                "category": item["category"],
                "dataset": "r1-onevision-bench"
            })
        except Exception as e:
            print(f"Error decoding image for sample {item['index']}: {e}")
            continue

    return dataset


def load_math12k_dataset(data_path: str) -> List[Dict]:
    """Load Math12K dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("hiyouga/math12k", split="test")
    
    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": idx,
            "question": item["problem"],
            "answer": item["answer"],
            "dataset": "math12k"
        })
    return dataset

def load_aime24_dataset(data_path: str) -> List[Dict]:
    """Load AIME24 dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    dataset = []
    for item in dataset_raw:
        dataset.append({
            "id": item["ID"],
            "question": item["Problem"],
            "answer": item["Answer"],
            "solution": item["Solution"],
            "dataset": "AIME24"
        })
    return dataset

def load_math500_dataset(data_path: str) -> List[Dict]:
    """Load MATH500 dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": idx,
            "question": item["problem"],
            "answer": item["answer"],
            "solution": item["solution"],
            "dataset": "MATH500"
        })
    return dataset

def load_gsm8k_dataset(data_path: str) -> List[Dict]:
    """Load GSM8K dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("openai/gsm8k", 'main', split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        raw_solution = item["answer"]
        think_process = raw_solution.split("\n####")[0].strip()
        gt_answer = raw_solution.split("\n####")[-1].strip()
        dataset.append({
            "id": idx,
            "question": item["question"],
            "answer": gt_answer,
            "solution": think_process + "\n" + f"The final answer is \\boxed{{{gt_answer}}}",
            "dataset": "GSM8K"
        })
    return dataset

def load_gpqa_dataset(data_path: str) -> List[Dict]:
    """Load GPQA dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("hendrydong/gpqa_diamond", split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        raw_problem = item["problem"]
        question = raw_problem.replace("\n\nPlease write your final answer in the form of \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}", "")
        dataset.append({
            "id": idx,
            "question": question,
            "answer": item["solution"],
            "dataset": "GPQA_diamond"
        })
    return dataset

def load_olympiadbench_dataset(data_path: str) -> List[Dict]:
    """Load OlympiadBench dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("KbsdJames/Omni-MATH", split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": idx,
            "question": item["problem"],
            "answer": item["answer"],
            "solution": item["solution"],
            "dataset": "OlympiadBench"
        })
    return dataset

def load_mmstar_dataset(data_path: str) -> List[Dict]:
    """Load MMStar dataset from Hugging Face (1.5k val, MCQ letters)"""
    dataset_raw = load_dataset("Lin-Chen/MMStar", split="val")

    dataset = []
    for item in tqdm(dataset_raw, desc="Loading MMStar from HuggingFace"):
        dataset.append({
            "id": item["index"],
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "category": item.get("category", ""),
            "dataset": "mmstar"
        })
    return dataset


def load_visualpuzzles_dataset(data_path: str) -> List[Dict]:
    """Load VisualPuzzles dataset from Hugging Face (1.17k, train split)"""
    dataset_raw = load_dataset("neulab/VisualPuzzles", split="train")

    dataset = []
    for item in tqdm(dataset_raw, desc="Loading VisualPuzzles from HuggingFace"):
        question = item["question"]
        options = item.get("options")
        if options:
            opts_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
            question = f"{question}\n\nOptions:\n{opts_str}"
        dataset.append({
            "id": item["id"],
            "image_path": item["image"],
            "question": question,
            "answer": item["answer"],
            "category": item.get("category", ""),
            "dataset": "visualpuzzles"
        })
    return dataset


def load_realworldqa_dataset(data_path: str) -> List[Dict]:
    """Load RealworldQA dataset from Hugging Face (765 rows, test split)"""
    dataset_raw = load_dataset("xai-org/RealworldQA", split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": f"rwqa_{idx}",
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "realworldqa"
        })
    return dataset


def load_mmmu_pro_dataset(data_path: str) -> List[Dict]:
    """Load MMMU_Pro vision config from Hugging Face (1.73k, test split).
    Vision subset: question and options are embedded in the image."""
    dataset_raw = load_dataset("MMMU/MMMU_Pro", "vision", split="test")

    dataset = []
    for item in tqdm(dataset_raw, desc="Loading MMMU_Pro vision from HuggingFace"):
        dataset.append({
            "id": item["id"],
            "image_path": item["image"],
            "question": "Look at the image and answer the question shown. Provide only the letter of the correct option.",
            "answer": item["answer"],
            "subject": item.get("subject", ""),
            "dataset": "mmmu_pro"
        })
    return dataset


def load_mmk12_dataset(data_path: str) -> List[Dict]:
    """Load MMK12 dataset from Hugging Face"""
    # Use Hugging Face dataset directly
    dataset_raw = load_dataset("FanqingM/MMK12", split="test")
    
    dataset = []
    for item in tqdm(dataset_raw, desc="Loading MMK12 data from Hugging Face"):
        # Prepare the data entry
        data_entry = {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "subject": item.get("subject", "unknown"),
            "dataset": "mmk12"
        }
        
        # Add image if it exists (some questions may not have images)
        if item.get("image") is not None:
            data_entry["image_path"] = item["image"]
        
        dataset.append(data_entry)
    
    return dataset