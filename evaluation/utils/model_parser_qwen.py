import time
import re
import os
import requests
import json


from utils.eval_templates import (
    get_gpt4_ICE, 
    get_gpt4_score_ICE, 
    get_gpt4_chartqa_score_ICE, 
    get_gpt4_logicvista_score_ICE,
    get_gpt4_r1_onevision_score_ICE,
    get_gpt4_extract_ICE
)

_session = None

def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64, pool_maxsize=128, max_retries=3
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session

def qwen_local_call(prompt, temperature=0.0, max_tokens=64):
    """Call locally deployed Qwen judge via vLLM OpenAI-compatible API."""
    api_url = os.getenv("QWEN_LOCAL_URL", "http://29.160.43.142:8000/v1/chat/completions")
    model = os.getenv("MODEL", "Qwen/Qwen2.5-72B-Instruct")

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = _get_session().post(api_url, json=data, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"Local Qwen API call failed: {e}")
        raise

def retry_with_backoff(func, max_retries=3, initial_delay=1, *args, **kwargs):
    """Retry with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 1.5
            else:
                raise

def build_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.\n\n
"""
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt

def build_chartqa_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a chart understanding question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.\n\n
"""
    demo_prompt = task_description
    examples = get_gpt4_chartqa_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt

def build_logicvista_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a logical reasoning question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.\n\n
"""
    demo_prompt = task_description
    examples = get_gpt4_logicvista_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt

def build_r1_onevision_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a multimodal reasoning question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.\n\n
"""
    demo_prompt = task_description
    examples = get_gpt4_r1_onevision_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt

def build_extract_prompt(prediction, question):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_mathverse_extract_prompt(prediction):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Model response: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_wemath_extract_prompt(prediction, question):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_chartqa_extract_prompt(prediction):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Model response: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_logicvista_extract_prompt(extraction: str, question: str) -> str:
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + extraction
    prompt += 'Extracted answer:'
    return prompt

def build_r1_onevision_extract_prompt(prediction):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Model response: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_mmk12_extract_prompt(prediction):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Model response: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_mmk12_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a multimodal math question from the MMK12 dataset. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm, or $$4$$ and 4.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.\n\n
"""
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt

def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format"""
    boxed_matches = re.findall(r'\\boxed{([^}]+)}', text)
    if boxed_matches:
        return boxed_matches[-1].strip(), True  # Return the last match
    return text, False

def llm_eval_score_retry(question, prediction, answer, dataset):
    """
    Evaluate prediction using local Qwen 72B model
    """
    
    if dataset.lower() == "mathverse":
        # extracted_answer, boxed_flag = extract_boxed_answer(prediction)
        # if not boxed_flag:
        #     extract_prompt = build_mathverse_extract_prompt(prediction)
            
        #     # Use retry mechanism to call local Qwen
        #     extracted_answer = retry_with_backoff(
        #         qwen_local_call,
        #         max_retries=3,
        #         initial_delay=2,
        #         prompt=extract_prompt,
        #         temperature=0.0,
        #         max_tokens=64
        #     )

        extract_prompt = build_mathverse_extract_prompt(prediction)
            
        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        )

        score_prompt = build_score_prompt(question, extracted_answer, answer)
        
        # Use retry mechanism to call local Qwen
        response_text = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

    elif dataset.lower() in ["mathvista", "mathvision"]:
        extract_prompt = build_extract_prompt(prediction, question)
        
        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        )

        if extracted_answer.strip() == answer:
            return 1.0
        else:
            return 0.0

    elif dataset.lower() == "wemath":
        extract_prompt = build_wemath_extract_prompt(prediction, question)
        
        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip().upper()
        
        if re.match(r'^[A-G]$', extracted_answer):
            accuracy = 1.0 if extracted_answer == answer else 0.0
            return accuracy
        else:
            return 0.0

    elif dataset.lower() == "chartqa":
        # extracted_answer, boxed_flag = extract_boxed_answer(prediction)
        # if not boxed_flag:
        extract_prompt = build_chartqa_extract_prompt(prediction)
        
        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        )

        score_prompt = build_chartqa_score_prompt(question, extracted_answer, answer)
        
        # Use retry mechanism to call local Qwen
        response_text = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

    elif dataset.lower() == "logicvista":
        extract_prompt = build_logicvista_extract_prompt(prediction, question)
        
        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip().upper()
        
        score_prompt = build_logicvista_score_prompt(question, extracted_answer, answer)
        
        # Use retry mechanism to call local Qwen
        response_text = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0
    
    elif dataset.lower() == "r1_onevision_bench":
        extract_prompt = build_r1_onevision_extract_prompt(prediction)
        
        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip().upper()

        score_prompt = build_r1_onevision_score_prompt(question, extracted_answer, answer)
        
        # Use retry mechanism to call local Qwen
        response_text = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0
    
    elif dataset.lower() == "mmk12":
        # extracted_answer, boxed_flag = extract_boxed_answer(prediction)
        # if not boxed_flag:
        extract_prompt = build_mmk12_extract_prompt(prediction)
        
        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        )

        score_prompt = build_mmk12_score_prompt(question, extracted_answer, answer)
        
        # Use retry mechanism to call local Qwen
        response_text = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0
    
    elif dataset.lower() in ("mmstar", "mmmu_pro"):
        extract_prompt = build_extract_prompt(prediction, question)

        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip().upper()

        if re.match(r'^[A-Z]$', extracted_answer):
            return 1.0 if extracted_answer == answer.strip().upper() else 0.0
        return 0.0

    elif dataset.lower() in ("visualpuzzles", "realworldqa"):
        extract_prompt = build_extract_prompt(prediction, question)

        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip().upper()

        score_prompt = build_score_prompt(question, extracted_answer, answer)

        response_text = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip()

        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

    elif dataset.lower() in ["math12k", "aime24", "math500", "gsm8k", "gpqa", "olympiadbench"]:
        extract_prompt = build_extract_prompt(prediction, question)

        # Use retry mechanism to call local Qwen
        extracted_answer = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip().upper()
        
        score_prompt = build_score_prompt(question, extracted_answer, answer)

        # Use retry mechanism to call local Qwen
        response_text = retry_with_backoff(
            qwen_local_call,
            max_retries=3,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0,
            max_tokens=64
        ).strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

if __name__ == "__main__":
    # Test the local Qwen API
    test_prompt = "What is 2+2? Answer with just the number."
    try:
        result = qwen_local_call(test_prompt, temperature=0.0, max_tokens=10)
        print(f"Local Qwen API test successful: {result}")
    except Exception as e:
        print(f"Local Qwen API test failed: {e}")
        print("Please check if your vLLM server is running on the correct port")