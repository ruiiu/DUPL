import time
import re
import os
import requests
import threading

from utils.eval_templates import (
    get_gpt4_ICE, 
    get_gpt4_score_ICE, 
    get_gpt4_chartqa_score_ICE, 
    get_gpt4_logicvista_score_ICE,
    get_gpt4_r1_onevision_score_ICE,
    get_gpt4_extract_ICE
)


class OptimizedQwenClient:
    def __init__(self, api_url: str = None, max_workers: int = 128):
        self.api_url = api_url or os.getenv("QWEN_LOCAL_URL", "http://29.232.228.185:8000/v1/chat/completions")
        self.model = os.getenv("MODEL", "Qwen/Qwen2.5-72B-Instruct")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        })

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2,
            max_retries=5
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.session.trust_env = False

    def single_call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 64) -> str:
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self.session.post(self.api_url, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def close(self):
        self.session.close()


_client = None
_lock = threading.Lock()

def get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = OptimizedQwenClient()
    return _client

def _call(prompt):
    """Call LLM judge with retries. Raises on total failure."""
    client = get_client()
    delay = 0.3
    for attempt in range(5):
        try:
            return client.single_call(prompt)
        except Exception:
            if attempt < 4:
                time.sleep(delay)
                delay *= 1.5
            else:
                raise

def _score(score_prompt):
    """Call LLM judge for 0/1 scoring."""
    resp = _call(score_prompt).strip()
    if resp in ('0', '1'):
        return int(resp)
    return 0.0

from utils.model_parser_qwen import (
    build_score_prompt,
    build_chartqa_score_prompt,
    build_logicvista_score_prompt,
    build_r1_onevision_score_prompt,
    build_extract_prompt,
    build_mathverse_extract_prompt,
    build_chartqa_extract_prompt,
    build_wemath_extract_prompt,
    build_logicvista_extract_prompt,
    build_r1_onevision_extract_prompt,
    build_mmk12_extract_prompt,
    build_mmk12_score_prompt,
)

def llm_eval_score_retry(question, prediction, answer, dataset):
    ds = dataset.lower()

    if ds == "mathverse":
        extracted = _call(build_mathverse_extract_prompt(prediction))
        return _score(build_score_prompt(question, extracted, answer))

    elif ds in ("mathvista", "mathvision"):
        extracted = _call(build_extract_prompt(prediction, question))
        return 1.0 if extracted.strip() == answer else 0.0

    elif ds == "wemath":
        extracted = _call(build_wemath_extract_prompt(prediction, question)).strip().upper()
        if re.match(r'^[A-G]$', extracted):
            return 1.0 if extracted == answer else 0.0
        return 0.0

    elif ds == "chartqa":
        extracted = _call(build_chartqa_extract_prompt(prediction))
        return _score(build_chartqa_score_prompt(question, extracted, answer))

    elif ds == "logicvista":
        extracted = _call(build_logicvista_extract_prompt(prediction, question)).strip().upper()
        return _score(build_logicvista_score_prompt(question, extracted, answer))

    elif ds == "r1_onevision_bench":
        extracted = _call(build_r1_onevision_extract_prompt(prediction)).strip().upper()
        return _score(build_r1_onevision_score_prompt(question, extracted, answer))

    elif ds == "mmk12":
        extracted = _call(build_mmk12_extract_prompt(prediction))
        return _score(build_mmk12_score_prompt(question, extracted, answer))

    elif ds in ("math12k", "aime24", "math500", "gsm8k", "gpqa", "olympiadbench"):
        extracted = _call(build_extract_prompt(prediction, question)).strip().upper()
        return _score(build_score_prompt(question, extracted, answer))

    else:
        return 0.0

def cleanup_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None
