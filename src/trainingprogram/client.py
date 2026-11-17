from __future__ import annotations

import json
import time
import httpx


class ModelClient:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="qwen", timeout=300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, max_tokens=2048, seed=2026, temperature=0, top_p=None):
        if not 0 <= temperature <= 2 or (top_p is not None and not 0 < top_p <= 1):
            raise ValueError("Invalid sampling parameters")
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "seed": seed, "max_tokens": max_tokens, "stream": True,
                   "stream_options": {"include_usage": True},
                   "chat_template_kwargs": {"enable_thinking": False}}
        if top_p is not None:
            payload["top_p"] = top_p
        text, reasoning, usage, finish, ttft = "", "", {}, None, None
        start = time.perf_counter()
        # No automatic retries: one sample is one model request.
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            with client.stream("POST", self.base_url + "/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    obj = json.loads(raw)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices", []):
                        delta = choice.get("delta", {})
                        token = delta.get("content") or ""
                        thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if (token or thought) and ttft is None:
                            ttft = time.perf_counter() - start
                        text += token
                        reasoning += thought
                        finish = choice.get("finish_reason") or finish
        elapsed = time.perf_counter() - start
        if finish is None:
            raise RuntimeError("Incomplete SSE response: missing finish_reason")
        return {"text": text, "reasoning": reasoning, "usage": usage, "finish_reason": finish,
                "ttft_s": ttft, "elapsed_s": elapsed,
                "output_tokens_per_s": (usage["completion_tokens"] / elapsed) if usage.get("completion_tokens") else None}

    def verify(self, expected_alias):
        with httpx.Client(timeout=20, trust_env=False) as client:
            r = client.get(self.base_url + "/models")
            r.raise_for_status()
            ids = [x["id"] for x in r.json()["data"]]
        if ids != [expected_alias]:
            raise ValueError(f"Wrong model service: expected {expected_alias}, got {ids}")
