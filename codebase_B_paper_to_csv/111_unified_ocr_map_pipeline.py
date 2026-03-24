from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import math
import os
import re
import sys
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd
from rate_limit_utils import (
    RateLimiterRegistry,
    estimate_text_tokens,
    load_rate_limit_overrides,
    normalize_headers,
    parse_retry_after_seconds,
    resolve_model_rate_limit_config,
    summarize_usage_records,
)

logger = logging.getLogger("unified_ocr_map")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RATE_LIMITER_REGISTRY = RateLimiterRegistry()
PHX_CHECKLIST_TRIGGER = "다음과 같은 질환을 앓고 있거나 과거에 앓은 적이 있습니까?"
PHX_EXPECTED_YES_NO_TOTAL = 28
PHX_OCR_MAX_ATTEMPTS = 3


def resolve_script_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def configure_logging(output_dir: Path, debug: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "unified_pipeline.log"
    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if not debug:
        for name in (
            "httpx",
            "httpcore",
            "urllib3",
            "huggingface_hub",
            "transformers",
            "accelerate",
            "torch",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Logging initialized: %s", log_path)
    return log_path


class TextAgent(Protocol):
    async def atext(self, system_prompt: str, user_text: str, label: str = "") -> str: ...


def parse_model_id_list(raw: str) -> List[str]:
    parts = [p.strip() for p in str(raw or "").split(",")]
    return [p for p in parts if p]


def analyze_phx_yes_no_markers(text: str) -> Dict[str, Any]:
    yes_count = text.count("[Yes]")
    no_count = text.count("[No]")
    total = yes_count + no_count
    contains_trigger = PHX_CHECKLIST_TRIGGER in text
    return {
        "contains_trigger": contains_trigger,
        "yes_count": yes_count,
        "no_count": no_count,
        "total": total,
        "is_complete": (not contains_trigger) or total == PHX_EXPECTED_YES_NO_TOTAL,
    }


def format_phx_retry_prompt(user_text: str, attempt_idx: int) -> str:
    label = f"medical history [yes]/[no] mark omission retrial {int(attempt_idx)}"
    body = str(user_text or "").strip()
    if not body:
        return label
    return f"{label}\n\n{body}"


class RemoteOpenAIResponsesTextAgent:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        max_inflight: int,
        timeout_sec: float,
        max_retries: int,
        api_key_env: str,
        rate_limiter: Any = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.timeout_sec = max(5.0, float(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.api_key_env = str(api_key_env or "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
        self._rate_limiter = rate_limiter
        self._sem = asyncio.Semaphore(max(1, int(max_inflight)))
        self._usage_lock = threading.Lock()
        self._usage_records: List[Dict[str, Any]] = []

    @staticmethod
    def _coerce_int(value: Any) -> int:
        try:
            if value is None:
                return 0
            return int(value)
        except Exception:
            return 0

    def usage_records(self) -> List[Dict[str, Any]]:
        with self._usage_lock:
            return [dict(item) for item in self._usage_records]

    @staticmethod
    def _extract_usage(payload: Dict[str, Any]) -> Dict[str, Any]:
        usage = payload.get("usage") or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return {
            "input_tokens": RemoteOpenAIResponsesTextAgent._coerce_int(usage.get("input_tokens")),
            "output_tokens": RemoteOpenAIResponsesTextAgent._coerce_int(usage.get("output_tokens")),
            "total_tokens": RemoteOpenAIResponsesTextAgent._coerce_int(usage.get("total_tokens")),
            "cached_input_tokens": RemoteOpenAIResponsesTextAgent._coerce_int(input_details.get("cached_tokens")),
            "reasoning_output_tokens": RemoteOpenAIResponsesTextAgent._coerce_int(output_details.get("reasoning_tokens")),
        }

    def _record_usage(
        self,
        payload: Dict[str, Any],
        label: str,
        request_id: str,
        started_at: float,
        rate_limit_meta: Optional[Dict[str, Any]],
    ) -> None:
        usage = self._extract_usage(payload)
        usage["label"] = str(label or "").strip() or "text"
        usage["model"] = self.model_id
        usage["request_id"] = request_id
        usage["started_at"] = started_at
        usage["finished_at"] = time.time()
        if rate_limit_meta:
            usage.update(rate_limit_meta)
        with self._usage_lock:
            self._usage_records.append(usage)

    def _read_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable: {self.api_key_env}")
        return api_key

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        if isinstance(output_text, list):
            parts = [str(x).strip() for x in output_text if str(x).strip()]
            if parts:
                return "\n".join(parts).strip()

        texts: List[str] = []
        for item in payload.get("output", []) or []:
            for content in item.get("content", []) or []:
                ctype = str(content.get("type", "")).lower()
                if ctype in {"output_text", "text"}:
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
        if texts:
            return "\n".join(texts).strip()
        raise RuntimeError("OpenAI response did not contain text")

    def _http_json(self, req: urlrequest.Request) -> Tuple[Dict[str, Any], Dict[str, str]]:
        with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read()
            headers = normalize_headers(resp.headers)
        return json.loads(raw.decode("utf-8")), headers

    def _text_sync(self, system_prompt: str, user_text: str, label: str) -> str:
        api_key = self._read_api_key()
        estimated_tokens = estimate_text_tokens(system_prompt, user_text) + self.max_new_tokens
        started_at = time.time()
        request_id = ""
        if self._rate_limiter is not None:
            request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label=label)
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                },
            ],
            "max_output_tokens": self.max_new_tokens,
        }
        if self.temperature > 0.0:
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p

        req = urlrequest.Request(
            url="https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                resp_payload, headers = self._http_json(req)
                rate_limit_meta = None
                if self._rate_limiter is not None and request_id:
                    rate_limit_meta = self._rate_limiter.release(
                        request_id,
                        actual_tokens=int((resp_payload.get("usage") or {}).get("total_tokens") or 0),
                        headers=headers,
                        status="ok",
                    )
                self._record_usage(resp_payload, label, request_id, started_at, rate_limit_meta)
                return self._extract_text(resp_payload)
            except urlerror.HTTPError as exc:
                body = ""
                headers = normalize_headers(getattr(exc, "headers", None))
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = str(exc)
                if self._rate_limiter is not None and request_id:
                    self._rate_limiter.release(
                        request_id,
                        actual_tokens=None,
                        headers=headers,
                        status=f"http_{exc.code}",
                        error=body[:500],
                    )
                    request_id = ""
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = parse_retry_after_seconds(headers, body)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    logger.warning(
                        "Transient OpenAI text error (status=%s, attempt=%d/%d). Retrying in %.1fs.",
                        exc.code,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label=label)
                    continue
                raise RuntimeError(f"OpenAI text request failed: status={exc.code} body={body}") from exc
            except Exception:
                if self._rate_limiter is not None and request_id:
                    self._rate_limiter.release(
                        request_id,
                        actual_tokens=None,
                        status="exception",
                        error="local_exception",
                    )
                    request_id = ""
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient OpenAI text failure (attempt=%d/%d). Retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label=label)
                    continue
                raise
        raise RuntimeError("OpenAI text request failed after retries")

    async def atext(self, system_prompt: str, user_text: str, label: str = "") -> str:
        async with self._sem:
            return await asyncio.to_thread(self._text_sync, system_prompt, user_text, label)


class RemoteGeminiTextAgent:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        max_inflight: int,
        timeout_sec: float,
        max_retries: int,
        api_key_env: str,
        rate_limiter: Any = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.timeout_sec = max(5.0, float(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.api_key_env = str(api_key_env or "GOOGLE_API_KEY").strip() or "GOOGLE_API_KEY"
        self._rate_limiter = rate_limiter
        self._sem = asyncio.Semaphore(max(1, int(max_inflight)))
        self._usage_lock = threading.Lock()
        self._usage_records: List[Dict[str, Any]] = []

    def _read_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable: {self.api_key_env}")
        return api_key

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        texts: List[str] = []
        for cand in payload.get("candidates", []) or []:
            content = cand.get("content") or {}
            for part in content.get("parts", []) or []:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if texts:
            return "\n".join(texts).strip()
        raise RuntimeError("Gemini response did not contain text")

    @staticmethod
    def _extract_usage(payload: Dict[str, Any]) -> Dict[str, Any]:
        usage = payload.get("usageMetadata") or {}
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        total_tokens = int(usage.get("totalTokenCount") or (input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
        }

    def usage_records(self) -> List[Dict[str, Any]]:
        with self._usage_lock:
            return [dict(item) for item in self._usage_records]

    def _record_usage(
        self,
        payload: Dict[str, Any],
        label: str,
        request_id: str,
        started_at: float,
        rate_limit_meta: Optional[Dict[str, Any]],
    ) -> None:
        usage = self._extract_usage(payload)
        usage["label"] = str(label or "").strip() or "text"
        usage["model"] = self.model_id
        usage["request_id"] = request_id
        usage["started_at"] = started_at
        usage["finished_at"] = time.time()
        if rate_limit_meta:
            usage.update(rate_limit_meta)
        with self._usage_lock:
            self._usage_records.append(usage)

    def _http_json(self, req: urlrequest.Request) -> Tuple[Dict[str, Any], Dict[str, str]]:
        with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read()
            headers = normalize_headers(resp.headers)
        return json.loads(raw.decode("utf-8")), headers

    def _text_sync(self, system_prompt: str, user_text: str, label: str) -> str:
        api_key = self._read_api_key()
        estimated_tokens = estimate_text_tokens(system_prompt, user_text) + self.max_new_tokens
        started_at = time.time()
        request_id = ""
        if self._rate_limiter is not None:
            request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label=label)
        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_new_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }
        req = urlrequest.Request(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                resp_payload, headers = self._http_json(req)
                rate_limit_meta = None
                if self._rate_limiter is not None and request_id:
                    rate_limit_meta = self._rate_limiter.release(
                        request_id,
                        actual_tokens=int((resp_payload.get("usageMetadata") or {}).get("totalTokenCount") or 0),
                        headers=headers,
                        status="ok",
                    )
                self._record_usage(resp_payload, label, request_id, started_at, rate_limit_meta)
                return self._extract_text(resp_payload)
            except urlerror.HTTPError as exc:
                body = ""
                headers = normalize_headers(getattr(exc, "headers", None))
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = str(exc)
                if self._rate_limiter is not None and request_id:
                    self._rate_limiter.release(
                        request_id,
                        actual_tokens=None,
                        headers=headers,
                        status=f"http_{exc.code}",
                        error=body[:500],
                    )
                    request_id = ""
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = parse_retry_after_seconds(headers, body)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    logger.warning(
                        "Transient Gemini text error (status=%s, attempt=%d/%d). Retrying in %.1fs.",
                        exc.code,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label=label)
                    continue
                raise RuntimeError(f"Gemini text request failed: status={exc.code} body={body}") from exc
            except Exception:
                if self._rate_limiter is not None and request_id:
                    self._rate_limiter.release(
                        request_id,
                        actual_tokens=None,
                        status="exception",
                        error="local_exception",
                    )
                    request_id = ""
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient Gemini text failure (attempt=%d/%d). Retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label=label)
                    continue
                raise
        raise RuntimeError("Gemini text request failed after retries")

    async def atext(self, system_prompt: str, user_text: str, label: str = "") -> str:
        async with self._sem:
            return await asyncio.to_thread(self._text_sync, system_prompt, user_text, label)


class LocalHFTextAgent:
    def __init__(
        self,
        model_id: str,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        max_inflight: int,
        trust_remote_code: bool,
        attn_implementation: str,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = (attn_implementation or "").strip()
        self._sem = asyncio.Semaphore(max(1, int(max_inflight)))
        self._load_lock = asyncio.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._usage_lock = threading.Lock()
        self._usage_records: List[Dict[str, Any]] = []

    def _resolve_dtype(self, torch_mod: Any) -> Any:
        alias = (self.dtype or "auto").strip().lower()
        if alias in {"", "auto"}:
            return "auto"
        if alias in {"bf16", "bfloat16"}:
            return torch_mod.bfloat16
        if alias in {"fp16", "float16", "half"}:
            return torch_mod.float16
        if alias in {"fp32", "float32"}:
            return torch_mod.float32
        raise ValueError(f"Unsupported --dtype value: {self.dtype}")

    async def warmup(self) -> None:
        await self._ensure_loaded()

    async def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None and self._torch is not None:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_value = self._resolve_dtype(torch)
        logger.info("Loading local text model: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype_value,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()
        self._torch = torch
        logger.info("Local text model ready: %s", self.model_id)

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> Any:
        assert self._tokenizer is not None
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            return self._tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _model_device(self) -> Any:
        assert self._model is not None and self._torch is not None
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")

    @staticmethod
    def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
        if hasattr(inputs, "items"):
            moved: Dict[str, Any] = {}
            for key, value in inputs.items():
                moved[key] = value.to(device) if hasattr(value, "to") else value
            return moved
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return inputs

    def usage_records(self) -> List[Dict[str, Any]]:
        with self._usage_lock:
            return [dict(item) for item in self._usage_records]

    def _record_usage(self, *, label: str, input_tokens: int, output_tokens: int) -> None:
        usage = {
            "input_tokens": max(0, int(input_tokens)),
            "output_tokens": max(0, int(output_tokens)),
            "total_tokens": max(0, int(input_tokens)) + max(0, int(output_tokens)),
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
            "label": str(label or "").strip() or "text",
            "model": self.model_id,
        }
        with self._usage_lock:
            self._usage_records.append(usage)

    def _text_sync(self, system_prompt: str, user_text: str, label: str) -> str:
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        raw_inputs = self._apply_chat_template(messages)
        device = self._model_device()
        inputs = self._move_inputs_to_device(raw_inputs, device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with self._torch.inference_mode():
            if isinstance(inputs, dict):
                outputs = self._model.generate(**inputs, **gen_kwargs)
                prompt_len = int(inputs["input_ids"].shape[1])
            else:
                outputs = self._model.generate(inputs, **gen_kwargs)
                prompt_len = int(inputs.shape[1])
        trimmed = outputs[:, prompt_len:]
        output_tokens = int(trimmed.shape[1]) if hasattr(trimmed, "shape") and len(trimmed.shape) >= 2 else 0
        self._record_usage(label=label, input_tokens=prompt_len, output_tokens=output_tokens)
        texts = self._tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (texts[0] if texts else "").strip()

    async def atext(self, system_prompt: str, user_text: str, label: str = "") -> str:
        await self._ensure_loaded()
        async with self._sem:
            return await asyncio.to_thread(self._text_sync, system_prompt, user_text, label)


def summarize_openai_usage(*agents: Optional[TextAgent]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    for agent in agents:
        if agent is None:
            continue
        if id(agent) in seen_ids:
            continue
        seen_ids.add(id(agent))
        getter = getattr(agent, "usage_records", None)
        if callable(getter):
            records.extend(getter())

    if not records:
        return {}

    deduped: List[Dict[str, Any]] = []
    seen_requests = set()
    for rec in records:
        request_id = str(rec.get("request_id") or "")
        if request_id and request_id in seen_requests:
            continue
        if request_id:
            seen_requests.add(request_id)
        deduped.append(rec)
    records = deduped

    by_label: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        label = str(rec.get("label") or "text")
        slot = by_label.setdefault(
            label,
            {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_output_tokens": 0,
            },
        )
        slot["requests"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_output_tokens"):
            slot[key] += int(rec.get(key) or 0)

    model = str(records[0].get("model") or "")
    summary = {
        "model": model,
        "request_count": len(records),
        "input_tokens": sum(int(rec.get("input_tokens") or 0) for rec in records),
        "output_tokens": sum(int(rec.get("output_tokens") or 0) for rec in records),
        "total_tokens": sum(int(rec.get("total_tokens") or 0) for rec in records),
        "cached_input_tokens": sum(int(rec.get("cached_input_tokens") or 0) for rec in records),
        "reasoning_output_tokens": sum(int(rec.get("reasoning_output_tokens") or 0) for rec in records),
        "by_label": by_label,
        "records": records,
    }
    return summary


def collect_usage_records(*agents: Optional[TextAgent]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_agent_ids = set()
    for agent in agents:
        if agent is None or id(agent) in seen_agent_ids:
            continue
        seen_agent_ids.add(id(agent))
        getter = getattr(agent, "usage_records", None)
        if callable(getter):
            records.extend(getter())
    return records


def _extract_json_dict(pipeline_mod: Any, raw_text: str) -> Dict[str, Any]:
    obj = pipeline_mod.safe_extract_json(raw_text)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def _looks_incomplete_json(raw_text: str) -> bool:
    txt = str(raw_text or "").strip()
    if not txt:
        return True
    if txt[-1:] not in {"}", "]"}:
        return True
    if txt.count("{") > txt.count("}"):
        return True
    if txt.count("[") > txt.count("]"):
        return True
    return False


def _summarize_json_failure(raw_text: str, exc: Exception) -> str:
    flags: List[str] = [type(exc).__name__]
    if _looks_incomplete_json(raw_text):
        flags.append("incomplete_json")
    if not str(raw_text or "").strip():
        flags.append("empty_output")
    return ",".join(flags)


async def text_to_json(
    llm: TextAgent,
    system_prompt: str,
    user_text: str,
    pipeline_mod: Any,
    schema_hint: str,
    max_attempts: int,
    label: str,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    last_raw = ""
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        retry_note = ""
        if attempt > 1:
            retry_note = (
                "\n\nIMPORTANT RETRY NOTICE:\n"
                "Your previous answer was malformed or incomplete JSON. "
                "Return ONLY one complete JSON object that follows the required schema exactly. "
                "Do not include markdown, prose, or trailing text."
            )
        raw = await llm.atext(system_prompt=system_prompt, user_text=user_text + retry_note, label=label)
        last_raw = raw
        try:
            return _extract_json_dict(pipeline_mod, raw)
        except Exception as exc:
            last_error = exc

        try:
            fixed = await llm.atext(
                system_prompt="You are a strict JSON repair assistant.",
                user_text=(
                    "Return ONLY valid JSON. No markdown, no prose.\n\n"
                    f"Required schema:\n{schema_hint}\n\n"
                    f"Previous invalid output:\n{raw}"
                ),
                label=f"{label}_repair",
            )
            last_raw = fixed
            return _extract_json_dict(pipeline_mod, fixed)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning(
                    "Invalid %s JSON. Retrying full generation (%d/%d): %s",
                    label,
                    attempt,
                    attempts,
                    _summarize_json_failure(last_raw, exc),
                )

    raise RuntimeError(
        f"Could not obtain valid {label} JSON after {attempts} attempts: "
        f"{_summarize_json_failure(last_raw, last_error or RuntimeError('unknown'))}"
    ) from last_error


def build_map_schema_hint() -> str:
    return (
        '{"CDM_KEY": {"CDM_Context": "<cdm context>", "value": <value>, '
        '"input_context": {"filled_by": "doctor|patient", '
        '"question": "<exact question/context that matches to the CDM key>"}}}'
    )


def build_route_schema_hint() -> str:
    return '{"route":"<route>","confidence":"high|medium|low","reason":"<short reason>"}'


def build_category_split_schema_hint() -> str:
    return (
        '{"phx_habit":[{"start_line":1,"end_line":2}],'
        '"sleep_behavior":[{"start_line":3,"end_line":4}],'
        '"psg":[{"start_line":5,"end_line":6}],'
        '"psqi":[],'
        '"sss":[],'
        '"ess":[],'
        '"fss":[],'
        '"berlin":[],'
        '"isi":[],'
        '"rls":[],'
        '"rbd":[],'
        '"phq":[],'
        '"bdi":[],'
        '"qol":[]}'
    )


async def route_ocr_text(
    llm: TextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    max_attempts: int,
) -> Dict[str, Any]:
    direct_or_heuristic = pipeline_mod.classify_map_route_heuristic(ocr_text)
    if str(direct_or_heuristic.get("reason") or "") == "direct_keyword_router":
        return direct_or_heuristic
    try:
        raw = await text_to_json(
            llm=llm,
            system_prompt=pipeline_mod.MAP_ROUTE_SYSTEM,
            user_text=pipeline_mod.build_route_user_prompt(ocr_text),
            pipeline_mod=pipeline_mod,
            schema_hint=build_route_schema_hint(),
            max_attempts=max_attempts,
            label="map_route",
        )
        return pipeline_mod.normalize_route_decision(raw, ocr_text)
    except Exception as exc:
        fallback = direct_or_heuristic
        fallback["reason"] = f"heuristic_fallback_after_route_error:{type(exc).__name__}"
        logger.warning("Route classifier failed, falling back to heuristic router: %s", exc)
        return fallback


async def split_patient_ocr_categories(
    llm: TextAgent,
    pipeline_mod: Any,
    image_name_text_pairs: Sequence[Tuple[str, str]],
    max_attempts: int,
) -> List[Dict[str, Any]]:
    merged_ocr_text = pipeline_mod.merge_ocr_text_blocks(list(image_name_text_pairs))
    try:
        raw = await text_to_json(
            llm=llm,
            system_prompt=pipeline_mod.CATEGORY_SPLIT_SYSTEM,
            user_text=pipeline_mod.build_category_split_user_prompt(
                merged_ocr_text=merged_ocr_text,
                source_images=[name for name, _ in image_name_text_pairs],
            ),
            pipeline_mod=pipeline_mod,
            schema_hint=build_category_split_schema_hint(),
            max_attempts=max_attempts,
            label="category_split",
        )
        records = pipeline_mod.normalize_category_split_decision(raw, image_name_text_pairs)
        assigned_ranges = pipeline_mod._extract_assigned_ranges_from_records(records)
        leftover_ranges = pipeline_mod.extract_uncategorized_informative_ranges(
            merged_ocr_text=merged_ocr_text,
            assigned_ranges=assigned_ranges,
        )
        if leftover_ranges:
            try:
                rescue_raw = await text_to_json(
                    llm=llm,
                    system_prompt=pipeline_mod.CATEGORY_SPLIT_SYSTEM,
                    user_text=pipeline_mod.build_leftover_rescue_user_prompt(
                        merged_ocr_text=merged_ocr_text,
                        leftover_ranges=leftover_ranges,
                    ),
                    pipeline_mod=pipeline_mod,
                    schema_hint=build_category_split_schema_hint(),
                    max_attempts=max_attempts,
                    label="category_split_rescue",
                )
                records = pipeline_mod.merge_rescued_category_ranges(
                    merged_ocr_text=merged_ocr_text,
                    records=records,
                    rescue_payload=rescue_raw,
                )
            except Exception as exc:
                logger.warning("Category split leftover rescue agent failed, falling back to deterministic assignment: %s", exc)

        final_assigned_ranges = pipeline_mod._extract_assigned_ranges_from_records(records)
        final_leftover_ranges = pipeline_mod.extract_uncategorized_informative_ranges(
            merged_ocr_text=merged_ocr_text,
            assigned_ranges=final_assigned_ranges,
        )
        if final_leftover_ranges:
            fallback_payload = pipeline_mod.assign_leftover_ranges_to_best_categories(
                merged_ocr_text=merged_ocr_text,
                leftover_ranges=final_leftover_ranges,
            )
            records = pipeline_mod.merge_rescued_category_ranges(
                merged_ocr_text=merged_ocr_text,
                records=records,
                rescue_payload=fallback_payload,
            )
        return records
    except Exception as exc:
        logger.warning("Category split agent failed, falling back to heuristic page categorization: %s", exc)
        return pipeline_mod.normalize_category_split_decision({}, image_name_text_pairs)


async def map_category_to_json(
    llm: TextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    candidates_block: str,
    map_category: str,
    max_attempts: int,
) -> Dict[str, Any]:
    return await text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.MAP_SYSTEM,
        user_text=pipeline_mod.build_category_map_user_prompt(
            ocr_text=ocr_text,
            candidates_block=candidates_block,
            map_category=map_category,
        ),
        pipeline_mod=pipeline_mod,
        schema_hint=build_map_schema_hint(),
        max_attempts=max_attempts,
        label=f"map_{map_category}",
    )


async def map_category_recall_to_json(
    llm: TextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    candidates_block: str,
    existing_json: Dict[str, Any],
    map_category: str,
    max_attempts: int,
) -> Dict[str, Any]:
    return await text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.MAP_RECALL_SYSTEM,
        user_text=pipeline_mod.build_category_map_recall_user_prompt(
            ocr_text=ocr_text,
            candidates_block=candidates_block,
            existing_json=existing_json,
            map_category=map_category,
        ),
        pipeline_mod=pipeline_mod,
        schema_hint=build_map_schema_hint(),
        max_attempts=max_attempts,
        label=f"map_recall_{map_category}",
    )


async def map_to_json(
    llm: TextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    candidates_block: str,
    route_name: str,
    official_questionnaire: bool,
    official_family: str,
    max_attempts: int,
) -> Dict[str, Any]:
    return await text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.MAP_SYSTEM,
        user_text=pipeline_mod.build_map_user_prompt(
            ocr_text,
            candidates_block,
            route_name=route_name,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
        ),
        pipeline_mod=pipeline_mod,
        schema_hint=build_map_schema_hint(),
        max_attempts=max_attempts,
        label="map",
    )


async def map_recall_to_json(
    llm: TextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    candidates_block: str,
    existing_json: Dict[str, Any],
    route_name: str,
    official_questionnaire: bool,
    official_family: str,
    max_attempts: int,
) -> Dict[str, Any]:
    return await text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.MAP_RECALL_SYSTEM,
        user_text=pipeline_mod.build_map_recall_user_prompt(
            ocr_text,
            candidates_block,
            existing_json,
            route_name=route_name,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
        ),
        pipeline_mod=pipeline_mod,
        schema_hint=build_map_schema_hint(),
        max_attempts=max_attempts,
        label="map_recall",
    )


def apply_core_backfill_to_stage(
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    stage_raw: Dict[str, Any],
    stage_valid: Dict[str, Any],
    stage_contexts: Dict[str, Dict[str, str]],
    stage_cdm_contexts: Dict[str, str],
    stage_rejected: Dict[str, Dict[str, Any]],
) -> None:
    backfill_additions, backfill_rejected = pipeline_mod.apply_core_backfill(stage_valid, retriever, ocr_text)
    for key, value in backfill_additions.items():
        stage_valid[key] = value
        stage_contexts.setdefault(key, {"filled_by": "", "question": "Derived from OCR header pattern", "page_type": ""})
        row = retriever.row_by_key.get(key)
        stage_cdm_contexts.setdefault(key, str(row.desc if row is not None else "").strip())
        stage_raw.setdefault(
            key,
            {
                "CDM_Context": stage_cdm_contexts[key],
                "value": value,
                "input_context": stage_contexts[key],
            },
        )
    for key, meta in backfill_rejected.items():
        stage_rejected.setdefault(key, meta)


async def map_ocr_text_for_category(
    llm: TextAgent,
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    map_category: str,
    json_retry_attempts: int,
    enable_recall: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_cdm_contexts: Dict[str, str] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}
    normalized_category = pipeline_mod.normalize_map_category_name(map_category)
    prompt_ocr_text = pipeline_mod.normalize_category_map_input_text(normalized_category, ocr_text)
    candidates_block = retriever.prompt_block_for_category(normalized_category, include_basic=True)

    raw = await map_category_to_json(
        llm=llm,
        pipeline_mod=pipeline_mod,
        ocr_text=prompt_ocr_text,
        candidates_block=candidates_block,
        map_category=normalized_category,
        max_attempts=json_retry_attempts,
    )
    pipeline_mod.merge_map_payload_into_stage(
        retriever=retriever,
        ocr_text=ocr_text,
        raw_payload=raw,
        route_name=normalized_category,
        stage_raw=stage_raw,
        stage_valid=stage_valid,
        stage_contexts=stage_contexts,
        stage_cdm_contexts=stage_cdm_contexts,
        stage_rejected=stage_rejected,
    )

    if enable_recall and pipeline_mod.should_run_recall_pass(prompt_ocr_text, stage_valid):
        recall_raw = await map_category_recall_to_json(
            llm=llm,
            pipeline_mod=pipeline_mod,
            ocr_text=prompt_ocr_text,
            candidates_block=candidates_block,
            existing_json=stage_valid,
            map_category=normalized_category,
            max_attempts=json_retry_attempts,
        )
        pipeline_mod.merge_map_payload_into_stage(
            retriever=retriever,
            ocr_text=ocr_text,
            raw_payload=recall_raw,
            route_name=normalized_category,
            stage_raw=stage_raw,
            stage_valid=stage_valid,
            stage_contexts=stage_contexts,
            stage_cdm_contexts=stage_cdm_contexts,
            stage_rejected=stage_rejected,
        )

    apply_core_backfill_to_stage(
        pipeline_mod=pipeline_mod,
        retriever=retriever,
        ocr_text=ocr_text,
        stage_raw=stage_raw,
        stage_valid=stage_valid,
        stage_contexts=stage_contexts,
        stage_cdm_contexts=stage_cdm_contexts,
        stage_rejected=stage_rejected,
    )
    return stage_raw, stage_valid, stage_contexts, stage_cdm_contexts, stage_rejected


async def map_ocr_text_single_agent(
    llm: TextAgent,
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    route_name: str,
    official_questionnaire: bool,
    official_family: str,
    json_retry_attempts: int,
    enable_recall: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_cdm_contexts: Dict[str, str] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}
    route_name = str(route_name or pipeline_mod.DEFAULT_MAP_ROUTE)
    candidates_block = retriever.prompt_block_for_route(route_name, official_questionnaire=official_questionnaire)
    ocr_chunks = pipeline_mod.split_ocr_text_for_map_route(ocr_text, route_name)

    for ocr_chunk in ocr_chunks:
        raw = await map_to_json(
            llm=llm,
            pipeline_mod=pipeline_mod,
            ocr_text=ocr_chunk,
            candidates_block=candidates_block,
            route_name=route_name,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
            max_attempts=json_retry_attempts,
        )
        pipeline_mod.merge_map_payload_into_stage(
            retriever=retriever,
            ocr_text=ocr_chunk,
            raw_payload=raw,
            route_name=route_name,
            stage_raw=stage_raw,
            stage_valid=stage_valid,
            stage_contexts=stage_contexts,
            stage_cdm_contexts=stage_cdm_contexts,
            stage_rejected=stage_rejected,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
        )

    if enable_recall and pipeline_mod.should_run_recall_pass(ocr_text, stage_valid):
        recall_raw = await map_recall_to_json(
            llm=llm,
            pipeline_mod=pipeline_mod,
            ocr_text=ocr_text,
            candidates_block=candidates_block,
            existing_json=stage_valid,
            route_name=route_name,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
            max_attempts=json_retry_attempts,
        )
        pipeline_mod.merge_map_payload_into_stage(
            retriever=retriever,
            ocr_text=ocr_text,
            raw_payload=recall_raw,
            route_name=route_name,
            stage_raw=stage_raw,
            stage_valid=stage_valid,
            stage_contexts=stage_contexts,
            stage_cdm_contexts=stage_cdm_contexts,
            stage_rejected=stage_rejected,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
        )

    apply_core_backfill_to_stage(
        pipeline_mod=pipeline_mod,
        retriever=retriever,
        ocr_text=ocr_text,
        stage_raw=stage_raw,
        stage_valid=stage_valid,
        stage_contexts=stage_contexts,
        stage_cdm_contexts=stage_cdm_contexts,
        stage_rejected=stage_rejected,
    )
    return stage_raw, stage_valid, stage_contexts, stage_cdm_contexts, stage_rejected


def _resolve_agent_backends(
    llm: Union[TextAgent, Sequence[TextAgent]],
    n_agents: int,
) -> List[TextAgent]:
    if isinstance(llm, Sequence) and not isinstance(llm, (str, bytes)):
        items = [item for item in llm if item is not None]
        if not items:
            raise ValueError("No map backends configured")
        if len(items) == 1:
            return [items[0] for _ in range(max(1, int(n_agents)))]
        out: List[TextAgent] = []
        for idx in range(max(1, int(n_agents))):
            out.append(items[idx % len(items)])
        return out
    return [llm for _ in range(max(1, int(n_agents)))]


async def map_ocr_text_multi_agent(
    llm: Union[TextAgent, Sequence[TextAgent]],
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    map_agent_count: int,
    map_agent_count_by_route: Optional[Dict[str, int]],
    route_name: str,
    official_questionnaire: bool,
    official_family: str,
    json_retry_attempts: int,
    enable_recall: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    route_name = str(route_name or pipeline_mod.DEFAULT_MAP_ROUTE)
    route_override = 0
    if map_agent_count_by_route:
        route_override = max(0, int(map_agent_count_by_route.get(route_name, 0) or 0))
    n_agents = max(1, int(route_override or map_agent_count))
    agent_backends = _resolve_agent_backends(llm, n_agents=n_agents)
    if n_agents <= 1:
        return await map_ocr_text_single_agent(
            llm=agent_backends[0],
            pipeline_mod=pipeline_mod,
            retriever=retriever,
            ocr_text=ocr_text,
            route_name=route_name,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
            json_retry_attempts=json_retry_attempts,
            enable_recall=enable_recall,
        )

    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_cdm_contexts: Dict[str, str] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}
    map_agents = pipeline_mod.build_map_agent_specs(
        retriever,
        n_agents,
        route_name=route_name,
        official_questionnaire=official_questionnaire,
    )
    ocr_chunks = pipeline_mod.split_ocr_text_for_map_route(ocr_text, route_name)

    async def _call(agent: Any, agent_backend: TextAgent, ocr_chunk: str) -> Tuple[Any, str, Dict[str, Any]]:
        payload = await map_to_json(
            llm=agent_backend,
            pipeline_mod=pipeline_mod,
            ocr_text=ocr_chunk,
            candidates_block=agent.candidates_block,
            route_name=agent.route_name,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
            max_attempts=json_retry_attempts,
        )
        return agent, ocr_chunk, payload

    outs = await asyncio.gather(
        *[
            _call(agent, agent_backends[agent_idx], ocr_chunk)
            for ocr_chunk in ocr_chunks
            for agent_idx, agent in enumerate(map_agents)
        ],
        return_exceptions=True,
    )
    for out in outs:
        if isinstance(out, Exception):
            logger.warning("Split map agent call failed: %s", out)
            continue
        _, ocr_chunk, payload = out
        pipeline_mod.merge_map_payload_into_stage(
            retriever=retriever,
            ocr_text=ocr_chunk,
            raw_payload=payload,
            route_name=route_name,
            stage_raw=stage_raw,
            stage_valid=stage_valid,
            stage_contexts=stage_contexts,
            stage_cdm_contexts=stage_cdm_contexts,
            stage_rejected=stage_rejected,
            official_questionnaire=official_questionnaire,
            official_family=official_family,
        )

    if enable_recall and pipeline_mod.should_run_recall_pass(ocr_text, stage_valid):
        async def _recall(agent: Any, agent_backend: TextAgent) -> Tuple[Any, Dict[str, Any]]:
            payload = await map_recall_to_json(
                llm=agent_backend,
                pipeline_mod=pipeline_mod,
                ocr_text=ocr_text,
                candidates_block=agent.candidates_block,
                existing_json=stage_valid,
                route_name=agent.route_name,
                official_questionnaire=official_questionnaire,
                official_family=official_family,
                max_attempts=json_retry_attempts,
            )
            return agent, payload

        recall_outs = await asyncio.gather(
            *[_recall(agent, agent_backends[agent_idx]) for agent_idx, agent in enumerate(map_agents)],
            return_exceptions=True,
        )
        for out in recall_outs:
            if isinstance(out, Exception):
                logger.warning("Split map recall agent call failed: %s", out)
                continue
            _, payload = out
            pipeline_mod.merge_map_payload_into_stage(
                retriever=retriever,
                ocr_text=ocr_text,
                raw_payload=payload,
                route_name=route_name,
                stage_raw=stage_raw,
                stage_valid=stage_valid,
                stage_contexts=stage_contexts,
                stage_cdm_contexts=stage_cdm_contexts,
                stage_rejected=stage_rejected,
                official_questionnaire=official_questionnaire,
                official_family=official_family,
            )

    apply_core_backfill_to_stage(
        pipeline_mod=pipeline_mod,
        retriever=retriever,
        ocr_text=ocr_text,
        stage_raw=stage_raw,
        stage_valid=stage_valid,
        stage_contexts=stage_contexts,
        stage_cdm_contexts=stage_cdm_contexts,
        stage_rejected=stage_rejected,
    )
    return stage_raw, stage_valid, stage_contexts, stage_cdm_contexts, stage_rejected


async def resolve_single_conflict(
    llm: TextAgent,
    pipeline_mod: Any,
    retriever: Any,
    patient_name: str,
    key: str,
    entries: List[Dict[str, Any]],
    json_retry_attempts: int,
) -> Tuple[int, str] | None:
    payload = pipeline_mod.build_single_conflict_payload(retriever=retriever, key=key, entries=entries)
    user = (
        f"PATIENT: {patient_name}\n"
        "Resolve one conflict candidate set.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        'Return JSON only: {"chosen_index": <int>, "reason": "<brief reason>"}'
    )
    raw = await text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.CONFLICT_RESOLVER_SYSTEM,
        user_text=user,
        pipeline_mod=pipeline_mod,
        schema_hint='{"chosen_index": <int>, "reason": "<brief reason>"}',
        max_attempts=json_retry_attempts,
        label="conflict_resolver_single",
    )
    idx = pipeline_mod._coerce_int(raw.get("chosen_index"))
    reason = str(raw.get("reason", "")).strip()
    if idx is None and isinstance(raw.get(key), dict):
        nested = raw.get(key) or {}
        idx = pipeline_mod._coerce_int(nested.get("chosen_index"))
        if not reason:
            reason = str(nested.get("reason", "")).strip()
    if idx is None or idx < 0 or idx >= len(entries):
        return None
    return idx, reason


async def resolve_conflicts(
    llm: TextAgent,
    pipeline_mod: Any,
    retriever: Any,
    patient_name: str,
    conflicts: Dict[str, List[Dict[str, Any]]],
    json_retry_attempts: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not conflicts:
        return {}, {}

    overrides: Dict[str, Any] = {}
    decisions: Dict[str, Any] = {}
    pending_conflicts: Dict[str, List[Dict[str, Any]]] = conflicts
    if hasattr(pipeline_mod, "resolve_conflicts_by_majority_vote"):
        try:
            code_overrides, code_decisions, pending_conflicts, _ = pipeline_mod.resolve_conflicts_by_majority_vote(conflicts)
            overrides.update(code_overrides)
            decisions.update(code_decisions)
        except Exception as exc:
            logger.warning("Code majority conflict pre-resolver failed for %s: %s", patient_name, exc)
            pending_conflicts = conflicts

    if not pending_conflicts:
        return overrides, decisions

    user = pipeline_mod.build_conflict_resolver_user_prompt(
        patient_name=patient_name,
        retriever=retriever,
        conflicts=pending_conflicts,
    )
    try:
        raw = await text_to_json(
            llm=llm,
            system_prompt=pipeline_mod.CONFLICT_RESOLVER_SYSTEM,
            user_text=user,
            pipeline_mod=pipeline_mod,
            schema_hint='{"resolved":{"CDM_KEY":{"chosen_index": <int>, "reason": "<brief reason>"}}}',
            max_attempts=json_retry_attempts,
            label="conflict_resolver",
        )
        resolved_obj = raw.get("resolved", raw)
    except Exception:
        resolved_obj = None

    if isinstance(resolved_obj, dict):
        for key, entries in pending_conflicts.items():
            item = resolved_obj.get(key)
            if not isinstance(item, dict):
                continue
            idx = pipeline_mod._coerce_int(item.get("chosen_index"))
            if idx is None or idx < 0 or idx >= len(entries):
                continue
            chosen = entries[idx]
            overrides[key] = chosen.get("value")
            decisions[key] = {
                "chosen_index": idx,
                "chosen_value": chosen.get("value"),
                "reason": str(item.get("reason", "")).strip(),
                "source_image": chosen.get("image"),
                "input_context": pipeline_mod._normalize_input_context(chosen.get("input_context")),
                "resolver_mode": "llm_batch",
            }

    pending = [key for key in pending_conflicts.keys() if key not in overrides]
    for key in pending:
        try:
            resolved = await resolve_single_conflict(
                llm=llm,
                pipeline_mod=pipeline_mod,
                retriever=retriever,
                patient_name=patient_name,
                key=key,
                entries=pending_conflicts[key],
                json_retry_attempts=json_retry_attempts,
            )
        except Exception as exc:
            logger.warning("Per-key conflict resolver failed for %s/%s: %s", patient_name, key, exc)
            continue
        if resolved is None:
            continue
        idx, reason = resolved
        chosen = pending_conflicts[key][idx]
        overrides[key] = chosen.get("value")
        decisions[key] = {
            "chosen_index": idx,
            "chosen_value": chosen.get("value"),
            "reason": reason,
            "source_image": chosen.get("image"),
            "input_context": pipeline_mod._normalize_input_context(chosen.get("input_context")),
            "resolver_mode": "llm_single",
        }
    return overrides, decisions


def resolve_text_backend_name(model_id: str) -> str:
    mid = str(model_id or "").strip().lower()
    if mid.startswith("gpt-"):
        return "openai"
    if mid.startswith("gemini-"):
        return "gemini"
    return "local_hf_text"


def build_text_backend(
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_inflight: int,
    timeout_sec: float,
    max_retries: int,
    openai_api_key_env: str,
    gemini_api_key_env: str,
    dtype: str,
    attn_implementation: str,
    disable_trust_remote_code: bool,
    rate_limit_overrides: Optional[Dict[str, Any]] = None,
    rate_limit_window_sec: float = 60.0,
    rate_limit_margin: float = 0.9,
) -> TextAgent:
    backend_name = resolve_text_backend_name(model_id)
    if backend_name == "openai":
        limiter = RATE_LIMITER_REGISTRY.get(
            provider="openai",
            model_id=model_id,
            config=resolve_model_rate_limit_config(
                rate_limit_overrides or {},
                provider="openai",
                model_id=model_id,
                default_window_sec=rate_limit_window_sec,
                default_margin=rate_limit_margin,
            ),
            default_window_sec=rate_limit_window_sec,
            default_margin=rate_limit_margin,
        )
        return RemoteOpenAIResponsesTextAgent(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_inflight=max_inflight,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            api_key_env=openai_api_key_env,
            rate_limiter=limiter,
        )
    if backend_name == "gemini":
        limiter = RATE_LIMITER_REGISTRY.get(
            provider="gemini",
            model_id=model_id,
            config=resolve_model_rate_limit_config(
                rate_limit_overrides or {},
                provider="gemini",
                model_id=model_id,
                default_window_sec=rate_limit_window_sec,
                default_margin=rate_limit_margin,
            ),
            default_window_sec=rate_limit_window_sec,
            default_margin=rate_limit_margin,
        )
        return RemoteGeminiTextAgent(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_inflight=max_inflight,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            api_key_env=gemini_api_key_env,
            rate_limiter=limiter,
        )
    return LocalHFTextAgent(
        model_id=model_id,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_inflight=max_inflight,
        trust_remote_code=(not disable_trust_remote_code),
        attn_implementation=attn_implementation,
    )


def build_ocr_backend(args: argparse.Namespace, ocr_mod: Any) -> Any:
    backend_name = ocr_mod.resolve_backend_name(args.ocr_model_id)
    if backend_name == "openai_responses":
        limiter = RATE_LIMITER_REGISTRY.get(
            provider="openai",
            model_id=args.ocr_model_id,
            config=resolve_model_rate_limit_config(
                getattr(args, "rate_limit_overrides", {}) or {},
                provider="openai",
                model_id=args.ocr_model_id,
                default_window_sec=args.rate_limit_window_sec,
                default_margin=args.rate_limit_margin,
            ),
            default_window_sec=args.rate_limit_window_sec,
            default_margin=args.rate_limit_margin,
        )
        return ocr_mod.RemoteOpenAIResponsesOCR(
            model_id=args.ocr_model_id,
            max_new_tokens=args.ocr_max_new_tokens,
            temperature=args.ocr_temperature,
            top_p=args.ocr_top_p,
            max_inflight=args.ocr_concurrency,
            timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            api_key_env=args.openai_api_key_env,
            image_max_side=args.image_max_side,
            rate_limiter=limiter,
        )
    if backend_name == "gemini_api":
        limiter = RATE_LIMITER_REGISTRY.get(
            provider="gemini",
            model_id=args.ocr_model_id,
            config=resolve_model_rate_limit_config(
                getattr(args, "rate_limit_overrides", {}) or {},
                provider="gemini",
                model_id=args.ocr_model_id,
                default_window_sec=args.rate_limit_window_sec,
                default_margin=args.rate_limit_margin,
            ),
            default_window_sec=args.rate_limit_window_sec,
            default_margin=args.rate_limit_margin,
        )
        return ocr_mod.RemoteGeminiOCR(
            model_id=args.ocr_model_id,
            max_new_tokens=args.ocr_max_new_tokens,
            temperature=args.ocr_temperature,
            top_p=args.ocr_top_p,
            max_inflight=args.ocr_concurrency,
            timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            api_key_env=args.gemini_api_key_env,
            image_max_side=args.image_max_side,
            rate_limiter=limiter,
        )
    if backend_name == "deepseek_vl2":
        return ocr_mod.LocalDeepSeekVLV2OCR(
            model_id=args.ocr_model_id,
            dtype=args.dtype,
            max_new_tokens=args.ocr_max_new_tokens,
            temperature=args.ocr_temperature,
            top_p=args.ocr_top_p,
            max_inflight=args.ocr_concurrency,
            trust_remote_code=(not args.disable_trust_remote_code),
            package_root=resolve_repo_path(args.deepseek_package_root),
        )
    local_vlm_mod = load_module(resolve_script_path(args.local_vlm_script), "local_qwen_vlm_unified")
    return local_vlm_mod.LocalQwenVLM(
        model_id=args.ocr_model_id,
        dtype=args.dtype,
        max_new_tokens=args.ocr_max_new_tokens,
        temperature=args.ocr_temperature,
        top_p=args.ocr_top_p,
        max_inflight=args.ocr_concurrency,
        trust_remote_code=(not args.disable_trust_remote_code),
        attn_implementation=args.attn_implementation,
        min_pixels=(None if args.min_pixels <= 0 else int(args.min_pixels)),
        max_pixels=(None if args.max_pixels <= 0 else int(args.max_pixels)),
        enable_thinking=args.qwen_enable_thinking,
    )


def parse_image_names(args: argparse.Namespace) -> List[str]:
    names: List[str] = []
    raw = str(args.image_names or "").strip()
    if raw:
        names.extend([part.strip() for part in raw.split(",") if part.strip()])
    if args.image_names_file:
        for line in resolve_repo_path(args.image_names_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                names.append(line)
    deduped: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def resolve_input_images(args: argparse.Namespace, pipeline_mod: Any) -> Tuple[str, List[Path], List[Dict[str, Any]], Optional[Path]]:
    single_image = str(args.single_image or "").strip()
    if single_image:
        img = resolve_repo_path(single_image)
        if not img.exists():
            raise FileNotFoundError(f"Single image not found: {img}")
        patient_name = str(args.patient_name or img.stem).strip() or img.stem
        return patient_name, [img], [], img.parent

    input_root = resolve_repo_path(args.input_root)
    patient_dir = input_root / args.patient_name
    if not patient_dir.exists() or not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    images = pipeline_mod.iter_images(patient_dir)
    if not images:
        raise RuntimeError(f"No images found in {patient_dir}")

    selected_names = parse_image_names(args)
    if selected_names:
        wanted = set(selected_names)
        images = [img for img in images if img.name in wanted]
        missing = [name for name in selected_names if name not in {img.name for img in images}]
        if missing:
            raise FileNotFoundError(f"Selected images not found in {patient_dir}: {missing}")

    duplicates: List[Dict[str, Any]] = []
    if not args.disable_dedup:
        images, duplicates = pipeline_mod.deduplicate_images(images, near_dup_hamming=args.near_dup_hamming)
    return args.patient_name, images, duplicates, patient_dir


def write_plan(output_dir: Path, plan: Dict[str, Any]) -> None:
    (output_dir / "unified_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "n/a", "na"}


def _normalize_semantic_value(value: Any) -> Any:
    if _is_missing_value(value):
        return None
    text = str(value).strip()
    numeric = text.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric):
        try:
            dec = Decimal(numeric).normalize()
            return format(dec, "f").rstrip("0").rstrip(".") if "." in format(dec, "f") else format(dec, "f")
        except InvalidOperation:
            pass
    return re.sub(r"\s+", " ", text)


def build_map_agent_count_by_route(args: argparse.Namespace, pipeline_mod: Any) -> Dict[str, int]:
    def _pos_int(v: Any) -> int:
        try:
            iv = int(v)
        except Exception:
            return 0
        return max(0, iv)

    return {
        str(pipeline_mod.MAP_ROUTE_NIGHT_QUESTIONNAIRE): _pos_int(getattr(args, "map_agent_count_night", 0)),
        str(pipeline_mod.MAP_ROUTE_MORNING_QUESTIONNAIRE): _pos_int(getattr(args, "map_agent_count_morning", 0)),
        str(pipeline_mod.MAP_ROUTE_PSG_REPORT_GENERAL): _pos_int(getattr(args, "map_agent_count_psg", 0)),
        str(pipeline_mod.MAP_ROUTE_PSG_REPORT_EXTENSIVE): _pos_int(getattr(args, "map_agent_count_psg", 0)),
        str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_GENERAL): _pos_int(getattr(args, "map_agent_count_cpap", 0)),
        str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE): _pos_int(getattr(args, "map_agent_count_cpap", 0)),
    }


def _json_safe_value(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def evaluate_against_reference(
    output_dir: Path,
    patient_name: str,
    row: Dict[str, Any],
    example_csv: Path,
    reference_name: str,
    reference_index: int,
) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    if not reference_name and reference_index <= 0:
        return None

    df = pd.read_csv(example_csv)
    if reference_name:
        matches = df[df["Name"].astype(str) == str(reference_name)]
        if matches.empty:
            skipped = {
                "patient_name": patient_name,
                "reference_name": reference_name,
                "evaluation_status": "skipped",
                "skip_reason": "reference_name_not_found",
                "semantic_accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "mismatches": None,
            }
            (output_dir / "evaluation.json").write_text(
                json.dumps(skipped, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning(
                "Skipping evaluation for %s: reference name not found in example.csv: %s",
                patient_name,
                reference_name,
            )
            return skipped
        ref_row = matches.iloc[0]
        selector = {"reference_name": reference_name}
    else:
        if reference_index < 1 or reference_index > len(df):
            skipped = {
                "patient_name": patient_name,
                "reference_index_1based": reference_index,
                "available_reference_rows": int(len(df)),
                "evaluation_status": "skipped",
                "skip_reason": "reference_index_out_of_range",
                "semantic_accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "mismatches": None,
            }
            (output_dir / "evaluation.json").write_text(
                json.dumps(skipped, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning(
                "Skipping evaluation for %s: reference index %d out of range for example.csv with %d rows",
                patient_name,
                reference_index,
                len(df),
            )
            return skipped
        ref_row = df.iloc[reference_index - 1]
        selector = {"reference_index_1based": reference_index}

    metrics = {
        "patient_name": patient_name,
        **selector,
        "evaluation_status": "completed",
        "total_columns": 0,
        "semantic_matches": 0,
        "semantic_accuracy": 0.0,
        "reference_non_null": 0,
        "predicted_non_null": 0,
        "correct_non_null": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "wrong_value_fields": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mismatches": [],
    }

    for col in df.columns:
        pred = row.get(col)
        ref = ref_row[col]
        pred_norm = _normalize_semantic_value(pred)
        ref_norm = _normalize_semantic_value(ref)
        metrics["total_columns"] += 1
        if pred_norm == ref_norm:
            metrics["semantic_matches"] += 1
        else:
            metrics["mismatches"].append(
                {
                    "field": col,
                    "predicted": None if _is_missing_value(pred) else _json_safe_value(pred),
                    "expected": None if _is_missing_value(ref) else _json_safe_value(ref),
                }
            )
        pred_present = pred_norm is not None
        ref_present = ref_norm is not None
        if pred_present:
            metrics["predicted_non_null"] += 1
        if ref_present:
            metrics["reference_non_null"] += 1
        if pred_present and ref_present and pred_norm == ref_norm:
            metrics["correct_non_null"] += 1
        elif pred_present and not ref_present:
            metrics["false_positives"] += 1
        elif ref_present and not pred_present:
            metrics["false_negatives"] += 1
        elif pred_present and ref_present:
            metrics["wrong_value_fields"] += 1

    total = max(1, int(metrics["total_columns"]))
    pred_non_null = max(1, int(metrics["predicted_non_null"]))
    ref_non_null = max(1, int(metrics["reference_non_null"]))
    metrics["semantic_accuracy"] = metrics["semantic_matches"] / total
    metrics["precision"] = metrics["correct_non_null"] / pred_non_null
    metrics["recall"] = metrics["correct_non_null"] / ref_non_null
    denom = metrics["precision"] + metrics["recall"]
    metrics["f1"] = (2 * metrics["precision"] * metrics["recall"] / denom) if denom > 0 else 0.0

    (output_dir / "evaluation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Evaluation complete: semantic_accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f mismatches=%d",
        metrics["semantic_accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        len(metrics["mismatches"]),
    )
    return metrics


def refresh_combined_patient_csv(output_dir: Path, output_columns: List[str]) -> Optional[Path]:
    roots_to_try: List[Path] = []
    seen_roots = set()
    for root in (Path(output_dir), Path(output_dir).parent):
        root = root.resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        roots_to_try.append(root)

    for root in roots_to_try:
        nested_csvs = sorted(
            p for p in root.glob("Patient_*/*.csv") if p.name == f"{p.parent.name}.csv"
        )
        direct_csvs = sorted(p for p in root.glob("Patient_*.csv") if p.stem.startswith("Patient_"))
        csv_paths = nested_csvs if len(nested_csvs) >= 2 else direct_csvs
        if len(csv_paths) < 2:
            continue

        rows: List[Dict[str, Any]] = []
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path, dtype=object)
            except Exception as exc:
                logger.warning("Skipping %s while building combined CSV: %s", csv_path, exc)
                continue
            if df.empty:
                continue
            row = {col: (df.iloc[0][col] if col in df.columns else None) for col in output_columns}
            rows.append(row)

        if len(rows) < 2:
            continue

        out_path = root / "all_patients.csv"
        pd.DataFrame(rows, columns=output_columns).to_csv(out_path, index=False)
        logger.info("Refreshed combined patient CSV: %s (%d rows)", out_path, len(rows))
        return out_path
    return None


async def maybe_warm_backend(backend: Any) -> None:
    warmup = getattr(backend, "warmup", None)
    if callable(warmup):
        result = warmup()
        if asyncio.iscoroutine(result):
            await result


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir=output_dir, debug=args.debug)
    args.rate_limit_overrides = load_rate_limit_overrides(args.rate_limits_json, args.rate_limits_file)

    pipeline_mod = load_module(resolve_script_path(args.pipeline_script), "paper_to_cdm_sa_unified")
    ocr_mod = load_module(resolve_script_path(args.ocr_script), "ocr_only_unified")
    if callable(getattr(pipeline_mod, "load_env", None)):
        pipeline_mod.load_env()

    patient_name, images, duplicates, patient_dir = resolve_input_images(args, pipeline_mod)
    cdm_csv = resolve_repo_path(args.cdm_csv)
    example_csv = resolve_repo_path(args.example_csv)
    reuse_ocr_dir = resolve_repo_path(args.reuse_ocr_dir) if args.reuse_ocr_dir else None
    if not cdm_csv.exists():
        raise FileNotFoundError(f"CDM CSV not found: {cdm_csv}")
    if not example_csv.exists():
        raise FileNotFoundError(f"example.csv not found: {example_csv}")
    if reuse_ocr_dir is not None and not reuse_ocr_dir.exists():
        raise FileNotFoundError(f"reuse OCR dir not found: {reuse_ocr_dir}")

    ocr_page_dir = output_dir / "ocr_pages"
    map_page_dir = output_dir / "map_pages"
    ocr_page_dir.mkdir(parents=True, exist_ok=True)
    if args.pipeline_mode != "ocr_only":
        map_page_dir.mkdir(parents=True, exist_ok=True)

    active_category_count = len(getattr(pipeline_mod, "PATIENT_MAP_CATEGORIES", ()))
    requested_map_agent_model_ids = parse_model_id_list(args.map_agent_model_ids)
    if requested_map_agent_model_ids:
        if len(requested_map_agent_model_ids) == 1:
            map_agent_model_ids = requested_map_agent_model_ids * max(1, active_category_count)
        elif len(requested_map_agent_model_ids) == max(1, active_category_count):
            map_agent_model_ids = requested_map_agent_model_ids
        else:
            raise ValueError(
                "--map_agent_model_ids length must be 1 or match the number of active map categories. "
                f"Got {len(requested_map_agent_model_ids)} vs {max(1, active_category_count)}."
            )
    else:
        map_agent_model_ids = [str(args.map_model_id)] * max(1, active_category_count)

    category_split_model_id = str(getattr(args, "category_split_model_id", "") or args.route_model_id or "").strip() or str(args.map_model_id)
    resolver_model_id = str(args.resolver_model_id or "").strip() or str(args.map_model_id)
    map_backend_kind_by_model = {
        model_id: resolve_text_backend_name(model_id) for model_id in sorted(set(map_agent_model_ids))
    }
    category_split_backend_kind = (
        "shared_map_backend"
        if category_split_model_id in map_backend_kind_by_model
        else resolve_text_backend_name(category_split_model_id)
    )
    resolver_backend_kind = (
        "shared_map_backend"
        if resolver_model_id in map_backend_kind_by_model
        else resolve_text_backend_name(resolver_model_id)
    )
    plan = {
        "patient_name": patient_name,
        "patient_dir": str(patient_dir) if patient_dir is not None else "",
        "single_image": str(resolve_repo_path(args.single_image)) if args.single_image else "",
        "selected_images": [img.name for img in images],
        "output_dir": str(output_dir),
        "pipeline_mode": args.pipeline_mode,
        "reuse_ocr_dir": str(reuse_ocr_dir) if reuse_ocr_dir is not None else "",
        "ocr_model_id": args.ocr_model_id,
        "ocr_backend": ocr_mod.resolve_backend_name(args.ocr_model_id) if not reuse_ocr_dir else "reused_frozen_ocr",
        "processing_design": "patient_level_category_mapping",
        "category_split_model_id": category_split_model_id,
        "category_split_backend": category_split_backend_kind,
        "map_categories": list(getattr(pipeline_mod, "PATIENT_MAP_CATEGORIES", ())),
        "map_model_id": args.map_model_id,
        "map_backend": resolve_text_backend_name(args.map_model_id),
        "map_agent_model_ids": map_agent_model_ids,
        "map_agent_backend_by_model": map_backend_kind_by_model,
        "resolver_model_id": resolver_model_id,
        "resolver_backend": resolver_backend_kind,
        "enable_recall": bool(args.enable_recall),
        "disable_conflict_resolver": bool(args.disable_conflict_resolver or args.pipeline_mode != "ocr_map_resolve"),
        "map_json_retry_attempts": max(1, int(args.map_json_retry_attempts)),
        "resolver_json_retry_attempts": max(1, int(args.resolver_json_retry_attempts)),
        "images_total": len(images),
        "duplicates_dropped": len(duplicates),
        "rate_limit_window_sec": float(args.rate_limit_window_sec),
        "rate_limit_margin": float(args.rate_limit_margin),
        "rate_limit_overrides": args.rate_limit_overrides,
    }
    write_plan(output_dir, plan)
    logger.info("Plan: %s", json.dumps(plan, ensure_ascii=False))

    if reuse_ocr_dir is None:
        ocr_backend = build_ocr_backend(args, ocr_mod)
        if args.preload_ocr_model:
            await maybe_warm_backend(ocr_backend)
    else:
        ocr_backend = None

    map_backends_by_model: Dict[str, TextAgent] = {}
    for model_id in sorted(set(map_agent_model_ids)):
        map_backends_by_model[model_id] = build_text_backend(
            model_id=model_id,
            max_new_tokens=args.map_max_new_tokens,
            temperature=args.map_temperature,
            top_p=args.map_top_p,
            max_inflight=args.map_concurrency,
            timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            openai_api_key_env=args.openai_api_key_env,
            gemini_api_key_env=args.gemini_api_key_env,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            disable_trust_remote_code=args.disable_trust_remote_code,
            rate_limit_overrides=args.rate_limit_overrides,
            rate_limit_window_sec=args.rate_limit_window_sec,
            rate_limit_margin=args.rate_limit_margin,
        )
    map_agent_backends = [map_backends_by_model[mid] for mid in map_agent_model_ids]
    map_backend = map_agent_backends[0]
    if args.preload_map_model:
        for backend in sorted(set(map_agent_backends), key=id):
            await maybe_warm_backend(backend)

    category_split_backend: TextAgent
    if category_split_model_id in map_backends_by_model:
        category_split_backend = map_backends_by_model[category_split_model_id]
    else:
        category_split_backend = build_text_backend(
            model_id=category_split_model_id,
            max_new_tokens=args.route_max_new_tokens,
            temperature=args.route_temperature,
            top_p=args.route_top_p,
            max_inflight=args.map_concurrency,
            timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            openai_api_key_env=args.openai_api_key_env,
            gemini_api_key_env=args.gemini_api_key_env,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            disable_trust_remote_code=args.disable_trust_remote_code,
            rate_limit_overrides=args.rate_limit_overrides,
            rate_limit_window_sec=args.rate_limit_window_sec,
            rate_limit_margin=args.rate_limit_margin,
        )
        if args.preload_map_model:
            await maybe_warm_backend(category_split_backend)

    conflict_backend: Optional[TextAgent]
    if args.disable_conflict_resolver or args.pipeline_mode != "ocr_map_resolve":
        conflict_backend = None
    elif resolver_model_id in map_backends_by_model:
        conflict_backend = map_backends_by_model[resolver_model_id]
    elif resolver_model_id == category_split_model_id:
        conflict_backend = category_split_backend
    else:
        conflict_backend = build_text_backend(
            model_id=resolver_model_id,
            max_new_tokens=args.resolver_max_new_tokens,
            temperature=args.resolver_temperature,
            top_p=args.resolver_top_p,
            max_inflight=args.map_concurrency,
            timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            openai_api_key_env=args.openai_api_key_env,
            gemini_api_key_env=args.gemini_api_key_env,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            disable_trust_remote_code=args.disable_trust_remote_code,
            rate_limit_overrides=args.rate_limit_overrides,
            rate_limit_window_sec=args.rate_limit_window_sec,
            rate_limit_margin=args.rate_limit_margin,
        )
        if args.preload_map_model:
            await maybe_warm_backend(conflict_backend)

    retriever = pipeline_mod.CDMRetriever(cdm_csv)
    output_columns = list(pd.read_csv(example_csv, nrows=0).columns)

    started = time.perf_counter()
    ocr_pairs: List[Tuple[Path, str]] = []
    page_errors: List[Dict[str, str]] = []
    phx_ocr_issues: List[Dict[str, Any]] = []
    ocr_sem = asyncio.Semaphore(max(1, int(args.ocr_concurrency)))
    map_sem = asyncio.Semaphore(max(1, int(args.map_concurrency)))

    async def _ocr_one(idx: int, img: Path) -> None:
        async with ocr_sem:
            t0 = time.perf_counter()
            ocr_image_path = img
            auto_rotated = False
            try:
                if reuse_ocr_dir is not None:
                    src_txt = reuse_ocr_dir / f"{img.stem}.txt"
                    src_meta = reuse_ocr_dir / f"{img.stem}.meta.json"
                    if not src_txt.exists():
                        raise FileNotFoundError(f"Frozen OCR text not found: {src_txt}")
                    text = src_txt.read_text(encoding="utf-8")
                    inherited_meta: Dict[str, Any] = {}
                    if src_meta.exists():
                        inherited_meta = json.loads(src_meta.read_text(encoding="utf-8"))
                    meta = {
                        "image": img.name,
                        "ocr_image": str(inherited_meta.get("ocr_image", img.name)),
                        "auto_rotated_landscape": bool(inherited_meta.get("auto_rotated_landscape", False)),
                        "ok": True,
                        "elapsed_seconds": time.perf_counter() - t0,
                        "text_chars": len(text),
                        "error": "",
                        "reused_ocr": True,
                        "reused_from": str(src_txt),
                    }
                else:
                    assert ocr_backend is not None
                    ocr_image_path, auto_rotated = ocr_mod.prepare_ocr_image(
                        image_path=img,
                        prepared_dir=None,
                        auto_rotate_landscape=False,
                        aspect_ratio_threshold=0.0,
                    )
                    best_attempt: Optional[Dict[str, Any]] = None
                    last_exc: Optional[Exception] = None
                    attempt_count = 0
                    for attempt_idx in range(1, PHX_OCR_MAX_ATTEMPTS + 1):
                        attempt_count = attempt_idx
                        attempt_user_prompt = pipeline_mod.OCR_USER_PROMPT
                        if attempt_idx > 1:
                            attempt_user_prompt = format_phx_retry_prompt(
                                pipeline_mod.OCR_USER_PROMPT,
                                attempt_idx - 1,
                            )
                        try:
                            attempt_text = await ocr_backend.aocr(
                                image_path=ocr_image_path,
                                system_prompt=pipeline_mod.OCR_SYSTEM,
                                user_text=attempt_user_prompt,
                            )
                        except Exception as exc:
                            last_exc = exc
                            logger.warning(
                                "OCR attempt %d/%d failed for %s: %s: %s",
                                attempt_idx,
                                PHX_OCR_MAX_ATTEMPTS,
                                img.name,
                                type(exc).__name__,
                                exc,
                            )
                            continue

                        marker_info = analyze_phx_yes_no_markers(attempt_text)
                        attempt_result = {
                            "text": attempt_text,
                            "marker_info": marker_info,
                            "attempt": attempt_idx,
                        }
                        if (
                            best_attempt is None
                            or marker_info["total"] > best_attempt["marker_info"]["total"]
                        ):
                            best_attempt = attempt_result
                        if marker_info["is_complete"]:
                            break
                        logger.warning(
                            "OCR retry %d/%d for %s due to PHx marker count %d/%d",
                            attempt_idx,
                            PHX_OCR_MAX_ATTEMPTS,
                            img.name,
                            marker_info["total"],
                            PHX_EXPECTED_YES_NO_TOTAL,
                        )

                    if best_attempt is None:
                        assert last_exc is not None
                        raise last_exc
                    text = str(best_attempt["text"])
                    marker_info = dict(best_attempt["marker_info"])
                    meta = {
                        "image": img.name,
                        "ocr_image": ocr_image_path.name,
                        "auto_rotated_landscape": auto_rotated,
                        "ok": True,
                        "elapsed_seconds": time.perf_counter() - t0,
                        "text_chars": len(text),
                        "error": "",
                        "reused_ocr": False,
                        "reused_from": "",
                        "ocr_attempts": attempt_count,
                        "phx_contains_trigger": marker_info["contains_trigger"],
                        "phx_yes_count": marker_info["yes_count"],
                        "phx_no_count": marker_info["no_count"],
                        "phx_yes_no_total": marker_info["total"],
                        "phx_expected_yes_no_total": PHX_EXPECTED_YES_NO_TOTAL,
                    }
                ocr_pairs.append((img, text))
                (ocr_page_dir / f"{img.stem}.txt").write_text(text, encoding="utf-8")
            except Exception as exc:
                meta = {
                    "image": img.name,
                    "ocr_image": ocr_image_path.name,
                    "auto_rotated_landscape": auto_rotated,
                    "ok": False,
                    "elapsed_seconds": time.perf_counter() - t0,
                    "text_chars": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                page_errors.append({"image": img.name, "error_type": type(exc).__name__, "error": str(exc)})
            (ocr_page_dir / f"{img.stem}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if (
                meta.get("ok")
                and meta.get("phx_contains_trigger")
                and int(meta.get("phx_yes_no_total") or 0) != PHX_EXPECTED_YES_NO_TOTAL
                and int(meta.get("ocr_attempts") or 0) >= PHX_OCR_MAX_ATTEMPTS
            ):
                phx_ocr_issues.append(
                    {
                        "image": img.name,
                        "status": "failed_after_max_retries",
                        "ocr_attempts": int(meta.get("ocr_attempts") or 0),
                        "phx_yes_count": int(meta.get("phx_yes_count") or 0),
                        "phx_no_count": int(meta.get("phx_no_count") or 0),
                        "phx_yes_no_total": int(meta.get("phx_yes_no_total") or 0),
                        "phx_expected_yes_no_total": PHX_EXPECTED_YES_NO_TOTAL,
                        "ocr_text_file": f"ocr_pages/{img.stem}.txt",
                        "ocr_meta_file": f"ocr_pages/{img.stem}.meta.json",
                    }
                )
            logger.info(
                "OCR %d/%d | %s | ok=%s | elapsed=%.1fs | chars=%d",
                idx,
                len(images),
                img.name,
                meta["ok"],
                meta["elapsed_seconds"],
                meta["text_chars"],
            )

    ocr_tasks = [asyncio.create_task(_ocr_one(i, img)) for i, img in enumerate(images, start=1)]
    for fut in asyncio.as_completed(ocr_tasks):
        await fut

    ordered_ocr_pairs = [(img, txt) for img in images for src_img, txt in ocr_pairs if src_img.name == img.name]
    ocr_merged_text = pipeline_mod.merge_ocr_text_blocks([(img.name, txt) for img, txt in ordered_ocr_pairs])
    (output_dir / f"{patient_name}_ocr_merged.txt").write_text(ocr_merged_text, encoding="utf-8")
    if hasattr(pipeline_mod, "build_numbered_merged_ocr_text"):
        numbered_ocr_text = pipeline_mod.build_numbered_merged_ocr_text(ocr_merged_text)
        (output_dir / f"{patient_name}_ocr_merged_numbered.txt").write_text(numbered_ocr_text, encoding="utf-8")

    if args.pipeline_mode == "ocr_only":
        summary = {
            "patient_name": patient_name,
            "images_total": len(images),
            "ocr_ok": len(ordered_ocr_pairs),
            "ocr_fail": len(images) - len(ordered_ocr_pairs),
            "duplicates_dropped": len(duplicates),
            "total_elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "ocr_map_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("OCR-only run complete: %s", json.dumps(summary, ensure_ascii=False))
        return

    image_name_text_pairs = [(img.name, txt) for img, txt in ordered_ocr_pairs]
    category_records = await split_patient_ocr_categories(
        llm=category_split_backend,
        pipeline_mod=pipeline_mod,
        image_name_text_pairs=image_name_text_pairs,
        max_attempts=max(1, min(2, int(args.map_json_retry_attempts))),
    )
    (output_dir / "category_split_result.json").write_text(
        json.dumps(category_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for record in category_records:
        (map_page_dir / f"category__{record['category']}.txt").write_text(
            str(record["merged_text"]),
            encoding="utf-8",
        )

    page_results: List[Any] = []

    async def _map_one(record: Dict[str, Any]) -> None:
        async with map_sem:
            idx = int(record["idx"])
            map_category = str(record["category"])
            source_images = list(record["source_images"])
            merged_text = str(record["merged_text"])
            category_name = f"category__{map_category}"

            t0 = time.perf_counter()
            try:
                backend = map_agent_backends[(idx - 1) % max(1, len(map_agent_backends))]
                raw_obj, valid_obj, valid_contexts, valid_cdm_contexts, rejected_fields = await map_ocr_text_for_category(
                    llm=backend,
                    pipeline_mod=pipeline_mod,
                    retriever=retriever,
                    ocr_text=merged_text,
                    map_category=map_category,
                    json_retry_attempts=args.map_json_retry_attempts,
                    enable_recall=args.enable_recall,
                )
                page_results.append(
                    pipeline_mod.PageResult(
                        image_name=category_name,
                        ocr_text=merged_text,
                        raw_json=raw_obj,
                        valid_json=valid_obj,
                        input_contexts=valid_contexts,
                        cdm_contexts=valid_cdm_contexts,
                        rejected_fields=rejected_fields,
                    )
                )
                owned_rows = retriever.category_rows(map_category, include_basic=True)
                owned_keys_payload = {
                    "category": map_category,
                    "owned_key_count": len(owned_rows),
                    "mapped_key_count": len(valid_obj),
                    "unmapped_key_count": sum(1 for row in owned_rows if row.key not in valid_obj),
                    "owned_keys": [
                        {
                            "key": row.key,
                            "map_category": getattr(row, "map_category", ""),
                            "mapped": row.key in valid_obj,
                            "value": valid_obj.get(row.key, ""),
                            "CDM_Context": valid_cdm_contexts.get(row.key, str(getattr(row, "desc", "") or "").strip()),
                            "input_context": valid_contexts.get(row.key, {}) if row.key in valid_obj else {},
                        }
                        for row in owned_rows
                    ],
                }
                (map_page_dir / f"{category_name}.raw.json").write_text(
                    json.dumps(raw_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (map_page_dir / f"{category_name}.valid.json").write_text(
                    json.dumps(valid_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (map_page_dir / f"{category_name}.contexts.json").write_text(
                    json.dumps(
                        {
                            k: {
                                "CDM_Context": valid_cdm_contexts.get(k, ""),
                                "input_context": valid_contexts.get(k, {}),
                            }
                            for k in valid_obj.keys()
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (map_page_dir / f"{category_name}.owned_keys.json").write_text(
                    json.dumps(owned_keys_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if rejected_fields:
                    (map_page_dir / f"{category_name}.rejected.json").write_text(
                        json.dumps(rejected_fields, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                ok = True
                valid_count = len(valid_obj)
                error = ""
            except Exception as exc:
                ok = False
                valid_count = 0
                error = f"{type(exc).__name__}: {exc}"
                page_errors.append({"image": category_name, "error_type": type(exc).__name__, "error": str(exc)})

        meta = {
            "category": map_category,
            "source_images": source_images,
            "ok": ok,
            "elapsed_seconds": time.perf_counter() - t0,
            "valid_keys": valid_count,
            "error": error,
        }
        (map_page_dir / f"{category_name}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "MAP %d/%d | %s | ok=%s | elapsed=%.1fs | valid_keys=%d",
            idx,
            len(category_records),
            category_name,
            ok,
            meta["elapsed_seconds"],
            valid_count,
        )

    for idx, record in enumerate(category_records, start=1):
        record["idx"] = idx
    map_tasks = [asyncio.create_task(_map_one(record)) for record in category_records]
    for fut in asyncio.as_completed(map_tasks):
        await fut

    page_results.sort(key=lambda x: x.image_name)
    patient_res = pipeline_mod.build_patient_result(
        patient_name=patient_name,
        page_results=page_results,
        duplicates=duplicates,
        page_errors=page_errors,
        output_columns=output_columns,
        save_intermediate=args.save_intermediate,
        out_dir=output_dir,
        elapsed_s=(time.perf_counter() - started),
    )
    patient_res["phx_ocr_issues"] = phx_ocr_issues

    if conflict_backend is not None and patient_res.get("row") is not None and patient_res.get("conflicts"):
        overrides, decisions = await resolve_conflicts(
            llm=conflict_backend,
            pipeline_mod=pipeline_mod,
            retriever=retriever,
            patient_name=patient_name,
            conflicts=patient_res.get("conflicts") or {},
            json_retry_attempts=args.resolver_json_retry_attempts,
        )
        if overrides:
            merged_like = dict(patient_res.get("merged") or {})
            for key, value in overrides.items():
                if key in output_columns:
                    merged_like[key] = value
            patient_res["merged"] = merged_like
            patient_res["row"] = pipeline_mod.build_output_row(merged_like, output_columns)
        patient_res["conflict_resolution"] = decisions
        logger.info(
            "Conflict resolution complete: conflict_keys=%d overrides=%d",
            len(patient_res.get("conflicts") or {}),
            len(overrides),
        )

    pipeline_mod.write_patient_outputs(
        output_dir=output_dir,
        patient_name=patient_name,
        res=patient_res,
        output_columns=output_columns,
    )
    refresh_combined_patient_csv(output_dir=output_dir, output_columns=output_columns)

    evaluation = evaluate_against_reference(
        output_dir=output_dir,
        patient_name=patient_name,
        row=patient_res.get("row") or {},
        example_csv=example_csv,
        reference_name=str(args.eval_reference_name or "").strip(),
        reference_index=int(args.eval_reference_index),
    )

    ocr_usage_summary = summarize_openai_usage(ocr_backend) if ocr_backend is not None else {}
    category_split_usage_summary = summarize_openai_usage(category_split_backend)
    map_usage_summary = summarize_openai_usage(*map_agent_backends)
    resolver_usage_summary = summarize_openai_usage(conflict_backend)
    unique_usage_records = collect_usage_records(ocr_backend, category_split_backend, *map_agent_backends, conflict_backend)
    rate_limit_usage_summary = summarize_usage_records(unique_usage_records)
    combined_usage_summary = {
        "ocr": ocr_usage_summary,
        "category_split": category_split_usage_summary,
        "map": map_usage_summary,
        "resolver": resolver_usage_summary,
        "totals": {
            "request_count": rate_limit_usage_summary.get("request_count", 0),
            "input_tokens": sum(int(rec.get("input_tokens") or 0) for rec in unique_usage_records),
            "output_tokens": sum(int(rec.get("output_tokens") or 0) for rec in unique_usage_records),
            "total_tokens": sum(int(rec.get("total_tokens") or 0) for rec in unique_usage_records),
            "cached_input_tokens": sum(int(rec.get("cached_input_tokens") or 0) for rec in unique_usage_records),
            "reasoning_output_tokens": sum(int(rec.get("reasoning_output_tokens") or 0) for rec in unique_usage_records),
        },
    }
    if unique_usage_records:
        (output_dir / "openai_usage_summary.json").write_text(
            json.dumps(combined_usage_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "rate_limit_usage_summary.json").write_text(
            json.dumps(rate_limit_usage_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "patient_name": patient_name,
        "images_total": len(images),
        "ocr_ok": len(ordered_ocr_pairs),
        "ocr_fail": len(images) - len(ordered_ocr_pairs),
        "map_ok": len(page_results),
        "map_fail": len(category_records) - len(page_results),
        "duplicates_dropped": len(duplicates),
        "total_elapsed_seconds": time.perf_counter() - started,
        "output_row_non_null_keys": (
            sum(1 for value in (patient_res.get("row") or {}).values() if not pipeline_mod._is_missing_value(value))
            if patient_res.get("row") is not None
            else 0
        ),
        "evaluation_semantic_accuracy": (evaluation or {}).get("semantic_accuracy"),
        "evaluation_precision": (evaluation or {}).get("precision"),
        "evaluation_recall": (evaluation or {}).get("recall"),
        "evaluation_f1": (evaluation or {}).get("f1"),
        "openai_usage_request_count": combined_usage_summary["totals"]["request_count"],
        "openai_usage_input_tokens": combined_usage_summary["totals"]["input_tokens"],
        "openai_usage_output_tokens": combined_usage_summary["totals"]["output_tokens"],
        "openai_usage_total_tokens": combined_usage_summary["totals"]["total_tokens"],
        "openai_usage_cached_input_tokens": combined_usage_summary["totals"]["cached_input_tokens"],
        "openai_usage_reasoning_output_tokens": combined_usage_summary["totals"]["reasoning_output_tokens"],
    }
    (output_dir / "ocr_map_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Unified run complete: %s", json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified OCR -> MAP -> RESOLVE runner with configurable backends and inputs.")
    ap.add_argument("--pipeline_script", type=str, default="103_paper_to_cdm_SA.py")
    ap.add_argument("--ocr_script", type=str, default="108_ocr_only_patient_local_qwen.py")
    ap.add_argument("--local_vlm_script", type=str, default="106_paper_to_cdm_SA_live_local_vlm.py")
    ap.add_argument("--deepseek_package_root", type=str, default="vendor/deepseek_vl2")

    ap.add_argument("--input_root", type=str, default="paper_patients")
    ap.add_argument("--patient_name", type=str, default="Patient_10")
    ap.add_argument("--single_image", type=str, default="")
    ap.add_argument("--image_names", type=str, default="", help="Comma-separated image filenames to run within a patient folder")
    ap.add_argument("--image_names_file", type=str, default="", help="Optional file with one image filename per line")
    ap.add_argument("--reuse_ocr_dir", type=str, default="", help="Reuse existing OCR txt/meta files from a prior run's ocr_pages dir")

    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--cdm_csv", type=str, default="cdm_new.csv")
    ap.add_argument("--example_csv", type=str, default="example.csv")
    ap.add_argument("--pipeline_mode", type=str, default="ocr_map_resolve", choices=["ocr_only", "ocr_map", "ocr_map_resolve"])

    ap.add_argument("--ocr_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--category_split_model_id", type=str, default="", help="Defaults to map_model_id when empty")
    ap.add_argument("--route_model_id", type=str, default="", help="Deprecated legacy alias for category split model")
    ap.add_argument("--map_model_id", type=str, default="gpt-5.4")
    ap.add_argument(
        "--map_agent_model_ids",
        type=str,
        default="",
        help="Optional comma-separated per-map-agent model IDs. Length must be 1 or --map_agent_count.",
    )
    ap.add_argument("--resolver_model_id", type=str, default="", help="Defaults to map_model_id when empty")

    ap.add_argument("--image_max_side", type=int, default=2048)
    ap.add_argument("--map_bundle_size", type=int, default=1)
    ap.add_argument("--map_agent_count", type=int, default=1)
    ap.add_argument("--map_agent_count_night", type=int, default=0, help="Per-route override for night questionnaire pages. 0 means use --map_agent_count.")
    ap.add_argument("--map_agent_count_morning", type=int, default=0, help="Per-route override for morning questionnaire pages. 0 means use --map_agent_count.")
    ap.add_argument("--map_agent_count_psg", type=int, default=0, help="Per-route override for PSG report pages. 0 means use --map_agent_count.")
    ap.add_argument("--map_agent_count_cpap", type=int, default=0, help="Per-route override for CPAP PSG report pages. 0 means use --map_agent_count.")
    ap.add_argument("--enable_recall", action="store_true")

    ap.add_argument("--ocr_max_new_tokens", type=int, default=4096)
    ap.add_argument("--route_max_new_tokens", type=int, default=512)
    ap.add_argument("--map_max_new_tokens", type=int, default=4096)
    ap.add_argument("--resolver_max_new_tokens", type=int, default=2048)
    ap.add_argument("--ocr_temperature", type=float, default=0.0)
    ap.add_argument("--route_temperature", type=float, default=0.0)
    ap.add_argument("--map_temperature", type=float, default=0.0)
    ap.add_argument("--resolver_temperature", type=float, default=0.0)
    ap.add_argument("--ocr_top_p", type=float, default=0.95)
    ap.add_argument("--route_top_p", type=float, default=0.95)
    ap.add_argument("--map_top_p", type=float, default=0.95)
    ap.add_argument("--resolver_top_p", type=float, default=0.95)

    ap.add_argument("--ocr_concurrency", type=int, default=1)
    ap.add_argument("--map_concurrency", type=int, default=1)
    ap.add_argument("--request_timeout_sec", type=float, default=180.0)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--map_json_retry_attempts", type=int, default=3)
    ap.add_argument("--resolver_json_retry_attempts", type=int, default=2)
    ap.add_argument("--disable_conflict_resolver", action="store_true")

    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--disable_trust_remote_code", action="store_true")
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0)
    ap.add_argument("--qwen_enable_thinking", action="store_true")
    ap.add_argument("--preload_ocr_model", action="store_true")
    ap.add_argument("--preload_map_model", action="store_true")

    ap.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--gemini_api_key_env", type=str, default="GOOGLE_API_KEY")
    ap.add_argument(
        "--rate_limits_json",
        type=str,
        default="",
        help=(
            "Inline JSON with per-provider per-model RPM/TPM limits, "
            'for example: {"openai":{"gpt-5-mini":{"rpm":30,"tpm":30000}}}'
        ),
    )
    ap.add_argument(
        "--rate_limits_file",
        type=str,
        default="",
        help="Optional JSON file with per-provider per-model RPM/TPM limits",
    )
    ap.add_argument("--rate_limit_window_sec", type=float, default=60.0)
    ap.add_argument("--rate_limit_margin", type=float, default=0.9)
    ap.add_argument("--disable_dedup", action="store_true")
    ap.add_argument("--near_dup_hamming", type=int, default=6)
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--eval_reference_name", type=str, default="")
    ap.add_argument("--eval_reference_index", type=int, default=0, help="1-based row index into example.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
