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
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

logger = logging.getLogger("unified_ocr_map")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


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
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.timeout_sec = max(5.0, float(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.api_key_env = str(api_key_env or "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
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

    def _record_usage(self, payload: Dict[str, Any], label: str) -> None:
        usage = self._extract_usage(payload)
        usage["label"] = str(label or "").strip() or "text"
        usage["model"] = self.model_id
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

    def _http_json(self, req: urlrequest.Request) -> Dict[str, Any]:
        with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))

    def _text_sync(self, system_prompt: str, user_text: str, label: str) -> str:
        api_key = self._read_api_key()
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
                resp_payload = self._http_json(req)
                self._record_usage(resp_payload, label)
                return self._extract_text(resp_payload)
            except urlerror.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = str(exc)
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    logger.warning(
                        "Transient OpenAI text error (status=%s, attempt=%d/%d). Retrying in %.1fs.",
                        exc.code,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise RuntimeError(f"OpenAI text request failed: status={exc.code} body={body}") from exc
            except Exception:
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient OpenAI text failure (attempt=%d/%d). Retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
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
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.timeout_sec = max(5.0, float(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.api_key_env = str(api_key_env or "GOOGLE_API_KEY").strip() or "GOOGLE_API_KEY"
        self._sem = asyncio.Semaphore(max(1, int(max_inflight)))

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

    def _http_json(self, req: urlrequest.Request) -> Dict[str, Any]:
        with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))

    def _text_sync(self, system_prompt: str, user_text: str) -> str:
        api_key = self._read_api_key()
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
                return self._extract_text(self._http_json(req))
            except urlerror.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = str(exc)
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    logger.warning(
                        "Transient Gemini text error (status=%s, attempt=%d/%d). Retrying in %.1fs.",
                        exc.code,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise RuntimeError(f"Gemini text request failed: status={exc.code} body={body}") from exc
            except Exception:
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient Gemini text failure (attempt=%d/%d). Retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise
        raise RuntimeError("Gemini text request failed after retries")

    async def atext(self, system_prompt: str, user_text: str, label: str = "") -> str:
        async with self._sem:
            return await asyncio.to_thread(self._text_sync, system_prompt, user_text)


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
        '{"CDM_KEY": {"value": <scalar>, '
        '"input_context": {"filled_by": "doctor|patient|unknown", '
        '"question": "<text>", "page": "<summary>"}}}'
    )


async def map_to_json(
    llm: TextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    candidates_block: str,
    max_attempts: int,
) -> Dict[str, Any]:
    return await text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.MAP_SYSTEM,
        user_text=pipeline_mod.build_map_user_prompt(ocr_text, candidates_block),
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
    max_attempts: int,
) -> Dict[str, Any]:
    return await text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.MAP_RECALL_SYSTEM,
        user_text=pipeline_mod.build_map_recall_user_prompt(ocr_text, candidates_block, existing_json),
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
    stage_rejected: Dict[str, Dict[str, Any]],
) -> None:
    backfill_additions, backfill_rejected = pipeline_mod.apply_core_backfill(stage_valid, retriever, ocr_text)
    for key, value in backfill_additions.items():
        stage_valid[key] = value
        stage_contexts.setdefault(key, {"filled_by": "unknown", "question": "Derived from OCR header pattern"})
        stage_raw.setdefault(key, {"value": value, "input_context": stage_contexts[key]})
    for key, meta in backfill_rejected.items():
        stage_rejected.setdefault(key, meta)


async def map_ocr_text_single_agent(
    llm: TextAgent,
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    json_retry_attempts: int,
    enable_recall: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}

    raw = await map_to_json(
        llm=llm,
        pipeline_mod=pipeline_mod,
        ocr_text=ocr_text,
        candidates_block=retriever.full_cdm_prompt_block(),
        max_attempts=json_retry_attempts,
    )
    pipeline_mod.merge_map_payload_into_stage(
        retriever=retriever,
        ocr_text=ocr_text,
        raw_payload=raw,
        stage_raw=stage_raw,
        stage_valid=stage_valid,
        stage_contexts=stage_contexts,
        stage_rejected=stage_rejected,
    )

    if enable_recall and pipeline_mod.should_run_recall_pass(ocr_text, stage_valid):
        recall_raw = await map_recall_to_json(
            llm=llm,
            pipeline_mod=pipeline_mod,
            ocr_text=ocr_text,
            candidates_block=retriever.full_cdm_prompt_block(),
            existing_json=stage_valid,
            max_attempts=json_retry_attempts,
        )
        pipeline_mod.merge_map_payload_into_stage(
            retriever=retriever,
            ocr_text=ocr_text,
            raw_payload=recall_raw,
            stage_raw=stage_raw,
            stage_valid=stage_valid,
            stage_contexts=stage_contexts,
            stage_rejected=stage_rejected,
        )

    apply_core_backfill_to_stage(
        pipeline_mod=pipeline_mod,
        retriever=retriever,
        ocr_text=ocr_text,
        stage_raw=stage_raw,
        stage_valid=stage_valid,
        stage_contexts=stage_contexts,
        stage_rejected=stage_rejected,
    )
    return stage_raw, stage_valid, stage_contexts, stage_rejected


async def map_ocr_text_multi_agent(
    llm: TextAgent,
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    map_agent_count: int,
    json_retry_attempts: int,
    enable_recall: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    n_agents = max(1, int(map_agent_count))
    if n_agents <= 1:
        return await map_ocr_text_single_agent(
            llm=llm,
            pipeline_mod=pipeline_mod,
            retriever=retriever,
            ocr_text=ocr_text,
            json_retry_attempts=json_retry_attempts,
            enable_recall=enable_recall,
        )

    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}
    map_agents = pipeline_mod.build_map_agent_specs(retriever, n_agents)

    async def _call(agent: Any) -> Tuple[Any, Dict[str, Any]]:
        payload = await map_to_json(
            llm=llm,
            pipeline_mod=pipeline_mod,
            ocr_text=ocr_text,
            candidates_block=agent.candidates_block,
            max_attempts=json_retry_attempts,
        )
        return agent, payload

    outs = await asyncio.gather(*[_call(agent) for agent in map_agents], return_exceptions=True)
    for out in outs:
        if isinstance(out, Exception):
            logger.warning("Split map agent call failed: %s", out)
            continue
        _, payload = out
        pipeline_mod.merge_map_payload_into_stage(
            retriever=retriever,
            ocr_text=ocr_text,
            raw_payload=payload,
            stage_raw=stage_raw,
            stage_valid=stage_valid,
            stage_contexts=stage_contexts,
            stage_rejected=stage_rejected,
        )

    if enable_recall and pipeline_mod.should_run_recall_pass(ocr_text, stage_valid):
        async def _recall(agent: Any) -> Tuple[Any, Dict[str, Any]]:
            payload = await map_recall_to_json(
                llm=llm,
                pipeline_mod=pipeline_mod,
                ocr_text=ocr_text,
                candidates_block=agent.candidates_block,
                existing_json=stage_valid,
                max_attempts=json_retry_attempts,
            )
            return agent, payload

        recall_outs = await asyncio.gather(*[_recall(agent) for agent in map_agents], return_exceptions=True)
        for out in recall_outs:
            if isinstance(out, Exception):
                logger.warning("Split map recall agent call failed: %s", out)
                continue
            _, payload = out
            pipeline_mod.merge_map_payload_into_stage(
                retriever=retriever,
                ocr_text=ocr_text,
                raw_payload=payload,
                stage_raw=stage_raw,
                stage_valid=stage_valid,
                stage_contexts=stage_contexts,
                stage_rejected=stage_rejected,
            )

    apply_core_backfill_to_stage(
        pipeline_mod=pipeline_mod,
        retriever=retriever,
        ocr_text=ocr_text,
        stage_raw=stage_raw,
        stage_valid=stage_valid,
        stage_contexts=stage_contexts,
        stage_rejected=stage_rejected,
    )
    return stage_raw, stage_valid, stage_contexts, stage_rejected


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

    user = pipeline_mod.build_conflict_resolver_user_prompt(
        patient_name=patient_name,
        retriever=retriever,
        conflicts=conflicts,
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

    overrides: Dict[str, Any] = {}
    decisions: Dict[str, Any] = {}
    if isinstance(resolved_obj, dict):
        for key, entries in conflicts.items():
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
            }

    pending = [key for key in conflicts.keys() if key not in overrides]
    for key in pending:
        try:
            resolved = await resolve_single_conflict(
                llm=llm,
                pipeline_mod=pipeline_mod,
                retriever=retriever,
                patient_name=patient_name,
                key=key,
                entries=conflicts[key],
                json_retry_attempts=json_retry_attempts,
            )
        except Exception as exc:
            logger.warning("Per-key conflict resolver failed for %s/%s: %s", patient_name, key, exc)
            continue
        if resolved is None:
            continue
        idx, reason = resolved
        chosen = conflicts[key][idx]
        overrides[key] = chosen.get("value")
        decisions[key] = {
            "chosen_index": idx,
            "chosen_value": chosen.get("value"),
            "reason": reason,
            "source_image": chosen.get("image"),
            "input_context": pipeline_mod._normalize_input_context(chosen.get("input_context")),
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
) -> TextAgent:
    backend_name = resolve_text_backend_name(model_id)
    if backend_name == "openai":
        return RemoteOpenAIResponsesTextAgent(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_inflight=max_inflight,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            api_key_env=openai_api_key_env,
        )
    if backend_name == "gemini":
        return RemoteGeminiTextAgent(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_inflight=max_inflight,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            api_key_env=gemini_api_key_env,
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
        )
    if backend_name == "gemini_api":
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
            raise RuntimeError(f"Reference name not found in example.csv: {reference_name}")
        ref_row = matches.iloc[0]
        selector = {"reference_name": reference_name}
    else:
        if reference_index < 1 or reference_index > len(df):
            raise RuntimeError(f"Reference index out of range: {reference_index}")
        ref_row = df.iloc[reference_index - 1]
        selector = {"reference_index_1based": reference_index}

    metrics = {
        "patient_name": patient_name,
        **selector,
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


async def maybe_warm_backend(backend: Any) -> None:
    warmup = getattr(backend, "warmup", None)
    if callable(warmup):
        result = warmup()
        if asyncio.iscoroutine(result):
            await result


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir=output_dir, debug=args.debug)

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
    prepared_image_dir = (output_dir / "prepared_images") if args.auto_rotate_landscape else None

    resolver_model_id = str(args.resolver_model_id or "").strip() or str(args.map_model_id)
    resolver_backend_kind = "shared_map_backend" if resolver_model_id == args.map_model_id else resolve_text_backend_name(resolver_model_id)
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
        "map_model_id": args.map_model_id,
        "map_backend": resolve_text_backend_name(args.map_model_id),
        "resolver_model_id": resolver_model_id,
        "resolver_backend": resolver_backend_kind,
        "map_bundle_size": max(1, int(args.map_bundle_size)),
        "map_agent_count": max(1, int(args.map_agent_count)),
        "enable_recall": bool(args.enable_recall),
        "disable_conflict_resolver": bool(args.disable_conflict_resolver or args.pipeline_mode != "ocr_map_resolve"),
        "map_json_retry_attempts": max(1, int(args.map_json_retry_attempts)),
        "resolver_json_retry_attempts": max(1, int(args.resolver_json_retry_attempts)),
        "images_total": len(images),
        "duplicates_dropped": len(duplicates),
    }
    write_plan(output_dir, plan)
    logger.info("Plan: %s", json.dumps(plan, ensure_ascii=False))

    if reuse_ocr_dir is None:
        ocr_backend = build_ocr_backend(args, ocr_mod)
        if args.preload_ocr_model:
            await maybe_warm_backend(ocr_backend)
    else:
        ocr_backend = None

    map_backend = build_text_backend(
        model_id=args.map_model_id,
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
    )
    if args.preload_map_model:
        await maybe_warm_backend(map_backend)

    conflict_backend: Optional[TextAgent]
    if args.disable_conflict_resolver or args.pipeline_mode != "ocr_map_resolve":
        conflict_backend = None
    elif resolver_model_id == args.map_model_id:
        conflict_backend = map_backend
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
        )
        if args.preload_map_model:
            await maybe_warm_backend(conflict_backend)

    retriever = pipeline_mod.CDMRetriever(cdm_csv)
    output_columns = list(pd.read_csv(example_csv, nrows=0).columns)

    started = time.perf_counter()
    ocr_pairs: List[Tuple[Path, str]] = []
    page_errors: List[Dict[str, str]] = []
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
                        prepared_dir=prepared_image_dir,
                        auto_rotate_landscape=args.auto_rotate_landscape,
                        aspect_ratio_threshold=args.auto_rotate_landscape_ratio,
                    )
                    text = await ocr_backend.aocr(
                        image_path=ocr_image_path,
                        system_prompt=pipeline_mod.OCR_SYSTEM,
                        user_text=pipeline_mod.OCR_USER_PROMPT,
                    )
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

    bundles = pipeline_mod.chunked(ordered_ocr_pairs, max(1, int(args.map_bundle_size)))
    page_results: List[Any] = []

    async def _map_one(idx: int, bundle: List[Tuple[Path, str]]) -> None:
        async with map_sem:
            image_names = [img.name for img, _ in bundle]
            bundle_name = pipeline_mod.make_bundle_image_name(idx, image_names)
            merged_text = pipeline_mod.merge_ocr_text_blocks([(img.name, txt) for img, txt in bundle]).strip()
            if not merged_text:
                page_errors.append({"image": bundle_name, "error_type": "EmptyMergedOCR", "error": "Merged OCR text is empty"})
                return

            t0 = time.perf_counter()
            try:
                raw_obj, valid_obj, valid_contexts, rejected_fields = await map_ocr_text_multi_agent(
                    llm=map_backend,
                    pipeline_mod=pipeline_mod,
                    retriever=retriever,
                    ocr_text=merged_text,
                    map_agent_count=args.map_agent_count,
                    json_retry_attempts=args.map_json_retry_attempts,
                    enable_recall=args.enable_recall,
                )
                page_results.append(
                    pipeline_mod.PageResult(
                        image_name=bundle_name,
                        ocr_text=merged_text,
                        raw_json=raw_obj,
                        valid_json=valid_obj,
                        input_contexts=valid_contexts,
                        rejected_fields=rejected_fields,
                    )
                )
                (map_page_dir / f"{Path(bundle_name).stem}.raw.json").write_text(
                    json.dumps(raw_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (map_page_dir / f"{Path(bundle_name).stem}.valid.json").write_text(
                    json.dumps(valid_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (map_page_dir / f"{Path(bundle_name).stem}.contexts.json").write_text(
                    json.dumps(valid_contexts, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if rejected_fields:
                    (map_page_dir / f"{Path(bundle_name).stem}.rejected.json").write_text(
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
                page_errors.append({"image": bundle_name, "error_type": type(exc).__name__, "error": str(exc)})

            meta = {
                "bundle": bundle_name,
                "source_images": image_names,
                "ok": ok,
                "elapsed_seconds": time.perf_counter() - t0,
                "valid_keys": valid_count,
                "error": error,
            }
            (map_page_dir / f"{Path(bundle_name).stem}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "MAP %d/%d | %s | ok=%s | elapsed=%.1fs | valid_keys=%d",
                idx,
                len(bundles),
                bundle_name,
                ok,
                meta["elapsed_seconds"],
                valid_count,
            )

    map_tasks = [asyncio.create_task(_map_one(i, bundle)) for i, bundle in enumerate(bundles, start=1)]
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

    evaluation = evaluate_against_reference(
        output_dir=output_dir,
        patient_name=patient_name,
        row=patient_res.get("row") or {},
        example_csv=example_csv,
        reference_name=str(args.eval_reference_name or "").strip(),
        reference_index=int(args.eval_reference_index),
    )

    ocr_usage_summary = summarize_openai_usage(ocr_backend) if ocr_backend is not None else {}
    map_usage_summary = summarize_openai_usage(map_backend, conflict_backend)
    combined_usage_summary = {
        "ocr": ocr_usage_summary,
        "map_and_resolver": map_usage_summary,
        "totals": {
            "request_count": int(ocr_usage_summary.get("request_count") or 0) + int(map_usage_summary.get("request_count") or 0),
            "input_tokens": int(ocr_usage_summary.get("input_tokens") or 0) + int(map_usage_summary.get("input_tokens") or 0),
            "output_tokens": int(ocr_usage_summary.get("output_tokens") or 0) + int(map_usage_summary.get("output_tokens") or 0),
            "total_tokens": int(ocr_usage_summary.get("total_tokens") or 0) + int(map_usage_summary.get("total_tokens") or 0),
            "cached_input_tokens": int(ocr_usage_summary.get("cached_input_tokens") or 0) + int(map_usage_summary.get("cached_input_tokens") or 0),
            "reasoning_output_tokens": int(ocr_usage_summary.get("reasoning_output_tokens") or 0) + int(map_usage_summary.get("reasoning_output_tokens") or 0),
        },
    }
    if ocr_usage_summary or map_usage_summary:
        (output_dir / "openai_usage_summary.json").write_text(
            json.dumps(combined_usage_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "patient_name": patient_name,
        "images_total": len(images),
        "ocr_ok": len(ordered_ocr_pairs),
        "ocr_fail": len(images) - len(ordered_ocr_pairs),
        "map_ok": len(page_results),
        "map_fail": len(bundles) - len(page_results),
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
    ap.add_argument("--cdm_csv", type=str, default="cdm_revised.csv")
    ap.add_argument("--example_csv", type=str, default="example.csv")
    ap.add_argument("--pipeline_mode", type=str, default="ocr_map_resolve", choices=["ocr_only", "ocr_map", "ocr_map_resolve"])

    ap.add_argument("--ocr_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--map_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--resolver_model_id", type=str, default="", help="Defaults to map_model_id when empty")

    ap.add_argument("--image_max_side", type=int, default=2048)
    ap.add_argument("--map_bundle_size", type=int, default=1)
    ap.add_argument("--map_agent_count", type=int, default=1)
    ap.add_argument("--enable_recall", action="store_true")

    ap.add_argument("--ocr_max_new_tokens", type=int, default=4096)
    ap.add_argument("--map_max_new_tokens", type=int, default=4096)
    ap.add_argument("--resolver_max_new_tokens", type=int, default=2048)
    ap.add_argument("--ocr_temperature", type=float, default=0.0)
    ap.add_argument("--map_temperature", type=float, default=0.0)
    ap.add_argument("--resolver_temperature", type=float, default=0.0)
    ap.add_argument("--ocr_top_p", type=float, default=0.95)
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
    ap.add_argument("--auto_rotate_landscape", action="store_true")
    ap.add_argument("--auto_rotate_landscape_ratio", type=float, default=1.05)
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
