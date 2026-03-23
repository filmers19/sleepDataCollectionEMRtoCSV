from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
from PIL import Image, ImageOps
from rate_limit_utils import estimate_text_tokens, normalize_headers, parse_retry_after_seconds

logger = logging.getLogger("ocr_only_local_qwen")
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
    log_path = output_dir / "ocr_only.log"
    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if not debug:
        for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Logging initialized: %s", log_path)
    return log_path


def normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def resolve_backend_name(model_id: str) -> str:
    model_id_l = str(model_id or "").lower()
    if model_id_l.startswith("gpt-"):
        return "openai_responses"
    if model_id_l.startswith("gemini-"):
        return "gemini_api"
    if "deepseek-vl2" in model_id_l:
        return "deepseek_vl2"
    return "qwen_vl"


def load_oriented_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def should_rotate_landscape(width: int, height: int, aspect_ratio_threshold: float) -> bool:
    if width <= 0 or height <= 0:
        return False
    return (float(width) / float(height)) >= max(1.0, float(aspect_ratio_threshold))


def prepare_ocr_image(
    image_path: Path,
    prepared_dir: Path | None,
    auto_rotate_landscape: bool,
    aspect_ratio_threshold: float,
) -> Tuple[Path, bool]:
    # Keep OCR input exactly as provided. Automatic orientation correction was removed
    # because aspect-ratio-based rotation is not reliable enough for these pages.
    return image_path, False


class LocalDeepSeekVLV2OCR:
    def __init__(
        self,
        model_id: str,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        max_inflight: int,
        trust_remote_code: bool,
        package_root: Path,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.trust_remote_code = trust_remote_code
        self.package_root = package_root

        self._load_lock = threading.Lock()
        self._sem = asyncio.Semaphore(max(1, int(max_inflight)))
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._load_pil_images: Any = None

    def _resolve_dtype(self, torch_mod: Any) -> Any:
        alias = self.dtype.strip().lower()
        if alias in {"", "auto", "bf16", "bfloat16"}:
            return torch_mod.bfloat16
        if alias in {"fp16", "float16", "half"}:
            return torch_mod.float16
        if alias in {"fp32", "float32"}:
            return torch_mod.float32
        raise ValueError(f"Unsupported --dtype value: {self.dtype}")

    def _model_device(self) -> Any:
        assert self._model is not None
        return next(self._model.parameters()).device

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        with self._load_lock:
            if self._model is not None and self._processor is not None and self._torch is not None:
                return

            package_parent = str(self.package_root.parent.resolve())
            if package_parent not in sys.path:
                sys.path.insert(0, package_parent)

            import torch
            from transformers import AutoModelForCausalLM

            import deepseek_vl2  # noqa: F401
            from deepseek_vl2.models import DeepseekVLV2Processor
            from deepseek_vl2.utils.io import load_pil_images

            dtype_value = self._resolve_dtype(torch)
            logger.info("Loading DeepSeek-VL2 processor: %s", self.model_id)
            self._processor = DeepseekVLV2Processor.from_pretrained(self.model_id)
            self._tokenizer = self._processor.tokenizer
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": dtype_value,
            }
            if torch.cuda.is_available():
                # Stream weights directly to the single local GPU to avoid the slow CPU-first load.
                load_kwargs["low_cpu_mem_usage"] = True
                load_kwargs["device_map"] = {"": 0}

            logger.info("Loading DeepSeek-VL2 model: %s", self.model_id)
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            except Exception as exc:
                logger.warning(
                    "DeepSeek-VL2 streaming load failed (%s). Falling back to CPU-first load.",
                    exc,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=dtype_value,
                )
                if torch.cuda.is_available():
                    model = model.cuda()
            self._model = model.eval()
            logger.info("DeepSeek-VL2 model ready")
            self._load_pil_images = load_pil_images
            self._torch = torch

    def _ocr_sync(self, image_path: Path, system_prompt: str, user_text: str) -> str:
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None and self._tokenizer is not None
        assert self._load_pil_images is not None and self._torch is not None

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{user_text}",
                "images": [str(image_path)],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = self._load_pil_images(conversation)
        prepare_inputs = self._processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=system_prompt,
        ).to(self._model_device())

        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
        gen_kwargs: Dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
            "pad_token_id": self._tokenizer.eos_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "use_cache": True,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with self._torch.inference_mode():
            language_model = getattr(self._model, "language_model", None)
            if language_model is not None and callable(getattr(language_model, "generate", None)):
                outputs = language_model.generate(**gen_kwargs)
            else:
                outputs = self._model.generate(**gen_kwargs)

        answer = self._tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        prompt = ""
        sft_format = getattr(prepare_inputs, "sft_format", None)
        if isinstance(sft_format, list) and sft_format:
            prompt = str(sft_format[0])
        if prompt and answer.startswith(prompt):
            answer = answer[len(prompt) :]
        return answer.strip()

    async def aocr(self, image_path: Path, system_prompt: str, user_text: str) -> str:
        async with self._sem:
            return await asyncio.to_thread(self._ocr_sync, image_path, system_prompt, user_text)

    def warmup(self) -> None:
        self._ensure_loaded()


class RemoteOpenAIResponsesOCR:
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
        image_max_side: int,
        rate_limiter: Any = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.timeout_sec = max(5.0, float(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.api_key_env = str(api_key_env or "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
        self.image_max_side = max(256, int(image_max_side))
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

    @staticmethod
    def _extract_usage(payload: Dict[str, Any]) -> Dict[str, Any]:
        usage = payload.get("usage") or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return {
            "input_tokens": RemoteOpenAIResponsesOCR._coerce_int(usage.get("input_tokens")),
            "output_tokens": RemoteOpenAIResponsesOCR._coerce_int(usage.get("output_tokens")),
            "total_tokens": RemoteOpenAIResponsesOCR._coerce_int(usage.get("total_tokens")),
            "cached_input_tokens": RemoteOpenAIResponsesOCR._coerce_int(input_details.get("cached_tokens")),
            "reasoning_output_tokens": RemoteOpenAIResponsesOCR._coerce_int(output_details.get("reasoning_tokens")),
        }

    def _record_usage(
        self,
        payload: Dict[str, Any],
        image_path: Path,
        request_id: str,
        started_at: float,
        rate_limit_meta: Dict[str, Any] | None,
    ) -> None:
        usage = self._extract_usage(payload)
        usage["label"] = "ocr"
        usage["image"] = image_path.name
        usage["model"] = self.model_id
        usage["request_id"] = request_id
        usage["started_at"] = started_at
        usage["finished_at"] = time.time()
        if rate_limit_meta:
            usage.update(rate_limit_meta)
        with self._usage_lock:
            self._usage_records.append(usage)

    def usage_records(self) -> List[Dict[str, Any]]:
        with self._usage_lock:
            return [dict(item) for item in self._usage_records]

    def _read_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable: {self.api_key_env}")
        return api_key

    @staticmethod
    def _resize_to_jpeg_bytes(image_path: Path, max_side: int) -> bytes:
        img = load_oriented_rgb_image(image_path)
        width, height = img.size
        scale = min(1.0, float(max_side) / float(max(width, height)))
        if scale < 1.0:
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _image_to_data_url(self, image_path: Path) -> str:
        raw = self._resize_to_jpeg_bytes(image_path, self.image_max_side)
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

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
        raise RuntimeError("OpenAI response did not contain OCR text")

    def _http_json(self, req: urlrequest.Request) -> Tuple[Dict[str, Any], Dict[str, str]]:
        with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read()
            headers = normalize_headers(resp.headers)
        return json.loads(raw.decode("utf-8")), headers

    def _ocr_sync(self, image_path: Path, system_prompt: str, user_text: str) -> str:
        api_key = self._read_api_key()
        estimated_tokens = estimate_text_tokens(system_prompt, user_text) + self.max_new_tokens
        started_at = time.time()
        request_id = ""
        if self._rate_limiter is not None:
            request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label="ocr")
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {
                            "type": "input_image",
                            "image_url": self._image_to_data_url(image_path),
                            "detail": "high",
                        },
                    ],
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
                data, headers = self._http_json(req)
                rate_limit_meta = None
                if self._rate_limiter is not None and request_id:
                    rate_limit_meta = self._rate_limiter.release(
                        request_id,
                        actual_tokens=int((data.get("usage") or {}).get("total_tokens") or 0),
                        headers=headers,
                        status="ok",
                    )
                self._record_usage(data, image_path, request_id, started_at, rate_limit_meta)
                return self._extract_text(data)
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
                        "Transient OpenAI OCR error for %s (status=%s, attempt=%d/%d). Retrying in %.1fs.",
                        image_path.name,
                        exc.code,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label="ocr")
                    continue
                raise RuntimeError(f"OpenAI OCR request failed: status={exc.code} body={body}") from exc
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
                        "Transient OpenAI OCR failure for %s (attempt=%d/%d). Retrying in %.1fs.",
                        image_path.name,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label="ocr")
                    continue
                raise
        raise RuntimeError(f"OpenAI OCR failed after retries: {image_path.name}")

    async def aocr(self, image_path: Path, system_prompt: str, user_text: str) -> str:
        async with self._sem:
            return await asyncio.to_thread(self._ocr_sync, image_path, system_prompt, user_text)

    def warmup(self) -> None:
        return None


class RemoteGeminiOCR:
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
        image_max_side: int,
        rate_limiter: Any = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.timeout_sec = max(5.0, float(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.api_key_env = str(api_key_env or "GOOGLE_API_KEY").strip() or "GOOGLE_API_KEY"
        self.image_max_side = max(256, int(image_max_side))
        self._rate_limiter = rate_limiter
        self._sem = asyncio.Semaphore(max(1, int(max_inflight)))
        self._usage_lock = threading.Lock()
        self._usage_records: List[Dict[str, Any]] = []

    def _read_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable: {self.api_key_env}")
        return api_key

    def _inline_image_part(self, image_path: Path) -> Dict[str, Any]:
        raw = RemoteOpenAIResponsesOCR._resize_to_jpeg_bytes(image_path, self.image_max_side)
        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(raw).decode("ascii"),
            }
        }

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
        raise RuntimeError("Gemini response did not contain OCR text")

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
        image_path: Path,
        request_id: str,
        started_at: float,
        rate_limit_meta: Dict[str, Any] | None,
    ) -> None:
        usage = self._extract_usage(payload)
        usage["label"] = "ocr"
        usage["image"] = image_path.name
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

    def _ocr_sync(self, image_path: Path, system_prompt: str, user_text: str) -> str:
        api_key = self._read_api_key()
        estimated_tokens = estimate_text_tokens(system_prompt, user_text) + self.max_new_tokens
        started_at = time.time()
        request_id = ""
        if self._rate_limiter is not None:
            request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label="ocr")
        payload: Dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                        self._inline_image_part(image_path),
                    ],
                }
            ],
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
                data, headers = self._http_json(req)
                rate_limit_meta = None
                if self._rate_limiter is not None and request_id:
                    rate_limit_meta = self._rate_limiter.release(
                        request_id,
                        actual_tokens=int((data.get("usageMetadata") or {}).get("totalTokenCount") or 0),
                        headers=headers,
                        status="ok",
                    )
                self._record_usage(data, image_path, request_id, started_at, rate_limit_meta)
                return self._extract_text(data)
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
                        "Transient Gemini OCR error for %s (status=%s, attempt=%d/%d). Retrying in %.1fs.",
                        image_path.name,
                        exc.code,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label="ocr")
                    continue
                raise RuntimeError(f"Gemini OCR request failed: status={exc.code} body={body}") from exc
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
                        "Transient Gemini OCR failure for %s (attempt=%d/%d). Retrying in %.1fs.",
                        image_path.name,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    if self._rate_limiter is not None:
                        request_id = self._rate_limiter.acquire(estimated_tokens=estimated_tokens, label="ocr")
                    continue
                raise
        raise RuntimeError(f"Gemini OCR failed after retries: {image_path.name}")

    async def aocr(self, image_path: Path, system_prompt: str, user_text: str) -> str:
        async with self._sem:
            return await asyncio.to_thread(self._ocr_sync, image_path, system_prompt, user_text)

    def warmup(self) -> None:
        return None


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir=output_dir, debug=args.debug)

    pipeline_mod = load_module(resolve_script_path(args.pipeline_script), "paper_to_cdm_sa_ocr_only")
    if callable(getattr(pipeline_mod, "load_env", None)):
        pipeline_mod.load_env()

    input_root = resolve_repo_path(args.input_root)
    patient_dir = input_root / args.patient_name
    if not patient_dir.exists() or not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    images = pipeline_mod.iter_images(patient_dir)
    if not images:
        raise RuntimeError(f"No images found in {patient_dir}")

    duplicates: List[Dict[str, Any]] = []
    if not args.disable_dedup:
        images, duplicates = pipeline_mod.deduplicate_images(images, near_dup_hamming=args.near_dup_hamming)

    page_out_dir = output_dir / "ocr_pages"
    page_out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "patient_name": args.patient_name,
        "input_root": str(input_root),
        "patient_dir": str(patient_dir),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "ocr_backend": resolve_backend_name(args.model_id),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_inflight": args.max_inflight,
        "concurrency": args.concurrency,
        "image_max_side": args.image_max_side,
        "request_timeout_sec": args.request_timeout_sec,
        "max_retries": args.max_retries,
        "disable_dedup": args.disable_dedup,
        "near_dup_hamming": args.near_dup_hamming,
        "images_total": len(images),
        "duplicates_dropped": len(duplicates),
    }
    (output_dir / "ocr_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Plan: %s", json.dumps(plan, ensure_ascii=False))

    backend_name = resolve_backend_name(args.model_id)
    use_deepseek_vl2 = backend_name == "deepseek_vl2"
    use_openai_responses = backend_name == "openai_responses"
    use_gemini_api = backend_name == "gemini_api"
    logger.info("Selected OCR backend: %s", backend_name)

    if use_openai_responses:
        backend = RemoteOpenAIResponsesOCR(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_inflight=args.max_inflight,
            timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            api_key_env=args.openai_api_key_env,
            image_max_side=args.image_max_side,
        )
    elif use_gemini_api:
        backend = RemoteGeminiOCR(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_inflight=args.max_inflight,
            timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            api_key_env=args.gemini_api_key_env,
            image_max_side=args.image_max_side,
        )
    elif use_deepseek_vl2:
        backend = LocalDeepSeekVLV2OCR(
            model_id=args.model_id,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_inflight=args.max_inflight,
            trust_remote_code=(not args.disable_trust_remote_code),
            package_root=resolve_repo_path(args.deepseek_package_root),
        )
    else:
        local_vlm_mod = load_module(resolve_script_path(args.local_vlm_script), "local_qwen_ocr_only")
        backend = local_vlm_mod.LocalQwenVLM(
            model_id=args.model_id,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_inflight=args.max_inflight,
            trust_remote_code=(not args.disable_trust_remote_code),
            attn_implementation=args.attn_implementation,
            min_pixels=(None if args.min_pixels <= 0 else int(args.min_pixels)),
            max_pixels=(None if args.max_pixels <= 0 else int(args.max_pixels)),
            enable_thinking=args.qwen_enable_thinking,
        )
    if args.preload_model:
        backend.warmup()

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    started = time.perf_counter()
    results_by_name: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    async def _ocr_one(idx: int, img: Path) -> None:
        async with sem:
            t0 = time.perf_counter()
            text = ""
            error = ""
            ok = False
            ocr_image_path = img
            auto_rotated = False
            user_prompt = getattr(pipeline_mod, "OCR_USER_PROMPT", "Please transcribe this image.")
            try:
                ocr_image_path, auto_rotated = prepare_ocr_image(
                    image_path=img,
                    prepared_dir=None,
                    auto_rotate_landscape=False,
                    aspect_ratio_threshold=0.0,
                )
                if use_openai_responses or use_gemini_api or use_deepseek_vl2:
                    text = await backend.aocr(
                        image_path=ocr_image_path,
                        system_prompt=pipeline_mod.OCR_SYSTEM,
                        user_text=user_prompt,
                    )
                else:
                    data_url = pipeline_mod.image_to_data_url(ocr_image_path, max_side=args.image_max_side)
                    msg = [
                        pipeline_mod.SystemMessage(content=pipeline_mod.OCR_SYSTEM),
                        pipeline_mod.HumanMessage(
                            content=[
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ]
                        ),
                    ]
                    resp = await backend.ainvoke(msg)
                    text = pipeline_mod.llm_content_to_text(resp.content)
                ok = True
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

            elapsed = time.perf_counter() - t0
            meta = {
                "image": img.name,
                "ocr_image": ocr_image_path.name,
                "auto_rotated_landscape": auto_rotated,
                "ok": ok,
                "elapsed_seconds": elapsed,
                "text_chars": len(text),
                "error": error,
            }
            results_by_name[img.name] = (text, meta)

            if ok:
                (page_out_dir / f"{img.stem}.txt").write_text(text, encoding="utf-8")
            (page_out_dir / f"{img.stem}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "OCR %d/%d | %s | ok=%s | elapsed=%.1fs | chars=%d",
                idx,
                len(images),
                img.name,
                ok,
                elapsed,
                len(text),
            )

    tasks = [asyncio.create_task(_ocr_one(i, img)) for i, img in enumerate(images, start=1)]
    for fut in asyncio.as_completed(tasks):
        await fut

    ordered_pairs: List[Tuple[str, str]] = []
    page_meta: List[Dict[str, Any]] = []
    for img in images:
        text, meta = results_by_name.get(img.name, ("", {"image": img.name, "ok": False, "error": "missing_result"}))
        ordered_pairs.append((img.name, text))
        page_meta.append(meta)

    merged_text = pipeline_mod.merge_ocr_text_blocks(ordered_pairs)
    (output_dir / f"{args.patient_name}_ocr_merged.txt").write_text(merged_text, encoding="utf-8")
    (output_dir / f"{args.patient_name}_ocr_page_meta.json").write_text(
        json.dumps(page_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok_count = sum(1 for m in page_meta if bool(m.get("ok")))
    fail_count = len(page_meta) - ok_count
    total_elapsed = time.perf_counter() - started
    elapsed_values = [float(m.get("elapsed_seconds", 0.0) or 0.0) for m in page_meta]
    summary = {
        "patient_name": args.patient_name,
        "images_total": len(images),
        "ocr_ok": ok_count,
        "ocr_fail": fail_count,
        "duplicates_dropped": len(duplicates),
        "total_elapsed_seconds": total_elapsed,
        "avg_elapsed_seconds_per_image": (sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0),
        "max_elapsed_seconds_per_image": (max(elapsed_values) if elapsed_values else 0.0),
    }
    (output_dir / "ocr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if duplicates:
        (output_dir / "duplicates_dropped.json").write_text(
            json.dumps(duplicates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    logger.info("OCR-only run complete: %s", json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run OCR-only pipeline for a single patient folder with a local open-source VLM.")
    ap.add_argument("--pipeline_script", type=str, default="103_paper_to_cdm_SA.py")
    ap.add_argument("--local_vlm_script", type=str, default="106_paper_to_cdm_SA_live_local_vlm.py")
    ap.add_argument("--deepseek_package_root", type=str, default="vendor/deepseek_vl2")
    ap.add_argument("--input_root", type=str, default="paper_patients")
    ap.add_argument("--patient_name", type=str, default="Patient_10")
    ap.add_argument("--output_dir", type=str, default="out_ocr_only_patient10")

    ap.add_argument("--model_id", type=str, default="")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="auto|bfloat16|float16|float32")
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--disable_trust_remote_code", action="store_true")
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0)
    ap.add_argument(
        "--qwen_enable_thinking",
        action="store_true",
        help="Enable Qwen thinking mode instead of the default non-thinking OCR baseline.",
    )
    ap.add_argument("--image_max_side", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_inflight", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--preload_model", action="store_true")
    ap.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--gemini_api_key_env", type=str, default="GOOGLE_API_KEY")
    ap.add_argument("--request_timeout_sec", type=float, default=180.0)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--disable_dedup", action="store_true")
    ap.add_argument("--near_dup_hamming", type=int, default=6)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not str(args.model_id).strip():
        raise ValueError("--model_id is required.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
