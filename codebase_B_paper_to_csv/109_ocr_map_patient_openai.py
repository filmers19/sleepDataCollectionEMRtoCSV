from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

logger = logging.getLogger("ocr_map_openai")
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
    log_path = output_dir / "ocr_map.log"
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

    def _text_sync(self, system_prompt: str, user_text: str) -> str:
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
                data = self._http_json(req)
                return self._extract_text(data)
            except urlerror.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = str(exc)
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    logger.warning(
                        "Transient OpenAI map error (status=%s, attempt=%d/%d). Retrying in %.1fs.",
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
                        "Transient OpenAI map failure (attempt=%d/%d). Retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise
        raise RuntimeError("OpenAI text request failed after retries")

    async def atext(self, system_prompt: str, user_text: str) -> str:
        async with self._sem:
            return await asyncio.to_thread(self._text_sync, system_prompt, user_text)


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


async def openai_map_to_json(
    llm: RemoteOpenAIResponsesTextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    candidates_block: str,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    user = pipeline_mod.build_map_user_prompt(ocr_text, candidates_block)
    schema_hint = (
        '{"CDM_KEY": {"value": <scalar>, '
        '"input_context": {"filled_by": "doctor|patient|unknown", "question": "<text>", "page": "<summary>"}}}'
    )
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
        raw = await llm.atext(system_prompt=pipeline_mod.MAP_SYSTEM, user_text=user + retry_note)
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
            )
            last_raw = fixed
            return _extract_json_dict(pipeline_mod, fixed)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning(
                    "Invalid OpenAI map JSON. Retrying full generation (%d/%d): %s",
                    attempt,
                    attempts,
                    _summarize_json_failure(last_raw, exc),
                )

    raise RuntimeError(
        f"Could not obtain valid map JSON after {attempts} attempts: "
        f"{_summarize_json_failure(last_raw, last_error or RuntimeError('unknown'))}"
    ) from last_error


async def openai_text_to_json(
    llm: RemoteOpenAIResponsesTextAgent,
    system_prompt: str,
    user_text: str,
    pipeline_mod: Any,
    schema_hint: str,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    last_raw = ""
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        raw = await llm.atext(system_prompt=system_prompt, user_text=user_text)
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
            )
            last_raw = fixed
            return _extract_json_dict(pipeline_mod, fixed)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning(
                    "Invalid OpenAI JSON response. Retrying full generation (%d/%d): %s",
                    attempt,
                    attempts,
                    _summarize_json_failure(last_raw, exc),
                )

    raise RuntimeError(
        f"Could not obtain valid JSON after {attempts} attempts: "
        f"{_summarize_json_failure(last_raw, last_error or RuntimeError('unknown'))}"
    ) from last_error


async def map_ocr_text_single_agent(
    llm: RemoteOpenAIResponsesTextAgent,
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    json_retry_attempts: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}

    raw = await openai_map_to_json(
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
    return stage_raw, stage_valid, stage_contexts, stage_rejected


async def map_ocr_text_multi_agent(
    llm: RemoteOpenAIResponsesTextAgent,
    pipeline_mod: Any,
    retriever: Any,
    ocr_text: str,
    map_agent_count: int,
    json_retry_attempts: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    n_agents = max(1, int(map_agent_count))
    if n_agents <= 1:
        return await map_ocr_text_single_agent(
            llm=llm,
            pipeline_mod=pipeline_mod,
            retriever=retriever,
            ocr_text=ocr_text,
            json_retry_attempts=json_retry_attempts,
        )

    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}
    map_agents = pipeline_mod.build_map_agent_specs(retriever, n_agents)

    async def _call(agent: Any) -> Tuple[Any, Dict[str, Any]]:
        payload = await openai_map_to_json(
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
    return stage_raw, stage_valid, stage_contexts, stage_rejected


async def resolve_single_conflict_with_openai(
    llm: RemoteOpenAIResponsesTextAgent,
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
    raw = await openai_text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.CONFLICT_RESOLVER_SYSTEM,
        user_text=user,
        pipeline_mod=pipeline_mod,
        schema_hint='{"chosen_index": <int>, "reason": "<brief reason>"}',
        max_attempts=json_retry_attempts,
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


async def resolve_conflicts_with_openai(
    llm: RemoteOpenAIResponsesTextAgent,
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
        raw = await openai_text_to_json(
            llm=llm,
            system_prompt=pipeline_mod.CONFLICT_RESOLVER_SYSTEM,
            user_text=user,
            pipeline_mod=pipeline_mod,
            schema_hint='{"resolved":{"CDM_KEY":{"chosen_index": <int>, "reason": "<brief reason>"}}}',
            max_attempts=json_retry_attempts,
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
            one = await resolve_single_conflict_with_openai(
                llm=llm,
                pipeline_mod=pipeline_mod,
                retriever=retriever,
                patient_name=patient_name,
                key=key,
                entries=conflicts[key],
                json_retry_attempts=json_retry_attempts,
            )
        except Exception as e:
            logger.warning("OpenAI per-key conflict resolver failed for %s/%s: %s", patient_name, key, e)
            continue
        if one is None:
            continue
        idx, reason = one
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


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir=output_dir, debug=args.debug)

    pipeline_mod = load_module(resolve_script_path(args.pipeline_script), "paper_to_cdm_sa_ocr_map")
    ocr_mod = load_module(resolve_script_path(args.ocr_script), "ocr_only_openai_map")
    if callable(getattr(pipeline_mod, "load_env", None)):
        pipeline_mod.load_env()

    input_root = resolve_repo_path(args.input_root)
    patient_dir = input_root / args.patient_name
    if not patient_dir.exists() or not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    cdm_csv = resolve_repo_path(args.cdm_csv)
    example_csv = resolve_repo_path(args.example_csv)
    reuse_ocr_dir = resolve_repo_path(args.reuse_ocr_dir) if args.reuse_ocr_dir else None
    if not cdm_csv.exists():
        raise FileNotFoundError(f"CDM CSV not found: {cdm_csv}")
    if not example_csv.exists():
        raise FileNotFoundError(f"example.csv not found: {example_csv}")
    if reuse_ocr_dir is not None and not reuse_ocr_dir.exists():
        raise FileNotFoundError(f"reuse OCR dir not found: {reuse_ocr_dir}")

    images = pipeline_mod.iter_images(patient_dir)
    if not images:
        raise RuntimeError(f"No images found in {patient_dir}")

    duplicates: List[Dict[str, Any]] = []
    if not args.disable_dedup:
        images, duplicates = pipeline_mod.deduplicate_images(images, near_dup_hamming=args.near_dup_hamming)

    ocr_page_dir = output_dir / "ocr_pages"
    map_page_dir = output_dir / "map_pages"
    ocr_page_dir.mkdir(parents=True, exist_ok=True)
    map_page_dir.mkdir(parents=True, exist_ok=True)
    prepared_image_dir = (output_dir / "prepared_images") if args.auto_rotate_landscape else None

    plan = {
        "patient_name": args.patient_name,
        "patient_dir": str(patient_dir),
        "output_dir": str(output_dir),
        "ocr_model_id": args.ocr_model_id,
        "map_model_id": args.map_model_id,
        "cdm_csv": str(cdm_csv),
        "example_csv": str(example_csv),
        "map_bundle_size": args.map_bundle_size,
        "single_map_agent_full_cdm": (max(1, int(args.map_agent_count)) == 1),
        "map_agent_count": max(1, int(args.map_agent_count)),
        "map_json_retry_attempts": max(1, int(args.map_json_retry_attempts)),
        "resolver_json_retry_attempts": max(1, int(args.resolver_json_retry_attempts)),
        "bundle_label_classification": False,
        "disable_recall": True,
        "disable_conflict_resolver": args.disable_conflict_resolver,
        "reuse_ocr_dir": str(reuse_ocr_dir) if reuse_ocr_dir is not None else "",
        "image_max_side": args.image_max_side,
        "auto_rotate_landscape": args.auto_rotate_landscape,
        "auto_rotate_landscape_ratio": args.auto_rotate_landscape_ratio,
        "ocr_concurrency": args.ocr_concurrency,
        "map_concurrency": args.map_concurrency,
        "images_total": len(images),
        "duplicates_dropped": len(duplicates),
    }
    (output_dir / "ocr_map_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Plan: %s", json.dumps(plan, ensure_ascii=False))

    ocr_backend = ocr_mod.RemoteOpenAIResponsesOCR(
        model_id=args.ocr_model_id,
        max_new_tokens=args.ocr_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_inflight=args.ocr_concurrency,
        timeout_sec=args.request_timeout_sec,
        max_retries=args.max_retries,
        api_key_env=args.openai_api_key_env,
        image_max_side=args.image_max_side,
    )
    map_backend = RemoteOpenAIResponsesTextAgent(
        model_id=args.map_model_id,
        max_new_tokens=args.map_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_inflight=args.map_concurrency,
        timeout_sec=args.request_timeout_sec,
        max_retries=args.max_retries,
        api_key_env=args.openai_api_key_env,
    )
    conflict_backend = map_backend

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
            except Exception as e:
                meta = {
                    "image": img.name,
                    "ocr_image": ocr_image_path.name,
                    "auto_rotated_landscape": auto_rotated,
                    "ok": False,
                    "elapsed_seconds": time.perf_counter() - t0,
                    "text_chars": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
                page_errors.append({"image": img.name, "error_type": type(e).__name__, "error": str(e)})
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
                error = ""
                valid_count = len(valid_obj)
            except Exception as e:
                ok = False
                error = f"{type(e).__name__}: {e}"
                valid_count = 0
                page_errors.append({"image": bundle_name, "error_type": type(e).__name__, "error": str(e)})

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
        patient_name=args.patient_name,
        page_results=page_results,
        duplicates=duplicates,
        page_errors=page_errors,
        output_columns=output_columns,
        save_intermediate=args.save_intermediate,
        out_dir=output_dir,
        elapsed_s=(time.perf_counter() - started),
    )
    if (
        not args.disable_conflict_resolver
        and patient_res.get("row") is not None
        and patient_res.get("conflicts")
    ):
        overrides, decisions = await resolve_conflicts_with_openai(
            llm=conflict_backend,
            pipeline_mod=pipeline_mod,
            retriever=retriever,
            patient_name=args.patient_name,
            conflicts=patient_res.get("conflicts") or {},
            json_retry_attempts=args.resolver_json_retry_attempts,
        )
        if overrides:
            merged_like = dict(patient_res.get("merged") or {})
            for k, v in overrides.items():
                if k in output_columns:
                    merged_like[k] = v
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
        patient_name=args.patient_name,
        res=patient_res,
        output_columns=output_columns,
    )

    ocr_merged_text = pipeline_mod.merge_ocr_text_blocks([(img.name, txt) for img, txt in ordered_ocr_pairs])
    (output_dir / f"{args.patient_name}_ocr_merged.txt").write_text(ocr_merged_text, encoding="utf-8")

    summary = {
        "patient_name": args.patient_name,
        "images_total": len(images),
        "ocr_ok": len(ordered_ocr_pairs),
        "ocr_fail": len(images) - len(ordered_ocr_pairs),
        "map_ok": len(page_results),
        "map_fail": len(bundles) - len(page_results),
        "duplicates_dropped": len(duplicates),
        "total_elapsed_seconds": time.perf_counter() - started,
        "output_row_non_null_keys": (
            sum(1 for v in (patient_res.get("row") or {}).values() if not pipeline_mod._is_missing_value(v))
            if patient_res.get("row") is not None
            else 0
        ),
    }
    (output_dir / "ocr_map_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("OCR->MAP run complete: %s", json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Two-stage OCR->MAP runner for a single patient using OpenAI Responses.")
    ap.add_argument("--pipeline_script", type=str, default="103_paper_to_cdm_SA.py")
    ap.add_argument("--ocr_script", type=str, default="108_ocr_only_patient_local_qwen.py")
    ap.add_argument("--input_root", type=str, default="paper_patients")
    ap.add_argument("--patient_name", type=str, default="Patient_10")
    ap.add_argument("--output_dir", type=str, default="out_ocr_map_patient10")
    ap.add_argument("--cdm_csv", type=str, default="cdm_revised.csv")
    ap.add_argument("--example_csv", type=str, default="example.csv")
    ap.add_argument("--ocr_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--map_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--image_max_side", type=int, default=2048)
    ap.add_argument("--map_bundle_size", type=int, default=1)
    ap.add_argument("--map_agent_count", type=int, default=1, help="Number of map agents. 1 = single full-CDM map agent.")
    ap.add_argument("--ocr_max_new_tokens", type=int, default=4096)
    ap.add_argument("--map_max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--ocr_concurrency", type=int, default=1)
    ap.add_argument("--map_concurrency", type=int, default=1)
    ap.add_argument("--request_timeout_sec", type=float, default=180.0)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--map_json_retry_attempts", type=int, default=3, help="Full regeneration attempts for malformed/incomplete map JSON")
    ap.add_argument("--resolver_json_retry_attempts", type=int, default=2, help="Full regeneration attempts for malformed/incomplete conflict-resolver JSON")
    ap.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--auto_rotate_landscape", action="store_true")
    ap.add_argument("--auto_rotate_landscape_ratio", type=float, default=1.05)
    ap.add_argument("--disable_dedup", action="store_true")
    ap.add_argument("--near_dup_hamming", type=int, default=6)
    ap.add_argument("--disable_conflict_resolver", action="store_true")
    ap.add_argument("--reuse_ocr_dir", type=str, default="", help="Reuse existing OCR txt/meta files from a prior run's ocr_pages dir")
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
