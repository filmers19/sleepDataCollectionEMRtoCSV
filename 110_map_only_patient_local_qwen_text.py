from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger("map_local_qwen_text")


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def configure_logging(output_dir: Path, debug: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "map_local_qwen.log"
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

    for name in ("transformers", "accelerate", "torch", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING if not debug else logging.INFO)

    logger.info("Logging initialized: %s", log_path)


class LocalQwenTextAgent:
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
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype_value,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs).eval()
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

    def _text_sync(self, system_prompt: str, user_text: str) -> str:
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        inputs = self._apply_chat_template(messages)
        device = self._model_device()
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        with self._torch.inference_mode():
            outputs = self._model.generate(inputs, **gen_kwargs)
        prompt_len = int(inputs.shape[1])
        trimmed = outputs[:, prompt_len:]
        texts = self._tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (texts[0] if texts else "").strip()

    async def atext(self, system_prompt: str, user_text: str) -> str:
        await self._ensure_loaded()
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


async def local_text_to_json(
    llm: LocalQwenTextAgent,
    system_prompt: str,
    user_text: str,
    pipeline_mod: Any,
    schema_hint: str,
    max_attempts: int,
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
                    "Invalid local JSON response. Retrying full generation (%d/%d): %s",
                    attempt,
                    attempts,
                    _summarize_json_failure(last_raw, exc),
                )
    raise RuntimeError(
        f"Could not obtain valid local JSON after {attempts} attempts: "
        f"{_summarize_json_failure(last_raw, last_error or RuntimeError('unknown'))}"
    ) from last_error


async def local_map_to_json(
    llm: LocalQwenTextAgent,
    pipeline_mod: Any,
    ocr_text: str,
    candidates_block: str,
    max_attempts: int,
) -> Dict[str, Any]:
    user = pipeline_mod.build_map_user_prompt(ocr_text, candidates_block)
    schema_hint = (
        '{"CDM_KEY": {"value": <scalar>, '
        '"input_context": {"filled_by": "doctor|patient|unknown", "question": "<text>", "page": "<summary>"}}}'
    )
    return await local_text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.MAP_SYSTEM,
        user_text=user,
        pipeline_mod=pipeline_mod,
        schema_hint=schema_hint,
        max_attempts=max_attempts,
    )


async def resolve_conflicts_with_local_qwen(
    llm: LocalQwenTextAgent,
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
    raw = await local_text_to_json(
        llm=llm,
        system_prompt=pipeline_mod.CONFLICT_RESOLVER_SYSTEM,
        user_text=user,
        pipeline_mod=pipeline_mod,
        schema_hint='{"resolved":{"CDM_KEY":{"chosen_index": <int>, "reason": "<brief reason>"}}}',
        max_attempts=json_retry_attempts,
    )
    resolved_obj = raw.get("resolved", raw)
    overrides: Dict[str, Any] = {}
    decisions: Dict[str, Any] = {}
    if not isinstance(resolved_obj, dict):
        return overrides, decisions
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
    return overrides, decisions


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir=output_dir, debug=args.debug)

    pipeline_mod = load_module(Path(args.pipeline_script).resolve(), "paper_to_cdm_sa_local_text")
    if callable(getattr(pipeline_mod, "load_env", None)):
        pipeline_mod.load_env()

    input_root = Path(args.input_root).resolve()
    patient_dir = input_root / args.patient_name
    reuse_ocr_dir = Path(args.reuse_ocr_dir).resolve()
    cdm_csv = Path(args.cdm_csv).resolve()
    example_csv = Path(args.example_csv).resolve()

    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")
    if not reuse_ocr_dir.exists():
        raise FileNotFoundError(f"reuse OCR dir not found: {reuse_ocr_dir}")

    images = pipeline_mod.iter_images(patient_dir)
    if not images:
        raise RuntimeError(f"No images found in {patient_dir}")

    ocr_page_dir = output_dir / "ocr_pages"
    map_page_dir = output_dir / "map_pages"
    ocr_page_dir.mkdir(parents=True, exist_ok=True)
    map_page_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "patient_name": args.patient_name,
        "patient_dir": str(patient_dir),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "reuse_ocr_dir": str(reuse_ocr_dir),
        "map_bundle_size": 1,
        "map_agent_count": 1,
        "disable_recall": True,
        "disable_conflict_resolver": args.disable_conflict_resolver,
        "map_json_retry_attempts": args.map_json_retry_attempts,
        "resolver_json_retry_attempts": args.resolver_json_retry_attempts,
    }
    (output_dir / "map_local_qwen_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Plan: %s", json.dumps(plan, ensure_ascii=False))

    llm = LocalQwenTextAgent(
        model_id=args.model_id,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_inflight=args.max_inflight,
        trust_remote_code=(not args.disable_trust_remote_code),
        attn_implementation=args.attn_implementation,
    )
    if args.preload_model:
        await llm._ensure_loaded()

    retriever = pipeline_mod.CDMRetriever(cdm_csv)
    output_columns = list(pd.read_csv(example_csv, nrows=0).columns)

    ordered_ocr_pairs: List[Tuple[Path, str]] = []
    for idx, img in enumerate(images, start=1):
        src_txt = reuse_ocr_dir / f"{img.stem}.txt"
        src_meta = reuse_ocr_dir / f"{img.stem}.meta.json"
        if not src_txt.exists():
            raise FileNotFoundError(f"Frozen OCR text not found: {src_txt}")
        text = src_txt.read_text(encoding="utf-8")
        ordered_ocr_pairs.append((img, text))
        (ocr_page_dir / f"{img.stem}.txt").write_text(text, encoding="utf-8")
        if src_meta.exists():
            (ocr_page_dir / f"{img.stem}.meta.json").write_text(src_meta.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("OCR %d/%d | %s | reused=True | chars=%d", idx, len(images), img.name, len(text))

    page_results: List[Any] = []
    page_errors: List[Dict[str, str]] = []
    started = time.perf_counter()

    for idx, (img, text) in enumerate(ordered_ocr_pairs, start=1):
        bundle_name = pipeline_mod.make_bundle_image_name(idx, [img.name])
        t0 = time.perf_counter()
        try:
            raw_obj = await local_map_to_json(
                llm=llm,
                pipeline_mod=pipeline_mod,
                ocr_text=text,
                candidates_block=retriever.full_cdm_prompt_block(),
                max_attempts=args.map_json_retry_attempts,
            )
            stage_raw: Dict[str, Any] = {}
            stage_valid: Dict[str, Any] = {}
            stage_contexts: Dict[str, Dict[str, str]] = {}
            stage_rejected: Dict[str, Dict[str, Any]] = {}
            pipeline_mod.merge_map_payload_into_stage(
                retriever=retriever,
                ocr_text=text,
                raw_payload=raw_obj,
                stage_raw=stage_raw,
                stage_valid=stage_valid,
                stage_contexts=stage_contexts,
                stage_rejected=stage_rejected,
            )
            page_results.append(
                pipeline_mod.PageResult(
                    image_name=bundle_name,
                    ocr_text=text,
                    raw_json=stage_raw,
                    valid_json=stage_valid,
                    input_contexts=stage_contexts,
                    rejected_fields=stage_rejected,
                )
            )
            (map_page_dir / f"{Path(bundle_name).stem}.raw.json").write_text(json.dumps(stage_raw, ensure_ascii=False, indent=2), encoding="utf-8")
            (map_page_dir / f"{Path(bundle_name).stem}.valid.json").write_text(json.dumps(stage_valid, ensure_ascii=False, indent=2), encoding="utf-8")
            (map_page_dir / f"{Path(bundle_name).stem}.contexts.json").write_text(json.dumps(stage_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
            if stage_rejected:
                (map_page_dir / f"{Path(bundle_name).stem}.rejected.json").write_text(json.dumps(stage_rejected, ensure_ascii=False, indent=2), encoding="utf-8")
            ok = True
            valid_count = len(stage_valid)
            error = ""
        except Exception as e:
            ok = False
            valid_count = 0
            error = f"{type(e).__name__}: {e}"
            page_errors.append({"image": bundle_name, "error_type": type(e).__name__, "error": str(e)})
        meta = {
            "bundle": bundle_name,
            "source_images": [img.name],
            "ok": ok,
            "elapsed_seconds": time.perf_counter() - t0,
            "valid_keys": valid_count,
            "error": error,
        }
        (map_page_dir / f"{Path(bundle_name).stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("MAP %d/%d | %s | ok=%s | elapsed=%.1fs | valid_keys=%d", idx, len(ordered_ocr_pairs), bundle_name, ok, meta["elapsed_seconds"], valid_count)

    patient_res = pipeline_mod.build_patient_result(
        patient_name=args.patient_name,
        page_results=page_results,
        duplicates=[],
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
        overrides, decisions = await resolve_conflicts_with_local_qwen(
            llm=llm,
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

    pipeline_mod.write_patient_outputs(
        output_dir=output_dir,
        patient_name=args.patient_name,
        res=patient_res,
        output_columns=output_columns,
    )
    merged_ocr = pipeline_mod.merge_ocr_text_blocks([(img.name, txt) for img, txt in ordered_ocr_pairs])
    (output_dir / f"{args.patient_name}_ocr_merged.txt").write_text(merged_ocr, encoding="utf-8")
    summary = {
        "patient_name": args.patient_name,
        "images_total": len(images),
        "ocr_ok": len(images),
        "ocr_fail": 0,
        "map_ok": len(page_results),
        "map_fail": len(images) - len(page_results),
        "total_elapsed_seconds": time.perf_counter() - started,
        "output_row_non_null_keys": (
            sum(1 for v in (patient_res.get("row") or {}).values() if not pipeline_mod._is_missing_value(v))
            if patient_res.get("row") is not None
            else 0
        ),
    }
    (output_dir / "map_local_qwen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Local map run complete: %s", json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map-only single-patient runner using frozen OCR and local Qwen text model.")
    ap.add_argument("--pipeline_script", type=str, default="103_paper_to_cdm_SA.py")
    ap.add_argument("--input_root", type=str, default="paper_patients")
    ap.add_argument("--patient_name", type=str, default="Patient_10")
    ap.add_argument("--reuse_ocr_dir", type=str, required=True)
    ap.add_argument("--cdm_csv", type=str, default="cdm_revised.csv")
    ap.add_argument("--example_csv", type=str, default="example.csv")
    ap.add_argument("--output_dir", type=str, default="out_map_patient10_local_qwen35")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--disable_trust_remote_code", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_inflight", type=int, default=1)
    ap.add_argument("--map_json_retry_attempts", type=int, default=3)
    ap.add_argument("--resolver_json_retry_attempts", type=int, default=2)
    ap.add_argument("--disable_conflict_resolver", action="store_true")
    ap.add_argument("--preload_model", action="store_true")
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
