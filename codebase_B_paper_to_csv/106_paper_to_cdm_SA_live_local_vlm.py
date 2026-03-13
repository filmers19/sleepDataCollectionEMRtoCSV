from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import io
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote_to_bytes

import pandas as pd
from PIL import Image

logger = logging.getLogger("local_qwen_vlm")
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


def configure_third_party_logging(quiet: bool) -> None:
    if not quiet:
        return
    noisy = (
        "google_genai.models",
        "google.genai._api_client",
        "google_genai._api_client",
        "httpx",
        "httpcore",
        "tenacity",
    )
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)


def classify_request_kind(messages: Iterable[Any]) -> str:
    text_parts: List[str] = []
    image_parts = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                    continue
                if isinstance(part, dict):
                    ptype = str(part.get("type", "")).lower()
                    if ptype == "image_url" or "image_url" in part or "inlineData" in part:
                        image_parts += 1
                    txt = part.get("text")
                    if isinstance(txt, str):
                        text_parts.append(txt)

    if image_parts > 0:
        return "ocr_remote"

    low = "\n".join(text_parts).lower()
    if "strict json repair assistant" in low or "fix into a valid json object only" in low:
        return "json_fix"
    if "resolve one conflict candidate set" in low or "conflict candidates json" in low:
        return "conflict"
    if "existing json (do not repeat these keys)" in low:
        return "map_recall"
    if "candidate cdm fields" in low:
        return "map"
    return "other"


def build_even_map_agent_specs(mod: Any, retriever: Any, agent_count: int) -> List[Any]:
    n = max(1, int(agent_count))
    rows = list(retriever.rows)
    total = len(rows)
    if total == 0:
        return []
    # Balance by approximate prompt/context size instead of raw row count.
    row_weights: List[int] = []
    for row in rows:
        block = mod.format_candidate_rows([(row, 1.0)], include_score=False, max_chars=200000)
        row_weights.append(max(1, len(block)))

    total_weight = sum(row_weights)
    target_thresholds = [(total_weight * i) / n for i in range(1, n)]
    cuts: List[int] = []

    cum_weight = 0
    threshold_idx = 0
    for i, w in enumerate(row_weights, start=1):
        cum_weight += w
        if threshold_idx >= len(target_thresholds):
            break
        if cum_weight < target_thresholds[threshold_idx]:
            continue
        remaining_rows = total - i
        groups_left_after_cut = n - (len(cuts) + 1)
        if remaining_rows >= groups_left_after_cut:
            cuts.append(i)
            threshold_idx += 1

    while len(cuts) < (n - 1):
        min_next = (cuts[-1] + 1) if cuts else 1
        max_next = total - ((n - 1) - len(cuts))
        cuts.append(max(min_next, max_next))

    boundaries = [0] + cuts + [total]
    specs: List[Any] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if start >= end:
            continue
        chunk_rows = rows[start:end]
        start_key = chunk_rows[0].key
        end_key = chunk_rows[-1].key
        cands = [(r, 1.0) for r in chunk_rows]
        block = mod.format_candidate_rows(cands, include_score=False, max_chars=50000)
        specs.append(
            mod.MapAgentSpec(
                name=f"agent_{idx + 1:02d}_even",
                start_key=start_key,
                end_key=end_key,
                rows=chunk_rows,
                candidates_block=block,
            )
        )

    chunk_stats = []
    for spec in specs:
        chunk_stats.append(f"{spec.name}:{len(spec.rows)}rows/{len(spec.candidates_block)}chars")
    mod.logger.info(
        "Configured %d even split map agents (total_rows=%d, total_chars=%d): %s",
        len(specs),
        total,
        total_weight,
        ", ".join(chunk_stats),
    )
    return specs


def load_pipeline_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("paper_to_cdm_sa", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    # Required for dataclass/type resolution on Python 3.13 during dynamic import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class LocalLLMResponse:
    content: str


class LocalQwenVLM:
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
        min_pixels: Optional[int],
        max_pixels: Optional[int],
        enable_thinking: bool = False,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = (attn_implementation or "").strip()
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.enable_thinking = bool(enable_thinking)

        self._load_lock = threading.Lock()
        self._sem = asyncio.Semaphore(max(1, int(max_inflight)))
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> Any:
        assert self._processor is not None
        kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        # Qwen3.5 enables thinking by default; keep it disabled unless explicitly requested.
        if self.enable_thinking:
            return self._processor.apply_chat_template(messages, **kwargs)
        try:
            return self._processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self._processor.apply_chat_template(messages, **kwargs)

    def _resolve_dtype(self, torch_mod: Any) -> Any:
        alias = self.dtype.strip().lower()
        if alias in {"", "auto"}:
            return "auto"
        if alias in {"bf16", "bfloat16"}:
            return torch_mod.bfloat16
        if alias in {"fp16", "float16", "half"}:
            return torch_mod.float16
        if alias in {"fp32", "float32"}:
            return torch_mod.float32
        raise ValueError(f"Unsupported --dtype value: {self.dtype}")

    def _pick_model_cls(self, transformers_mod: Any) -> Any:
        model_id_l = self.model_id.lower()

        cls_qwen3_moe = getattr(transformers_mod, "Qwen3VLMoeForConditionalGeneration", None)
        cls_qwen3_dense = getattr(transformers_mod, "Qwen3VLForConditionalGeneration", None)
        cls_qwen25 = getattr(transformers_mod, "Qwen2_5_VLForConditionalGeneration", None)

        if "qwen3-vl" in model_id_l and cls_qwen3_moe is not None and ("a3b" in model_id_l or "moe" in model_id_l):
            return cls_qwen3_moe
        if "qwen3-vl" in model_id_l and cls_qwen3_dense is not None:
            return cls_qwen3_dense
        if "qwen2.5-vl" in model_id_l and cls_qwen25 is not None:
            return cls_qwen25

        for auto_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            auto_cls = getattr(transformers_mod, auto_name, None)
            if auto_cls is not None:
                return auto_cls

        raise RuntimeError(
            "No compatible transformers class found for Qwen VL models. "
            "Install newer transformers from source."
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        with self._load_lock:
            if self._model is not None and self._processor is not None and self._torch is not None:
                return

            try:
                import torch
                import transformers
                from transformers import AutoProcessor
            except Exception as e:
                raise RuntimeError(
                    "Missing local VLM dependencies. Install torch/transformers/accelerate first."
                ) from e

            model_cls = self._pick_model_cls(transformers)
            dtype_value = self._resolve_dtype(torch)

            processor_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
            if self.min_pixels is not None:
                processor_kwargs["min_pixels"] = int(self.min_pixels)
            if self.max_pixels is not None:
                processor_kwargs["max_pixels"] = int(self.max_pixels)
            self._processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)

            attn_pref = (self.attn_implementation or "").strip()
            if attn_pref == "flash_attention_2":
                attn_attempts = ["flash_attention_2", "sdpa", "eager"]
            elif attn_pref:
                attn_attempts = [attn_pref]
            else:
                attn_attempts = [""]

            last_exc: Optional[Exception] = None
            for idx, attn_impl in enumerate(attn_attempts):
                model_kwargs: Dict[str, Any] = {
                    "torch_dtype": dtype_value,
                    "device_map": "auto",
                    "trust_remote_code": self.trust_remote_code,
                }
                if attn_impl:
                    model_kwargs["attn_implementation"] = attn_impl
                try:
                    self._model = model_cls.from_pretrained(self.model_id, **model_kwargs)
                    self._model.eval()
                    if idx > 0:
                        logger.warning(
                            "Fell back attention backend from '%s' to '%s' for %s.",
                            attn_pref or "auto",
                            attn_impl or "auto",
                            self.model_id,
                        )
                    self.attn_implementation = attn_impl
                    break
                except ImportError as e:
                    last_exc = e
                    msg = str(e).lower()
                    is_flash_missing = ("flashattention2" in msg) or ("flash_attn" in msg)
                    if (attn_impl == "flash_attention_2") and is_flash_missing and (idx + 1) < len(attn_attempts):
                        logger.warning(
                            "flash_attention_2 unavailable; retrying model load with '%s'.",
                            attn_attempts[idx + 1],
                        )
                        continue
                    raise

            if self._model is None:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError(f"Failed to load model: {self.model_id}")
            self._torch = torch

    @staticmethod
    def _message_role(msg: Any) -> str:
        role = str(getattr(msg, "type", "")).strip().lower()
        if role in {"human", "user"}:
            return "user"
        if role in {"ai", "assistant"}:
            return "assistant"
        if role == "system":
            return "system"
        return "user"

    @staticmethod
    def _decode_data_url(data_url: str) -> Image.Image:
        head, payload = data_url.split(",", 1)
        if ";base64" in head:
            raw = base64.b64decode(payload)
        else:
            raw = unquote_to_bytes(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _parse_image_ref(self, ref: Any) -> Any:
        if ref is None:
            raise ValueError("Image ref is None")
        if isinstance(ref, Image.Image):
            return ref.convert("RGB")
        if isinstance(ref, bytes):
            return Image.open(io.BytesIO(ref)).convert("RGB")

        s = str(ref).strip()
        if not s:
            raise ValueError("Image ref is empty")
        if s.startswith("data:image/"):
            return self._decode_data_url(s)
        if s.startswith("http://") or s.startswith("https://"):
            return s
        p = Path(s)
        if p.exists() and p.is_file():
            return str(p.resolve())
        return s

    def _normalize_message_content(self, content: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        if isinstance(content, str):
            txt = content.strip()
            if txt:
                out.append({"type": "text", "text": txt})
            return out

        if not isinstance(content, list):
            txt = str(content).strip()
            if txt:
                out.append({"type": "text", "text": txt})
            return out

        for part in content:
            if isinstance(part, str):
                txt = part.strip()
                if txt:
                    out.append({"type": "text", "text": txt})
                continue

            if not isinstance(part, dict):
                txt = str(part).strip()
                if txt:
                    out.append({"type": "text", "text": txt})
                continue

            ptype = str(part.get("type", "")).lower()
            if ptype == "text" or ("text" in part and ptype != "image_url"):
                txt = str(part.get("text", "")).strip()
                if txt:
                    out.append({"type": "text", "text": txt})
                continue

            if ptype == "image_url" or "image_url" in part:
                img_obj = part.get("image_url")
                ref = None
                if isinstance(img_obj, dict):
                    ref = img_obj.get("url")
                elif isinstance(img_obj, str):
                    ref = img_obj
                if ref is None and "url" in part:
                    ref = part.get("url")
                if ref is not None:
                    out.append({"type": "image", "image": self._parse_image_ref(ref)})
                continue

            if ptype == "image" and "image" in part:
                out.append({"type": "image", "image": self._parse_image_ref(part.get("image"))})
                continue

            txt = json.dumps(part, ensure_ascii=False)
            if txt:
                out.append({"type": "text", "text": txt})

        return out

    def _to_qwen_messages(self, messages: Iterable[Any]) -> List[Dict[str, Any]]:
        qmsgs: List[Dict[str, Any]] = []
        for msg in messages:
            role = self._message_role(msg)
            content = self._normalize_message_content(getattr(msg, "content", ""))
            if not content:
                content = [{"type": "text", "text": ""}]
            qmsgs.append({"role": role, "content": content})
        if not qmsgs:
            qmsgs = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        return qmsgs

    def _model_device(self) -> Any:
        assert self._model is not None
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")

    def _invoke_sync(self, messages: List[Any]) -> LocalLLMResponse:
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None and self._torch is not None

        qmsgs = self._to_qwen_messages(messages)
        model_inputs = self._apply_chat_template(qmsgs)

        device = self._model_device()
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(device)
        else:
            model_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(model_inputs).items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with self._torch.inference_mode():
            generated_ids = self._model.generate(**model_inputs, **gen_kwargs)

        input_ids = model_inputs["input_ids"]
        if hasattr(input_ids, "shape"):
            prompt_len = int(input_ids.shape[1])
            generated_trimmed = generated_ids[:, prompt_len:]
        else:
            generated_trimmed = generated_ids

        text_list = self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        out_text = text_list[0] if text_list else ""
        return LocalLLMResponse(content=str(out_text).strip())

    async def ainvoke(self, messages: List[Any], **_kwargs: Any) -> LocalLLMResponse:
        async with self._sem:
            return await asyncio.to_thread(self._invoke_sync, messages)

    def warmup(self) -> None:
        self._ensure_loaded()


def collect_dataset_overview(mod: Any, input_root: Path, near_dup_hamming: int) -> Dict[str, Any]:
    patient_dirs = mod.iter_patient_folders(input_root)
    raw_pages_total = 0
    kept_pages_total = 0
    for pdir in patient_dirs:
        imgs = mod.iter_images(pdir)
        raw_pages_total += len(imgs)
        kept_imgs, _dups = mod.deduplicate_images(imgs, near_dup_hamming=near_dup_hamming)
        kept_pages_total += len(kept_imgs)
    return {
        "patients": len(patient_dirs),
        "raw_pages_total": raw_pages_total,
        "kept_pages_total": kept_pages_total,
        "deduped_pages_total": max(0, raw_pages_total - kept_pages_total),
    }


async def run_pipeline_incremental_live(
    mod: Any,
    input_root: Path,
    cdm_csv: Path,
    example_csv: Path,
    output_dir: Path,
    map_bundle_size: int,
    use_split_map_agents: bool,
    concurrency: int,
    patient_concurrency: int,
    request_delay_sec: float,
    top_k: int,
    near_dup_hamming: int,
    debug: bool,
    log_filename: str,
    save_intermediate: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mod.configure_logging(output_dir=output_dir, debug=debug, log_filename=log_filename)
    mod.REQUEST_THROTTLE.configure(request_delay_sec)
    t_run0 = mod.time.perf_counter()

    example_df = pd.read_csv(example_csv)
    output_columns = list(example_df.columns)

    patient_dirs = mod.iter_patient_folders(input_root)
    mod.logger.info("Found %d patient folders", len(patient_dirs))
    mod.logger.info(
        (
            "Run config: use_batch_api=%s, patient_concurrency=%d, page_concurrency=%d, top_k=%d (ignored in full-CDM mode), "
            "near_dup_hamming=%d, save_intermediate=%s, map_bundle_size=%d, split_map_agents=%s, request_delay_sec=%.2f"
        ),
        False,
        patient_concurrency,
        concurrency,
        top_k,
        near_dup_hamming,
        save_intermediate,
        map_bundle_size,
        use_split_map_agents,
        request_delay_sec,
    )

    retriever = mod.CDMRetriever(cdm_csv)
    map_agents = mod.build_map_agent_specs(retriever) if use_split_map_agents else []
    resolver_llm: Optional[Any] = None
    try:
        resolver_llm = mod.build_gemini()
    except Exception as e:
        mod.logger.warning("Conflict resolver LLM is unavailable. Conflicting keys will remain unresolved. (%s)", e)

    async def _maybe_resolve_conflicts(res: Dict[str, Any]) -> None:
        res.setdefault("conflict_resolution", {})
        if resolver_llm is None:
            return
        if res.get("row") is None:
            return
        conflicts = res.get("conflicts") or {}
        if not conflicts:
            return
        try:
            overrides, decisions = await mod.resolve_conflicts_with_llm(
                llm=resolver_llm,
                retriever=retriever,
                patient_name=str(res.get("patient", "")),
                conflicts=conflicts,
            )
            if overrides:
                merged_like = {
                    k: v
                    for k, v in (res["row"] or {}).items()
                    if k in output_columns and not mod._is_missing_value(v)
                }
                for k, v in overrides.items():
                    if k in output_columns:
                        merged_like[k] = v
                res["row"] = mod.build_output_row(merged_like, output_columns)
            res["conflict_resolution"] = decisions
            mod.logger.info(
                "Patient %s conflict resolution: conflict_keys=%d overrides=%d",
                str(res.get("patient", "")),
                len(conflicts),
                len(overrides),
            )
        except Exception as e:
            mod.logger.warning("Conflict resolver failed for %s: %s", str(res.get("patient", "")), e)

    llm = resolver_llm if resolver_llm is not None else mod.build_gemini()
    done_count = 0
    patient_failures = 0
    results_by_patient: Dict[str, Dict[str, Any]] = {}

    sem = asyncio.Semaphore(max(1, int(patient_concurrency)))

    async def _process_patient_slot(idx: int, pdir: Path) -> Tuple[Path, Optional[Dict[str, Any]], Optional[Exception]]:
        async with sem:
            try:
                mod.logger.info("Processing patient %d/%d: %s", idx, len(patient_dirs), pdir.name)
                res = await mod.process_one_patient(
                    patient_dir=pdir,
                    llm=llm,
                    retriever=retriever,
                    map_agents=map_agents,
                    output_columns=output_columns,
                    concurrency=concurrency,
                    map_bundle_size=map_bundle_size,
                    top_k=top_k,
                    near_dup_hamming=near_dup_hamming,
                    save_intermediate=save_intermediate,
                    out_dir=output_dir,
                )
                return pdir, res, None
            except Exception as e:
                return pdir, None, e

    tasks = [
        asyncio.create_task(_process_patient_slot(idx=idx, pdir=pdir))
        for idx, pdir in enumerate(patient_dirs, start=1)
    ]

    for fut in asyncio.as_completed(tasks):
        pdir, res, err = await fut
        done_count += 1
        if err is not None or res is None:
            patient_failures += 1
            mod.logger.error("Failed processing patient folder %s: %s", pdir, err)
            mod.logger.info("Incremental progress: %d/%d patients completed (failures=%d)", done_count, len(patient_dirs), patient_failures)
            continue

        await _maybe_resolve_conflicts(res)
        mod.write_patient_outputs(output_dir=output_dir, patient_name=pdir.name, res=res, output_columns=output_columns)
        results_by_patient[pdir.name] = res
        mod.logger.info(
            "Incremental write complete: %s (%d/%d patients, failures=%d)",
            pdir.name,
            done_count,
            len(patient_dirs),
            patient_failures,
        )

    ordered_results = [results_by_patient[p.name] for p in patient_dirs if p.name in results_by_patient]

    rows = [r["row"] for r in ordered_results if r.get("row") is not None]
    if rows:
        df_all = pd.DataFrame(rows, columns=output_columns)
        df_all.to_csv(output_dir / "all_patients.csv", index=False)
        mod.logger.info("Wrote %d rows to %s", len(rows), output_dir / "all_patients.csv")
    else:
        mod.logger.warning("No patient rows produced.")

    total_page_errors = sum(len(r.get("page_errors", [])) for r in ordered_results)
    total_conflicts = sum(len(r.get("conflicts", {})) for r in ordered_results)
    total_duplicates = sum(len(r.get("duplicates", [])) for r in ordered_results)
    elapsed = mod.time.perf_counter() - t_run0
    mod.logger.info(
        (
            "Run summary: patients=%d patient_failures=%d rows=%d page_errors=%d "
            "conflicts=%d duplicates=%d elapsed_s=%.1f"
        ),
        len(patient_dirs),
        patient_failures,
        len(rows),
        total_page_errors,
        total_conflicts,
        total_duplicates,
        elapsed,
    )


async def run(args: argparse.Namespace) -> None:
    module_path = resolve_script_path(args.pipeline_script)
    mod = load_pipeline_module(module_path)
    mod.load_env()
    configure_third_party_logging(quiet=(not args.no_quiet_sdk_logs))
    t_run_wall_start = time.perf_counter()

    input_root = resolve_repo_path(args.input_root)
    cdm_csv = resolve_repo_path(args.cdm_csv)
    example_csv = resolve_repo_path(args.example_csv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = LocalQwenVLM(
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

    orig_transient = mod.is_transient_llm_error

    def _local_or_remote_transient(exc: Exception) -> bool:
        s = str(exc).lower()
        local_markers = (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "broken pipe",
            "429",
            "503",
        )
        return bool(orig_transient(exc) or any(m in s for m in local_markers))

    if args.llm_routing == "all_local":
        def _build_local_vlm() -> LocalQwenVLM:
            return backend

        mod.build_gemini = _build_local_vlm
        mod.is_transient_llm_error = _local_or_remote_transient
    elif args.llm_routing == "ocr_local_map_gemini":
        gemini_model = (args.gemini_model or "").strip()
        if not gemini_model:
            raise ValueError("--gemini_model is required when --llm_routing=ocr_local_map_gemini")
        os.environ["GEMINI_MODEL"] = gemini_model
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY is required for MAP/RECALL/CONFLICT steps "
                "when --llm_routing=ocr_local_map_gemini."
            )

        async def _ocr_with_local_qwen(llm: Any, image_path: Path, **_kwargs: Any) -> str:
            data_url = mod.image_to_data_url(image_path)
            user_prompt = getattr(mod, "OCR_USER_PROMPT", "Please transcribe this image.")
            msg = [
                mod.SystemMessage(content=mod.OCR_SYSTEM),
                mod.HumanMessage(
                    content=[
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                ),
            ]

            attempts = max(0, int(args.ocr_local_retries))
            for attempt in range(attempts + 1):
                try:
                    resp = await backend.ainvoke(msg)
                    return mod.llm_content_to_text(resp.content)
                except Exception as e:
                    if attempt >= attempts or (not _local_or_remote_transient(e)):
                        raise
                    delay = min(20.0, 1.5 * (2**attempt))
                    mod.logger.warning(
                        "Local OCR transient error (%s). Retrying in %.1fs (%d/%d).",
                        e,
                        delay,
                        attempt + 1,
                        attempts,
                    )
                    await asyncio.sleep(delay)

            raise RuntimeError(f"Local OCR failed unexpectedly for image={image_path}")

        mod.gemini_ocr = _ocr_with_local_qwen
    else:
        raise ValueError(f"Unsupported --llm_routing value: {args.llm_routing}")

    if not args.disable_split_map_agents:
        map_agent_count = max(1, int(args.map_agent_count))

        def _build_even_split_specs(retriever: Any) -> List[Any]:
            return build_even_map_agent_specs(mod=mod, retriever=retriever, agent_count=map_agent_count)

        mod.build_map_agent_specs = _build_even_split_specs

    mod.REQUEST_THROTTLE.configure(max(0.0, float(args.request_delay_sec)))

    dataset = collect_dataset_overview(mod, input_root=input_root, near_dup_hamming=args.near_dup_hamming)
    progress_state = {
        "ocr_total": int(dataset.get("kept_pages_total", 0)),
        "ocr_ok": 0,
        "ocr_fail": 0,
        "patients_total": int(dataset.get("patients", 0)),
        "patients_done": 0,
    }
    progress_lock = asyncio.Lock()
    timing_lock = threading.Lock()
    timing: Dict[str, Dict[str, float]] = {}

    def _record_timing(stage: str, elapsed_s: float) -> None:
        with timing_lock:
            stat = timing.setdefault(
                stage,
                {"calls": 0.0, "total_s": 0.0, "max_s": 0.0, "min_s": float("inf")},
            )
            stat["calls"] += 1.0
            stat["total_s"] += float(elapsed_s)
            stat["max_s"] = max(stat["max_s"], float(elapsed_s))
            stat["min_s"] = min(stat["min_s"], float(elapsed_s))

    async def _record_ocr_progress(image_path: Optional[Path], ok: bool) -> None:
        if progress_state["ocr_total"] <= 0:
            return
        async with progress_lock:
            if ok:
                progress_state["ocr_ok"] += 1
            else:
                progress_state["ocr_fail"] += 1
            done = progress_state["ocr_ok"] + progress_state["ocr_fail"]
            every_n = max(1, int(args.progress_every_pages))
            if done == 1 or done == progress_state["ocr_total"] or (done % every_n == 0):
                latest = ""
                if image_path is not None:
                    latest = f" | latest={image_path.parent.name}/{image_path.name}"
                mod.logger.info(
                    "OCR progress: %d/%d pages (ok=%d, fail=%d)%s",
                    done,
                    progress_state["ocr_total"],
                    progress_state["ocr_ok"],
                    progress_state["ocr_fail"],
                    latest,
                )

    original_gemini_ocr = mod.gemini_ocr

    async def _ocr_with_progress(*f_args: Any, **f_kwargs: Any) -> str:
        image_path: Optional[Path] = None
        if "image_path" in f_kwargs and f_kwargs["image_path"] is not None:
            image_path = Path(str(f_kwargs["image_path"]))
        elif len(f_args) >= 2 and f_args[1] is not None:
            image_path = Path(str(f_args[1]))
        t0 = time.perf_counter()
        try:
            out = await original_gemini_ocr(*f_args, **f_kwargs)
            _record_timing("ocr_local", time.perf_counter() - t0)
            await _record_ocr_progress(image_path, ok=True)
            return out
        except Exception:
            _record_timing("ocr_local", time.perf_counter() - t0)
            await _record_ocr_progress(image_path, ok=False)
            raise

    mod.gemini_ocr = _ocr_with_progress

    original_ainvoke_with_retry = mod.ainvoke_with_retry

    async def _ainvoke_with_retry_timed(llm: Any, messages: List[Any], *a: Any, **k: Any) -> Any:
        kind = classify_request_kind(messages)
        t0 = time.perf_counter()
        try:
            return await original_ainvoke_with_retry(llm, messages, *a, **k)
        finally:
            _record_timing(f"live_{kind}", time.perf_counter() - t0)

    mod.ainvoke_with_retry = _ainvoke_with_retry_timed

    original_build_patient_result = mod.build_patient_result

    def _build_patient_result_with_progress(*f_args: Any, **f_kwargs: Any) -> Dict[str, Any]:
        res = original_build_patient_result(*f_args, **f_kwargs)
        patient_name = f_kwargs.get("patient_name")
        if patient_name is None and f_args:
            patient_name = str(f_args[0])
        progress_state["patients_done"] += 1
        mod.logger.info(
            "Patient progress: %d/%d completed (latest=%s)",
            progress_state["patients_done"],
            progress_state["patients_total"],
            str(patient_name or "unknown"),
        )
        return res

    mod.build_patient_result = _build_patient_result_with_progress

    run_plan = {
        "pipeline_script": str(module_path),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_inflight": args.max_inflight,
        "llm_routing": args.llm_routing,
        "gemini_model_for_non_ocr": (args.gemini_model if args.llm_routing == "ocr_local_map_gemini" else None),
        "ocr_local_retries": args.ocr_local_retries,
        "dataset": dataset,
        "runtime": {
            "patient_concurrency": args.patient_concurrency,
            "page_concurrency": args.concurrency,
            "map_bundle_size": args.map_bundle_size,
            "split_map_agents": (not args.disable_split_map_agents),
            "map_agent_count": (0 if args.disable_split_map_agents else int(args.map_agent_count)),
            "request_delay_sec": args.request_delay_sec,
            "progress_every_pages": int(args.progress_every_pages),
            "quiet_sdk_logs": (not args.no_quiet_sdk_logs),
            "incremental_write": (not args.disable_incremental_write),
        },
    }
    (output_dir / "local_vlm_plan.json").write_text(json.dumps(run_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_plan, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("Dry-run only: plan written, pipeline execution skipped.")
        return

    try:
        if args.disable_incremental_write:
            await mod.run_pipeline(
                input_root=input_root,
                cdm_csv=cdm_csv,
                example_csv=example_csv,
                output_dir=output_dir,
                use_batch_api=False,
                batch_model="",
                batch_poll_interval_sec=15,
                batch_timeout_sec=7200,
                batch_image_max_side=2338,
                batch_ocr_retry_rounds=2,
                batch_retry_pause_sec=10.0,
                map_bundle_size=max(1, int(args.map_bundle_size)),
                use_split_map_agents=(not args.disable_split_map_agents),
                concurrency=max(1, int(args.concurrency)),
                patient_concurrency=max(1, int(args.patient_concurrency)),
                request_delay_sec=max(0.0, float(args.request_delay_sec)),
                top_k=int(args.top_k),
                near_dup_hamming=int(args.near_dup_hamming),
                debug=args.debug,
                log_filename=args.log_filename,
                save_intermediate=args.save_intermediate,
            )
        else:
            await run_pipeline_incremental_live(
                mod=mod,
                input_root=input_root,
                cdm_csv=cdm_csv,
                example_csv=example_csv,
                output_dir=output_dir,
                map_bundle_size=max(1, int(args.map_bundle_size)),
                use_split_map_agents=(not args.disable_split_map_agents),
                concurrency=max(1, int(args.concurrency)),
                patient_concurrency=max(1, int(args.patient_concurrency)),
                request_delay_sec=max(0.0, float(args.request_delay_sec)),
                top_k=int(args.top_k),
                near_dup_hamming=int(args.near_dup_hamming),
                debug=args.debug,
                log_filename=args.log_filename,
                save_intermediate=args.save_intermediate,
            )
    finally:
        stages: Dict[str, Any] = {}
        with timing_lock:
            for stage, stat in timing.items():
                calls = int(stat["calls"])
                total_s = float(stat["total_s"])
                stages[stage] = {
                    "calls": calls,
                    "total_seconds": total_s,
                    "avg_seconds": (total_s / calls if calls > 0 else None),
                    "max_seconds": float(stat["max_s"]),
                    "min_seconds": (float(stat["min_s"]) if stat["min_s"] != float("inf") else None),
                }
        timing_summary = {
            "wall_elapsed_seconds": time.perf_counter() - t_run_wall_start,
            "stages": stages,
        }
        timing_path = output_dir / "timing_summary.json"
        timing_path.write_text(json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        mod.logger.info("Timing summary written: %s", timing_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run 103_paper_to_cdm_SA.py with a local open-source VLM backend "
            "(default routing: OCR=local Qwen, MAP/RECALL/CONFLICT=Gemini)."
        )
    )
    ap.add_argument("--pipeline_script", type=str, default="103_paper_to_cdm_SA.py")
    ap.add_argument("--input_root", type=str, default="paper_patients")
    ap.add_argument("--cdm_csv", type=str, default="cdm_revised.csv")
    ap.add_argument("--example_csv", type=str, default="example.csv")
    ap.add_argument("--output_dir", type=str, default="out_sa_local_qwen3_vl")

    ap.add_argument("--model_id", type=str, default="")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="auto|bfloat16|float16|float32")
    ap.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    ap.add_argument("--disable_trust_remote_code", action="store_true")
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0)
    ap.add_argument(
        "--qwen_enable_thinking",
        action="store_true",
        help="Enable Qwen thinking mode instead of the default non-thinking OCR baseline.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_inflight", type=int, default=1, help="Max concurrent local VLM generations")
    ap.add_argument("--preload_model", action="store_true", help="Load model before starting pipeline")
    ap.add_argument(
        "--llm_routing",
        type=str,
        default="ocr_local_map_gemini",
        choices=["ocr_local_map_gemini", "all_local"],
        help="OCR-only local mode or all-local mode",
    )
    ap.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model for non-OCR steps when llm_routing=ocr_local_map_gemini",
    )
    ap.add_argument(
        "--ocr_local_retries",
        type=int,
        default=2,
        help="Retry count for local OCR transient failures",
    )

    ap.add_argument("--patient_concurrency", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--map_bundle_size", type=int, default=1)
    ap.add_argument(
        "--map_agent_count",
        type=int,
        default=3,
        help="Number of evenly split map agents when split-map mode is enabled",
    )
    ap.add_argument("--disable_split_map_agents", action="store_true")
    ap.add_argument("--request_delay_sec", type=float, default=0.0)
    ap.add_argument("--progress_every_pages", type=int, default=5, help="Log OCR progress every N pages")
    ap.add_argument("--no_quiet_sdk_logs", action="store_true", help="Do not suppress noisy SDK INFO logs")
    ap.add_argument(
        "--disable_incremental_write",
        action="store_true",
        help="Use original write-at-end behavior instead of incremental per-patient writes",
    )
    ap.add_argument("--top_k", type=int, default=220)
    ap.add_argument("--near_dup_hamming", type=int, default=6)
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log_filename", type=str, default="pipeline_local_vlm.log")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not str(args.model_id).strip():
        raise ValueError("--model_id is required.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
