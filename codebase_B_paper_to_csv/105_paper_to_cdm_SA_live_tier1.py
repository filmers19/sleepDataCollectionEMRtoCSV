from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

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


def load_pipeline_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("paper_to_cdm_sa", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    # Required for dataclass/type resolution on Python 3.13 during dynamic import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class TierLimits:
    rpm: int
    tpm: Optional[int]
    rpd: int
    tpd: Optional[int] = None


# Source: Google AI for Developers Gemini API quota table (Tier 1).
TIER1_MODEL_LIMITS: List[Tuple[str, TierLimits]] = [
    ("gemini-2.5-flash-lite", TierLimits(rpm=4000, tpm=3_000_000, rpd=10_000)),
    ("gemini-2.5-flash", TierLimits(rpm=1000, tpm=1_000_000, rpd=10_000)),
    ("gemini-2.5-pro", TierLimits(rpm=5, tpm=150_000, rpd=1_000)),
]


DEFAULT_OUTPUT_TOKENS: Dict[str, int] = {
    "ocr": 800,
    "map": 1000,
    "map_recall": 1000,
    "conflict": 300,
    "json_fix": 350,
    "other": 500,
}


def normalize_model_name(model_name: str) -> str:
    s = model_name.strip().lower()
    s = s.replace("_", "-")
    if "/" in s:
        s = s.split("/")[-1]
    return s


def find_tier1_preset(model_name: str) -> Optional[TierLimits]:
    norm = normalize_model_name(model_name)
    for prefix, limits in TIER1_MODEL_LIMITS:
        if norm.startswith(prefix):
            return limits
    return None


def resolve_limits_for_model(model_name: str, args: argparse.Namespace) -> Tuple[TierLimits, str]:
    preset = find_tier1_preset(model_name)

    rpm = args.rpm if args.rpm is not None else (preset.rpm if preset else None)
    tpm = args.tpm if args.tpm is not None else (preset.tpm if preset else None)
    rpd = args.rpd if args.rpd is not None else (preset.rpd if preset else None)
    tpd = args.tpd if args.tpd is not None else (preset.tpd if preset else None)

    missing = []
    if rpm is None:
        missing.append("--rpm")
    if rpd is None:
        missing.append("--rpd")
    if missing:
        hint = (
            f"No built-in Tier 1 preset for model '{model_name}'. "
            f"Provide {' and '.join(missing)} (and optionally --tpm/--tpd)."
        )
        raise ValueError(hint)

    source = "tier1_preset"
    if args.rpm is not None or args.tpm is not None or args.rpd is not None or args.tpd is not None:
        source = "cli_override"
    return TierLimits(rpm=int(rpm), tpm=(int(tpm) if tpm is not None else None), rpd=int(rpd), tpd=(int(tpd) if tpd is not None else None)), source


@dataclass
class DatasetStats:
    patient_count: int
    raw_pages_total: int
    kept_pages_total: int
    deduped_pages_total: int
    kept_pages_by_patient: Dict[str, int]
    raw_pages_by_patient: Dict[str, int]


def collect_dataset_stats(mod: Any, input_root: Path, near_dup_hamming: int) -> DatasetStats:
    patient_dirs = mod.iter_patient_folders(input_root)
    kept_pages_by_patient: Dict[str, int] = {}
    raw_pages_by_patient: Dict[str, int] = {}
    raw_total = 0
    kept_total = 0
    dedup_total = 0

    for pdir in patient_dirs:
        images = mod.iter_images(pdir)
        raw_n = len(images)
        kept_imgs, _duplicates = mod.deduplicate_images(images, near_dup_hamming=near_dup_hamming)
        kept_n = len(kept_imgs)

        raw_pages_by_patient[pdir.name] = raw_n
        kept_pages_by_patient[pdir.name] = kept_n

        raw_total += raw_n
        kept_total += kept_n
        dedup_total += max(0, raw_n - kept_n)

    return DatasetStats(
        patient_count=len(patient_dirs),
        raw_pages_total=raw_total,
        kept_pages_total=kept_total,
        deduped_pages_total=dedup_total,
        kept_pages_by_patient=kept_pages_by_patient,
        raw_pages_by_patient=raw_pages_by_patient,
    )


@dataclass
class RequestEstimate:
    bundle_size: int
    agent_count: int
    bundles_total: int
    ocr_requests: int
    map_requests: int
    recall_requests: int
    conflict_requests: int
    base_requests: int
    projected_requests: int


def estimate_requests(
    stats: DatasetStats,
    bundle_size: int,
    agent_count: int,
    recall_multiplier: float,
    include_conflicts: bool = True,
) -> RequestEstimate:
    bsize = max(1, int(bundle_size))
    bundles_total = 0
    for kept_n in stats.kept_pages_by_patient.values():
        if kept_n <= 0:
            continue
        bundles_total += int(math.ceil(kept_n / bsize))

    ocr_requests = stats.kept_pages_total
    map_requests = bundles_total * agent_count
    recall_requests = int(math.ceil(map_requests * max(0.0, float(recall_multiplier))))
    conflict_requests = stats.patient_count if include_conflicts else 0
    base_requests = ocr_requests + map_requests + conflict_requests
    projected_requests = base_requests + recall_requests

    return RequestEstimate(
        bundle_size=bsize,
        agent_count=agent_count,
        bundles_total=bundles_total,
        ocr_requests=ocr_requests,
        map_requests=map_requests,
        recall_requests=recall_requests,
        conflict_requests=conflict_requests,
        base_requests=base_requests,
        projected_requests=projected_requests,
    )


def choose_bundle_size(
    stats: DatasetStats,
    limits: TierLimits,
    agent_count: int,
    daily_budget_ratio: float,
    recall_multiplier: float,
    preferred_bundle_size: int,
    max_bundle_size: int,
) -> Tuple[int, RequestEstimate, int]:
    target_rpd = max(1, int(limits.rpd * max(0.01, min(1.0, daily_budget_ratio))))

    if preferred_bundle_size > 0:
        est = estimate_requests(
            stats=stats,
            bundle_size=preferred_bundle_size,
            agent_count=agent_count,
            recall_multiplier=recall_multiplier,
            include_conflicts=True,
        )
        return max(1, preferred_bundle_size), est, target_rpd

    chosen_size = max(1, max_bundle_size)
    chosen_est = estimate_requests(
        stats=stats,
        bundle_size=chosen_size,
        agent_count=agent_count,
        recall_multiplier=recall_multiplier,
        include_conflicts=True,
    )

    for bsize in range(1, max(1, max_bundle_size) + 1):
        est = estimate_requests(
            stats=stats,
            bundle_size=bsize,
            agent_count=agent_count,
            recall_multiplier=recall_multiplier,
            include_conflicts=True,
        )
        chosen_size = bsize
        chosen_est = est
        if est.projected_requests <= target_rpd:
            break

    return chosen_size, chosen_est, target_rpd


class AsyncQuotaGuard:
    def __init__(self, limits: TierLimits) -> None:
        self.limits = limits
        self._req_window: deque[float] = deque()
        self._tok_window: deque[Tuple[float, int, str]] = deque()
        self._reserved_tokens: Dict[str, int] = {}
        self._requests_today = 0
        self._tokens_today = 0
        self._day = datetime.now().date()
        self._lock = asyncio.Lock()

    def _maybe_reset_day(self) -> None:
        today = datetime.now().date()
        if today != self._day:
            self._day = today
            self._requests_today = 0
            self._tokens_today = 0

    def _prune(self, now_mono: float) -> None:
        while self._req_window and (now_mono - self._req_window[0]) >= 60.0:
            self._req_window.popleft()
        while self._tok_window and (now_mono - self._tok_window[0][0]) >= 60.0:
            self._tok_window.popleft()

    def _token_wait_seconds(self, now_mono: float, reserve_tokens: int) -> float:
        if self.limits.tpm is None:
            return 0.0
        used = sum(t for _, t, _ in self._tok_window)
        if (used + reserve_tokens) <= self.limits.tpm:
            return 0.0

        excess = (used + reserve_tokens) - self.limits.tpm
        freed = 0
        for ts, tok, _ in self._tok_window:
            freed += tok
            if freed >= excess:
                return max(0.0, (ts + 60.0) - now_mono)
        return 0.0

    async def reserve(self, reserve_tokens: int) -> str:
        reserve_tokens = max(1, int(reserve_tokens))
        while True:
            wait_for = 0.0
            async with self._lock:
                now_mono = time.monotonic()
                self._maybe_reset_day()
                self._prune(now_mono)

                if self._requests_today + 1 > self.limits.rpd:
                    raise RuntimeError(
                        f"Daily request budget exceeded: used={self._requests_today}, limit={self.limits.rpd}"
                    )
                if self.limits.tpd is not None and (self._tokens_today + reserve_tokens) > self.limits.tpd:
                    raise RuntimeError(
                        f"Daily token budget exceeded: used={self._tokens_today}, limit={self.limits.tpd}"
                    )

                wait_rpm = 0.0
                if self.limits.rpm > 0 and len(self._req_window) >= self.limits.rpm:
                    wait_rpm = max(0.0, (self._req_window[0] + 60.0) - now_mono)
                wait_tpm = self._token_wait_seconds(now_mono, reserve_tokens)
                wait_for = max(wait_rpm, wait_tpm)

                if wait_for <= 0.0:
                    rid = uuid4().hex
                    self._req_window.append(now_mono)
                    self._tok_window.append((now_mono, reserve_tokens, rid))
                    self._reserved_tokens[rid] = reserve_tokens
                    self._requests_today += 1
                    self._tokens_today += reserve_tokens
                    return rid

            await asyncio.sleep(wait_for + 0.01)

    async def finalize(self, reservation_id: str, actual_tokens: Optional[int]) -> None:
        async with self._lock:
            estimated = self._reserved_tokens.pop(reservation_id, None)
            if estimated is None:
                return
            if actual_tokens is None:
                actual = estimated
            else:
                actual = max(1, int(actual_tokens))

            delta = actual - estimated
            self._tokens_today += delta

            for i, (ts, tok, rid) in enumerate(self._tok_window):
                if rid == reservation_id:
                    self._tok_window[i] = (ts, actual, rid)
                    break

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            now_mono = time.monotonic()
            self._maybe_reset_day()
            self._prune(now_mono)
            tokens_last_min = sum(t for _, t, _ in self._tok_window)
            return {
                "requests_today": self._requests_today,
                "tokens_today": self._tokens_today,
                "requests_last_min": len(self._req_window),
                "tokens_last_min": tokens_last_min,
                "day": str(self._day),
            }


class RunMetrics:
    def __init__(self) -> None:
        self.calls_by_kind: Dict[str, int] = defaultdict(int)
        self.failures_by_kind: Dict[str, int] = defaultdict(int)
        self.estimated_tokens_by_kind: Dict[str, int] = defaultdict(int)
        self.actual_tokens_by_kind: Dict[str, int] = defaultdict(int)

    def record(self, kind: str, estimated_tokens: int, actual_tokens: int, success: bool) -> None:
        self.calls_by_kind[kind] += 1
        self.estimated_tokens_by_kind[kind] += max(0, int(estimated_tokens))
        self.actual_tokens_by_kind[kind] += max(0, int(actual_tokens))
        if not success:
            self.failures_by_kind[kind] += 1

    def average_actual_tokens(self, kind: str) -> Optional[float]:
        n = self.calls_by_kind.get(kind, 0)
        if n <= 0:
            return None
        return self.actual_tokens_by_kind.get(kind, 0) / n

    def total_calls(self) -> int:
        return sum(self.calls_by_kind.values())

    def total_actual_tokens(self) -> int:
        return sum(self.actual_tokens_by_kind.values())

    def as_dict(self) -> Dict[str, Any]:
        kinds = sorted(set(self.calls_by_kind.keys()) | set(self.failures_by_kind.keys()))
        by_kind: Dict[str, Any] = {}
        for k in kinds:
            n = self.calls_by_kind.get(k, 0)
            actual = self.actual_tokens_by_kind.get(k, 0)
            by_kind[k] = {
                "calls": n,
                "failures": self.failures_by_kind.get(k, 0),
                "estimated_tokens": self.estimated_tokens_by_kind.get(k, 0),
                "actual_tokens": actual,
                "avg_actual_tokens": (actual / n if n else None),
            }
        return {
            "total_calls": self.total_calls(),
            "total_actual_tokens": self.total_actual_tokens(),
            "by_kind": by_kind,
        }


def _iter_message_text_and_images(messages: Iterable[Any]) -> Tuple[int, int, str]:
    text_chars = 0
    image_parts = 0
    text_fragments: List[str] = []

    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            text_chars += len(content)
            text_fragments.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    text_chars += len(part)
                    text_fragments.append(part)
                    continue
                if isinstance(part, dict):
                    ptype = str(part.get("type", "")).lower()
                    if ptype == "image_url" or ("image_url" in part) or ("inlineData" in part):
                        image_parts += 1
                    txt = part.get("text")
                    if isinstance(txt, str):
                        text_chars += len(txt)
                        text_fragments.append(txt)
    return text_chars, image_parts, "\n".join(text_fragments)


def classify_request_kind(messages: Iterable[Any]) -> str:
    text_chars, image_parts, all_text = _iter_message_text_and_images(messages)
    if image_parts > 0:
        return "ocr"
    low = all_text.lower()
    if "strict json repair assistant" in low or "fix into a valid json object only" in low:
        return "json_fix"
    if "resolve one conflict candidate set" in low or "conflict candidates json" in low:
        return "conflict"
    if "existing json (do not repeat these keys)" in low:
        return "map_recall"
    if "candidate cdm fields" in low:
        return "map"
    if text_chars < 10:
        return "other"
    return "other"


def estimate_prompt_tokens(messages: Iterable[Any]) -> int:
    text_chars, image_parts, _ = _iter_message_text_and_images(messages)
    text_tokens = max(1, int(math.ceil(text_chars / 4.0)))
    # Reserve a fixed token-equivalent cost per image request.
    image_tokens = image_parts * 768
    return text_tokens + image_tokens


def estimate_reservation_tokens(kind: str, prompt_tokens: int, metrics: RunMetrics) -> int:
    learned_avg = metrics.average_actual_tokens(kind)
    if learned_avg is not None:
        # Keep a safety margin and avoid under-reserving.
        return max(prompt_tokens + 128, int(math.ceil(learned_avg * 1.10)))
    return prompt_tokens + DEFAULT_OUTPUT_TOKENS.get(kind, DEFAULT_OUTPUT_TOKENS["other"])


def extract_total_tokens_from_response(resp: Any) -> Optional[int]:
    def _extract_from_dict(obj: Any) -> Optional[int]:
        if not isinstance(obj, dict):
            return None
        for key in ("total_tokens", "totalTokenCount", "total_token_count", "totalTokens"):
            v = obj.get(key)
            if isinstance(v, (int, float)):
                return int(v)

        usage = obj.get("usage_metadata")
        if isinstance(usage, dict):
            nested = _extract_from_dict(usage)
            if nested is not None:
                return nested

        inp = obj.get("input_tokens", obj.get("prompt_tokens"))
        out = obj.get("output_tokens", obj.get("completion_tokens"))
        if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
            return int(inp + out)
        return None

    for attr in ("usage_metadata", "response_metadata", "additional_kwargs"):
        if hasattr(resp, attr):
            v = getattr(resp, attr)
            got = _extract_from_dict(v)
            if got is not None:
                return got

    # Some SDK layers expose plain dict payloads.
    got = _extract_from_dict(resp if isinstance(resp, dict) else None)
    if got is not None:
        return got
    return None


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


def build_quota_aware_ainvoke(
    mod: Any,
    quota: AsyncQuotaGuard,
    metrics: RunMetrics,
    extra_delay_sec: float,
    transient_max_retries: int,
    transient_base_delay: float,
    transient_max_delay: float,
    sdk_max_retries: int,
    disable_afc: bool,
    sdk_timeout_sec: Optional[float],
):
    async def _ainvoke_with_retry(
        llm: Any,
        messages: List[Any],
        max_retries: int = 5,
        base_delay: float = 1.5,
        max_delay: float = 20.0,
    ):
        effective_max_retries = max(0, int(max_retries if max_retries is not None else transient_max_retries))
        effective_base_delay = max(0.1, float(base_delay if base_delay is not None else transient_base_delay))
        effective_max_delay = max(
            effective_base_delay, float(max_delay if max_delay is not None else transient_max_delay)
        )
        invoke_kwargs: Dict[str, Any] = {}
        invoke_kwargs["max_retries"] = max(0, int(sdk_max_retries))
        if disable_afc:
            invoke_kwargs["automatic_function_calling"] = {"disable": True}
        if sdk_timeout_sec is not None:
            invoke_kwargs["timeout"] = float(sdk_timeout_sec)

        for attempt in range(effective_max_retries + 1):
            kind = classify_request_kind(messages)
            prompt_tokens = estimate_prompt_tokens(messages)
            reserved_tokens = estimate_reservation_tokens(kind, prompt_tokens, metrics)
            reservation_id = await quota.reserve(reserved_tokens)

            try:
                if extra_delay_sec > 0:
                    await asyncio.sleep(extra_delay_sec)
                resp = await llm.ainvoke(messages, **invoke_kwargs)
                actual_tokens = extract_total_tokens_from_response(resp)
                if actual_tokens is None:
                    actual_tokens = reserved_tokens
                await quota.finalize(reservation_id, actual_tokens)
                metrics.record(kind, reserved_tokens, actual_tokens, success=True)
                return resp
            except Exception as e:
                # Keep conservative accounting for failed requests.
                await quota.finalize(reservation_id, reserved_tokens)
                metrics.record(kind, reserved_tokens, reserved_tokens, success=False)
                if attempt >= effective_max_retries or not mod.is_transient_llm_error(e):
                    raise
                delay = min(effective_max_delay, effective_base_delay * (2**attempt)) + random.uniform(0.0, 0.5)
                mod.logger.warning(
                    "Transient Gemini error (%s). Retrying in %.1fs (%d/%d).",
                    e,
                    delay,
                    attempt + 1,
                    effective_max_retries,
                )
                await asyncio.sleep(delay)

    return _ainvoke_with_retry


async def run(args: argparse.Namespace) -> None:
    module_path = resolve_script_path(args.pipeline_script)
    mod = load_pipeline_module(module_path)

    mod.load_env()
    model = (args.model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    os.environ["GEMINI_MODEL"] = model

    limits, limit_source = resolve_limits_for_model(model, args)
    configure_third_party_logging(quiet=(not args.no_quiet_sdk_logs))

    input_root = resolve_repo_path(args.input_root)
    cdm_csv = resolve_repo_path(args.cdm_csv)
    example_csv = resolve_repo_path(args.example_csv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = collect_dataset_stats(mod=mod, input_root=input_root, near_dup_hamming=args.near_dup_hamming)
    use_split_map_agents = not args.disable_split_map_agents
    agent_count = 6 if use_split_map_agents else 1
    bundle_size, req_est, target_rpd = choose_bundle_size(
        stats=stats,
        limits=limits,
        agent_count=agent_count,
        daily_budget_ratio=args.daily_budget_ratio,
        recall_multiplier=args.recall_multiplier,
        preferred_bundle_size=args.map_bundle_size,
        max_bundle_size=args.max_bundle_size,
    )

    if (req_est.projected_requests > limits.rpd) and (not args.allow_rpd_overflow):
        raise RuntimeError(
            "Projected requests exceed RPD limit. "
            f"projected={req_est.projected_requests}, rpd_limit={limits.rpd}. "
            "Increase map_bundle_size, disable split map agents, or use a higher-quota model."
        )

    est_runtime_min = req_est.projected_requests / max(1, limits.rpm)
    run_plan = {
        "model": model,
        "limit_source": limit_source,
        "limits": {
            "rpm": limits.rpm,
            "tpm": limits.tpm,
            "rpd": limits.rpd,
            "tpd": limits.tpd,
        },
        "dataset": {
            "patients": stats.patient_count,
            "raw_pages_total": stats.raw_pages_total,
            "kept_pages_total": stats.kept_pages_total,
            "deduped_pages_total": stats.deduped_pages_total,
        },
        "query_strategy": {
            "split_map_agents": use_split_map_agents,
            "agent_count": agent_count,
            "map_bundle_size": bundle_size,
            "recall_multiplier_for_projection": args.recall_multiplier,
            "daily_budget_ratio": args.daily_budget_ratio,
            "target_rpd": target_rpd,
        },
        "request_estimate": {
            "ocr_requests": req_est.ocr_requests,
            "map_requests": req_est.map_requests,
            "recall_requests": req_est.recall_requests,
            "conflict_requests": req_est.conflict_requests,
            "base_requests": req_est.base_requests,
            "projected_requests": req_est.projected_requests,
            "estimated_min_runtime_at_rpm_cap": est_runtime_min,
        },
        "llm_runtime_strategy": {
            "transient_max_retries": max(0, int(args.llm_transient_max_retries)),
            "transient_base_delay_sec": max(0.1, float(args.llm_transient_base_delay_sec)),
            "transient_max_delay_sec": max(0.1, float(args.llm_transient_max_delay_sec)),
            "sdk_max_retries": max(0, int(args.sdk_max_retries)),
            "disable_afc": (not args.enable_afc),
            "sdk_timeout_sec": (None if args.sdk_timeout_sec is None else float(args.sdk_timeout_sec)),
            "quiet_sdk_logs": (not args.no_quiet_sdk_logs),
        },
    }
    (output_dir / "quota_plan.json").write_text(json.dumps(run_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_plan, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("Dry-run only: quota plan written, pipeline execution skipped.")
        return

    quota = AsyncQuotaGuard(limits=limits)
    metrics = RunMetrics()
    mod.ainvoke_with_retry = build_quota_aware_ainvoke(
        mod=mod,
        quota=quota,
        metrics=metrics,
        extra_delay_sec=max(0.0, float(args.request_delay_sec)),
        transient_max_retries=max(0, int(args.llm_transient_max_retries)),
        transient_base_delay=max(0.1, float(args.llm_transient_base_delay_sec)),
        transient_max_delay=max(0.1, float(args.llm_transient_max_delay_sec)),
        sdk_max_retries=max(0, int(args.sdk_max_retries)),
        disable_afc=(not args.enable_afc),
        sdk_timeout_sec=(None if args.sdk_timeout_sec is None else float(args.sdk_timeout_sec)),
    )

    # Disable original fixed-interval throttling; this runner uses quota-aware throttling.
    mod.REQUEST_THROTTLE.configure(0.0)

    started = time.perf_counter()
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
        map_bundle_size=bundle_size,
        use_split_map_agents=use_split_map_agents,
        concurrency=max(1, int(args.concurrency)),
        patient_concurrency=max(1, int(args.patient_concurrency)),
        request_delay_sec=0.0,
        top_k=args.top_k,
        near_dup_hamming=args.near_dup_hamming,
        debug=args.debug,
        log_filename=args.log_filename,
        save_intermediate=args.save_intermediate,
    )
    elapsed_s = time.perf_counter() - started

    quota_snapshot = await quota.snapshot()
    runtime_summary = {
        "elapsed_seconds": elapsed_s,
        "quota_counters": quota_snapshot,
        "llm_metrics": metrics.as_dict(),
        "output_dir": str(output_dir),
    }
    (output_dir / "quota_runtime_summary.json").write_text(
        json.dumps(runtime_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(runtime_summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live Gemini runner based on 103_paper_to_cdm_SA.py with Tier-1 quota planning and throttling."
    )
    ap.add_argument("--pipeline_script", type=str, default="103_paper_to_cdm_SA.py")
    ap.add_argument("--input_root", type=str, default="paper_patients")
    ap.add_argument("--cdm_csv", type=str, default="cdm_revised.csv")
    ap.add_argument("--example_csv", type=str, default="example.csv")
    ap.add_argument("--output_dir", type=str, default="out_sa_live_tier1")

    ap.add_argument("--model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--rpm", type=int, default=None, help="Override requests per minute limit")
    ap.add_argument("--tpm", type=int, default=None, help="Override tokens per minute limit")
    ap.add_argument("--rpd", type=int, default=None, help="Override requests per day limit")
    ap.add_argument("--tpd", type=int, default=None, help="Override tokens per day limit")

    ap.add_argument(
        "--daily_budget_ratio",
        type=float,
        default=0.90,
        help="Target fraction of RPD used by projected requests (for auto bundle sizing)",
    )
    ap.add_argument(
        "--recall_multiplier",
        type=float,
        default=1.0,
        help="Projected recall request multiplier vs MAP requests",
    )
    ap.add_argument(
        "--map_bundle_size",
        type=int,
        default=0,
        help="0 = auto choose bundle size; >0 = fixed bundle size",
    )
    ap.add_argument("--max_bundle_size", type=int, default=6, help="Max bundle size for auto mode")
    ap.add_argument("--allow_rpd_overflow", action="store_true", help="Allow run even if projection exceeds RPD")
    ap.add_argument("--disable_split_map_agents", action="store_true")

    ap.add_argument("--patient_concurrency", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--request_delay_sec", type=float, default=0.0, help="Extra delay before each request")
    ap.add_argument("--top_k", type=int, default=220)
    ap.add_argument("--near_dup_hamming", type=int, default=6)
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log_filename", type=str, default="pipeline_live_tier1.log")
    ap.add_argument(
        "--llm_transient_max_retries",
        type=int,
        default=10,
        help="Retries for transient 429/503 handled by this runner (outside SDK retries)",
    )
    ap.add_argument("--llm_transient_base_delay_sec", type=float, default=2.0, help="Base backoff delay for transient retries")
    ap.add_argument("--llm_transient_max_delay_sec", type=float, default=90.0, help="Max backoff delay for transient retries")
    ap.add_argument(
        "--sdk_max_retries",
        type=int,
        default=0,
        help="Gemini SDK internal retries per call (0 recommended; runner handles retries)",
    )
    ap.add_argument("--sdk_timeout_sec", type=float, default=None, help="Per-request SDK timeout in seconds")
    ap.add_argument("--enable_afc", action="store_true", help="Enable SDK automatic function calling (default: disabled)")
    ap.add_argument("--no_quiet_sdk_logs", action="store_true", help="Do not suppress noisy SDK INFO logs")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
