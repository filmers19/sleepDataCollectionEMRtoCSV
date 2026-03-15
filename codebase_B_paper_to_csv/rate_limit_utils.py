from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def estimate_text_tokens(*parts: Any) -> int:
    total_chars = sum(len(str(part or "")) for part in parts)
    if total_chars <= 0:
        return 0
    return max(1, (total_chars + 3) // 4)


def parse_duration_seconds(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    text = str(raw).strip().lower()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except Exception:
        pass

    total = 0.0
    number = ""
    matched = False
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isdigit() or ch == ".":
            number += ch
            i += 1
            continue
        if not number:
            i += 1
            continue
        unit = ch
        i += 1
        if unit == "m" and i < len(text) and text[i] == "s":
            unit = "ms"
            i += 1
        mult = units.get(unit)
        if mult is None:
            number = ""
            continue
        try:
            total += float(number) * mult
            matched = True
        except Exception:
            pass
        number = ""
    if number:
        try:
            total += float(number)
            matched = True
        except Exception:
            pass
    return total if matched else None


def parse_retry_after_seconds(headers: Dict[str, Any], body_text: str = "") -> Optional[float]:
    retry_after = headers.get("retry-after")
    seconds = parse_duration_seconds(retry_after)
    if seconds is not None and seconds > 0:
        return seconds

    candidates = [
        headers.get("x-ratelimit-reset-requests"),
        headers.get("x-ratelimit-reset-tokens"),
    ]
    for raw in candidates:
        seconds = parse_duration_seconds(raw)
        if seconds is not None and seconds > 0:
            return seconds

    lower = str(body_text or "").lower()
    for marker in ("retry after", "try again in", "please try again in"):
        idx = lower.find(marker)
        if idx == -1:
            continue
        tail = lower[idx : idx + 80]
        digits = []
        for ch in tail:
            if ch.isdigit() or ch == ".":
                digits.append(ch)
            elif digits:
                break
        if digits:
            try:
                return max(0.0, float("".join(digits)))
            except Exception:
                return None
    return None


def load_rate_limit_overrides(inline_json: str = "", file_path: str = "") -> Dict[str, Any]:
    raw = str(inline_json or "").strip()
    if file_path:
        raw = Path(file_path).read_text(encoding="utf-8")
    if not raw:
        return {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def normalize_headers(headers: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if headers is None:
        return out
    items = getattr(headers, "items", None)
    if callable(items):
        for key, value in headers.items():
            out[str(key).strip().lower()] = str(value).strip()
        return out
    if isinstance(headers, dict):
        for key, value in headers.items():
            out[str(key).strip().lower()] = str(value).strip()
    return out


def extract_rate_limit_values(headers: Dict[str, Any]) -> Dict[str, Optional[float]]:
    normalized = normalize_headers(headers)
    return {
        "limit_requests": parse_duration_seconds(normalized.get("x-ratelimit-limit-requests")),
        "remaining_requests": parse_duration_seconds(normalized.get("x-ratelimit-remaining-requests")),
        "reset_requests_sec": parse_duration_seconds(normalized.get("x-ratelimit-reset-requests")),
        "limit_tokens": parse_duration_seconds(normalized.get("x-ratelimit-limit-tokens")),
        "remaining_tokens": parse_duration_seconds(normalized.get("x-ratelimit-remaining-tokens")),
        "reset_tokens_sec": parse_duration_seconds(normalized.get("x-ratelimit-reset-tokens")),
        "retry_after_sec": parse_retry_after_seconds(normalized),
    }


class SlidingWindowRateLimiter:
    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        window_sec: float = 60.0,
        margin: float = 0.9,
        requests_per_window: Optional[float] = None,
        tokens_per_window: Optional[float] = None,
    ) -> None:
        self.provider = str(provider or "").strip().lower()
        self.model_id = str(model_id or "").strip()
        self.window_sec = max(1.0, float(window_sec))
        self.margin = min(1.0, max(0.1, float(margin)))
        self.requests_per_window = self._sanitize_limit(requests_per_window)
        self.tokens_per_window = self._sanitize_limit(tokens_per_window)
        self._request_events: Deque[Tuple[float, int]] = deque()
        self._token_events: Deque[Tuple[float, int]] = deque()
        self._reservations: Dict[str, Dict[str, Any]] = {}
        self._condition = threading.Condition()
        self._bootstrap_inflight = False
        self._history: List[Dict[str, Any]] = []

    @staticmethod
    def _sanitize_limit(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            ivalue = int(float(value))
        except Exception:
            return None
        return ivalue if ivalue > 0 else None

    def configure(
        self,
        *,
        requests_per_window: Optional[float] = None,
        tokens_per_window: Optional[float] = None,
        window_sec: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> None:
        with self._condition:
            if window_sec is not None:
                self.window_sec = max(1.0, float(window_sec))
            if margin is not None:
                self.margin = min(1.0, max(0.1, float(margin)))
            reqs = self._sanitize_limit(requests_per_window)
            toks = self._sanitize_limit(tokens_per_window)
            if reqs is not None:
                self.requests_per_window = reqs
            if toks is not None:
                self.tokens_per_window = toks
            self._condition.notify_all()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._request_events and self._request_events[0][0] <= cutoff:
            self._request_events.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _requests_used(self) -> int:
        return sum(amount for _, amount in self._request_events)

    def _tokens_used(self) -> int:
        return sum(amount for _, amount in self._token_events)

    def _effective_request_limit(self) -> Optional[int]:
        if self.requests_per_window is None:
            return None
        return max(1, int(self.requests_per_window * self.margin))

    def _effective_token_limit(self) -> Optional[int]:
        if self.tokens_per_window is None:
            return None
        return max(1, int(self.tokens_per_window * self.margin))

    def _wait_for_request_budget(self, now: float, current_used: int, limit: int) -> float:
        if current_used + 1 <= limit:
            return 0.0
        if not self._request_events:
            return 0.25
        return max(0.05, (self._request_events[0][0] + self.window_sec) - now)

    def _wait_for_token_budget(self, now: float, current_used: int, needed: int, limit: int) -> float:
        if current_used + needed <= limit:
            return 0.0
        excess = (current_used + needed) - limit
        released = 0
        for ts, amount in self._token_events:
            released += amount
            if released >= excess:
                return max(0.05, (ts + self.window_sec) - now)
        return self.window_sec

    def acquire(self, estimated_tokens: int, label: str = "") -> str:
        estimated = max(0, int(estimated_tokens))
        with self._condition:
            while True:
                now = time.monotonic()
                self._prune(now)

                if (
                    self.requests_per_window is None
                    and self.tokens_per_window is None
                    and self._bootstrap_inflight
                ):
                    self._condition.wait(0.1)
                    continue

                request_limit = self._effective_request_limit()
                token_limit = self._effective_token_limit()
                wait_req = 0.0
                wait_tok = 0.0
                if request_limit is not None:
                    wait_req = self._wait_for_request_budget(now, self._requests_used(), request_limit)
                if token_limit is not None:
                    wait_tok = self._wait_for_token_budget(now, self._tokens_used(), estimated, token_limit)

                wait_for = max(wait_req, wait_tok)
                if wait_for > 0.0:
                    self._condition.wait(wait_for)
                    continue

                request_id = uuid.uuid4().hex
                self._request_events.append((now, 1))
                if estimated > 0:
                    self._token_events.append((now, estimated))
                self._reservations[request_id] = {
                    "reserved_at_monotonic": now,
                    "reserved_at_epoch": time.time(),
                    "estimated_tokens": estimated,
                    "label": str(label or "").strip(),
                }
                self._bootstrap_inflight = True
                return request_id

    def update_from_headers(self, headers: Dict[str, Any]) -> Dict[str, Optional[float]]:
        values = extract_rate_limit_values(headers)
        req_limit = values.get("limit_requests")
        tok_limit = values.get("limit_tokens")
        if req_limit is not None or tok_limit is not None:
            self.configure(requests_per_window=req_limit, tokens_per_window=tok_limit)
        return values

    def release(
        self,
        request_id: str,
        *,
        actual_tokens: Optional[int],
        headers: Optional[Dict[str, Any]] = None,
        status: str = "ok",
        error: str = "",
    ) -> Dict[str, Any]:
        normalized_headers = normalize_headers(headers)
        with self._condition:
            now = time.monotonic()
            self._prune(now)
            header_values = self.update_from_headers(normalized_headers)
            reservation = self._reservations.pop(request_id, None) or {}
            reserved_tokens = int(reservation.get("estimated_tokens") or 0)
            actual = None if actual_tokens is None else max(0, int(actual_tokens))
            if actual is not None and actual != reserved_tokens:
                self._token_events.append((now, actual - reserved_tokens))
            self._bootstrap_inflight = False

            record = {
                "request_id": request_id,
                "provider": self.provider,
                "model": self.model_id,
                "label": str(reservation.get("label") or ""),
                "status": str(status or "ok"),
                "error": str(error or ""),
                "started_at": reservation.get("reserved_at_epoch"),
                "finished_at": time.time(),
                "estimated_tokens": reserved_tokens,
                "actual_tokens": actual,
                "window_requests_used": self._requests_used(),
                "window_tokens_used": self._tokens_used(),
                "window_sec": self.window_sec,
                "configured_limit_requests": self.requests_per_window,
                "configured_limit_tokens": self.tokens_per_window,
                **header_values,
            }
            self._history.append(record)
            if len(self._history) > 5000:
                self._history = self._history[-5000:]
            self._condition.notify_all()
            return record

    def usage_history(self) -> List[Dict[str, Any]]:
        with self._condition:
            return [dict(item) for item in self._history]


class RateLimiterRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limiters: Dict[Tuple[str, str], SlidingWindowRateLimiter] = {}

    def get(
        self,
        *,
        provider: str,
        model_id: str,
        config: Optional[Dict[str, Any]] = None,
        default_window_sec: float = 60.0,
        default_margin: float = 0.9,
    ) -> SlidingWindowRateLimiter:
        key = (str(provider or "").strip().lower(), str(model_id or "").strip())
        with self._lock:
            limiter = self._limiters.get(key)
            if limiter is None:
                limiter = SlidingWindowRateLimiter(
                    provider=key[0],
                    model_id=key[1],
                    window_sec=float(config.get("window_sec", default_window_sec) if config else default_window_sec),
                    margin=float(config.get("margin", default_margin) if config else default_margin),
                    requests_per_window=(config or {}).get("rpm"),
                    tokens_per_window=(config or {}).get("tpm"),
                )
                self._limiters[key] = limiter
            elif config:
                limiter.configure(
                    requests_per_window=config.get("rpm"),
                    tokens_per_window=config.get("tpm"),
                    window_sec=config.get("window_sec"),
                    margin=config.get("margin"),
                )
            return limiter


def resolve_model_rate_limit_config(
    overrides: Dict[str, Any],
    *,
    provider: str,
    model_id: str,
    default_window_sec: float,
    default_margin: float,
) -> Dict[str, Any]:
    provider_key = str(provider or "").strip().lower()
    model_key = str(model_id or "").strip()
    provider_cfg = overrides.get(provider_key) if isinstance(overrides, dict) else None
    model_cfg = provider_cfg.get(model_key, {}) if isinstance(provider_cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    return {
        "rpm": model_cfg.get("rpm") or model_cfg.get("requests_per_minute") or model_cfg.get("requests_per_window"),
        "tpm": model_cfg.get("tpm") or model_cfg.get("tokens_per_minute") or model_cfg.get("tokens_per_window"),
        "window_sec": model_cfg.get("window_sec") or model_cfg.get("window_seconds") or default_window_sec,
        "margin": model_cfg.get("margin") or default_margin,
    }


def summarize_usage_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for rec in records:
        request_id = str(rec.get("request_id") or "")
        if request_id and request_id in seen:
            continue
        if request_id:
            seen.add(request_id)
        deduped.append(rec)

    per_model: Dict[str, Dict[str, Any]] = {}
    per_minute: Dict[str, Dict[str, Any]] = {}
    for rec in deduped:
        model = str(rec.get("model") or "")
        started_at = rec.get("started_at")
        minute_key = ""
        if isinstance(started_at, (int, float)) and started_at > 0:
            minute_key = time.strftime("%Y-%m-%dT%H:%M:00", time.localtime(float(started_at)))
        model_slot = per_model.setdefault(
            model,
            {
                "requests": 0,
                "estimated_tokens": 0,
                "actual_tokens": 0,
                "max_window_requests_used": 0,
                "max_window_tokens_used": 0,
            },
        )
        model_slot["requests"] += 1
        model_slot["estimated_tokens"] += int(rec.get("estimated_tokens") or 0)
        model_slot["actual_tokens"] += int(rec.get("actual_tokens") or 0)
        model_slot["max_window_requests_used"] = max(
            int(model_slot["max_window_requests_used"]),
            int(rec.get("window_requests_used") or 0),
        )
        model_slot["max_window_tokens_used"] = max(
            int(model_slot["max_window_tokens_used"]),
            int(rec.get("window_tokens_used") or 0),
        )

        if minute_key:
            minute_slot = per_minute.setdefault(minute_key, {"requests": 0, "estimated_tokens": 0, "actual_tokens": 0})
            minute_slot["requests"] += 1
            minute_slot["estimated_tokens"] += int(rec.get("estimated_tokens") or 0)
            minute_slot["actual_tokens"] += int(rec.get("actual_tokens") or 0)

    return {
        "request_count": len(deduped),
        "per_model": per_model,
        "per_minute": dict(sorted(per_minute.items())),
        "records": deduped,
    }
