from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
from PIL import Image, ImageFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("sleep_cdm_gemini_single_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# Always load the full CDM. How much of it each map agent sees is controlled
# by the split-agent count when building agent specs.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CPAP_PRESSURE_STEP_START = 5
CPAP_PRESSURE_STEP_END = 29

CORE_ALWAYS_KEYS = [
    "Hospital_ID",
    "Name",
    "Lab_ID",
    "Device_Type",
    "PSG_Date",
    "PSG_No",
    "PSG_Type",
    "Database_ID",
    "Previous_Data",
    "SEX",
    "AGE",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Neckcir_cm",
    "Occupation",
    "Shiftwork",
]

TRIGGER_PREFIX_RULES: List[Tuple[re.Pattern, Tuple[str, ...]]] = [
    (re.compile(r"\bpsqi\b|pittsburgh sleep quality index", re.I), ("PSQI",)),
    (re.compile(r"\bess\b|epworth sleepiness", re.I), ("ESS",)),
    (re.compile(r"\bfss\b|fatigue severity", re.I), ("FSS",)),
    (re.compile(r"\bbq\b|berlin questionnaire", re.I), ("BQ",)),
    (re.compile(r"stop[- ]?bang|stop questionnaire", re.I), ("STOP", "STOPBANG")),
    (re.compile(r"\bisi\b|insomnia severity", re.I), ("ISI",)),
    (re.compile(r"\brls\b|restless legs", re.I), ("RLS", "IRLS")),
    (re.compile(r"\birls\b|international restless", re.I), ("IRLS", "RLS")),
    (re.compile(r"\brbd\b|rbdsq|rem sleep behavior", re.I), ("RBDSQ", "RBD")),
    (re.compile(r"\bphq\b|patient health questionnaire", re.I), ("PHQ",)),
    (re.compile(r"\bbdi\b|beck depression", re.I), ("BDI",)),
    (re.compile(r"whoqol|quality of life", re.I), ("WHOQOL", "QOL")),
    (re.compile(r"\bmslt\b|multiple sleep latency", re.I), ("MSLT",)),
    (re.compile(r"\bnap\b|sleep attack", re.I), ("Nap", "N")),
]

PSG_TRIGGER_RE = re.compile(
    r"polysomnography|sleep architecture|respiratory event|rdi|ahi|arousal index|"
    r"total sleep time|sleep latency|rem latency|snoring|stage n1|stage n2|stage n3|"
    r"sleep efficiency|lowest sao2",
    re.I,
)

PSG_METRIC_KEY_RE = re.compile(
    r"^(?:"
    r"TST_min|SL_min|REM_SL_min|Sleep_Eff|"
    r"Arousal_(?:no|idx|resp_idx|snoring_idx|PLM_idx|spont_idx|PLM_no|PLM_idx_re|LM_no|LM_idx)|"
    r"REM_pct|N1_pct|N2_pct|N3_pct|WASO_pct|"
    r"AI(?:_|$)|HI(?:_|$)|AHI_|RDI_|"
    r"Lowest_SpO2|REM_(?:sup|lat)_min|NREM_(?:sup|lat)_min|PLM_idx|LM_idx|"
    r"Pressure_\d{2}|Pr\d{2}_"
    r")",
    re.I,
)

RECALL_HINT_RE = re.compile(
    r"selected|circled|checked|\[x\]|questionnaire|pittsburgh|epworth|beck depression|"
    r"insomnia severity|restless|whoqol|sleep architecture|patient information|"
    r"등록번호|성명|name:|study date|total sleep time|sleep latency|rdi|ahi|arousal",
    re.I,
)

DOCUMENT_LABEL_SPECS: Dict[str, Dict[str, Any]] = {
    "psg_report": {
        "description": "Polysomnography report/summary page with sleep architecture, respiratory indices, oxygen saturation, diagnosis, or impression text.",
        "prefixes": ("TST", "SL", "REM", "N1", "N2", "N3", "WASO", "AI", "HI", "AHI", "RDI", "Arousal", "Lowest", "PLM", "LM"),
        "regexes": (r"^PSG_(?!M_)", r"^Sleep_Eff$", r"^Diagnosis_etc$"),
        "extra_keys": ("Diagnosis_etc",),
    },
    "cpap_pressure": {
        "description": "CPAP titration pressure-step metrics for each tested pressure level, including Pressure_XX and PrXX_* rows.",
        "prefixes": (),
        "regexes": (r"^Pressure_\d{2}$", r"^Pr\d{2}_"),
        "extra_keys": (),
    },
    "psg_morning": {
        "description": "Morning-after PSG questionnaire page asking subjective sleep latency, sleep duration, awakenings, and sleep-quality scales.",
        "prefixes": (),
        "regexes": (r"^PSG_M_",),
        "extra_keys": (),
    },
    "psqi": {
        "description": "Official Pittsburgh Sleep Quality Index questionnaire page.",
        "prefixes": ("PSQI",),
        "regexes": (),
        "extra_keys": (),
    },
    "ess": {
        "description": "Official Epworth Sleepiness Scale questionnaire page.",
        "prefixes": ("ESS",),
        "regexes": (),
        "extra_keys": (),
    },
    "fss": {
        "description": "Fatigue Severity Scale questionnaire page.",
        "prefixes": ("FSS",),
        "regexes": (),
        "extra_keys": (),
    },
    "bq": {
        "description": "Berlin Questionnaire page.",
        "prefixes": ("BQ",),
        "regexes": (),
        "extra_keys": (),
    },
    "isi": {
        "description": "Insomnia Severity Index questionnaire page.",
        "prefixes": ("ISI",),
        "regexes": (),
        "extra_keys": (),
    },
    "rls_irls": {
        "description": "Restless legs / IRLS questionnaire page.",
        "prefixes": ("RLS", "IRLS"),
        "regexes": (),
        "extra_keys": (),
    },
    "rbd_rbdsq": {
        "description": "RBD or RBDSQ questionnaire page.",
        "prefixes": ("RBD", "RBDSQ"),
        "regexes": (),
        "extra_keys": (),
    },
    "mood": {
        "description": "Mood / depression questionnaire page such as BDI or PHQ.",
        "prefixes": ("BDI", "PHQ"),
        "regexes": (),
        "extra_keys": (),
    },
    "qol": {
        "description": "Quality-of-life questionnaire page.",
        "prefixes": ("QOL",),
        "regexes": (),
        "extra_keys": (),
    },
    "sleep_history": {
        "description": "General sleep-history, symptom, habit, or medical-history questionnaire page.",
        "prefixes": ("Habit", "PHx", "Occupation", "Shiftwork", "SSS", "SQ", "Nap", "N"),
        "regexes": (),
        "extra_keys": (),
    },
}

MAP_ROUTE_PSG_SIGNALS = "map_route_polysomnography_signals"
MAP_ROUTE_PSG_REPORT_GENERAL = "map_route_psg_report_general"
MAP_ROUTE_PSG_REPORT_EXTENSIVE = "map_route_psg_report_extensive"
MAP_ROUTE_CPAP_PSG_REPORT_GENERAL = "map_route_cpap_psg_report_general"
MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE = "map_route_cpap_psg_report_extensive"
MAP_ROUTE_PSG_REPORT = MAP_ROUTE_PSG_REPORT_GENERAL
MAP_ROUTE_MORNING_QUESTIONNAIRE = "map_route_morning_questionnaire"
MAP_ROUTE_NIGHT_QUESTIONNAIRE = "map_route_night_questionnaire"
DEFAULT_MAP_ROUTE = MAP_ROUTE_NIGHT_QUESTIONNAIRE

MAP_ROUTE_DESCRIPTIONS: Dict[str, str] = {
    MAP_ROUTE_PSG_SIGNALS: "Signal-graph polysomnography tracing page dominated by stacked PSG channel waveforms and labels. Map only base identity/demographic keys that are directly visible on the page.",
    MAP_ROUTE_PSG_REPORT_GENERAL: "General sleep lab polysomnography report page with PSG metrics, signal/channel labels, respiratory tables, clinician interpretation, or diagnosis/impression notes.",
    MAP_ROUTE_PSG_REPORT_EXTENSIVE: "Extensive polysomnography report page with the same PSG-report characteristics plus a RESPIRATORY DISTURBANCE INDEX section or similarly dense respiratory/position/stage tables. This route should use two map passes over split OCR text.",
    MAP_ROUTE_CPAP_PSG_REPORT_GENERAL: "General CPAP titration polysomnography report page with standard PSG summary metrics plus CPAP pressure-step titration metrics.",
    MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE: "Extensive CPAP titration polysomnography report page, especially pages headed by FULL NIGHT CPAP POLYSOMNOGRAPHY REPORT or similarly dense CPAP pressure tables. This route should use two map passes over split OCR text.",
    MAP_ROUTE_MORNING_QUESTIONNAIRE: "Morning-after PSG questionnaire page asking about last night's sleep, awakenings, dreams, alertness, and waking experience.",
    MAP_ROUTE_NIGHT_QUESTIONNAIRE: "Patient-filled night questionnaire or sleep-history page, including official sleep scales and general symptom/history questionnaires.",
}

MORNING_QUESTIONNAIRE_ROUTE_LABELS: Tuple[str, ...] = (
    "psg_morning",
)

NIGHT_QUESTIONNAIRE_ROUTE_LABELS: Tuple[str, ...] = (
    "psqi",
    "ess",
    "fss",
    "bq",
    "isi",
    "rls_irls",
    "rbd_rbdsq",
    "mood",
    "qol",
    "sleep_history",
)

DIRECT_ROUTE_KEYWORD_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE,
        (
            "full night cpap polysomnography report",
        ),
    ),
    (
        MAP_ROUTE_PSG_REPORT_EXTENSIVE,
        (
            "night polysomnography report",
        ),
    ),
    (
        MAP_ROUTE_MORNING_QUESTIONNAIRE,
        (
            "아침 질문 사항",
        ),
    ),
    (
        MAP_ROUTE_NIGHT_QUESTIONNAIRE,
        (
            "병력과 가족력",
            "수면 습관",
            "생활 습관",
            "Stanford Sleepiness Scale",
            "Epworth Sleepiness Scale",
            "Pittsburgh Sleep Quality Index",
            "피로 정도에 대한 설문",
            "Fatigue Severity Scale",
            "Berlin Questionnaire",
            "Insomnia Severity Index",
            "하지불안증후군/주기성사지운동증후군",
            "하지불안증후군에 대한 설문",
            "우울증에 관한 설문",
            "Beck Depression Inventory",
            "삶의 질 척도",
            "WHOQOL-BREF",
        ),
    ),
)

PSG_REPORT_HINT_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"신경과\s*수면검사|수면다원검사|polysomnography(?:\s+(?:data|report))?", re.I), 3, "psg_title"),
    (re.compile(r"\b(?:tst|ahi|rdi|sleep efficiency|sleep latency|rem latency|lowest\s*sao2|lowest\s*spo2)\b", re.I), 2, "psg_metrics"),
    (re.compile(r"\b(?:eeg|eog|emg|ecg|ekg|airflow|thorax|abd(?:omen)?|spo2|snor(?:e|ing)?|position|leg|chin)\b", re.I), 2, "psg_channels"),
    (re.compile(r"\b(?:c3-a2|c4-a1|o1-a2|o2-a1|f3-m2|f4-m1)\b", re.I), 2, "psg_leads"),
    (re.compile(r"diagnosis|impression|interpretation|clinical\s+correlation|technologist|physician", re.I), 2, "psg_clinician_note"),
    (re.compile(r"respiratory event|apnea|hypopnea|arousal index|stage n1|stage n2|stage n3|rem/tst", re.I), 2, "psg_event_table"),
]

PSG_SIGNAL_GRAPH_HINT_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"this page contains psg signal graphs\.", re.I), 6, "ocr_signal_note"),
    (
        re.compile(
            r"\b(?:f3-a2|c3-a2|f4-a1|c4-a1|o1-a2|o2-a1|le-a2|re-a1|so-a2|io-a2|chin1-chin2|snore-ref|emg-ref|ptaf-ref|chest-ref|abd-ref|lleg-ref|rleg-ref)\b",
            re.I,
        ),
        4,
        "psg_channel_bundle",
    ),
    (re.compile(r"\b(?:epoch|flow|snore|spo2|hr|body|position|stage|events?)\b", re.I), 1, "signal_graph_terms"),
]

PSG_REPORT_EXTENSIVE_HINT_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"night polysomnography report", re.I), 3, "night_polysomnography_report"),
]

CPAP_REPORT_HINT_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"cpap polysomnography|cpap titration|pap titration|nasal cpap titration", re.I), 3, "cpap_title"),
    (re.compile(r"\bcpap\b|\bpap\b|cmh2o|cmh_?2o", re.I), 2, "cpap_units"),
    (re.compile(r"optimal cpap pressure|pressure\s*(?:\(|\d)|mask leak|mouth breathing", re.I), 2, "cpap_footer"),
    (re.compile(r"(?:^|\n)\s*pressure\s+\d+\s*cmh2o", re.I), 2, "cpap_pressure_lines"),
]

CPAP_REPORT_EXTENSIVE_HINT_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"full night cpap polysomnography report", re.I), 4, "full_night_cpap_polysomnography_report"),
]

MORNING_QUESTIONNAIRE_HINT_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"어젯밤|오늘\s*아침|보통\s*집에서|잠에서\s*깨어|얼마나\s*오랫동안\s*잠을", re.I), 2, "morning_questionnaire_text"),
    (re.compile(r"수면제|꿈을\s*기억|잠자는\s*동안\s*몇\s*번\s*깨|어떻게\s*잠에서\s*깨어", re.I), 2, "morning_questionnaire_items"),
    (re.compile(r"어젯밤\s*당신의\s*수면에\s*대한\s*평가|오늘\s*아침\s*신체적으로\s*불편", re.I), 3, "morning_questionnaire_title"),
]

NIGHT_QUESTIONNAIRE_HINT_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"questionnaire|설문지|문진표|자가기입|자가\s*보고|지난\s*한\s*달간", re.I), 2, "questionnaire_title"),
    (re.compile(r"\b(?:psqi|epworth|ess|fss|bq|berlin questionnaire|isi|rbdsq|phq|whoqol)\b", re.I), 2, "questionnaire_scale_name"),
    (re.compile(r"\[\s*selected\s*\]|◯|○|☑|✓|체크|예\s*/\s*아니오|예\s+\[selected\]|아니오\s+\[selected\]", re.I), 2, "questionnaire_marks"),
    (re.compile(r"(?:^|\n)\s*\d{1,2}\.\s", re.I), 1, "questionnaire_numbered_items"),
    (re.compile(r"환자|본인|주중|주말|잠이|졸립|피곤|깼|수면", re.I), 1, "questionnaire_language"),
]


def normalize_map_route_name(route_raw: Any) -> str:
    raw = str(route_raw or "").strip().lower()
    aliases = {
        "polysomnography signals": MAP_ROUTE_PSG_SIGNALS,
        "polysomnography_signals": MAP_ROUTE_PSG_SIGNALS,
        "psg_signals": MAP_ROUTE_PSG_SIGNALS,
        "signal_graphs": MAP_ROUTE_PSG_SIGNALS,
        "map_route_polysomnography_signals": MAP_ROUTE_PSG_SIGNALS,
        "psg_report_general": MAP_ROUTE_PSG_REPORT_GENERAL,
        "general_polysomnography_report": MAP_ROUTE_PSG_REPORT_GENERAL,
        "map_route_psg_report_general": MAP_ROUTE_PSG_REPORT_GENERAL,
        "psg_report_extensive": MAP_ROUTE_PSG_REPORT_EXTENSIVE,
        "extensive_polysomnography_report": MAP_ROUTE_PSG_REPORT_EXTENSIVE,
        "map_route_psg_report_extensive": MAP_ROUTE_PSG_REPORT_EXTENSIVE,
        "psg_report": MAP_ROUTE_PSG_REPORT,
        "polysomnography_report": MAP_ROUTE_PSG_REPORT,
        "map_route_psg_report": MAP_ROUTE_PSG_REPORT,
        "cpap_psg_report_general": MAP_ROUTE_CPAP_PSG_REPORT_GENERAL,
        "cpap_polysomnography_report": MAP_ROUTE_CPAP_PSG_REPORT_GENERAL,
        "map_route_cpap_psg_report_general": MAP_ROUTE_CPAP_PSG_REPORT_GENERAL,
        "cpap_psg_report_extensive": MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE,
        "cpap_polysomnography_report_extensive": MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE,
        "map_route_cpap_psg_report_extensive": MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE,
        "morning_questionnaire": MAP_ROUTE_MORNING_QUESTIONNAIRE,
        "psg_morning": MAP_ROUTE_MORNING_QUESTIONNAIRE,
        "map_route_morning_questionnaire": MAP_ROUTE_MORNING_QUESTIONNAIRE,
        "night_questionnaire": MAP_ROUTE_NIGHT_QUESTIONNAIRE,
        "patient_questionnaire": MAP_ROUTE_NIGHT_QUESTIONNAIRE,
        "questionnaire": MAP_ROUTE_NIGHT_QUESTIONNAIRE,
        "map_route_night_questionnaire": MAP_ROUTE_NIGHT_QUESTIONNAIRE,
    }
    return aliases.get(raw, DEFAULT_MAP_ROUTE)


def prepare_ocr_text_for_router(ocr_text: str) -> str:
    text = str(ocr_text or "")
    if not text:
        return ""
    if re.search(r"night polysomnography report", text, flags=re.I):
        stripped = text.lstrip()
        prefix = "NIGHT POLYSOMNOGRAPHY REPORT"
        if not stripped.startswith(prefix):
            return f"{prefix}\n{text}"
    return text


def normalize_router_keyword_text(text: str) -> str:
    raw = str(text or "").lower()
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def detect_direct_route_keyword_match(ocr_text: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_router_keyword_text(ocr_text)
    if not normalized:
        return None
    for route_name, phrases in DIRECT_ROUTE_KEYWORD_RULES:
        hits = [phrase for phrase in phrases if normalize_router_keyword_text(phrase) in normalized]
        if hits:
            return {
                "route": route_name,
                "confidence": "high",
                "direct_keyword_hits": hits,
                "reason": "direct_keyword_router",
            }
    return None


def classify_map_route_heuristic(ocr_text: str) -> Dict[str, Any]:
    text = prepare_ocr_text_for_router(ocr_text)
    direct = detect_direct_route_keyword_match(text)
    if direct:
        return {
            "route": direct["route"],
            "confidence": direct["confidence"],
            "signal_score": 0,
            "cpap_score": 0,
            "cpap_extensive_score": 0,
            "report_score": 0,
            "extensive_report_score": 0,
            "morning_score": 0,
            "night_score": 0,
            "signal_hits": [],
            "cpap_hits": [],
            "cpap_extensive_hits": [],
            "report_hits": [],
            "extensive_report_hits": [],
            "morning_hits": [],
            "night_hits": [],
            "direct_keyword_hits": list(direct.get("direct_keyword_hits") or []),
            "reason": str(direct.get("reason") or "direct_keyword_router"),
        }
    signal_score = 0
    report_score = 0
    extensive_report_score = 0
    cpap_score = 0
    cpap_extensive_score = 0
    morning_score = 0
    night_score = 0
    signal_hits: List[str] = []
    report_hits: List[str] = []
    extensive_report_hits: List[str] = []
    cpap_hits: List[str] = []
    cpap_extensive_hits: List[str] = []
    morning_hits: List[str] = []
    night_hits: List[str] = []

    for pattern, weight, label in PSG_SIGNAL_GRAPH_HINT_PATTERNS:
        if pattern.search(text):
            signal_score += weight
            signal_hits.append(label)

    for pattern, weight, label in PSG_REPORT_HINT_PATTERNS:
        if pattern.search(text):
            report_score += weight
            report_hits.append(label)

    for pattern, weight, label in PSG_REPORT_EXTENSIVE_HINT_PATTERNS:
        if pattern.search(text):
            extensive_report_score += weight
            extensive_report_hits.append(label)

    for pattern, weight, label in CPAP_REPORT_HINT_PATTERNS:
        if pattern.search(text):
            cpap_score += weight
            cpap_hits.append(label)

    for pattern, weight, label in CPAP_REPORT_EXTENSIVE_HINT_PATTERNS:
        if pattern.search(text):
            cpap_extensive_score += weight
            cpap_extensive_hits.append(label)

    for pattern, weight, label in MORNING_QUESTIONNAIRE_HINT_PATTERNS:
        if pattern.search(text):
            morning_score += weight
            morning_hits.append(label)

    for pattern, weight, label in NIGHT_QUESTIONNAIRE_HINT_PATTERNS:
        if pattern.search(text):
            night_score += weight
            night_hits.append(label)

    scores = {
        MAP_ROUTE_PSG_SIGNALS: signal_score,
        MAP_ROUTE_CPAP_PSG_REPORT_GENERAL: cpap_score,
        MAP_ROUTE_PSG_REPORT_GENERAL: report_score,
        MAP_ROUTE_MORNING_QUESTIONNAIRE: morning_score,
        MAP_ROUTE_NIGHT_QUESTIONNAIRE: night_score,
    }
    best_route = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best_route]
    runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

    has_extensive_signature = "night_polysomnography_report" in set(extensive_report_hits)
    has_cpap_extensive_signature = "full_night_cpap_polysomnography_report" in set(cpap_extensive_hits)
    report_hit_set = set(report_hits)
    signal_hit_set = set(signal_hits)
    has_report_table = any(x in report_hit_set for x in {"psg_metrics", "psg_event_table", "psg_clinician_note"})
    has_signal_note = "ocr_signal_note" in signal_hit_set
    looks_like_signal_graph_page = signal_score >= 4 and not has_report_table and not has_extensive_signature and not has_cpap_extensive_signature

    if has_signal_note or (best_route == MAP_ROUTE_PSG_SIGNALS and looks_like_signal_graph_page and signal_score >= max(4, runner_up + 1)):
        route = MAP_ROUTE_PSG_SIGNALS
    elif cpap_score >= max(3, runner_up + 2):
        if has_cpap_extensive_signature:
            route = MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE
        else:
            route = MAP_ROUTE_CPAP_PSG_REPORT_GENERAL
    elif report_score >= max(3, runner_up + 2):
        if has_extensive_signature:
            route = MAP_ROUTE_PSG_REPORT_EXTENSIVE
        else:
            route = MAP_ROUTE_PSG_REPORT_GENERAL
    elif best_route == MAP_ROUTE_MORNING_QUESTIONNAIRE and best_score >= max(3, runner_up + 2):
        route = MAP_ROUTE_MORNING_QUESTIONNAIRE
    else:
        route = best_route

    diff = best_score - runner_up
    if diff >= 4:
        confidence = "high"
    elif diff >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "route": route,
        "confidence": confidence,
        "signal_score": signal_score,
        "cpap_score": cpap_score,
        "cpap_extensive_score": cpap_extensive_score,
        "report_score": report_score,
        "extensive_report_score": extensive_report_score,
        "morning_score": morning_score,
        "night_score": night_score,
        "signal_hits": signal_hits,
        "cpap_hits": cpap_hits,
        "cpap_extensive_hits": cpap_extensive_hits,
        "report_hits": report_hits,
        "extensive_report_hits": extensive_report_hits,
        "morning_hits": morning_hits,
        "night_hits": night_hits,
        "direct_keyword_hits": [],
        "reason": "heuristic_router",
    }


def classify_map_route(ocr_text: str) -> Dict[str, Any]:
    return classify_map_route_heuristic(ocr_text)


def normalize_route_decision(decision: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
    fallback = classify_map_route_heuristic(ocr_text)
    if not isinstance(decision, dict):
        return fallback
    route = normalize_map_route_name(decision.get("route"))
    confidence_raw = str(decision.get("confidence", "")).strip().lower()
    confidence = confidence_raw if confidence_raw in {"high", "medium", "low"} else fallback["confidence"]
    reason = str(decision.get("reason", "")).strip()
    if not reason:
        reason = "llm_router"
    out = dict(fallback)
    out["route"] = route
    out["confidence"] = confidence
    out["reason"] = reason
    return out


MAP_ROUTE_SYSTEM = """
# Role: You are a routing classifier for OCR pages from a sleep clinic.
# Task: Choose exactly one of 7 possible map route types depending on the content of the page.
1) map_route_polysomnography_signals
- authors: psg machine.
- keywords: 'This page contains psg signal graphs.'
- Typical cues: EEG, EOG, EMG, airflow, psg channel labels.
2) map_route_psg_report_general
- authors: doctor or staffs, not patients.
- keywords
    - '신경과 수면검사', '수면다원검사', 'polysomnography data', or 'polysomnography report'. OR
    - PSG metrics and diagnostic notes OR
    - clinical notes mentioning diagnosis, impression, interpretation, or clinical correlation.
- caution: do NOT choose this for raw signal-graph pages without report metrics/tables; use map_route_polysomnography_signals instead.
3) map_route_psg_report_extensive
- authors: doctor or staffs, not patients.
- pre-condition: map_route_psg_report_general
- keywords: 'NIGHT POLYSOMNOGRAPHY REPORT'
4) map_route_cpap_psg_report_general
- authors: doctor or staffs, not patients.
- keywords: 'CPAP polysomnography', 'CPAP titration', 'PAP titration', repeated pressure rows, or cmH2O pressure-step content.
5) map_route_cpap_psg_report_extensive
- authors: doctor or staffs, not patients.
- pre-condition: map_route_cpap_psg_report_general
- keywords: 'FULL NIGHT CPAP POLYSOMNOGRAPHY REPORT'
6) map_route_morning_questionnaire
- authors: patients, not doctors or staffs.
- title: '아침 질문' or 'morning questionnaire'
- characteristic: asks questions comparing the patient's last psg night sleep to their home sleep.
- cautions: Just because it has questions related to morning time, it's not necessarily a morning questionnaire. 
- cautions: Just because it is titled as 'wake questionnaire', it's not necessarily a morning questionnaire.
7) map_route_night_questionnaire
- authors: patients, not doctors or staffs.
- pre-condition: not fitting in any of the above 6 categories.
- examples: PSQI, ESS, SSS, FSS, BQ, ISI, RLS, IRLS, RBDSQ, PHQ, BDI, QOL, habits/history, symptom checklists, and all other questions.

Output JSON only:
{
  "route": "<one route name>",
  "confidence": "high|medium|low",
  "reason": "<short reason>"
}
"""


def build_route_user_prompt(ocr_text: str) -> str:
    lines = [f"- {key}: {value}" for key, value in MAP_ROUTE_DESCRIPTIONS.items()]
    route_catalog = "\n".join(lines)
    route_text = prepare_ocr_text_for_router(ocr_text)
    return f"""OCR TEXT:
\"\"\"{route_text[:10000]}\"\"\"

AVAILABLE ROUTES:
{route_catalog}

Choose the single best route for this OCR text.
Return ONE JSON object only.
"""


def split_ocr_text_for_map_route(ocr_text: str, route_name: str) -> List[str]:
    route = normalize_map_route_name(route_name)
    text = str(ocr_text or "").strip()
    if not text:
        return [""]
    if route not in {MAP_ROUTE_PSG_REPORT_EXTENSIVE, MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE}:
        return [text]

    lines = text.splitlines()
    if len(lines) < 20:
        return [text]

    header_keep = min(12, max(4, len(lines) // 6))
    shared_header = lines[:header_keep]
    body = lines[header_keep:]
    if len(body) < 10:
        return [text]

    mid = len(body) // 2
    chunks: List[str] = []
    for body_part in (body[:mid], body[mid:]):
        part = "\n".join(shared_header + body_part).strip()
        if part:
            chunks.append(part)
    return chunks or [text]


def build_document_label_catalog_text() -> str:
    lines: List[str] = []
    for label, spec in DOCUMENT_LABEL_SPECS.items():
        desc = str(spec.get("description", "")).strip()
        lines.append(f"- {label}: {desc}")
    return "\n".join(lines)


class GlobalRequestThrottle:
    def __init__(self) -> None:
        self.min_interval_sec: float = 0.0
        self._next_allowed_at: float = 0.0
        self._lock: Optional[asyncio.Lock] = None

    def configure(self, min_interval_sec: float) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._next_allowed_at = 0.0
        self._lock = None

    async def wait_turn(self) -> None:
        if self.min_interval_sec <= 0.0:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_at:
                await asyncio.sleep(self._next_allowed_at - now)
                now = time.monotonic()
            self._next_allowed_at = now + self.min_interval_sec


REQUEST_THROTTLE = GlobalRequestThrottle()


def load_env() -> None:
    dotenv_path = REPO_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        load_dotenv(override=False)


def configure_logging(output_dir: Path, debug: bool = False, log_filename: str = "pipeline.log") -> Path:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    logger.setLevel(level)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / log_filename).resolve()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    has_file_handler = False
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).resolve() == log_path:
            has_file_handler = True
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)
            h.setFormatter(formatter)

    if not has_file_handler:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    logger.info("Logging initialized: level=%s, file=%s", logging.getLevelName(level), log_path)
    return log_path


# -----------------------------
# Filesystem helpers
# -----------------------------
def iter_patient_folders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def iter_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)


def image_to_data_url(image_path: Path, max_side: int = 2048) -> str:
    """
    Convert image to a base64 data URL. Downscale large images for payload size.
    """
    img: Optional[Image.Image] = None
    try:
        with Image.open(image_path) as raw:
            img = raw.convert("RGB")
    except OSError as e:
        # Some scans are slightly truncated but still decodable enough for OCR.
        if "truncated" not in str(e).lower():
            raise
        logger.warning(
            "Image appears truncated, retrying with tolerant loader: %s (%s)",
            image_path.name,
            e,
        )
        prev = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(image_path) as raw:
                img = raw.convert("RGB")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev

    if img is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# -----------------------------
# JSON helpers
# -----------------------------
def llm_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
                continue
            parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def safe_extract_json(text: Any) -> Dict[str, Any]:
    text = llm_content_to_text(text)

    # pure JSON?
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # try substring {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not parse JSON. Output starts with: {text[:200]!r}")


def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"null", "none", "n/a", "na"}:
            return None
        return s
    return v


def _contains_hangul(text: Any) -> bool:
    if text is None:
        return False
    return bool(re.search(r"[가-힣]", str(text)))


def _is_missing_value(v: Any) -> bool:
    return normalize_value(v) is None


def _normalize_filled_by(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    if any(t in s for t in ["doctor", "physician", "md", "의사", "전문의", "clinician", "lab"]):
        return "doctor"
    if any(t in s for t in ["patient", "self", "subject", "respondent", "환자", "본인", "보호자"]):
        return "patient"
    return ""


def _default_input_context() -> Dict[str, Any]:
    return {"filled_by": "", "question": "", "page_type": ""}


OFFICIAL_QUESTIONNAIRE_RULE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "MQ": ("아침 질문 사항",),
    "SSS": ("Stanford Sleepiness Scale", "Stanford sleepiness scale"),
    "ESS": ("The Epworth Sleepiness Scale", "Epworth Sleepiness Scale"),
    "FSS": ("Fatigue Severity Scale", "피로 정도에 대한 설문"),
    "BQ": ("Berlin Questionnaire",),
    "ISI": ("Insomnia Severity Index", "불면증에 관한 설문", "불면증에 관한 질문"),
    "RLS": ("하지불안증후군/주기성사지운동증후군", "Restless Legs Syndromes and PLMS questions"),
    "IRLS": ("하지불안증후군에 대한 설문",),
    "PSQI": ("수면의 질 지수", "Pittsburgh Sleep Quality Index", "PITTSBURGH SLEEP QUALITY INDEX"),
    "BDI": ("우울증에 관한 설문", "Beck Depression Inventory", "Beck depression inventory"),
    "QOL": ("삶의 질 척도", "WHOQOL-BREF"),
}
MULTIPAGE_OFFICIAL_FAMILIES: Tuple[str, ...] = ("PSQI", "BDI", "QOL")
OFFICIAL_GENERIC_BREAK_PATTERNS: Tuple[str, ...] = (
    "생활 습관",
    "병력과 가족력",
    "수면 습관",
    "수면에 관한 설문지",
    "수면다원검사 설문지",
    r"POLYSOMNOGRAPHY\s*\|\s*QUESTIONNAIRE",
    "SLEEP QUESTIONNAIRE",
    "SLEEP - WAKE QUESTIONNAIRE",
    "Living habit",
)
PSQI_PAGE2_CUE_PATTERNS: Tuple[str, ...] = (
    r"During the past month,\s*how would you rate your sleep quality overall\?",
    r"rate your sleep quality overall",
    r"지난 한달 동안,\s*당신의 전반적인 수면의 질은 어떠하였습니까",
)


def _extract_page_type_from_text(text_like: Any) -> str:
    text = str(text_like or "").strip()
    m = re.match(r"\[page_type:\s*([^\]]+)\]\s*", text)
    if not m:
        return ""
    return normalize_map_route_name(m.group(1))


def _extract_page_type_from_context(ctx: Any) -> str:
    if isinstance(ctx, dict):
        raw_direct = str(ctx.get("page_type") or "").strip()
        direct = normalize_map_route_name(raw_direct) if raw_direct else ""
        if direct:
            return direct
        for legacy_key in ("relevance", "page", "page_summary", "source_page", "page_context", "summary"):
            legacy = _extract_page_type_from_text(ctx.get(legacy_key))
            if legacy:
                return legacy
    elif isinstance(ctx, str):
        legacy = _extract_page_type_from_text(ctx)
        if legacy:
            return legacy
    return ""


def _normalize_input_context(ctx: Any) -> Dict[str, Any]:
    out = _default_input_context()
    if isinstance(ctx, str):
        out["question"] = ctx.strip()
        return out
    if not isinstance(ctx, dict):
        return out

    filled_by_raw = (
        ctx.get("filled_by")
        or ctx.get("who_filled")
        or ctx.get("source_filler")
        or ctx.get("filler")
    )
    question_raw = (
        ctx.get("question")
        or ctx.get("source_question")
        or ctx.get("exact_question")
        or ctx.get("prompt")
        or ctx.get("text")
    )
    out["filled_by"] = _normalize_filled_by(filled_by_raw)
    out["question"] = str(question_raw or "").strip()
    out["page_type"] = _extract_page_type_from_context(ctx)
    return out


def parse_value_context_map(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    Normalize model output into:
      - values: key -> mapped value candidate
      - contexts: key -> {filled_by, question, page_type}
      - cdm_contexts: key -> string explaining what the CDM key is about

    Accepts both old format:
      {"KEY": 1}
    and new format:
      {"KEY": {"CDM_Context": "...", "value": 1, "input_context": {...}}}
    """
    values: Dict[str, Any] = {}
    contexts: Dict[str, Dict[str, Any]] = {}
    cdm_contexts: Dict[str, str] = {}
    for raw_k, raw_v in obj.items():
        key = str(raw_k).strip()
        if not key:
            continue

        if isinstance(raw_v, dict) and "value" in raw_v:
            values[key] = raw_v.get("value")
            contexts[key] = _normalize_input_context(raw_v.get("input_context"))
            cdm_contexts[key] = str(
                raw_v.get("CDM_Context")
                or raw_v.get("cdm_context")
                or raw_v.get("CDM_context")
                or ""
            ).strip()
            # Allow flattened context fields.
            if contexts[key] == _default_input_context():
                contexts[key] = _normalize_input_context(raw_v)
            continue

        values[key] = raw_v
        contexts[key] = _default_input_context()
        cdm_contexts[key] = ""
    return values, contexts, cdm_contexts


def _type_a_official_family_prefix(key: str) -> str:
    family_prefixes = (
        "MQ_",
        "SSS_",
        "ESS_",
        "FSS_",
        "BQ_",
        "ISI_",
        "RLS_",
        "IRLS_",
        "RBDSQ_",
        "PHQ_",
        "BDI_",
        "QOL_",
        "PSQI_",
    )
    for prefix in family_prefixes:
        if key.startswith(prefix):
            return prefix
    return ""


def _ocr_text_matches_type_a_official_family(key: str, ocr_text: str) -> bool:
    return True


def classify_official_questionnaire_family(text: Any) -> str:
    raw = str(text or "")
    for family, patterns in OFFICIAL_QUESTIONNAIRE_RULE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, raw, flags=re.I):
                return family
    return "NON"


def looks_like_official_generic_break(text: Any) -> bool:
    raw = str(text or "")
    return any(re.search(pattern, raw, flags=re.I) for pattern in OFFICIAL_GENERIC_BREAK_PATTERNS)


def looks_like_psqi_page2(text: Any) -> bool:
    raw = str(text or "")
    return any(re.search(pattern, raw, flags=re.I) for pattern in PSQI_PAGE2_CUE_PATTERNS)


def classify_official_questionnaire_sequence(
    bundle_infos: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    active_family = "NON"
    for info in bundle_infos:
        bundle_name = str(info.get("bundle_name") or "").strip()
        route_name = normalize_map_route_name(info.get("route_name"))
        ocr_text = str(info.get("ocr_text") or "")
        if route_name not in {MAP_ROUTE_NIGHT_QUESTIONNAIRE, MAP_ROUTE_MORNING_QUESTIONNAIRE}:
            out[bundle_name] = {"official_questionnaire": False, "official_family": "NON"}
            active_family = "NON"
            continue

        family = classify_official_questionnaire_family(ocr_text)
        if family != "NON":
            out[bundle_name] = {"official_questionnaire": True, "official_family": family}
            active_family = family if family in MULTIPAGE_OFFICIAL_FAMILIES else "NON"
            continue

        if looks_like_official_generic_break(ocr_text):
            out[bundle_name] = {"official_questionnaire": False, "official_family": "NON"}
            active_family = "NON"
            continue

        if active_family == "PSQI":
            if looks_like_psqi_page2(ocr_text):
                out[bundle_name] = {"official_questionnaire": True, "official_family": "PSQI"}
            else:
                out[bundle_name] = {"official_questionnaire": False, "official_family": "NON"}
                active_family = "NON"
            continue

        if active_family in {"BDI", "QOL"}:
            out[bundle_name] = {"official_questionnaire": True, "official_family": active_family}
            continue

        out[bundle_name] = {"official_questionnaire": False, "official_family": "NON"}
    return out


def _coerce_int(v: Any) -> Optional[int]:
    nv = normalize_value(v)
    if nv is None:
        return None
    if isinstance(nv, bool):
        return None
    if isinstance(nv, int):
        return int(nv)
    if isinstance(nv, float):
        if not (float("-inf") < nv < float("inf")):
            return None
        return int(round(nv))
    s = str(nv).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return int(round(float(m.group(0))))
    except Exception:
        return None


def _norm_cmp(v: Any) -> str:
    iv = _coerce_int(v)
    if iv is not None:
        return str(iv)
    nv = normalize_value(v)
    if nv is None:
        return ""
    return str(nv).strip().lower()


def _value_token(v: Any) -> str:
    return json.dumps(normalize_value(v), ensure_ascii=False, sort_keys=True)


def _normalize_text_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^\d+\s*[\.\)]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _to_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None


def _to_yyyymmdd(v: Any) -> Optional[str]:
    s = str(v).strip()
    m = re.search(r"\b(\d{8})\b", s)
    if m:
        cand = m.group(1)
    else:
        digits = re.sub(r"\D", "", s)
        if len(digits) != 8:
            return None
        cand = digits
    try:
        datetime.strptime(cand, "%Y%m%d")
        return cand
    except Exception:
        return None


def _parse_numeric_range(format_range: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", format_range)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_choice_tokens(format_range: str) -> Optional[set]:
    if not format_range:
        return None
    if "YYYYMMDD" in format_range.upper():
        return None
    if re.search(r"\d+\s*-\s*\d+", format_range):
        return None
    if "," not in format_range:
        return None
    tokens = [t.strip() for t in format_range.split(",")]
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return None
    if any(len(t) > 16 for t in tokens):
        return None
    return {t.upper() for t in tokens}


def _parse_explicit_numeric_choices(format_range: str) -> set[int]:
    """
    Parse explicit numeric tokens from format strings like "1-4, 9999".
    Ranges are ignored; only standalone integers are returned.
    """
    out: set[int] = set()
    if not format_range:
        return out
    for tok in [t.strip() for t in str(format_range).split(",")]:
        if re.fullmatch(r"-?\d+", tok or ""):
            out.add(int(tok))
    return out


def _normalize_numeric_value(x: float) -> Any:
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return round(x, 4)


def _is_pure_numeric_string(s: str) -> bool:
    s = s.strip().replace(",", "")
    return re.fullmatch(r"-?\d+(?:\.\d+)?", s) is not None


def _extract_numeric_range_pair(v: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    # Normalize separators: "~", "～", "to", "between ... and ..."
    s_norm = s.replace("～", "~").replace("∼", "~").replace("–", "-").replace("—", "-")
    m = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:~|-|to)\s*(-?\d+(?:\.\d+)?)",
        s_norm,
        flags=re.I,
    )
    if not m:
        m2 = re.search(
            r"between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)",
            s_norm,
            flags=re.I,
        )
        if not m2:
            return None
        m = m2
    try:
        a = float(m.group(1))
        b = float(m.group(2))
        return (a, b)
    except Exception:
        return None


def _is_time_like_field(row: "CDMRow") -> bool:
    key = str(row.key).lower()
    desc = str(row.desc).lower()
    hints = [
        "_hh",
        "_mm",
        "latency",
        "time",
        "duration",
        "sleep",
        "wake",
        "tst",
        "weso",
        "waso",
        "분",
        "시간",
        "시각",
    ]
    joined = key + " " + desc
    return any(h in joined for h in hints)


def _is_severity_like_field(row: "CDMRow") -> bool:
    key = str(row.key).lower()
    desc = str(row.desc).lower()
    hints = [
        "severity",
        "freq",
        "frequency",
        "scale",
        "grade",
        "정도",
        "빈도",
        "심함",
        "심각",
    ]
    joined = key + " " + desc
    return any(h in joined for h in hints)


PSQI_0104_BASES = (
    "PSQI_01_BedIn",
    "PSQI_02_Latency",
    "PSQI_03_BedOut",
    "PSQI_04_SD",
)


def _detect_psqi_version_mode(ocr_text: str) -> str:
    """
    Returns:
      - "weekfree" if OCR explicitly indicates weekday/weekend form.
      - "single" otherwise.
    """
    txt = str(ocr_text or "")
    if re.search(r"주중|주말|weekday|weekend", txt, flags=re.I):
        return "weekfree"
    return "single"


def _is_psqi_0104_weekfree_key(key: str) -> bool:
    for base in PSQI_0104_BASES:
        if key.startswith(base + "_") and (key.endswith("_week") or key.endswith("_free")):
            return True
    return False


def _is_psqi_0104_single_key(key: str) -> bool:
    for base in PSQI_0104_BASES:
        if not key.startswith(base + "_"):
            continue
        if key.endswith("_week") or key.endswith("_free"):
            return False
        suffix = key[len(base) + 1 :]
        if suffix in {"HH", "MM"}:
            return True
    return False


def validate_value_with_cdm(row: "CDMRow", value: Any) -> Tuple[Any, Optional[str]]:
    v = normalize_value(value)
    if v is None:
        return None, "empty"
    if isinstance(v, (list, dict, tuple, set)):
        return None, "non_scalar"

    # Option-coded field: enforce code output.
    if row.options:
        code: Optional[str] = None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if float(v).is_integer():
                code = str(int(v))
        elif isinstance(v, str):
            sv = v.strip()
            if re.fullmatch(r"-?\d+(?:\.0+)?", sv):
                code = str(int(float(sv)))

        if code is not None and code in row.options:
            return int(code), None

        # If an option-coded answer is given as a range (e.g., "1~2"),
        # keep the more severe side as requested.
        rng_pair = _extract_numeric_range_pair(v)
        if rng_pair is not None:
            lo_i = int(round(min(rng_pair[0], rng_pair[1])))
            hi_i = int(round(max(rng_pair[0], rng_pair[1])))
            if str(hi_i) in row.options:
                return hi_i, None
            if str(lo_i) in row.options:
                return lo_i, None

        # Some CDM rows include extra sentinel values in Format/Range
        # that are not enumerated in option columns (e.g., "..., 9999").
        if code is not None:
            iv = int(code)
            if iv in _parse_explicit_numeric_choices(row.format_range or ""):
                return iv, None

        text_v = _normalize_text_token(str(v))
        for opt_code, opt_label in row.options.items():
            text_label = _normalize_text_token(opt_label)
            if text_v == text_label or text_v in text_label or text_label in text_v:
                return int(opt_code), None
        return None, "invalid_option"

    fr = row.format_range or ""
    fr_upper = fr.upper()

    if "YYYYMMDD" in fr_upper:
        d = _to_yyyymmdd(v)
        if d is None:
            return None, "invalid_date"
        return d, None

    enum_tokens = _parse_choice_tokens(fr)
    if enum_tokens is not None:
        token = str(v).strip().upper()
        if token in enum_tokens:
            return token, None
        return None, "invalid_choice"

    num_range = _parse_numeric_range(fr)
    if num_range is not None:
        x = _to_number(v)
        if x is None:
            rng_pair = _extract_numeric_range_pair(v)
            if rng_pair is not None:
                lo_raw = min(rng_pair[0], rng_pair[1])
                hi_raw = max(rng_pair[0], rng_pair[1])
                # User rule:
                # - time range => median
                # - severity range => more severe
                if _is_time_like_field(row):
                    x = (lo_raw + hi_raw) / 2.0
                elif _is_severity_like_field(row):
                    x = hi_raw
                else:
                    x = (lo_raw + hi_raw) / 2.0
        if x is None:
            return None, "not_numeric"
        lo, hi = num_range
        if not (lo <= x <= hi):
            return None, "out_of_range"
        return _normalize_numeric_value(x), None

    # Keep raw string-like identifiers (e.g., PSG_No, Database_ID) as text.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _normalize_numeric_value(float(v)), None
    if isinstance(v, str) and _is_pure_numeric_string(v):
        return _normalize_numeric_value(float(v)), None

    return str(v).strip(), None


def validate_extracted_json(
    obj: Dict[str, Any],
    retriever: "CDMRetriever",
    ocr_text: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    valid: Dict[str, Any] = {}
    rejected: Dict[str, Dict[str, Any]] = {}
    psqi_mode = _detect_psqi_version_mode(ocr_text)

    for k, v in obj.items():
        key = str(k).strip()
        if key not in retriever.row_by_key:
            rejected[key] = {"value": v, "reason": "unknown_key"}
            continue

        if psqi_mode == "weekfree" and _is_psqi_0104_single_key(key):
            rejected[key] = {"value": v, "reason": "psqi_single_variant_disallowed_by_ocr"}
            continue
        if psqi_mode == "single" and _is_psqi_0104_weekfree_key(key):
            rejected[key] = {"value": v, "reason": "psqi_weekfree_variant_disallowed_by_ocr"}
            continue

        norm, reason = validate_value_with_cdm(retriever.row_by_key[key], v)
        if norm is None:
            rejected[key] = {"value": v, "reason": reason}
            continue
        valid[key] = norm

    return valid, rejected


def _to_yyyymmdd_from_parts(year: int, month: int, day: int) -> Optional[str]:
    try:
        return datetime(year=year, month=month, day=day).strftime("%Y%m%d")
    except Exception:
        return None


def extract_core_fields_from_ocr(ocr_text: str) -> Dict[str, Any]:
    """
    Regex backfill for high-value demographics/report-header fields.
    This is intentionally conservative and only adds values when patterns are explicit.
    """
    out: Dict[str, Any] = {}
    txt = ocr_text

    # Hospital/registration id: prefer long digit ids.
    id_patterns = [
        r"(?:등록번호|registration\s*no\.?|sleep study number|hospi\.?\s*no\.?)\D{0,12}([0-9]{6,12})",
        r"\bID\s*(?:[:#|]\s*)?([0-9]{6,12})\b",
    ]
    for pat in id_patterns:
        m = re.search(pat, txt, re.I)
        if m:
            out["Hospital_ID"] = m.group(1).strip()
            break

    # Name
    name_patterns = [
        r"(?:성명|name)\s*[:#]\s*([^\n\r|]{2,80})",
        r"(?:환자명)\s*[:#]\s*([^\n\r|]{2,80})",
    ]
    for pat in name_patterns:
        m = re.search(pat, txt, re.I)
        if not m:
            continue
        name = m.group(1).strip(" -\t")
        name = name.replace("*", "")
        name = re.sub(r"^[^0-9A-Za-z가-힣]+", "", name)
        name = re.sub(r"\s{2,}", " ", name)
        # Trim obvious trailing mixed fields.
        name = re.sub(r"\b(?:id|sex|age|dept|psg#?)\b.*$", "", name, flags=re.I).strip(" ,;")
        name = re.sub(r"[, ]+\d{4,}$", "", name).strip(" ,;")
        if name and len(name) >= 2:
            out["Name"] = name
            break

    # PSG number
    psg_patterns = [
        r"(?:test\s*no\.?|psg#?)\s*[:#]?\s*([A-Za-z]?\d{4}\s*[-/]\s*\d+)",
        r"\b(P\d{4}\s*[-/]\s*\d+)\b",
    ]
    for pat in psg_patterns:
        m = re.search(pat, txt, re.I)
        if m:
            out["PSG_No"] = re.sub(r"\s+", "", m.group(1))
            break

    # Date
    date_candidates: List[str] = []
    # 2012.05.22 / 2012-05-22 / 2012/05/22
    for y, m, d in re.findall(r"\b(20\d{2})[./-]\s*(\d{1,2})[./-]\s*(\d{1,2})\b", txt):
        val = _to_yyyymmdd_from_parts(int(y), int(m), int(d))
        if val:
            date_candidates.append(val)
    # 22 day 05 mo 2012 year
    for d, m, y in re.findall(
        r"\b(\d{1,2})\s*day\s*(\d{1,2})\s*mo\s*(20\d{2})\s*year\b",
        txt,
        flags=re.I,
    ):
        val = _to_yyyymmdd_from_parts(int(y), int(m), int(d))
        if val:
            date_candidates.append(val)
    if date_candidates:
        out["PSG_Date"] = date_candidates[0]

    # Sex + age
    sex_age = re.search(r"sex\s*/\s*age\s*[:#]?\s*([MF])\s*/\s*(\d{1,3})", txt, re.I)
    if sex_age:
        sx = sex_age.group(1).upper()
        out["SEX"] = "Male" if sx == "M" else "Female"
        out["AGE"] = int(sex_age.group(2))
    else:
        sex = re.search(r"\bsex\b\s*[:#]?\s*(male|female|m|f)\b", txt, re.I)
        if sex:
            sx = sex.group(1).strip().lower()
            out["SEX"] = "Male" if sx in {"male", "m"} else "Female"
        age = re.search(r"\bage\b\s*[:#]?\s*(\d{1,3})\b", txt, re.I)
        if age:
            out["AGE"] = int(age.group(1))

    # Anthropometrics
    h = re.search(r"\bheight\b\s*[:#]?\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", txt, re.I)
    if h:
        out["Height_cm"] = float(h.group(1))
    w = re.search(r"\bweight\b\s*[:#]?\s*(\d{2,3}(?:\.\d+)?)\s*kg\b", txt, re.I)
    if w:
        out["Weight_kg"] = float(w.group(1))
    bmi = re.search(r"\b(?:bmi|body mass index)\b\s*[:#]?\s*(\d{1,2}(?:\.\d+)?)", txt, re.I)
    if bmi:
        out["BMI"] = float(bmi.group(1))

    # Common report totals often shown in header.
    ess = re.search(r"\bESS\b\s*[:#]?\s*(\d{1,2})\b", txt, re.I)
    if ess:
        out["ESS_Total"] = int(ess.group(1))
    bdi = re.search(r"\bBDI\b\s*[:#]?\s*(\d{1,2})\b", txt, re.I)
    if bdi:
        out["BDI_Total"] = int(bdi.group(1))

    return out


def apply_core_backfill(
    valid_obj: Dict[str, Any],
    retriever: "CDMRetriever",
    ocr_text: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    additions: Dict[str, Any] = {}
    rejected: Dict[str, Dict[str, Any]] = {}

    extracted = extract_core_fields_from_ocr(ocr_text)
    for key, val in extracted.items():
        row = retriever.row_by_key.get(key)
        if row is None:
            continue
        norm, reason = validate_value_with_cdm(row, val)
        if norm is None:
            rejected[key] = {"value": val, "reason": f"backfill_{reason}"}
            continue

        # Keep original script for names. If current name is romanized but OCR has Hangul, prefer Hangul.
        if key == "Name" and key in valid_obj:
            current = valid_obj.get("Name")
            if (not _contains_hangul(current)) and _contains_hangul(norm):
                additions[key] = norm
            continue

        if key in valid_obj:
            continue
        additions[key] = norm
    return additions, rejected


def should_run_recall_pass(ocr_text: str, valid_obj: Dict[str, Any], min_keys: int = 12) -> bool:
    if len(valid_obj) >= min_keys:
        return False
    if len(ocr_text.strip()) < 180:
        return False

    hint_hits = len(RECALL_HINT_RE.findall(ocr_text))
    numeric_hits = len(re.findall(r"\b\d+(?:\.\d+)?\b", ocr_text))
    keyval_hits = len(re.findall(r":\s*[^\n]{1,60}", ocr_text))

    structured_hint = bool(
        re.search(
            r"selected|circled|checked|\[x\]|name:|등록번호|study date|sex/age|height|weight|bmi|"
            r"total sleep time|sleep latency|rem latency|sleep efficiency|arousal index|\|",
            ocr_text,
            re.I,
        )
    )

    return hint_hits >= 1 and structured_hint and (numeric_hits >= 8 or keyval_hits >= 4)


def _psg_type_to_token(v: Any) -> Optional[str]:
    """
    Normalize PSG_Type into one of: P, PE, C, SP, M.
    Accepts code values (1..5) or text labels.
    """
    if v is None:
        return None

    code_map = {
        "1": "P",
        "2": "PE",
        "3": "C",
        "4": "SP",
        "5": "M",
    }

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if float(v).is_integer():
            return code_map.get(str(int(v)))

    s = str(v).strip().upper()
    if not s:
        return None

    if re.fullmatch(r"\d+(?:\.0+)?", s):
        return code_map.get(str(int(float(s))))

    if s in {"P", "PE", "C", "SP", "M"}:
        return s

    m = re.search(r"\b(PE|SP|P|C|M)\b", s)
    if m:
        return m.group(1)
    return None


def _psg_no_suffix(psg_no: Any) -> Optional[str]:
    if psg_no is None:
        return None
    s = str(psg_no).strip()
    m = re.search(r"(\d+)\s*$", s)
    if not m:
        return None
    return str(int(m.group(1)))


def synthesize_database_id(row: Dict[str, Any]) -> Optional[str]:
    """
    Rule:
      Database_ID = 001_A_B_C
      A: PSG_Date (YYYYMMDD)
      B: ending number of PSG_No
      C: PSG_Type (P/PE/C/SP/M)
    """
    a = _to_yyyymmdd(row.get("PSG_Date"))
    b = _psg_no_suffix(row.get("PSG_No"))
    c = _psg_type_to_token(row.get("PSG_Type"))
    if not (a and b and c):
        return None
    return f"001_{a}_{b}_{c}"


PSQI_BASE_GROUPS = [
    "PSQI_01_BedIn",
    "PSQI_02_Latency",
    "PSQI_03_BedOut",
    "PSQI_04_SD",
]


def _normalize_psqi_clock_hour(group: str, hour: int) -> Optional[int]:
    x = hour
    if x == 24:
        x = 0

    # Bedtime fields are often written in 12h style without AM/PM on paper forms.
    # Heuristic: 6-11 likely PM (18-23), while 0-5 are commonly after-midnight bedtimes.
    if group == "PSQI_01_BedIn":
        if x == 12:
            x = 0
        elif 6 <= x <= 11:
            x += 12

    if 0 <= x <= 23:
        return x
    return None


def apply_psqi_format_and_time_rules(row: Dict[str, Any]) -> None:
    for base in PSQI_BASE_GROUPS:
        for suffix in ("", "_week", "_free"):
            hh_k = f"{base}_HH{suffix}"
            mm_k = f"{base}_MM{suffix}"
            hh_has = hh_k in row and not _is_missing_value(row.get(hh_k))
            mm_has = mm_k in row and not _is_missing_value(row.get(mm_k))

            if hh_has and not mm_has:
                row[mm_k] = 0
                mm_has = True
            elif mm_has and not hh_has:
                row[hh_k] = 0
                hh_has = True

            for unit, tk in (("HH", hh_k), ("MM", mm_k)):
                if tk not in row or _is_missing_value(row.get(tk)):
                    continue
                if unit == "MM":
                    nv = _to_number(row.get(tk))
                    if nv is None:
                        continue
                    if abs(nv - 60.0) < 1e-9:
                        nv = 0.0
                    row[tk] = _normalize_numeric_value(nv) if 0 <= nv <= 59 else None
                elif unit == "HH":
                    iv = _coerce_int(row.get(tk))
                    if iv is None:
                        continue
                    if base in {"PSQI_01_BedIn", "PSQI_03_BedOut"}:
                        row[tk] = _normalize_psqi_clock_hour(base, iv)
                    else:
                        row[tk] = iv


MORNING_TIME_BASE_GROUPS = [
    "PSG_M_02_SubSL",
    "PSG_M_03_SubSD",
]


def apply_morning_questionnaire_time_rules(row: Dict[str, Any]) -> None:
    for base in MORNING_TIME_BASE_GROUPS:
        hh_k = f"{base}_HH"
        mm_k = f"{base}_MM"
        hh_has = hh_k in row and not _is_missing_value(row.get(hh_k))
        mm_has = mm_k in row and not _is_missing_value(row.get(mm_k))

        if hh_has and not mm_has:
            row[mm_k] = 0
        elif mm_has and not hh_has:
            row[hh_k] = 0

        if hh_k in row and not _is_missing_value(row.get(hh_k)):
            iv = _coerce_int(row.get(hh_k))
            row[hh_k] = iv if iv is not None and 0 <= iv <= 23 else None

        if mm_k in row and not _is_missing_value(row.get(mm_k)):
            nv = _to_number(row.get(mm_k))
            if nv is None:
                row[mm_k] = None
            else:
                if abs(nv - 60.0) < 1e-9:
                    nv = 0.0
                row[mm_k] = _normalize_numeric_value(nv) if 0 <= nv <= 59 else None


def apply_phx_default_rules(row: Dict[str, Any]) -> None:
    phx_cols = [k for k in row.keys() if k.startswith("PHx_")]
    if not phx_cols:
        return

    for k in phx_cols:
        if _is_missing_value(row.get(k)):
            continue
        iv = _coerce_int(row.get(k))
        if iv is not None:
            row[k] = iv


CPAP_ROUTE_NAMES = {
    MAP_ROUTE_CPAP_PSG_REPORT_GENERAL,
    MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE,
}
CPAP_PRESSURE_KEY_RE = re.compile(r"^Pressure_(\d{2})$")
CPAP_PRESSURE_METRIC_KEY_RE = re.compile(r"^Pr(\d{2})_(.+)$")
CPAP_SNORING_MISSING_TOKENS = {"-", "--", "—", "–"}
CPAP_PRESSURE_CUE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(\d{1,2})\s*(?:/|\|)"),
    re.compile(r"(?i)\bat\s+cpap\s+pressure\s*(\d{1,2})\s*cm\s*h2o\b"),
    re.compile(r"(?i)\bcpap\s+pressure\s*(\d{1,2})\s*cm\s*h2o\b"),
    re.compile(r"(?i)\bpressure\s*(\d{1,2})\s*cm\s*h2o\b"),
)


def _extract_cpap_pressure_cues(question: Any) -> List[int]:
    text = str(question or "").strip()
    if not text:
        return []
    steps: List[int] = []
    for pat in CPAP_PRESSURE_CUE_PATTERNS:
        for m in pat.finditer(text):
            try:
                step = int(m.group(1))
            except Exception:
                continue
            if CPAP_PRESSURE_STEP_START <= step <= CPAP_PRESSURE_STEP_END and step not in steps:
                steps.append(step)
    return steps


def _normalize_cpap_position_value(v: Any) -> Any:
    s = normalize_value(v)
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s).strip())
    return s.lower()


def _normalize_cpap_stage_value(v: Any) -> Any:
    s = normalize_value(v)
    if s is None:
        return None
    text = re.sub(r"\s+", "", str(s).strip())
    return re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), text)


def _normalize_cpap_snoring_value(v: Any) -> Any:
    s = normalize_value(v)
    if s is None:
        return None
    text = str(s).strip()
    if text in CPAP_SNORING_MISSING_TOKENS:
        return None
    return text.lower()


def _normalize_cpap_field_value(key: str, value: Any) -> Any:
    if key.endswith("_position"):
        return _normalize_cpap_position_value(value)
    if key.endswith("_stage"):
        return _normalize_cpap_stage_value(value)
    if key.endswith("_snoring"):
        return _normalize_cpap_snoring_value(value)
    if CPAP_PRESSURE_KEY_RE.fullmatch(key):
        iv = _coerce_int(value)
        return iv if iv is not None else normalize_value(value)
    return normalize_value(value)


def _apply_cpap_output_rules(row: Dict[str, Any]) -> None:
    for key in list(row.keys()):
        pressure_m = CPAP_PRESSURE_KEY_RE.fullmatch(key)
        metric_m = CPAP_PRESSURE_METRIC_KEY_RE.fullmatch(key)
        if not pressure_m and not metric_m:
            continue

        row[key] = _normalize_cpap_field_value(key, row.get(key))

        if pressure_m and not _is_missing_value(row.get(key)):
            expected_step = int(pressure_m.group(1))
            iv = _coerce_int(row.get(key))
            row[key] = iv if iv == expected_step else None

    for step in range(CPAP_PRESSURE_STEP_START, CPAP_PRESSURE_STEP_END + 1):
        pos_k = f"Pr{step:02d}_position"
        stage_k = f"Pr{step:02d}_stage"
        pos_v = normalize_value(row.get(pos_k))
        stage_v = normalize_value(row.get(stage_k))
        if pos_v is None or stage_v is None:
            continue
        parts = str(pos_v).split()
        if len(parts) < 2:
            continue
        trailing = parts[-1].lower()
        if trailing not in {"w", "r", "1", "2", "3"}:
            continue
        base_position = " ".join(parts[:-1]).strip()
        if not base_position:
            continue
        stage_tokens = [tok for tok in re.split(r"\s*,\s*", str(stage_v).strip()) if tok]
        stage_tokens = [re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), tok) for tok in stage_tokens]
        if trailing not in stage_tokens:
            stage_tokens.append(trailing)
        row[pos_k] = base_position
        row[stage_k] = ",".join(stage_tokens)


def _apply_cpap_page_guardrails(
    valid: Dict[str, Any],
    raw_contexts: Dict[str, Dict[str, Any]],
    rejected: Dict[str, Dict[str, Any]],
) -> None:
    supported_steps: set[int] = set()

    for key, value in list(valid.items()):
        m = CPAP_PRESSURE_KEY_RE.fullmatch(key)
        if not m:
            continue
        step = int(m.group(1))
        iv = _coerce_int(value)
        if iv is None or iv != step:
            ctx = _normalize_input_context(raw_contexts.get(key))
            valid.pop(key, None)
            rejected[key] = {
                "value": value,
                "reason": "cpap_pressure_value_mismatch",
                "input_context": ctx,
            }
            continue
        valid[key] = iv
        supported_steps.add(step)

    for key, value in list(valid.items()):
        m = CPAP_PRESSURE_METRIC_KEY_RE.fullmatch(key)
        if not m:
            continue
        step = int(m.group(1))
        ctx = _normalize_input_context(raw_contexts.get(key))
        cues = _extract_cpap_pressure_cues(ctx.get("question"))
        if cues and step not in cues:
            valid.pop(key, None)
            rejected[key] = {
                "value": value,
                "reason": "cpap_pressure_row_mismatch",
                "input_context": ctx,
            }
            continue
        if cues and step in cues:
            supported_steps.add(step)

    for key, value in list(valid.items()):
        m = CPAP_PRESSURE_METRIC_KEY_RE.fullmatch(key)
        if not m:
            continue
        step = int(m.group(1))
        ctx = _normalize_input_context(raw_contexts.get(key))
        cues = _extract_cpap_pressure_cues(ctx.get("question"))
        if not cues and step not in supported_steps:
            valid.pop(key, None)
            rejected[key] = {
                "value": value,
                "reason": "cpap_pressure_row_missing_evidence",
                "input_context": ctx,
            }
            continue

        norm = _normalize_cpap_field_value(key, value)
        if norm is None and not _is_missing_value(value):
            valid.pop(key, None)
            rejected[key] = {
                "value": value,
                "reason": "cpap_placeholder_blank",
                "input_context": ctx,
            }
            continue
        valid[key] = norm


def normalize_diagnosis_etc_value(v: Any) -> Optional[str]:
    nv = normalize_value(v)
    if nv is None:
        return None
    text = str(nv).strip()
    if not text:
        return None

    # Keep only #... fragments from II. Diagnosis text.
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    hash_lines = [ln for ln in lines if ln.startswith("#")]
    if hash_lines:
        return " ".join(hash_lines)

    # Fallback: recover inline hash phrases.
    inline = re.findall(r"(#[^#\n\r]+)", text)
    inline = [x.strip() for x in inline if x.strip()]
    if inline:
        return " ".join(inline)
    return None


def merge_diagnosis_etc_values(values: List[Any]) -> Optional[str]:
    merged_lines: List[str] = []
    seen = set()
    for v in values:
        norm = normalize_diagnosis_etc_value(v)
        if not norm:
            continue
        for ln in [x.strip() for x in re.split(r"[\r\n]+", norm) if x.strip()]:
            if not ln.startswith("#"):
                continue
            if ln in seen:
                continue
            seen.add(ln)
            merged_lines.append(ln)
    if not merged_lines:
        return None
    return " ".join(merged_lines)


def merge_page_results(
    page_results: List["PageResult"],
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pr in page_results:
        for k, v in pr.valid_json.items():
            by_key[k].append(
                {
                    "image": pr.image_name,
                    "value": v,
                    "cdm_context": str(pr.cdm_contexts.get(k, "")).strip(),
                    "input_context": _normalize_input_context(pr.input_contexts.get(k)),
                }
            )

    merged: Dict[str, Any] = {}
    conflicts: Dict[str, List[Dict[str, Any]]] = {}
    provenance: Dict[str, List[Dict[str, Any]]] = {}

    for k, entries in by_key.items():
        provenance[k] = entries

        conflict_entries: List[Dict[str, Any]] = []
        for e in entries:
            conflict_entries.append(
                {
                    "image": e.get("image"),
                    "value": e.get("value"),
                    "cdm_context": str(e.get("cdm_context", "")).strip(),
                    "input_context": _normalize_input_context(e.get("input_context")),
                }
            )
        value_tokens = {_value_token(normalize_value(e.get("value"))) for e in conflict_entries}
        if k == "Diagnosis_etc":
            merged_diag = merge_diagnosis_etc_values([e.get("value") for e in conflict_entries])
            if merged_diag is not None:
                merged[k] = merged_diag
            elif len(value_tokens) > 1:
                conflicts[k] = conflict_entries
            elif conflict_entries:
                merged[k] = conflict_entries[0]["value"]
        elif len(value_tokens) <= 1:
            merged[k] = conflict_entries[0]["value"]
        elif len(value_tokens) > 1:
            conflicts[k] = conflict_entries

    return merged, conflicts, provenance


def compute_exact_hash(image_path: Path) -> str:
    h = sha1()
    with image_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_ahash(image_path: Path, size: int = 16) -> int:
    with Image.open(image_path) as img:
        gray = img.convert("L").resize((size, size))
        px = list(gray.tobytes())
    mean = sum(px) / len(px)
    bits = 0
    for p in px:
        bits = (bits << 1) | (1 if p >= mean else 0)
    return bits


def deduplicate_images(images: List[Path], near_dup_hamming: int = 6) -> Tuple[List[Path], List[Dict[str, Any]]]:
    kept: List[Path] = []
    dropped: List[Dict[str, Any]] = []

    exact_seen: Dict[str, str] = {}
    ahash_seen: List[Tuple[int, str]] = []

    for img in images:
        try:
            exact = compute_exact_hash(img)
            if exact in exact_seen:
                dropped.append({"image": img.name, "reason": "exact_duplicate", "matched_with": exact_seen[exact]})
                continue

            ah = compute_ahash(img)
            near_match = None
            for prev_ah, prev_name in ahash_seen:
                if (ah ^ prev_ah).bit_count() <= near_dup_hamming:
                    near_match = prev_name
                    break
            if near_match:
                dropped.append({"image": img.name, "reason": "near_duplicate", "matched_with": near_match})
                continue

            kept.append(img)
            exact_seen[exact] = img.name
            ahash_seen.append((ah, img.name))
        except Exception as e:
            # If dedup fails for a page, keep it to avoid data loss.
            logger.warning("Dedup hash failed for %s (%s). Keeping page.", img.name, e)
            kept.append(img)

    return kept, dropped


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    n = max(1, int(size))
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def merge_ocr_text_blocks(image_name_text_pairs: List[Tuple[str, str]]) -> str:
    parts: List[str] = []
    for image_name, txt in image_name_text_pairs:
        body = str(txt or "").strip()
        if not body:
            continue
        parts.append(f"[SOURCE_IMAGE: {image_name}]\n{body}")
    return "\n\n".join(parts).strip()


def make_bundle_image_name(bundle_idx: int, image_names: List[str]) -> str:
    lead = image_names[0] if image_names else f"bundle_{bundle_idx:04d}"
    return f"bundle_{bundle_idx:04d}__{Path(lead).stem}.txt"


# -----------------------------
# CDM TF-IDF retriever (local RAG)
# -----------------------------
@dataclass
class CDMRow:
    key: str
    desc: str
    format_range: str
    options: Dict[str, str]  # code -> label
    type_label: str = "B"
    semantic_allowed: int = 1


@dataclass
class PageResult:
    image_name: str
    ocr_text: str
    raw_json: Dict[str, Any]
    valid_json: Dict[str, Any]
    input_contexts: Dict[str, Dict[str, str]]
    cdm_contexts: Dict[str, str]
    rejected_fields: Dict[str, Dict[str, Any]]


@dataclass
class MapAgentSpec:
    name: str
    start_key: str
    end_key: str
    rows: List["CDMRow"]
    candidates_block: str
    route_name: str


class CDMRetriever:
    """
    Local retrieval over CDM rows using TF-IDF char-ngrams (good for Korean/English mix).
    Expects cdm.csv columns:
      - 'csv key'
      - one of:
        - '설명'
        - 'Korean_Context' and/or 'English_Context'
      - 'Format/Range'
      - option columns named like '0','1','2',...
    """
    def __init__(self, cdm_csv_path: Path):
        self.cdm_df = pd.read_csv(cdm_csv_path)
        self.cdm_df = self._expand_cpap_pressure_rows(self.cdm_df)

        option_cols = [c for c in self.cdm_df.columns if re.fullmatch(r"\d+", str(c))]

        self.rows: List[CDMRow] = []
        self._texts: List[str] = []

        for _, r in self.cdm_df.iterrows():
            key = str(r.get("csv key", "")).strip()
            if not key or key.lower() == "nan":
                continue

            desc_candidates = [
                r.get("설명", ""),
                r.get("Korean_Context", ""),
                r.get("English_Context", ""),
            ]
            fr_v = r.get("Format/Range", "")
            desc_parts: List[str] = []
            for dv in desc_candidates:
                if pd.isna(dv):
                    continue
                ds = str(dv).strip()
                if not ds or ds.lower() == "nan":
                    continue
                if ds not in desc_parts:
                    desc_parts.append(ds)
            desc = " | ".join(desc_parts)
            fr = "" if pd.isna(fr_v) else str(fr_v).strip()

            opts: Dict[str, str] = {}
            for c in option_cols:
                val = r.get(c)
                if pd.isna(val):
                    continue
                label = str(val).strip()
                if label:
                    opts[str(c)] = label
            semantic_allowed_raw = r.get("semantic_allowed", 1)
            semantic_allowed = _coerce_int(semantic_allowed_raw)
            if semantic_allowed not in {0, 1}:
                semantic_allowed = 1
            type_label = str(r.get("Type", "B") or "B").strip().upper()
            if type_label not in {"A", "B"}:
                type_label = "B"

            row = CDMRow(
                key=key,
                desc=desc,
                format_range=fr,
                options=opts,
                type_label=type_label,
                semantic_allowed=int(semantic_allowed),
            )
            self.rows.append(row)

            opt_str = " | ".join([f"{code}:{label}" for code, label in sorted(opts.items(), key=lambda x: int(x[0]))])
            self._texts.append(
                f"KEY={key}\nDESC={desc}\nTYPE={type_label}\nSEMANTIC_ALLOWED={int(semantic_allowed)}\nFORMAT={fr}\nOPTIONS={opt_str}"
            )
        logger.info("Loaded full CDM rows: %d", len(self.rows))

        self.key_set = {r.key for r in self.rows}
        self.row_by_key = {r.key: r for r in self.rows}
        self.rows_by_prefix: Dict[str, List[CDMRow]] = defaultdict(list)
        for row in self.rows:
            prefix = row.key.split("_", 1)[0]
            self.rows_by_prefix[prefix].append(row)
        self.label_to_rows = self._build_document_label_rows()

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self._texts)
        self._full_cdm_prompt_block = self._build_full_cdm_prompt_block()
        self._route_prompt_blocks: Dict[Tuple[str, str], str] = {}

    def _expand_cpap_pressure_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        key_col = "csv key"
        if key_col not in df.columns:
            return df
        keys = df[key_col].fillna("").astype(str).str.strip()
        prototype_mask = keys.str.fullmatch(r"Pressure_05|Pr05_.+")
        if not prototype_mask.any():
            return df

        prototype_rows = df[prototype_mask].copy()
        existing_keys = {k for k in keys.tolist() if k}
        generated_rows: List[pd.Series] = []

        for step in range(CPAP_PRESSURE_STEP_START + 1, CPAP_PRESSURE_STEP_END + 1):
            step_text = f"{step:02d}"
            for _, proto in prototype_rows.iterrows():
                proto_key = str(proto.get(key_col, "")).strip()
                if not proto_key:
                    continue
                if proto_key == "Pressure_05":
                    new_key = f"Pressure_{step_text}"
                elif proto_key.startswith("Pr05_"):
                    new_key = f"Pr{step_text}_{proto_key[len('Pr05_'):]}"
                else:
                    continue
                if new_key in existing_keys:
                    continue
                new_row = proto.copy()
                new_row[key_col] = new_key
                generated_rows.append(new_row)
                existing_keys.add(new_key)

        if not generated_rows:
            return df
        extra_df = pd.DataFrame(generated_rows, columns=df.columns)
        return pd.concat([df, extra_df], ignore_index=True)

    def _build_full_cdm_prompt_block(self) -> str:
        parts: List[str] = []
        for row in self.rows:
            opt_items = sorted(row.options.items(), key=lambda x: int(x[0]))
            opt_str = ", ".join([f"{code}={label}" for code, label in opt_items])
            parts.append(
                f"- {row.key}\n"
                f"  desc: {row.desc}\n"
                f"  format/range: {row.format_range}\n"
                f"  options: {opt_str}\n"
            )
        return "\n".join(parts)

    def full_cdm_prompt_block(self) -> str:
        return self._full_cdm_prompt_block

    def route_rows(self, route_name: str, official_questionnaire: Optional[bool] = None) -> List["CDMRow"]:
        route = str(route_name or DEFAULT_MAP_ROUTE).strip() or DEFAULT_MAP_ROUTE
        if route == MAP_ROUTE_PSG_SIGNALS:
            rows: List[CDMRow] = []
            for key in CORE_ALWAYS_KEYS:
                row = self.row_by_key.get(key)
                if row is not None:
                    rows.append(row)
            return rows
        if route in {MAP_ROUTE_PSG_REPORT_GENERAL, MAP_ROUTE_PSG_REPORT_EXTENSIVE, MAP_ROUTE_PSG_REPORT}:
            return [row for row, _ in self.select_candidate_rows_for_labels(["psg_report"])]
        if route in {MAP_ROUTE_CPAP_PSG_REPORT_GENERAL, MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE}:
            return [row for row, _ in self.select_candidate_rows_for_labels(["psg_report", "cpap_pressure"])]
        if route == MAP_ROUTE_MORNING_QUESTIONNAIRE:
            rows = [row for row, _ in self.select_candidate_rows_for_labels(list(MORNING_QUESTIONNAIRE_ROUTE_LABELS))]
            return self._filter_questionnaire_rows_by_type(rows, official_questionnaire)
        if route == MAP_ROUTE_NIGHT_QUESTIONNAIRE:
            rows = [row for row, _ in self.select_candidate_rows_for_labels(list(NIGHT_QUESTIONNAIRE_ROUTE_LABELS))]
            return self._filter_questionnaire_rows_by_type(rows, official_questionnaire)
        return list(self.rows)

    def _filter_questionnaire_rows_by_type(
        self,
        rows: List["CDMRow"],
        official_questionnaire: Optional[bool],
    ) -> List["CDMRow"]:
        if official_questionnaire is None:
            return list(rows)
        want_type = "A" if bool(official_questionnaire) else "B"
        return [row for row in rows if str(getattr(row, "type_label", "B") or "B").upper() == want_type]

    def prompt_block_for_route(self, route_name: str, official_questionnaire: Optional[bool] = None) -> str:
        route = str(route_name or DEFAULT_MAP_ROUTE).strip() or DEFAULT_MAP_ROUTE
        cache_key = (route, "any" if official_questionnaire is None else ("official" if official_questionnaire else "non_official"))
        cached = self._route_prompt_blocks.get(cache_key)
        if cached:
            return cached
        rows = self.route_rows(route, official_questionnaire=official_questionnaire)
        if route in {MAP_ROUTE_CPAP_PSG_REPORT_GENERAL, MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE}:
            block = format_cpap_candidate_rows_compact(rows, max_chars=50000)
        else:
            block = format_candidate_rows([(row, 1.0) for row in rows], include_score=False, max_chars=50000)
        self._route_prompt_blocks[cache_key] = block
        return block

    def _build_document_label_rows(self) -> Dict[str, List["CDMRow"]]:
        out: Dict[str, List[CDMRow]] = {}
        for label, spec in DOCUMENT_LABEL_SPECS.items():
            prefixes = tuple(spec.get("prefixes", ()))
            regexes = [re.compile(x) for x in spec.get("regexes", ())]
            extra_keys = set(spec.get("extra_keys", ()))
            rows: List[CDMRow] = []
            seen = set()
            for row in self.rows:
                matched = False
                prefix = row.key.split("_", 1)[0]
                if prefix in prefixes:
                    matched = True
                if not matched:
                    for rgx in regexes:
                        if rgx.search(row.key):
                            matched = True
                            break
                if not matched and row.key in extra_keys:
                    matched = True
                if not matched or row.key in seen:
                    continue
                seen.add(row.key)
                rows.append(row)
            out[label] = rows
        return out

    def document_label_catalog(self) -> Dict[str, str]:
        return {label: str(spec.get("description", "")).strip() for label, spec in DOCUMENT_LABEL_SPECS.items()}

    def select_candidate_rows_for_labels(self, labels: List[str]) -> List[Tuple["CDMRow", float]]:
        merged: List[Tuple[CDMRow, float]] = []
        seen = set()

        def _add_row(row: "CDMRow") -> None:
            if row.key in seen:
                return
            seen.add(row.key)
            merged.append((row, 1.0))

        for key in CORE_ALWAYS_KEYS:
            row = self.row_by_key.get(key)
            if row is not None:
                _add_row(row)

        for label in labels:
            for row in self.label_to_rows.get(label, []):
                _add_row(row)

        return merged

    def search(self, query: str, k: int = 60) -> List[Tuple[CDMRow, float]]:
        qv = self.vectorizer.transform([query[:8000]])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        return [(self.rows[int(i)], float(sims[int(i)])) for i in idxs]

    def select_candidate_rows(self, ocr_text: str, top_k: int) -> List[Tuple[CDMRow, float]]:
        """
        Hybrid candidate selection:
        - lexical retrieval top-k
        - always-include core identity keys
        - keyword-triggered field families
        - PSG metric bundle when report-like text is detected
        """
        base = self.search(ocr_text, k=top_k)
        text = ocr_text

        forced: List[Tuple[CDMRow, float]] = []
        for key in CORE_ALWAYS_KEYS:
            row = self.row_by_key.get(key)
            if row is not None:
                forced.append((row, 1.0))

        for pattern, prefixes in TRIGGER_PREFIX_RULES:
            if not pattern.search(text):
                continue
            for p in prefixes:
                for row in self.rows_by_prefix.get(p, []):
                    forced.append((row, 1.0))

        if PSG_TRIGGER_RE.search(text):
            for row in self.rows:
                if PSG_METRIC_KEY_RE.search(row.key):
                    forced.append((row, 1.0))

        merged: List[Tuple[CDMRow, float]] = []
        seen = set()
        for row, score in forced + base:
            if row.key in seen:
                continue
            seen.add(row.key)
            merged.append((row, score))
        return merged


def _clip_prompt_text(s: str, max_len: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def format_cpap_candidate_rows_compact(
    rows: List[CDMRow],
    max_chars: int = 50000,
) -> str:
    non_cpap_rows: List[Tuple[CDMRow, float]] = []
    cpap_proto_rows: List[Tuple[CDMRow, float]] = []
    cpap_family_rows: List[CDMRow] = []
    for row in rows:
        if row.key == "Pressure_05" or row.key.startswith("Pr05_"):
            cpap_proto_rows.append((row, 1.0))
        elif re.fullmatch(r"Pressure_\d{2}|Pr\d{2}_.+", row.key):
            cpap_family_rows.append(row)
        else:
            non_cpap_rows.append((row, 1.0))

    explicit_block = format_candidate_rows(
        non_cpap_rows + cpap_proto_rows,
        include_score=False,
        max_chars=max_chars,
    )
    if not cpap_family_rows:
        return explicit_block

    pressure_proto = next((r for r, _ in cpap_proto_rows if r.key == "Pressure_05"), None)
    pr_protos = [r for r, _ in cpap_proto_rows if r.key.startswith("Pr05_")]
    pr_protos.sort(key=lambda r: r.key)

    lines: List[str] = []
    lines.append("CPAP pressure-step key family:")
    lines.append(
        f"- Allowed pressure steps in this summary: {CPAP_PRESSURE_STEP_START + 1:02d} to {CPAP_PRESSURE_STEP_END:02d}"
    )
    lines.append("- Pressure_05 and all Pr05_* keys are already listed explicitly above.")
    lines.append(f"- Apply the same key pattern to every visible pressure step from {CPAP_PRESSURE_STEP_START + 1:02d} through the allowed maximum.")
    lines.append("- If later pressure rows are visible, continue the pattern instead of stopping at early steps.")
    lines.append("- Allowed template keys for each pressure step XX:")
    if pressure_proto is not None:
        lines.append(
            f"  - Pressure_XX | desc={_clip_prompt_text(pressure_proto.desc, 120)} | "
            f"type={getattr(pressure_proto, 'type_label', 'B')} | "
            f"semantic_allowed={int(getattr(pressure_proto, 'semantic_allowed', 1))} | "
            f"format={_clip_prompt_text(pressure_proto.format_range, 60)}"
        )
    for proto in pr_protos:
        suffix = proto.key[len('Pr05_'):]
        opt_items = sorted(proto.options.items(), key=lambda x: int(x[0]))
        opt_str = ", ".join([f"{code}={label}" for code, label in opt_items])
        lines.append(
            f"  - PrXX_{suffix} | desc={_clip_prompt_text(proto.desc, 120)} | "
            f"type={getattr(proto, 'type_label', 'B')} | "
            f"semantic_allowed={int(getattr(proto, 'semantic_allowed', 1))} | "
            f"format={_clip_prompt_text(proto.format_range, 60)} | "
            f"options={_clip_prompt_text(opt_str, 120)}"
        )
    lines.append(
        f"- Example concrete keys in this summary: Pressure_{CPAP_PRESSURE_STEP_START + 1:02d}, Pr{CPAP_PRESSURE_STEP_START + 1:02d}_AHI, "
        f"Pressure_{CPAP_PRESSURE_STEP_START + 2:02d}, Pr{CPAP_PRESSURE_STEP_START + 2:02d}_time_min, ..., "
        f"Pressure_{CPAP_PRESSURE_STEP_END:02d}, Pr{CPAP_PRESSURE_STEP_END:02d}_arousal_spont_idx"
    )
    cpap_block = "\n".join(lines)
    if explicit_block:
        combined = explicit_block.rstrip() + "\n\n" + cpap_block
    else:
        combined = cpap_block
    return combined[:max_chars]


def format_candidate_rows(
    cands: List[Tuple[CDMRow, float]],
    max_chars: int = 26000,
    include_score: bool = True,
    compact: bool = False,
) -> str:
    parts: List[str] = []
    total = 0
    for row, score in cands:
        opt_items = sorted(row.options.items(), key=lambda x: int(x[0]))
        opt_str = ", ".join([f"{code}={label}" for code, label in opt_items])
        type_str = f"type={getattr(row, 'type_label', 'B')}"
        semantic_str = f"semantic_allowed={int(getattr(row, 'semantic_allowed', 1))}"
        if compact:
            block = (
                f"- {row.key} | desc={_clip_prompt_text(row.desc, 80)} | "
                f"{type_str} | "
                f"{semantic_str} | "
                f"format={_clip_prompt_text(row.format_range, 40)} | "
                f"options={_clip_prompt_text(opt_str, 180)}\n"
            )
        else:
            block = (
                f"- {row.key}\n"
                f"  desc: {_clip_prompt_text(row.desc, 220)}\n"
                f"  type: {getattr(row, 'type_label', 'B')}\n"
                f"  semantic_allowed: {int(getattr(row, 'semantic_allowed', 1))}\n"
                f"  format/range: {_clip_prompt_text(row.format_range, 80)}\n"
                f"  options: {_clip_prompt_text(opt_str, 260)}\n"
            )
            if include_score:
                block += f"  (retrieval_score={score:.4f})\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def build_map_agent_specs(
    retriever: CDMRetriever,
    num_agents: int,
    route_name: str = DEFAULT_MAP_ROUTE,
    official_questionnaire: Optional[bool] = None,
) -> List[MapAgentSpec]:
    n_agents = max(1, int(num_agents))
    route_rows = retriever.route_rows(route_name, official_questionnaire=official_questionnaire)
    total = len(route_rows)
    if total == 0:
        return []

    specs: List[MapAgentSpec] = []
    for idx in range(n_agents):
        start = (idx * total) // n_agents
        end = ((idx + 1) * total) // n_agents
        rows = route_rows[start:end]
        if not rows:
            continue
        start_key = rows[0].key
        end_key = rows[-1].key
        name = f"agent_{idx + 1}"
        cands = [(r, 1.0) for r in rows]
        block = format_candidate_rows(cands, include_score=False, max_chars=50000)
        specs.append(
            MapAgentSpec(
                name=name,
                start_key=start_key,
                end_key=end_key,
                rows=rows,
                candidates_block=block,
                route_name=route_name,
            )
        )
    logger.info(
        "Configured %d split map agents over %d CDM rows (about %d rows/agent)",
        len(specs),
        total,
        max(1, total // max(1, len(specs))),
    )
    return specs
# - Convert table into human interpretable texts.
OCR_SYSTEM = """
# Role: You are a literal OCR engine for sleep-clinic questionnaires with Korean/English printed text, handwritings, tables, and marked answers by circles, checks, crosses, and other symbols.
# Task: Perform OCR on all visible content on the scanned page as accurately as possible.
# Guideline
- Preserve original wording, script, numbers, punctuation, units, and visible structure as faithfully as plain text allows.
- Construct a table as much as possible by texts. (Should be interpretable by later LLM agent)
- Multiple choices can be given as numbers encircled, plain numbers, plain texts, square boxes, and empty slots for users to mark answers with circles, checks, crosses, or other symbols.
- Mark a visibly chosen option inline as '[selected]'. Selected option should be clearly expressed without ambiguity. keep question-answer association explicit.
- Use '[corrected from X to Y]' when a correction or overwrite is visible.
- Use '[crossed out]' when text is visibly struck through or crossed out.
- Use '[unclear]' when the content/selection is unclear.
- Use '[not filled/answered]' when a visible blank answer field is clearly intended and relevant.
- Special case: For medical history checklist with '그 외 다음과 같은 질환을 앓고 있거나 과거에 앓은 적이 있습니까', use '[No]' for unmarked conditions and '[Yes]' for marked conditions.
    e.g., 뇌졸중 [No], 파킨슨씨 병 [No], ..., 공황장애 [Yes], 자살시도 [No]
- Special case: if the page primarily contains polysomnography channel signal graphs/tracings, prepend the exact line 'This page contains psg signal graphs.' before the rest of the transcription. This note is allowed even though it is not literal page text.

# Output format
- Only the pure, transcribed text that covers the full page.
"""

OCR_USER_PROMPT = (
    "Transcribe the full page in natural reading order."
)

MAP_SYSTEM = """
# Role: You are a clinical data mapping and parsing expert.
# Task: Map OCR text from sleep-clinic reports and questionnaires to exact CDM (common data model) keys exactly. Parse values to correct keys and create the final JSON.

# Guideline
1. INPUT
You will get two inputs:
    - OCR text from a sleep questionnaire image
    - Candidate CDM fields (keys) with descriptions/ranges/options
2. CDM KEY
    - Read the provided OCR text line by line.
    - Based on the candidate cdm keys provided and the OCR text, find the EXACT CDM key that the Korean_Context/English_Context has the exactly same wordings or meaning when comparing to the OCR text.
        * Note: For official questionnaire cases, being semantically same is NOT sufficient. Each wording should be the same for these cases.
            e.g., PSQI, ESS, SSS, FSS, BQ, ISI, RLS, IRLS, RBDSQ, PHQ, BDI, QOL, MQ
        * Note: Each candidate includes `semantic_allowed`.
            - If `semantic_allowed = 0`, direct wording equivalence or direct official-item identity is required. Do not map from merely similar wording or broad summaries.
            - If `semantic_allowed = 1`, semantic matching is allowed, but only when the OCR evidence directly supports the same clinical meaning.
            - Use this field conservatively to avoid over-mapping.
3. CDM VALUE
    - For the correct key, fill a value following the candidate field format/range/options exactly.
        - e.g., answer can be scaled as yes: 1, no:2, but cdm might require no:0, yes:1
    - For PSG tables, read row and column intersections explicitly. Do not guess from nearby values. 
    - Do NOT invent values if not allowed by 'Special Rules'.
    - Special Rules for filling CDM values
        - For map_route_psg_report_general and map_route_psg_report_extensive
            - Always attempt to emit Hospital_ID, Name, PSG_Date, PSG_No, PSG_Type, SEX, AGE, Height_cm, Weight_kg, BMI, Neckcir_cm when directly supported by the OCR text. Missing these fields is a serious error.
        - Name
            - If the patient name is written in Korean, output the Korean name in Korean.
            - If the patient name is written in English, output the English name in English.
            - Do not romanize Korean names, and do not translate English names into Korean.
        - Numeric ranges (i.e. 'a~b')
            - If time values measured in time units (e.g., hrs, min, sec, day), store the median.
            - If severity/frequency reported, store the more severe one. (e.g., waking frequency 3 times > 1 time)
        - Time values in HH:MM (Exceptional case that allows value invention.)
            - If a time answer is shown only in minutes, emit HH=0 and MM=<minutes>. If shown only in hours, emit HH=<hours> and MM=0.
        - Occupation categorization
            - Normalize occupation to Korean wording when possible.
            - if CDM options exist for the occupation, map to the correct option code.
            - if OCR answer indicates job-seeking/leave (e.g., 취준, 취업준비, 휴직), omit Occupation.
        - PHx, Medical History Sections
            - PHx_* keys must come only from the explicit medical-history checklist/list on the page.
            - Do not infer PHx_* from diagnosis summary, medication list, free-text complaint, or family history.
            - If a PHx condition is visibly listed and marked/checked, output 1.
            - If a PHx condition is visibly listed and unmarked/not checked, output 0.
            - If the condition is not visible on the page, omit the key.
        - Diagnosis_etc:
            - 1st Position: `II. Diagnosis` section after `I. Result`.
            - 2nd Position: Before `III. Conclusion and Recommendation`.
            - Extract only the lines that begin with # from the source text. If multiple lines match, preserve their order and join them with newline characters.        
        - PSQI 01-04 version rule:
            - Find the OFFICIAL PSQI questions, not similar questions.
            - If OCR text explicitly contains even single `주중`/`주말` (or weekday/weekend wording), map to ONLY `_week` / `_free` keys for PSQI 01-04.
            - If OCR text does not contain `주중`/`주말` (or weekday/weekend wording), map to ONLY non-week/free keys (`..._HH`, `..._MM`) for PSQI 01-04.
4. CONTEXT
    - Context is used by later resolver agent to resolve between multiple values of the same CDM key.
    - 'filled_by': doctor for diagnosis and diagnostic report, patient when self-reported questionnaire.
    - 'question': The one sentence - exact question/context that matches to the CDM key in OCR text.

# Caution
- OCR text -> CDM Keys
    - Write all cdm keys applicable to the OCR text. 
    - Be careful not to connect wrong cdm key to irrelevant text.
- CDM Keys -> CDM Values
    - Review your key-value decision carefully to reduce mistakes.
- Before finalizing, check whether any obvious directly supported candidate fields from the page header, questionnaire item list, or PSG tables were omitted.

# Output format
Output JSON object only. Return ONE JSON object that maps CDM keys to objects with this schema:
{
  "CDM_KEY": {
    "CDM_Context": "<brief explanation copied from the Korean_Context/English_Context>",
    "value": <value>,
    "input_context": {
      "filled_by": "doctor|patient",
      "question": "<one sentence - exact question/context that matches to the CDM key in OCR text>"
    }
  }
}
"""

MAP_RECALL_SYSTEM = """You are a clinical data extraction assistant performing a second-pass recall step.
You will be given:
1) OCR text
2) Candidate CDM fields
3) Existing extracted JSON

Task:
- First, internally categorize the OCR text into one or more relevant document/questionnaire types.
- Use that internal categorization to decide which candidate CDM keys are relevant to the OCR text.
- Do not output the categorization itself.
- Return ONLY additional key-value pairs that are clearly supported by OCR text.
- Do NOT repeat keys already present in existing JSON.
- Use ONLY keys from the candidate list.
- Always follow candidate field format/range/options exactly.
- Do not translate or transliterate person names; preserve OCR script.
- Occupation translation/categorization must be done in this step:
  - normalize occupation to Korean wording when possible.
  - if CDM options exist for Occupation, map to the correct option code.
  - if OCR indicates 취준/취업준비/휴직, omit Occupation.
- For numeric ranges like `a~b`: time-like => median; severity/frequency scale-like => more severe side.
- PSQI 01-04 version rule:
  - If OCR text explicitly contains `주중`/`주말` (or weekday/weekend wording), extract ONLY `_week` / `_free` keys for PSQI 01-04.
  - If OCR text does NOT contain those cues, extract ONLY non-week/free keys (`..._HH`, `..._MM`) for PSQI 01-04.
- Diagnosis_etc must come only from Polysomnography Data `II. Diagnosis` lines starting with `#`.
- Follow coded options/range/date rules exactly.
- If uncertain, omit the key.
- Use the same output schema as MAP step:
  {"CDM_KEY":{"CDM_Context":"...","value":..., "input_context":{"filled_by":"doctor|patient","question":"..."}}}
Output JSON object only.
"""

CONFLICT_RESOLVER_SYSTEM = """
You are resolving only the remaining ambiguous CDM conflicts after deterministic code resolution.
Your job is NOT to extract new values.
Your job is NOT to re-run majority voting.
Your job is to choose the best candidate only from the provided candidates for each CDM key.

# Input per candidate
- value
- CDM_Context
- input_context.question
- page_type
- CDM metadata (description, format/range, options)

# Critical decision rules
1) Remove obvious context mismatches first.
- Candidate question/page_type must match the CDM key semantics.
2) Enforce CDM validity.
- Prefer candidates consistent with options, format/range, and date constraints.
3) Tie-break with source clarity.
- Prefer candidates with clearer, more specific question evidence.
4) Keep reason short and concrete.

# Special case
- Diagnosis_etc: if multiple candidates are valid diagnosis statements, merge them.

# Output Format
Output JSON only and STRICTLY follow the instructed JSON format:
{
  "resolved": {
    "CDM_KEY": {"chosen_index": <int>, "reason": "<brief reason>"}
  }
}
"""


# -----------------------------
# Gemini model builder
# -----------------------------
def build_gemini() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")  # set to your real Gemini model name
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        max_output_tokens=4096,
    )


# -----------------------------
# Two-stage pipeline (Gemini OCR -> local CDM retrieval -> Gemini JSON)
# -----------------------------
def is_transient_llm_error(exc: Exception) -> bool:
    s = str(exc).upper()
    return ("503" in s) or ("UNAVAILABLE" in s) or ("429" in s) or ("RESOURCE_EXHAUSTED" in s)


async def ainvoke_with_retry(
    llm: ChatGoogleGenerativeAI,
    messages: List[Any],
    max_retries: int = 5,
    base_delay: float = 1.5,
    max_delay: float = 20.0,
):
    for attempt in range(max_retries + 1):
        try:
            await REQUEST_THROTTLE.wait_turn()
            return await llm.ainvoke(messages)
        except Exception as e:
            if attempt >= max_retries or not is_transient_llm_error(e):
                raise
            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0.0, 0.5)
            logger.warning(
                "Transient Gemini error (%s). Retrying in %.1fs (%d/%d).",
                e,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)


async def gemini_ocr(llm: ChatGoogleGenerativeAI, image_path: Path) -> str:
    data_url = image_to_data_url(image_path)
    msg = [
        SystemMessage(content=OCR_SYSTEM),
        HumanMessage(
            content=[
                {"type": "text", "text": OCR_USER_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]
    resp = await ainvoke_with_retry(llm, msg)
    return llm_content_to_text(resp.content)


async def gemini_map_to_json(
    llm: ChatGoogleGenerativeAI,
    ocr_text: str,
    candidates_block: str,
    route_name: str = DEFAULT_MAP_ROUTE,
    official_questionnaire: bool = False,
    official_family: str = "NON",
) -> Dict[str, Any]:
    user = build_map_user_prompt(
        ocr_text,
        candidates_block,
        route_name=route_name,
        official_questionnaire=official_questionnaire,
        official_family=official_family,
    )
    msg = [
        SystemMessage(content=MAP_SYSTEM),
        HumanMessage(content=user),
    ]
    resp = await ainvoke_with_retry(llm, msg)
    raw = llm_content_to_text(resp.content)

    try:
        return safe_extract_json(raw)
    except Exception:
        # One repair attempt
        fix = await ainvoke_with_retry(
            llm,
            [
                SystemMessage(content="Fix into a valid JSON object only. No explanation."),
                HumanMessage(content=raw),
            ],
        )
        return safe_extract_json(fix.content)


async def gemini_map_additional_json(
    llm: ChatGoogleGenerativeAI,
    ocr_text: str,
    candidates_block: str,
    existing_json: Dict[str, Any],
    route_name: str = DEFAULT_MAP_ROUTE,
    official_questionnaire: bool = False,
    official_family: str = "NON",
) -> Dict[str, Any]:
    user = build_map_recall_user_prompt(
        ocr_text,
        candidates_block,
        existing_json,
        route_name=route_name,
        official_questionnaire=official_questionnaire,
        official_family=official_family,
    )
    msg = [
        SystemMessage(content=MAP_RECALL_SYSTEM),
        HumanMessage(content=user),
    ]
    resp = await ainvoke_with_retry(llm, msg)
    raw = llm_content_to_text(resp.content)

    try:
        return safe_extract_json(raw)
    except Exception:
        fix = await ainvoke_with_retry(
            llm,
            [
                SystemMessage(content="Fix into a valid JSON object only. No explanation."),
                HumanMessage(content=raw),
            ],
        )
        return safe_extract_json(fix.content)


async def gemini_route_ocr_text(
    llm: ChatGoogleGenerativeAI,
    ocr_text: str,
) -> Dict[str, Any]:
    user = build_route_user_prompt(ocr_text)
    msg = [
        SystemMessage(content=MAP_ROUTE_SYSTEM),
        HumanMessage(content=user),
    ]
    try:
        resp = await ainvoke_with_retry(llm, msg)
        raw = llm_content_to_text(resp.content)
        return normalize_route_decision(safe_extract_json(raw), ocr_text)
    except Exception as exc:
        fallback = classify_map_route_heuristic(ocr_text)
        fallback["reason"] = f"heuristic_fallback_after_route_error:{type(exc).__name__}"
        logger.warning("Route classifier failed, falling back to heuristic router: %s", exc)
        return fallback


def merge_map_payload_into_stage(
    retriever: CDMRetriever,
    ocr_text: str,
    raw_payload: Dict[str, Any],
    route_name: str,
    stage_raw: Dict[str, Any],
    stage_valid: Dict[str, Any],
    stage_contexts: Dict[str, Dict[str, str]],
    stage_cdm_contexts: Dict[str, str],
    stage_rejected: Dict[str, Dict[str, Any]],
    official_questionnaire: bool = False,
    official_family: str = "NON",
) -> None:
    raw_values, raw_contexts, raw_cdm_contexts = parse_value_context_map(raw_payload)
    add_valid, add_rejected = validate_extracted_json(raw_values, retriever, ocr_text=ocr_text)
    if route_name in CPAP_ROUTE_NAMES:
        _apply_cpap_page_guardrails(add_valid, raw_contexts, add_rejected)

    for k, v in raw_payload.items():
        stage_raw.setdefault(k, v)
    for k, meta in add_rejected.items():
        stage_rejected.setdefault(k, meta)

    for k, v in add_valid.items():
        row = retriever.row_by_key.get(k)
        ctx = _normalize_input_context(raw_contexts.get(k))
        if row is not None and route_name in {MAP_ROUTE_NIGHT_QUESTIONNAIRE, MAP_ROUTE_MORNING_QUESTIONNAIRE}:
            row_type = str(getattr(row, "type_label", "B") or "B").upper()
            if row_type == "A" and not official_questionnaire:
                stage_rejected.setdefault(
                    k,
                    {
                        "value": v,
                        "reason": "type_a_requires_official_questionnaire_page_tag",
                        "input_context": ctx,
                    },
                )
                continue
            if row_type == "B" and official_questionnaire:
                stage_rejected.setdefault(
                    k,
                    {
                        "value": v,
                        "reason": "type_b_requires_non_official_questionnaire_page_tag",
                        "input_context": ctx,
                    },
                )
                continue
        if k in stage_valid:
            if _norm_cmp(stage_valid.get(k)) != _norm_cmp(v):
                logger.debug(
                    "Split-map overlap conflict on key=%s. Keeping first value=%r, dropping=%r",
                    k,
                    stage_valid.get(k),
                    v,
                )
            continue
        stage_valid[k] = v
        ctx["page_type"] = normalize_map_route_name(ctx.get("page_type") or route_name)
        stage_contexts[k] = ctx
        stage_cdm_contexts[k] = str(raw_cdm_contexts.get(k) or (row.desc if row is not None else "")).strip()


async def map_ocr_text_with_split_agents_live(
    llm: ChatGoogleGenerativeAI,
    retriever: CDMRetriever,
    ocr_text: str,
    map_agents: List[MapAgentSpec],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    stage_raw: Dict[str, Any] = {}
    stage_valid: Dict[str, Any] = {}
    stage_contexts: Dict[str, Dict[str, str]] = {}
    stage_cdm_contexts: Dict[str, str] = {}
    stage_rejected: Dict[str, Dict[str, Any]] = {}
    route_info = await gemini_route_ocr_text(llm, ocr_text)
    route_name = str(route_info.get("route") or DEFAULT_MAP_ROUTE)

    async def _call(agent: MapAgentSpec):
        payload = await gemini_map_to_json(
            llm=llm,
            ocr_text=ocr_text,
            candidates_block=agent.candidates_block,
            route_name=agent.route_name,
        )
        return agent, payload

    if not map_agents:
        # Fallback to full-CDM single-agent.
        raw = await gemini_map_to_json(
            llm=llm,
            ocr_text=ocr_text,
            candidates_block=retriever.prompt_block_for_route(route_name),
            route_name=route_name,
        )
        merge_map_payload_into_stage(
            retriever=retriever,
            ocr_text=ocr_text,
            raw_payload=raw,
            route_name=route_name,
            stage_raw=stage_raw,
            stage_valid=stage_valid,
            stage_contexts=stage_contexts,
            stage_cdm_contexts=stage_cdm_contexts,
            stage_rejected=stage_rejected,
        )
    else:
        outs = await asyncio.gather(*[_call(agent) for agent in map_agents], return_exceptions=True)
        for out in outs:
            if isinstance(out, Exception):
                logger.warning("Split map agent call failed: %s", out)
                continue
            _, payload = out
            merge_map_payload_into_stage(
                retriever=retriever,
                ocr_text=ocr_text,
                raw_payload=payload,
                route_name=route_name,
                stage_raw=stage_raw,
                stage_valid=stage_valid,
                stage_contexts=stage_contexts,
                stage_cdm_contexts=stage_cdm_contexts,
                stage_rejected=stage_rejected,
            )

    # Optional recall on the same split-agent setup.
    if should_run_recall_pass(ocr_text, stage_valid):
        async def _recall(agent: MapAgentSpec):
            payload = await gemini_map_additional_json(
                llm=llm,
                ocr_text=ocr_text,
                candidates_block=agent.candidates_block,
                existing_json=stage_valid,
                route_name=agent.route_name,
            )
            return agent, payload

        if map_agents:
            recall_outs = await asyncio.gather(*[_recall(agent) for agent in map_agents], return_exceptions=True)
            for out in recall_outs:
                if isinstance(out, Exception):
                    logger.warning("Split map recall agent call failed: %s", out)
                    continue
                _, payload = out
                merge_map_payload_into_stage(
                    retriever=retriever,
                    ocr_text=ocr_text,
                    raw_payload=payload,
                    route_name=route_name,
                    stage_raw=stage_raw,
                    stage_valid=stage_valid,
                    stage_contexts=stage_contexts,
                    stage_cdm_contexts=stage_cdm_contexts,
                    stage_rejected=stage_rejected,
                )
        else:
            recall_raw = await gemini_map_additional_json(
                llm=llm,
                ocr_text=ocr_text,
                candidates_block=retriever.prompt_block_for_route(route_name),
                existing_json=stage_valid,
                route_name=route_name,
            )
            merge_map_payload_into_stage(
                retriever=retriever,
                ocr_text=ocr_text,
                raw_payload=recall_raw,
                route_name=route_name,
                stage_raw=stage_raw,
                stage_valid=stage_valid,
                stage_contexts=stage_contexts,
                stage_cdm_contexts=stage_cdm_contexts,
                stage_rejected=stage_rejected,
            )

    backfill_additions, backfill_rejected = apply_core_backfill(stage_valid, retriever, ocr_text)
    for k, v in backfill_additions.items():
        stage_valid[k] = v
        stage_contexts.setdefault(k, {"filled_by": "", "question": "Derived from OCR header pattern", "page_type": route_name})
        stage_cdm_contexts.setdefault(k, str(retriever.row_by_key.get(k).desc if retriever.row_by_key.get(k) is not None else "").strip())
        stage_raw.setdefault(
            k,
            {
                "CDM_Context": stage_cdm_contexts.get(k, ""),
                "value": v,
                "input_context": stage_contexts[k],
            },
        )
    for k, meta in backfill_rejected.items():
        stage_rejected.setdefault(k, meta)

    return stage_raw, stage_valid, stage_contexts, stage_cdm_contexts, stage_rejected


def build_conflict_resolver_user_prompt(
    patient_name: str,
    retriever: "CDMRetriever",
    conflicts: Dict[str, List[Dict[str, Any]]],
) -> str:
    payload: List[Dict[str, Any]] = []
    for key, entries in conflicts.items():
        row = retriever.row_by_key.get(key)
        opt_items: List[Tuple[str, str]] = []
        if row is not None:
            opt_items = sorted(row.options.items(), key=lambda x: int(x[0]))
        candidates: List[Dict[str, Any]] = []
        for idx, e in enumerate(entries):
            ctx = _normalize_input_context(e.get("input_context"))
            page_type = normalize_map_route_name(ctx.get("page_type"))
            candidates.append(
                {
                    "index": idx,
                    "cdm_context": str(e.get("cdm_context") or (row.desc if row is not None else "")).strip(),
                    "value": e.get("value"),
                    "question": _clip_prompt_text(ctx.get("question", ""), 260),
                    "page_type": page_type,
                }
            )

        payload.append(
            {
                "key": key,
                "cdm_context": row.desc if row is not None else "",
                "format_range": row.format_range if row is not None else "",
                "options": {k: v for k, v in opt_items},
                "candidates": candidates,
            }
        )

    return (
        f"PATIENT: {patient_name}\n\n"
        "CONFLICT CANDIDATES JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return JSON only in the required schema."
    )


def _dedupe_question_list(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        q = str(v or "").strip()
        if not q:
            continue
        token = _normalize_text_token(q)
        if token in seen:
            continue
        seen.add(token)
        out.append(q)
    return out


def build_conflict_count_dataframe(
    conflicts: Dict[str, List[Dict[str, Any]]],
) -> pd.DataFrame:
    """
    Build a single conflict vote table:
      CDM_KEY, value, count, input_context.question(list)
    Group identity:
      (CDM_KEY, normalized value token)
    """
    rows: List[Dict[str, Any]] = []
    for key, entries in conflicts.items():
        for e in entries:
            ctx = _normalize_input_context(e.get("input_context"))
            norm_value = normalize_value(e.get("value"))
            rows.append(
                {
                    "CDM_KEY": key,
                    "value": norm_value,
                    "value_token": _value_token(norm_value),
                    "input_context.question": str(ctx.get("question") or "").strip(),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "CDM_KEY",
                "value",
                "count",
                "input_context.question",
            ]
        )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["CDM_KEY", "value_token"], dropna=False, as_index=False)
        .agg(
            value=("value", "first"),
            count=("value_token", "size"),
            **{"input_context.question": ("input_context.question", _dedupe_question_list)},
        )
        .reset_index(drop=True)
    )
    grouped["count"] = grouped["count"].astype(int)
    grouped = grouped.sort_values(
        ["CDM_KEY", "count"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)
    return grouped[
        [
            "CDM_KEY",
            "value",
            "count",
            "input_context.question",
        ]
    ]


def _token_count_map(df_key: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if df_key.empty:
        return out
    for _, row in df_key.iterrows():
        token = str(row.get("value_token", ""))
        if not token:
            continue
        out[token] = out.get(token, 0) + int(row.get("count", 0) or 0)
    return out


def _unique_argmax_token(counts: Dict[str, int]) -> Optional[str]:
    if not counts:
        return None
    top = max(counts.values())
    winners = [tok for tok, c in counts.items() if c == top]
    if len(winners) != 1:
        return None
    return winners[0]


def _pick_entry_index_by_token(
    entries: List[Dict[str, Any]],
    chosen_token: str,
) -> Optional[int]:
    for idx, e in enumerate(entries):
        if _value_token(normalize_value(e.get("value"))) == chosen_token:
            return idx
    return None


def resolve_conflicts_by_majority_vote(
    conflicts: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[Dict[str, Any]]], pd.DataFrame]:
    """
    Deterministic pre-resolver:
    1) Aggregate conflicts into a single count dataframe.
    2) Resolve by plain majority vote across normalized values.
    3) Keep unresolved ties for LLM fallback.
    """
    vote_df = build_conflict_count_dataframe(conflicts)
    vote_df_internal = vote_df.copy()
    vote_df_internal["value_token"] = vote_df_internal["value"].map(_value_token)
    overrides: Dict[str, Any] = {}
    decisions: Dict[str, Any] = {}
    pending: Dict[str, List[Dict[str, Any]]] = {}

    for key, entries in conflicts.items():
        key_df = vote_df_internal[vote_df_internal["CDM_KEY"] == key]
        if key_df.empty:
            pending[key] = entries
            continue

        majority_df = key_df
        majority_entries = entries
        unique_tokens = majority_df["value_token"].dropna().astype(str).unique().tolist()
        chosen_token: Optional[str] = None
        rule_used = ""

        if len(unique_tokens) == 1:
            chosen_token = unique_tokens[0]
            rule_used = "single_unique_value"
        else:
            counts = _token_count_map(majority_df)
            chosen_token = _unique_argmax_token(counts)
            rule_used = "overall_majority"

        if not chosen_token:
            pending[key] = majority_entries
            continue

        idx = _pick_entry_index_by_token(
            entries=entries,
            chosen_token=chosen_token,
        )
        if idx is None:
            pending[key] = majority_entries
            continue

        chosen = entries[idx]
        vote_rows: List[Dict[str, Any]] = []
        for _, row in key_df.iterrows():
            questions_raw = row.get("input_context.question", [])
            question_list: List[str] = []
            if isinstance(questions_raw, list):
                question_list = [str(q).strip() for q in questions_raw if str(q or "").strip()]
            elif isinstance(questions_raw, tuple):
                question_list = [str(q).strip() for q in list(questions_raw) if str(q or "").strip()]
            try:
                vote_value = json.loads(str(row.get("value_token", "null")))
            except Exception:
                vote_value = normalize_value(row.get("value"))
            vote_rows.append(
                {
                    "value": vote_value,
                    "count": int(row.get("count", 0) or 0),
                    "question": question_list,
                }
            )

        overrides[key] = chosen.get("value")
        decisions[key] = {
            "chosen_index": idx,
            "chosen_value": chosen.get("value"),
            "reason": f"code_majority:{rule_used}",
            "source_image": chosen.get("image"),
            "input_context": _normalize_input_context(chosen.get("input_context")),
            "resolver_mode": "code_majority",
            "vote_table": vote_rows,
        }

    return overrides, decisions, pending, vote_df


async def parse_json_with_feedback_repair(
    llm: ChatGoogleGenerativeAI,
    raw_content: Any,
    schema_hint: str,
    failed_context: str,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    raw_text = llm_content_to_text(raw_content)
    try:
        return safe_extract_json(raw_text)
    except Exception as e:
        last_error = e

    prior_output = raw_text
    clipped_ctx = failed_context if len(failed_context) <= 14000 else failed_context[:14000] + "\n...(truncated)"
    for _ in range(max_attempts):
        fix_prompt = (
            "Your previous output was not valid JSON.\n"
            "Return ONLY valid JSON. No markdown, no prose.\n\n"
            f"Required schema:\n{schema_hint}\n\n"
            f"Context:\n{clipped_ctx}\n\n"
            f"Previous invalid output:\n{prior_output}"
        )
        fix = await ainvoke_with_retry(
            llm,
            [
                SystemMessage(content="You are a strict JSON repair assistant."),
                HumanMessage(content=fix_prompt),
            ],
            max_retries=3,
        )
        prior_output = llm_content_to_text(fix.content)
        try:
            return safe_extract_json(prior_output)
        except Exception as e:
            last_error = e

    raise ValueError(f"Could not parse JSON after repair attempts: {last_error}")


def build_single_conflict_payload(
    retriever: "CDMRetriever",
    key: str,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    row = retriever.row_by_key.get(key)
    opt_items: List[Tuple[str, str]] = []
    if row is not None:
        opt_items = sorted(row.options.items(), key=lambda x: int(x[0]))
    candidates: List[Dict[str, Any]] = []
    for idx, e in enumerate(entries):
        ctx = _normalize_input_context(e.get("input_context"))
        page_type = normalize_map_route_name(ctx.get("page_type"))
        candidates.append(
            {
                "index": idx,
                "cdm_context": str(e.get("cdm_context") or (row.desc if row is not None else "")).strip(),
                "value": e.get("value"),
                "question": _clip_prompt_text(ctx.get("question", ""), 260),
                "page_type": page_type,
            }
        )
    return {
        "key": key,
        "cdm_context": row.desc if row is not None else "",
        "format_range": row.format_range if row is not None else "",
        "options": {k: v for k, v in opt_items},
        "candidates": candidates,
    }

async def resolve_single_conflict_with_llm(
    llm: ChatGoogleGenerativeAI,
    patient_name: str,
    retriever: "CDMRetriever",
    key: str,
    entries: List[Dict[str, Any]],
) -> Optional[Tuple[int, str]]:
    payload = build_single_conflict_payload(retriever=retriever, key=key, entries=entries)
    user = (
        f"PATIENT: {patient_name}\n"
        "Resolve one conflict candidate set.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        'Return JSON only: {"chosen_index": <int>, "reason": "<brief reason>"}'
    )
    resp = await ainvoke_with_retry(
        llm,
        [
            SystemMessage(content=CONFLICT_RESOLVER_SYSTEM),
            HumanMessage(content=user),
        ],
    )
    raw = await parse_json_with_feedback_repair(
        llm=llm,
        raw_content=resp.content,
        schema_hint='{"chosen_index": <int>, "reason": "<brief reason>"}',
        failed_context=f"patient={patient_name}\nkey={key}\npayload={json.dumps(payload, ensure_ascii=False)}",
        max_attempts=2,
    )
    idx = _coerce_int(raw.get("chosen_index"))
    reason = str(raw.get("reason", "")).strip()
    if idx is None and isinstance(raw.get(key), dict):
        nested = raw.get(key) or {}
        idx = _coerce_int(nested.get("chosen_index"))
        if not reason:
            reason = str(nested.get("reason", "")).strip()
    if idx is None or idx < 0 or idx >= len(entries):
        return None
    return idx, reason


async def resolve_conflicts_keywise_fallback(
    llm: ChatGoogleGenerativeAI,
    retriever: "CDMRetriever",
    patient_name: str,
    conflicts: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    overrides: Dict[str, Any] = {}
    decisions: Dict[str, Any] = {}
    for key, entries in conflicts.items():
        try:
            one = await resolve_single_conflict_with_llm(
                llm=llm,
                patient_name=patient_name,
                retriever=retriever,
                key=key,
                entries=entries,
            )
        except Exception as e:
            logger.warning("Per-key conflict resolver failed for %s/%s: %s", patient_name, key, e)
            continue
        if one is None:
            continue
        idx, reason = one
        chosen = entries[idx]
        overrides[key] = chosen.get("value")
        decisions[key] = {
            "chosen_index": idx,
            "chosen_value": chosen.get("value"),
            "reason": reason,
            "source_image": chosen.get("image"),
            "input_context": _normalize_input_context(chosen.get("input_context")),
            "resolver_mode": "llm_single",
        }
    return overrides, decisions


async def resolve_conflicts_with_llm(
    llm: ChatGoogleGenerativeAI,
    retriever: "CDMRetriever",
    patient_name: str,
    conflicts: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not conflicts:
        return {}, {}

    overrides: Dict[str, Any] = {}
    decisions: Dict[str, Any] = {}
    pending_conflicts: Dict[str, List[Dict[str, Any]]] = conflicts
    try:
        code_overrides, code_decisions, pending_conflicts, _ = resolve_conflicts_by_majority_vote(conflicts)
        overrides.update(code_overrides)
        decisions.update(code_decisions)
    except Exception as e:
        logger.warning("Code majority conflict pre-resolver failed for %s: %s", patient_name, e)
        pending_conflicts = conflicts

    if not pending_conflicts:
        return overrides, decisions

    user = build_conflict_resolver_user_prompt(
        patient_name=patient_name,
        retriever=retriever,
        conflicts=pending_conflicts,
    )
    resp = await ainvoke_with_retry(
        llm,
        [
            SystemMessage(content=CONFLICT_RESOLVER_SYSTEM),
            HumanMessage(content=user),
        ],
    )
    try:
        raw = await parse_json_with_feedback_repair(
            llm=llm,
            raw_content=resp.content,
            schema_hint='{"resolved":{"CDM_KEY":{"chosen_index": <int>, "reason": "<brief reason>"}}}',
            failed_context=f"patient={patient_name}\nkeys={json.dumps(list(pending_conflicts.keys()), ensure_ascii=False)}\nrequest={user}",
            max_attempts=2,
        )
    except Exception as e:
        logger.warning(
            "Conflict resolver JSON parse failed for %s (%s). Falling back to per-key resolver.",
            patient_name,
            e,
        )
        fb_overrides, fb_decisions = await resolve_conflicts_keywise_fallback(
            llm=llm,
            retriever=retriever,
            patient_name=patient_name,
            conflicts=pending_conflicts,
        )
        overrides.update(fb_overrides)
        decisions.update(fb_decisions)
        return overrides, decisions
    resolved_obj = raw.get("resolved", raw)
    if not isinstance(resolved_obj, dict):
        logger.warning(
            "Conflict resolver returned non-dict payload for %s. Falling back to per-key resolver.",
            patient_name,
        )
        fb_overrides, fb_decisions = await resolve_conflicts_keywise_fallback(
            llm=llm,
            retriever=retriever,
            patient_name=patient_name,
            conflicts=pending_conflicts,
        )
        overrides.update(fb_overrides)
        decisions.update(fb_decisions)
        return overrides, decisions

    for key, entries in pending_conflicts.items():
        item = resolved_obj.get(key)
        if not isinstance(item, dict):
            continue
        idx = _coerce_int(item.get("chosen_index"))
        if idx is None:
            continue
        if idx < 0 or idx >= len(entries):
            continue
        chosen = entries[idx]
        overrides[key] = chosen.get("value")
        decisions[key] = {
            "chosen_index": idx,
            "chosen_value": chosen.get("value"),
            "reason": str(item.get("reason", "")).strip(),
            "source_image": chosen.get("image"),
            "input_context": _normalize_input_context(chosen.get("input_context")),
            "resolver_mode": "llm_batch",
        }
    still_pending = {key: entries for key, entries in pending_conflicts.items() if key not in overrides}
    if still_pending:
        fallback_overrides, fallback_decisions = await resolve_conflicts_keywise_fallback(
            llm=llm,
            retriever=retriever,
            patient_name=patient_name,
            conflicts=still_pending,
        )
        overrides.update(fallback_overrides)
        decisions.update(fallback_decisions)
    return overrides, decisions


def build_map_user_prompt(
    ocr_text: str,
    candidates_block: str,
    route_name: str = DEFAULT_MAP_ROUTE,
    official_questionnaire: bool = False,
    official_family: str = "NON",
) -> str:
    page_tag = "official_questionnaire" if official_questionnaire else "non_official_questionnaire"
    return f"""OCR TEXT:
\"\"\"{ocr_text[:12000]}\"\"\"

PAGE QUESTIONNAIRE TAG:
- tag: {page_tag}
- family: {official_family}

CANDIDATE CDM FIELDS (use ONLY these keys):
{candidates_block}

Map only keys that are directly supported by the OCR text and fit the candidate CDM fields shown below.
Extract as many clearly supported keys as possible.
Return ONE JSON object only.

Output schema reminder:
{{
  "CDM_KEY": {{
    "CDM_Context": "<brief explanation copied from the Korean_Context/English_Context>",
    "value": <value>,
    "input_context": {{
      "filled_by": "doctor|patient",
      "question": "<one sentence - exact question/context that matches to the CDM key in OCR text>"
    }}
  }}
}}"""


def build_map_recall_user_prompt(
    ocr_text: str,
    candidates_block: str,
    existing_json: Dict[str, Any],
    route_name: str = DEFAULT_MAP_ROUTE,
    official_questionnaire: bool = False,
    official_family: str = "NON",
) -> str:
    page_tag = "official_questionnaire" if official_questionnaire else "non_official_questionnaire"
    return f"""OCR TEXT:
\"\"\"{ocr_text[:12000]}\"\"\"

PAGE QUESTIONNAIRE TAG:
- tag: {page_tag}
- family: {official_family}

EXISTING JSON (do not repeat these keys):
{json.dumps(existing_json, ensure_ascii=False)}

CANDIDATE CDM FIELDS (use ONLY these keys):
{candidates_block}

Add only additional keys that are directly supported by the OCR text and fit the candidate CDM fields shown below.
Return ONLY additional key-value pairs as ONE JSON object.

Output schema reminder:
{{
  "CDM_KEY": {{
    "CDM_Context": "<brief explanation copied from the Korean_Context/English_Context>",
    "value": <value>,
    "input_context": {{
      "filled_by": "doctor|patient",
      "question": "<one sentence - exact question/context that matches to the CDM key in OCR text>"
    }}
  }}
}}"""


def create_genai_batch_client():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")
    try:
        from google import genai  # type: ignore
    except Exception as e:
        raise RuntimeError("Batch mode requires `google-genai`. Install with: pip install -U google-genai") from e
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_batch_state_name(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    if state is None:
        return "UNKNOWN"
    name = getattr(state, "name", None)
    if name:
        return str(name)
    return str(state)


def build_ocr_batch_request(image_path: Path, batch_image_max_side: int) -> Dict[str, Any]:
    data_url = image_to_data_url(image_path, max_side=batch_image_max_side)
    b64_data = data_url.split(",", 1)[1] if "," in data_url else data_url
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": OCR_USER_PROMPT},
                    {"inlineData": {"mimeType": "image/jpeg", "data": b64_data}},
                ],
            }
        ],
        "systemInstruction": {"parts": [{"text": OCR_SYSTEM}]},
        "generationConfig": {
            "temperature": 0.0,
        },
    }


def build_map_batch_request(ocr_text: str, candidates_block: str, route_name: str = DEFAULT_MAP_ROUTE) -> Dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": build_map_user_prompt(ocr_text, candidates_block, route_name=route_name)}]}],
        "systemInstruction": {"parts": [{"text": MAP_SYSTEM}]},
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }


def build_map_recall_batch_request(
    ocr_text: str,
    candidates_block: str,
    existing_json: Dict[str, Any],
    route_name: str = DEFAULT_MAP_ROUTE,
) -> Dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_map_recall_user_prompt(ocr_text, candidates_block, existing_json, route_name=route_name)}],
            }
        ],
        "systemInstruction": {"parts": [{"text": MAP_RECALL_SYSTEM}]},
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }


def submit_batch_jsonl_and_wait(
    client: Any,
    model: str,
    requests_jsonl: Path,
    display_name: str,
    poll_interval_sec: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    req_file = client.files.upload(
        file=str(requests_jsonl),
        config={"display_name": f"{display_name}_requests", "mime_type": "application/jsonl"},
    )
    batch_job = client.batches.create(model=model, src=req_file.name, config={"display_name": display_name})
    logger.info("Submitted batch job: name=%s display_name=%s", getattr(batch_job, "name", "unknown"), display_name)

    started = time.perf_counter()
    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    while True:
        current = client.batches.get(name=batch_job.name)
        state_name = get_batch_state_name(current)
        logger.debug("Batch poll: name=%s state=%s", batch_job.name, state_name)
        if state_name in terminal_states:
            batch_job = current
            break
        if (time.perf_counter() - started) > timeout_sec:
            raise TimeoutError(f"Batch job timed out after {timeout_sec}s: {batch_job.name}")
        time.sleep(max(1, poll_interval_sec))

    final_state = get_batch_state_name(batch_job)
    if final_state != "JOB_STATE_SUCCEEDED":
        err = getattr(batch_job, "error", None)
        raise RuntimeError(f"Batch job failed: name={batch_job.name}, state={final_state}, error={err}")

    dest = getattr(batch_job, "dest", None)
    result_file_name = getattr(dest, "file_name", None)
    if not result_file_name:
        raise RuntimeError(f"Batch job has no result file: {batch_job.name}")

    downloaded = client.files.download(file=result_file_name)
    if isinstance(downloaded, (bytes, bytearray)):
        raw_bytes = bytes(downloaded)
    elif hasattr(downloaded, "read"):
        raw_bytes = downloaded.read()
    else:
        raw_bytes = str(downloaded).encode("utf-8")

    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
    by_key: Dict[str, Dict[str, Any]] = {}
    for ln in lines:
        if not ln.strip():
            continue
        obj = json.loads(ln)
        key = str(obj.get("key", "")).strip()
        if key:
            by_key[key] = obj

    meta = {
        "job_name": getattr(batch_job, "name", ""),
        "state": final_state,
        "result_file": result_file_name,
        "request_file": getattr(req_file, "name", ""),
        "line_count": len(lines),
    }
    logger.info(
        "Batch completed: name=%s state=%s records=%d",
        meta["job_name"],
        meta["state"],
        len(by_key),
    )
    return by_key, meta


def batch_record_error_message(record: Dict[str, Any]) -> Optional[str]:
    err = record.get("error")
    if not err:
        return None
    if isinstance(err, dict):
        msg = err.get("message")
        if msg:
            return str(msg)
        return json.dumps(err, ensure_ascii=False)
    return str(err)


def _collect_text_fields_recursive(obj: Any, out: List[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if isinstance(v, str) and key in {"text", "output_text", "generated_text", "answer"}:
                s = v.strip()
                if s:
                    out.append(s)
            elif isinstance(v, (dict, list)):
                _collect_text_fields_recursive(v, out)
    elif isinstance(obj, list):
        for it in obj:
            _collect_text_fields_recursive(it, out)


def batch_record_response_text(record: Dict[str, Any]) -> str:
    resp = record.get("response")
    if not isinstance(resp, dict):
        return ""

    candidates = resp.get("candidates") or []
    if not candidates:
        return ""

    first = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first.get("content") if isinstance(first, dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else []

    texts: List[str] = []
    if isinstance(parts, list):
        for p in parts:
            if isinstance(p, dict):
                txt = p.get("text")
                if txt is not None:
                    texts.append(str(txt))
    if not texts and isinstance(first, dict) and first.get("text") is not None:
        texts.append(str(first.get("text")))

    # Fallback: some batch responses can place text in alternate nested fields.
    if not texts:
        _collect_text_fields_recursive(resp, texts)

    # Keep order, drop duplicates/empties.
    deduped: List[str] = []
    seen = set()
    for t in texts:
        s = str(t).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return "\n".join(deduped).strip()


# -----------------------------
# Main per-image: fixed two-stage flow
# -----------------------------
async def image_to_cdm_json(
    llm: ChatGoogleGenerativeAI,
    retriever: CDMRetriever,
    image_path: Path,
    map_agents: Optional[List[MapAgentSpec]] = None,
    top_k: int = 80,
) -> PageResult:
    t0 = time.perf_counter()
    logger.debug("Page start: image=%s", image_path.name)

    t_ocr0 = time.perf_counter()
    ocr_text = await gemini_ocr(llm, image_path)
    t_ocr = time.perf_counter() - t_ocr0

    t_map0 = time.perf_counter()
    raw_obj, valid_obj, valid_contexts, rejected_fields = await map_ocr_text_with_split_agents_live(
        llm=llm,
        retriever=retriever,
        ocr_text=ocr_text,
        map_agents=map_agents or [],
    )
    recall_added_keys = 0

    t_map = time.perf_counter() - t_map0

    total_t = time.perf_counter() - t0
    logger.debug(
        (
            "Page done: image=%s ocr_chars=%d cdm_keys=%d raw_keys=%d "
            "valid_keys=%d rejected_keys=%d backfill_added=%d recall_added=%d "
            "ocr_s=%.2f map_s=%.2f total_s=%.2f"
        ),
        image_path.name,
        len(ocr_text),
        len(retriever.rows),
        len(raw_obj),
        len(valid_obj),
        len(rejected_fields),
        0,
        recall_added_keys,
        t_ocr,
        t_map,
        total_t,
    )
    return PageResult(
        image_name=image_path.name,
        ocr_text=ocr_text,
        raw_json=raw_obj,
        valid_json=valid_obj,
        input_contexts=valid_contexts,
        cdm_contexts={k: str(retriever.row_by_key.get(k).desc if retriever.row_by_key.get(k) is not None else "").strip() for k in valid_obj.keys()},
        rejected_fields=rejected_fields,
    )


# -----------------------------
# Concurrency helper
# -----------------------------
async def gather_with_concurrency(n: int, coros: Iterable):
    sem = asyncio.Semaphore(n)

    async def _wrap(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_wrap(c) for c in coros], return_exceptions=True)


def build_output_row(merged: Dict[str, Any], output_columns: List[str]) -> Dict[str, Any]:
    row = {c: None for c in output_columns}
    for k, v in merged.items():
        if k in row:
            row[k] = v

    # Fixed site defaults.
    if "Lab_ID" in row:
        row["Lab_ID"] = 1
    if "Device_Type" in row:
        row["Device_Type"] = 1

    # Canonicalize PSG_No prefix.
    if "PSG_No" in row and not _is_missing_value(row.get("PSG_No")):
        s = str(row["PSG_No"]).strip()
        m = re.match(r"^p(\d{4}\s*[-/]\s*\d+)$", s, flags=re.I)
        if m:
            row["PSG_No"] = "P" + re.sub(r"\s+", "", m.group(1))
        else:
            row["PSG_No"] = s

    # Questionnaire-specific normalization.
    apply_psqi_format_and_time_rules(row)
    apply_morning_questionnaire_time_rules(row)
    apply_phx_default_rules(row)
    _apply_cpap_output_rules(row)

    if "Diagnosis_etc" in row:
        row["Diagnosis_etc"] = normalize_diagnosis_etc_value(row.get("Diagnosis_etc"))

    # IRLS/RLS consistency rule:
    # - If IRLS_Category is 9999, force RLS_Category to 0.
    # - If RLS_Category is 0, force IRLS_Category to 9999.
    rls_cat = _coerce_int(row.get("RLS_Category")) if "RLS_Category" in row else None
    irls_cat = _coerce_int(row.get("IRLS_Category")) if "IRLS_Category" in row else None
    if irls_cat == 9999 and "RLS_Category" in row:
        row["RLS_Category"] = 0
    if rls_cat == 0 and "IRLS_Category" in row:
        row["IRLS_Category"] = 9999

    # Database_ID is deterministic from key fields.
    dbid = synthesize_database_id(row)
    if dbid and "Database_ID" in row:
        row["Database_ID"] = dbid

    # Keep these empty in final export by policy.
    if "PSG_Type" in row:
        row["PSG_Type"] = None
    if "Previous_Data" in row:
        row["Previous_Data"] = None

    return row


def build_patient_result(
    patient_name: str,
    page_results: List[PageResult],
    duplicates: List[Dict[str, Any]],
    page_errors: List[Dict[str, str]],
    output_columns: List[str],
    save_intermediate: bool,
    out_dir: Path,
    elapsed_s: float,
) -> Dict[str, Any]:
    if not page_results:
        logger.warning("No successful page results for %s", patient_name)
        return {
            "patient": patient_name,
            "row": None,
            "conflicts": {},
            "provenance": {},
            "duplicates": duplicates,
            "validation_rejections": {},
            "page_errors": page_errors,
            "conflict_resolution": {},
        }

    if save_intermediate:
        (out_dir / "intermediate" / patient_name).mkdir(parents=True, exist_ok=True)
        for pr in page_results:
            stem = Path(pr.image_name).stem
            (out_dir / "intermediate" / patient_name / f"{stem}.txt").write_text(
                pr.ocr_text,
                encoding="utf-8",
            )
            (out_dir / "intermediate" / patient_name / f"{stem}.raw.json").write_text(
                json.dumps(pr.raw_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            context_json = {
                k: {
                    "CDM_Context": str(pr.cdm_contexts.get(k, "")).strip(),
                    "value": v,
                    "input_context": _normalize_input_context(pr.input_contexts.get(k)),
                }
                for k, v in pr.valid_json.items()
            }
            (out_dir / "intermediate" / patient_name / f"{stem}.json").write_text(
                json.dumps(context_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if pr.rejected_fields:
                (out_dir / "intermediate" / patient_name / f"{stem}.rejected.json").write_text(
                    json.dumps(pr.rejected_fields, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    merged, conflicts, provenance = merge_page_results(page_results)
    row = build_output_row(merged, output_columns)
    validation_rejections = {pr.image_name: pr.rejected_fields for pr in page_results if pr.rejected_fields}
    total_valid_keys = sum(len(pr.valid_json) for pr in page_results)
    total_rejected_keys = sum(len(pr.rejected_fields) for pr in page_results)
    reject_reason_counter: Counter = Counter()
    for pr in page_results:
        for meta in pr.rejected_fields.values():
            reason = str(meta.get("reason", "unknown"))
            reject_reason_counter[reason] += 1
    logger.info(
        (
            "Patient %s summary: pages_ok=%d pages_failed=%d merged_keys=%d conflicts=%d "
            "rejected_keys=%d elapsed_s=%.1f"
        ),
        patient_name,
        len(page_results),
        len(page_errors),
        len(merged),
        len(conflicts),
        total_rejected_keys,
        elapsed_s,
    )
    logger.debug(
        "Patient %s key stats: total_valid_keys_from_pages=%d, merged_keys=%d, reject_reasons=%s",
        patient_name,
        total_valid_keys,
        len(merged),
        dict(reject_reason_counter),
    )

    return {
        "patient": patient_name,
        "row": row,
        "merged": merged,
        "conflicts": conflicts,
        "provenance": provenance,
        "duplicates": duplicates,
        "validation_rejections": validation_rejections,
        "page_errors": page_errors,
        "conflict_resolution": {},
    }


def _format_markdown_scalar(v: Any) -> str:
    nv = normalize_value(v)
    if nv is None:
        return "(blank)"
    if isinstance(nv, str):
        s = nv.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not s:
            return "(blank)"
        return s
    return str(nv)


def _markdown_cell(v: Any) -> str:
    text = _format_markdown_scalar(v)
    return text.replace("\n", "<br>").replace("|", "\\|")


def _value_sort_key(v: Any) -> Tuple[int, str]:
    nv = normalize_value(v)
    if nv is None:
        return (2, "")
    if isinstance(nv, (int, float)) and not isinstance(nv, bool):
        return (0, f"{float(nv):020.6f}")
    return (1, str(nv))


def _format_markdown_list(items: Iterable[Any], empty_text: str = "(none)") -> str:
    vals = [str(x).strip() for x in items if str(x or "").strip()]
    if not vals:
        return empty_text
    return "; ".join(vals)


def _aggregate_conflict_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        value = normalize_value(entry.get("value"))
        token = _value_token(value)
        ctx = _normalize_input_context(entry.get("input_context"))
        slot = grouped.setdefault(
            token,
            {
                "value": value,
                "count": 0,
                "questions": [],
                "images": [],
                "page_types": [],
                "filled_bys": [],
            },
        )
        slot["count"] += 1

        question = str(ctx.get("question") or "").strip()
        if question and question not in slot["questions"]:
            slot["questions"].append(question)

        image = str(entry.get("image") or "").strip()
        if image and image not in slot["images"]:
            slot["images"].append(image)

        page_type = normalize_map_route_name(ctx.get("page_type"))
        if page_type and page_type not in slot["page_types"]:
            slot["page_types"].append(page_type)

        filled_by = _normalize_filled_by(ctx.get("filled_by"))
        if filled_by and filled_by not in slot["filled_bys"]:
            slot["filled_bys"].append(filled_by)

    rows = list(grouped.values())
    rows.sort(key=lambda row: (-int(row.get("count", 0) or 0), _value_sort_key(row.get("value"))))
    return rows


def build_conflict_markdown_report(
    patient_name: str,
    row: Optional[Dict[str, Any]],
    conflicts: Dict[str, List[Dict[str, Any]]],
    conflict_resolution: Dict[str, Any],
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_counts = Counter(
        str(v.get("resolver_mode") or "unresolved")
        for v in (conflict_resolution or {}).values()
    )
    lines: List[str] = [
        f"# Conflict Report: {patient_name}",
        "",
        f"- Generated: `{now_str}`",
        "- Conflict definition: same `CDM_KEY` with more than one different normalized value.",
        f"- Conflict keys: `{len(conflicts)}`",
        f"- Resolved by code: `{mode_counts.get('code_majority', 0)}`",
        f"- Resolved by LLM batch: `{mode_counts.get('llm_batch', 0)}`",
        f"- Resolved by LLM single: `{mode_counts.get('llm_single', 0)}`",
        "",
    ]

    if not conflicts:
        lines.extend(
            [
                "## Summary",
                "",
                "No conflicts were detected for this patient.",
                "",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    lines.extend(
        [
            "## Conflict Index",
            "",
        ]
    )
    conflict_keys = sorted(conflicts.keys())
    for idx, key in enumerate(conflict_keys, start=1):
        decision = conflict_resolution.get(key, {}) or {}
        candidate_values = _aggregate_conflict_entries(conflicts[key])
        candidate_text = ", ".join(
            f"`{_format_markdown_scalar(item.get('value'))}` x{int(item.get('count', 0) or 0)}"
            for item in candidate_values
        )
        final_value = row.get(key) if isinstance(row, dict) else decision.get("chosen_value")
        resolver_mode = str(decision.get("resolver_mode") or "unresolved")
        lines.append(
            f"{idx}. `{key}` -> final `{_format_markdown_scalar(final_value)}` via `{resolver_mode}`"
        )
        lines.append(f"   Candidates: {candidate_text}")
        lines.append("")

    for idx, key in enumerate(conflict_keys, start=1):
        entries = conflicts[key]
        decision = conflict_resolution.get(key, {}) or {}
        chosen_idx = _coerce_int(decision.get("chosen_index"))
        chosen_image = str(decision.get("source_image") or "").strip()
        chosen_value = decision.get("chosen_value")
        if chosen_value is None and isinstance(row, dict):
            chosen_value = row.get(key)
        reason = str(decision.get("reason") or "").strip()
        resolver_mode = str(decision.get("resolver_mode") or "unresolved")
        chosen_ctx = _normalize_input_context(decision.get("input_context"))
        aggregated = _aggregate_conflict_entries(entries)

        lines.extend(
            [
                "",
                f"## {idx}. `{key}`",
                "",
                f"- Final value: `{_format_markdown_scalar(chosen_value)}`",
                f"- Resolver: `{resolver_mode}`",
            ]
        )
        if chosen_idx is not None:
            lines.append(f"- Chosen candidate index: `{chosen_idx}`")
        if chosen_image:
            lines.append(f"- Chosen source image: `{chosen_image}`")
        if chosen_ctx.get("page_type"):
            lines.append(f"- Chosen page type: `{chosen_ctx.get('page_type')}`")
        if chosen_ctx.get("filled_by"):
            lines.append(f"- Chosen source type: `{chosen_ctx.get('filled_by')}`")
        if chosen_ctx.get("question"):
            lines.append(f"- Chosen question/context: `{chosen_ctx.get('question')}`")
        if reason:
            lines.append(f"- Reason: {reason}")

        lines.extend(
            [
                "",
                "### Candidate Summary",
                "",
            ]
        )
        for option_idx, item in enumerate(aggregated, start=1):
            lines.append(f"#### Option {option_idx}")
            lines.append("")
            lines.append(f"- Value: `{_format_markdown_scalar(item.get('value'))}`")
            lines.append(f"- Count: `{int(item.get('count', 0) or 0)}`")
            lines.append(f"- Filled by: {_format_markdown_list(item.get('filled_bys') or [])}")
            lines.append(f"- Page types: {_format_markdown_list(item.get('page_types') or [])}")
            lines.append(f"- Source images: {_format_markdown_list(item.get('images') or [])}")
            lines.append("- Questions / contexts:")
            question_list = item.get("questions") or []
            if question_list:
                for q in question_list:
                    lines.append(f"  - `{str(q).strip()}`")
            else:
                lines.append("  - (none)")
            lines.append("")

        lines.extend(
            [
                "### Raw Candidates",
                "",
            ]
        )
        for entry_idx, entry in enumerate(entries):
            ctx = _normalize_input_context(entry.get("input_context"))
            marker = " <- chosen" if chosen_idx is not None and entry_idx == chosen_idx else ""
            lines.append(f"#### Candidate {entry_idx}{marker}")
            lines.append("")
            lines.append(f"- Value: `{_format_markdown_scalar(entry.get('value'))}`")
            lines.append(f"- Image: `{str(entry.get('image') or '').strip()}`")
            lines.append(f"- Filled by: {_format_markdown_scalar(ctx.get('filled_by'))}")
            lines.append(f"- Page type: `{_format_markdown_scalar(ctx.get('page_type'))}`")
            lines.append(f"- Question / context: `{_format_markdown_scalar(ctx.get('question'))}`")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_patient_outputs(output_dir: Path, patient_name: str, res: Dict[str, Any], output_columns: List[str]) -> None:
    # Per-patient CSV
    if res["row"] is not None:
        df_one = pd.DataFrame([res["row"]], columns=output_columns)
        df_one.to_csv(output_dir / f"{patient_name}.csv", index=False)

    # Conflicts report
    if res["conflicts"]:
        (output_dir / "conflicts").mkdir(exist_ok=True)
        (output_dir / "conflicts" / f"{patient_name}_conflicts.json").write_text(
            json.dumps(res["conflicts"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Provenance report: which image contributed each key/value.
    if res.get("provenance"):
        (output_dir / "provenance").mkdir(exist_ok=True)
        (output_dir / "provenance" / f"{patient_name}_provenance.json").write_text(
            json.dumps(res["provenance"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Validation rejection report: keys dropped by CDM schema checks.
    if res.get("validation_rejections"):
        (output_dir / "validation").mkdir(exist_ok=True)
        (output_dir / "validation" / f"{patient_name}_rejected.json").write_text(
            json.dumps(res["validation_rejections"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Dedup report: exact/near duplicate pages skipped.
    if res.get("duplicates"):
        (output_dir / "dedup").mkdir(exist_ok=True)
        (output_dir / "dedup" / f"{patient_name}_duplicates.json").write_text(
            json.dumps(res["duplicates"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Page-level runtime errors.
    if res.get("page_errors"):
        (output_dir / "errors").mkdir(exist_ok=True)
        (output_dir / "errors" / f"{patient_name}_errors.json").write_text(
            json.dumps(res["page_errors"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # LLM conflict resolution decisions.
    if res.get("conflict_resolution"):
        (output_dir / "conflict_resolution").mkdir(exist_ok=True)
        (output_dir / "conflict_resolution" / f"{patient_name}_resolution.json").write_text(
            json.dumps(res["conflict_resolution"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if res.get("conflicts") or res.get("conflict_resolution"):
        (output_dir / "conflict_reports").mkdir(exist_ok=True)
        report_md = build_conflict_markdown_report(
            patient_name=patient_name,
            row=res.get("row"),
            conflicts=res.get("conflicts") or {},
            conflict_resolution=res.get("conflict_resolution") or {},
        )
        (output_dir / "conflict_reports" / f"{patient_name}_conflict_report.md").write_text(
            report_md,
            encoding="utf-8",
        )


# -----------------------------
# Patient processing
# -----------------------------
async def process_one_patient(
    patient_dir: Path,
    llm: ChatGoogleGenerativeAI,
    retriever: CDMRetriever,
    map_agents: List[MapAgentSpec],
    output_columns: List[str],
    concurrency: int,
    map_bundle_size: int,
    top_k: int,
    near_dup_hamming: int,
    save_intermediate: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    t_patient0 = time.perf_counter()
    images = iter_images(patient_dir)
    orig_n_images = len(images)
    if not images:
        logger.warning("No images found in %s", patient_dir)
        return {
            "patient": patient_dir.name,
            "row": None,
            "conflicts": {},
            "provenance": {},
            "duplicates": [],
            "validation_rejections": {},
            "page_errors": [],
            "conflict_resolution": {},
        }

    images, duplicates = deduplicate_images(images, near_dup_hamming=near_dup_hamming)
    if duplicates:
        logger.info(
            "Patient %s: dropped %d duplicate/near-duplicate pages (%d -> %d)",
            patient_dir.name,
            len(duplicates),
            orig_n_images,
            len(images),
        )
        logger.debug(
            "Patient %s duplicate details: %s",
            patient_dir.name,
            json.dumps(duplicates[:10], ensure_ascii=False),
        )
    if not images:
        logger.warning("All pages were filtered as duplicates in %s", patient_dir)
        return {
            "patient": patient_dir.name,
            "row": None,
            "conflicts": {},
            "provenance": {},
            "duplicates": duplicates,
            "validation_rejections": {},
            "page_errors": [],
            "conflict_resolution": {},
        }

    logger.info("Patient %s: %d images", patient_dir.name, len(images))
    logger.debug(
        "Patient %s images: %s",
        patient_dir.name,
        ", ".join([img.name for img in images]),
    )

    # 1) OCR each image first.
    ocr_tasks = [gemini_ocr(llm=llm, image_path=img) for img in images]
    ocr_outputs = await gather_with_concurrency(concurrency, ocr_tasks)

    page_errors: List[Dict[str, str]] = []
    ocr_pairs: List[Tuple[Path, str]] = []
    for img, out in zip(images, ocr_outputs):
        if isinstance(out, Exception):
            logger.warning("OCR failed: %s/%s (%s)", patient_dir.name, img.name, out)
            page_errors.append({"image": img.name, "error_type": type(out).__name__, "error": str(out)})
            continue
        txt = str(out).strip()
        if not txt:
            page_errors.append({"image": img.name, "error_type": "EmptyOCR", "error": "Empty OCR text"})
            continue
        ocr_pairs.append((img, txt))

    # 2) Merge OCR texts from n images, then map once per bundle.
    bundles = chunked(ocr_pairs, map_bundle_size)
    bundle_meta: List[Tuple[str, str, List[str]]] = []
    for idx, b in enumerate(bundles, start=1):
        image_names = [img.name for img, _ in b]
        merged_text = merge_ocr_text_blocks([(img.name, txt) for img, txt in b])
        if not merged_text:
            page_errors.append(
                {
                    "image": ",".join(image_names),
                    "error_type": "EmptyMergedOCR",
                    "error": "Merged OCR text is empty",
                }
            )
            continue
        bundle_name = make_bundle_image_name(idx, image_names)
        bundle_meta.append((bundle_name, merged_text, image_names))

    map_tasks = [
        map_ocr_text_with_split_agents_live(
            llm=llm,
            retriever=retriever,
            ocr_text=merged_text,
            map_agents=map_agents,
        )
        for _, merged_text, _ in bundle_meta
    ]
    map_outputs = await gather_with_concurrency(max(1, min(concurrency, 4)), map_tasks)

    page_results: List[PageResult] = []
    for (bundle_name, merged_text, image_names), out in zip(bundle_meta, map_outputs):
        if isinstance(out, Exception):
            logger.warning("Bundle map failed: %s/%s (%s)", patient_dir.name, bundle_name, out)
            page_errors.append(
                {
                    "image": bundle_name,
                    "error_type": type(out).__name__,
                    "error": f"{out}; source_images={','.join(image_names)}",
                }
            )
            continue
        raw_obj, valid_obj, valid_contexts, valid_cdm_contexts, rejected_fields = out
        page_results.append(
            PageResult(
                image_name=bundle_name,
                ocr_text=merged_text,
                raw_json=raw_obj,
                valid_json=valid_obj,
                input_contexts=valid_contexts,
                cdm_contexts=valid_cdm_contexts,
                rejected_fields=rejected_fields,
            )
        )

    t_patient = time.perf_counter() - t_patient0
    return build_patient_result(
        patient_name=patient_dir.name,
        page_results=page_results,
        duplicates=duplicates,
        page_errors=page_errors,
        output_columns=output_columns,
        save_intermediate=save_intermediate,
        out_dir=out_dir,
        elapsed_s=t_patient,
    )


def make_page_key(patient_name: str, image_name: str) -> str:
    return f"{patient_name}::{image_name}"


def process_patients_with_batch_api(
    input_root: Path,
    retriever: CDMRetriever,
    map_agents: List[MapAgentSpec],
    output_columns: List[str],
    output_dir: Path,
    map_bundle_size: int,
    top_k: int,
    near_dup_hamming: int,
    batch_model: str,
    batch_poll_interval_sec: int,
    batch_timeout_sec: int,
    batch_image_max_side: int,
    batch_ocr_retry_rounds: int,
    batch_retry_pause_sec: float,
    save_intermediate: bool,
) -> List[Dict[str, Any]]:
    client = create_genai_batch_client()
    patient_dirs = iter_patient_folders(input_root)
    logger.info("Found %d patient folders", len(patient_dirs))

    contexts: Dict[str, Dict[str, Any]] = {}
    page_plan: List[Dict[str, Any]] = []

    for pdir in patient_dirs:
        started = time.perf_counter()
        images = iter_images(pdir)
        duplicates: List[Dict[str, Any]] = []
        page_errors: List[Dict[str, str]] = []

        if not images:
            logger.warning("No images found in %s", pdir)
            contexts[pdir.name] = {
                "started": started,
                "duplicates": [],
                "page_errors": page_errors,
                "page_results": [],
                "images": [],
            }
            continue

        orig_n = len(images)
        images, duplicates = deduplicate_images(images, near_dup_hamming=near_dup_hamming)
        if duplicates:
            logger.info(
                "Patient %s: dropped %d duplicate/near-duplicate pages (%d -> %d)",
                pdir.name,
                len(duplicates),
                orig_n,
                len(images),
            )
        if not images:
            logger.warning("All pages were filtered as duplicates in %s", pdir)
            contexts[pdir.name] = {
                "started": started,
                "duplicates": duplicates,
                "page_errors": page_errors,
                "page_results": [],
                "images": [],
            }
            continue

        logger.info("Patient %s: %d images", pdir.name, len(images))
        contexts[pdir.name] = {
            "started": started,
            "duplicates": duplicates,
            "page_errors": page_errors,
            "page_results": [],
            "images": [img.name for img in images],
        }
        for img in images:
            key = make_page_key(pdir.name, img.name)
            page_plan.append({"key": key, "patient": pdir.name, "image_path": img})

    if page_plan:
        tmp_dir = output_dir / "_batch_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        ocr_plan: List[Dict[str, Any]] = []
        ocr_requests_path = tmp_dir / f"ocr_requests_{uuid4().hex}.jsonl"
        with ocr_requests_path.open("w", encoding="utf-8") as f:
            for p in page_plan:
                try:
                    req = build_ocr_batch_request(p["image_path"], batch_image_max_side=batch_image_max_side)
                except Exception as e:
                    ctx = contexts[p["patient"]]
                    ctx["page_errors"].append(
                        {"image": p["image_path"].name, "error_type": type(e).__name__, "error": str(e)}
                    )
                    logger.warning(
                        "Skipping page due to request build failure: %s/%s (%s)",
                        p["patient"],
                        p["image_path"].name,
                        e,
                    )
                    continue
                f.write(json.dumps({"key": p["key"], "request": req}, ensure_ascii=False) + "\n")
                ocr_plan.append(p)

        if ocr_plan:
            ocr_file_mb = ocr_requests_path.stat().st_size / (1024 * 1024)
            logger.info("Submitting OCR batch with %d requests (input_jsonl=%.2f MB)", len(ocr_plan), ocr_file_mb)
            ocr_records, ocr_meta = submit_batch_jsonl_and_wait(
                client=client,
                model=batch_model,
                requests_jsonl=ocr_requests_path,
                display_name=f"sleep-ocr-{uuid4().hex[:8]}",
                poll_interval_sec=batch_poll_interval_sec,
                timeout_sec=batch_timeout_sec,
            )
            logger.info("OCR batch metadata: %s", json.dumps(ocr_meta, ensure_ascii=False))

            ocr_text_by_key: Dict[str, str] = {}
            ocr_plan_by_key: Dict[str, Dict[str, Any]] = {str(p["key"]): p for p in ocr_plan}
            pending_plan: List[Dict[str, Any]] = list(ocr_plan)
            unresolved: Dict[str, Dict[str, str]] = {}

            # Batch OCR retry loop:
            # attempt=0 uses the already-submitted initial OCR batch.
            for attempt in range(batch_ocr_retry_rounds + 1):
                if not pending_plan:
                    break

                attempt_records = ocr_records
                if attempt > 0:
                    if batch_retry_pause_sec > 0:
                        logger.info(
                            "Waiting %.1fs before OCR retry batch %d to reduce transient empty responses.",
                            batch_retry_pause_sec,
                            attempt,
                        )
                        time.sleep(batch_retry_pause_sec)

                    retry_path = tmp_dir / f"ocr_retry_{attempt}_{uuid4().hex}.jsonl"
                    retry_plan: List[Dict[str, Any]] = []
                    with retry_path.open("w", encoding="utf-8") as f:
                        for p in pending_plan:
                            key = str(p["key"])
                            prev_err_type = str(unresolved.get(key, {}).get("error_type", ""))
                            retry_max_side = batch_image_max_side
                            # Empty OCR pages occasionally recover with a smaller payload.
                            if prev_err_type == "EmptyOCR":
                                retry_max_side = max(1280, int(batch_image_max_side * 0.85))
                            try:
                                req = build_ocr_batch_request(
                                    p["image_path"],
                                    batch_image_max_side=retry_max_side,
                                )
                            except Exception as e:
                                unresolved[key] = {
                                    "image": p["image_path"].name,
                                    "error_type": type(e).__name__,
                                    "error": str(e),
                                }
                                logger.warning(
                                    "Skipping OCR retry request build failure: %s/%s (%s)",
                                    p["patient"],
                                    p["image_path"].name,
                                    e,
                                )
                                continue
                            f.write(json.dumps({"key": p["key"], "request": req}, ensure_ascii=False) + "\n")
                            retry_plan.append(p)

                    if not retry_plan:
                        break

                    retry_mb = retry_path.stat().st_size / (1024 * 1024)
                    logger.info(
                        "Submitting OCR retry batch %d/%d with %d requests (input_jsonl=%.2f MB)",
                        attempt,
                        batch_ocr_retry_rounds,
                        len(retry_plan),
                        retry_mb,
                    )
                    attempt_records, retry_meta = submit_batch_jsonl_and_wait(
                        client=client,
                        model=batch_model,
                        requests_jsonl=retry_path,
                        display_name=f"sleep-ocr-retry{attempt}-{uuid4().hex[:8]}",
                        poll_interval_sec=batch_poll_interval_sec,
                        timeout_sec=batch_timeout_sec,
                    )
                    logger.info(
                        "OCR retry batch %d metadata: %s",
                        attempt,
                        json.dumps(retry_meta, ensure_ascii=False),
                    )
                    pending_plan = retry_plan

                next_pending: List[Dict[str, Any]] = []
                for p in pending_plan:
                    key = str(p["key"])
                    rec = attempt_records.get(key)
                    if rec is None:
                        unresolved[key] = {
                            "image": p["image_path"].name,
                            "error_type": "MissingResult",
                            "error": "Missing OCR batch result",
                        }
                        next_pending.append(p)
                        continue

                    err = batch_record_error_message(rec)
                    if err:
                        unresolved[key] = {
                            "image": p["image_path"].name,
                            "error_type": "BatchError",
                            "error": err,
                        }
                        next_pending.append(p)
                        continue

                    ocr_text = batch_record_response_text(rec)
                    if not ocr_text:
                        if logger.isEnabledFor(logging.DEBUG):
                            resp_obj = rec.get("response")
                            try:
                                resp_str = json.dumps(resp_obj, ensure_ascii=False)
                            except Exception:
                                resp_str = str(resp_obj)
                            logger.debug(
                                "Empty OCR response payload for %s/%s: %s",
                                p["patient"],
                                p["image_path"].name,
                                _clip_prompt_text(resp_str, 1500),
                            )
                        unresolved[key] = {
                            "image": p["image_path"].name,
                            "error_type": "EmptyOCR",
                            "error": "Empty OCR text from batch response",
                        }
                        next_pending.append(p)
                        continue

                    ocr_text_by_key[key] = ocr_text
                    unresolved.pop(key, None)

                pending_plan = next_pending
                if pending_plan and attempt < batch_ocr_retry_rounds:
                    logger.warning(
                        "OCR batch unresolved pages after attempt %d: %d (retrying)",
                        attempt,
                        len(pending_plan),
                    )

            if unresolved:
                logger.warning(
                    "OCR retries exhausted: unresolved pages=%d",
                    len(unresolved),
                )
                for key, meta in unresolved.items():
                    p = ocr_plan_by_key.get(key)
                    if p is None:
                        continue
                    ctx = contexts[p["patient"]]
                    ctx["page_errors"].append(meta)

            map_source_plan = [p for p in ocr_plan if str(p["key"]) in ocr_text_by_key]
            if map_source_plan:
                # Build patient-level OCR bundles (n images -> one merged map input).
                map_bundle_plan: List[Dict[str, Any]] = []
                for patient_name, ctx in contexts.items():
                    order = {name: i for i, name in enumerate(ctx.get("images", []))}
                    patient_pages = [p for p in map_source_plan if p["patient"] == patient_name]
                    patient_pages.sort(key=lambda x: order.get(x["image_path"].name, 10**9))

                    pairs: List[Tuple[Dict[str, Any], str]] = []
                    for p in patient_pages:
                        k = str(p["key"])
                        txt = ocr_text_by_key.get(k, "").strip()
                        if not txt:
                            continue
                        pairs.append((p, txt))

                    for bidx, chunk in enumerate(chunked(pairs, map_bundle_size), start=1):
                        image_names = [it[0]["image_path"].name for it in chunk]
                        merged_text = merge_ocr_text_blocks([(it[0]["image_path"].name, it[1]) for it in chunk])
                        if not merged_text:
                            ctx["page_errors"].append(
                                {
                                    "image": ",".join(image_names),
                                    "error_type": "EmptyMergedOCR",
                                    "error": "Merged OCR text is empty",
                                }
                            )
                            continue
                        bundle_key = f"{patient_name}::bundle{bidx:04d}"
                        bundle_name = make_bundle_image_name(bidx, image_names)
                        map_bundle_plan.append(
                            {
                                "bundle_key": bundle_key,
                                "bundle_name": bundle_name,
                                "patient": patient_name,
                                "image_names": image_names,
                                "ocr_text": merged_text,
                            }
                        )

                if map_bundle_plan:
                    map_requests_path = tmp_dir / f"map_requests_{uuid4().hex}.jsonl"
                    req_meta: Dict[str, Dict[str, Any]] = {}
                    with map_requests_path.open("w", encoding="utf-8") as f:
                        for bundle in map_bundle_plan:
                            route_name = str(classify_map_route(bundle["ocr_text"]).get("route") or DEFAULT_MAP_ROUTE)
                            if map_agents:
                                route_agents = build_map_agent_specs(retriever, len(map_agents), route_name=route_name)
                                for aidx, agent in enumerate(route_agents, start=1):
                                    req_key = f"{bundle['bundle_key']}::A{aidx}"
                                    req = build_map_batch_request(bundle["ocr_text"], agent.candidates_block, route_name=agent.route_name)
                                    f.write(json.dumps({"key": req_key, "request": req}, ensure_ascii=False) + "\n")
                                    req_meta[req_key] = {"bundle_key": bundle["bundle_key"], "agent_name": agent.name}
                            else:
                                req_key = f"{bundle['bundle_key']}::A1"
                                req = build_map_batch_request(
                                    bundle["ocr_text"],
                                    retriever.prompt_block_for_route(route_name),
                                    route_name=route_name,
                                )
                                f.write(json.dumps({"key": req_key, "request": req}, ensure_ascii=False) + "\n")
                                req_meta[req_key] = {"bundle_key": bundle["bundle_key"], "agent_name": "single_full"}

                    map_file_mb = map_requests_path.stat().st_size / (1024 * 1024)
                    logger.info(
                        "Submitting MAP batch with %d requests across %d bundles (input_jsonl=%.2f MB)",
                        len(req_meta),
                        len(map_bundle_plan),
                        map_file_mb,
                    )
                    map_records, map_meta = submit_batch_jsonl_and_wait(
                        client=client,
                        model=batch_model,
                        requests_jsonl=map_requests_path,
                        display_name=f"sleep-map-{uuid4().hex[:8]}",
                        poll_interval_sec=batch_poll_interval_sec,
                        timeout_sec=batch_timeout_sec,
                    )
                    logger.info("MAP batch metadata: %s", json.dumps(map_meta, ensure_ascii=False))

                    stage_by_bundle: Dict[str, Dict[str, Any]] = {}
                    for bundle in map_bundle_plan:
                        stage_by_bundle[bundle["bundle_key"]] = {
                            "patient": bundle["patient"],
                            "image_name": bundle["bundle_name"],
                            "source_images": bundle["image_names"],
                            "ocr_text": bundle["ocr_text"],
                            "raw_json": {},
                            "valid_json": {},
                            "input_contexts": {},
                            "cdm_contexts": {},
                            "rejected_fields": {},
                        }

                    for req_key, meta in req_meta.items():
                        bundle_key = str(meta.get("bundle_key", ""))
                        stage = stage_by_bundle.get(bundle_key)
                        if stage is None:
                            continue
                        patient_name = str(stage["patient"])
                        ctx = contexts[patient_name]
                        rec = map_records.get(req_key)
                        if rec is None:
                            ctx["page_errors"].append(
                                {
                                    "image": str(stage["image_name"]),
                                    "error_type": "MissingResult",
                                    "error": f"Missing MAP batch result for {meta.get('agent_name')}",
                                }
                            )
                            continue
                        err = batch_record_error_message(rec)
                        if err:
                            ctx["page_errors"].append(
                                {
                                    "image": str(stage["image_name"]),
                                    "error_type": "BatchError",
                                    "error": f"{meta.get('agent_name')}: {err}",
                                }
                            )
                            continue
                        raw_text = batch_record_response_text(rec)
                        if not raw_text:
                            ctx["page_errors"].append(
                                {
                                    "image": str(stage["image_name"]),
                                    "error_type": "EmptyMapOutput",
                                    "error": f"Empty MAP response text from {meta.get('agent_name')}",
                                }
                            )
                            continue
                        try:
                            raw_payload = safe_extract_json(raw_text)
                        except Exception as e:
                            ctx["page_errors"].append(
                                {
                                    "image": str(stage["image_name"]),
                                    "error_type": type(e).__name__,
                                    "error": f"{meta.get('agent_name')}: {e}",
                                }
                            )
                            continue
                        merge_map_payload_into_stage(
                            retriever=retriever,
                            ocr_text=str(stage.get("ocr_text", "")),
                            raw_payload=raw_payload,
                            route_name=str(meta.get("route_name") or DEFAULT_MAP_ROUTE),
                            stage_raw=stage["raw_json"],
                            stage_valid=stage["valid_json"],
                            stage_contexts=stage["input_contexts"],
                            stage_cdm_contexts=stage["cdm_contexts"],
                            stage_rejected=stage["rejected_fields"],
                        )

                    # Backfill and finalize bundle-level page results.
                    for bundle_key, stage in stage_by_bundle.items():
                        patient_name = str(stage["patient"])
                        ctx = contexts[patient_name]
                        backfill_additions, backfill_rejected = apply_core_backfill(
                            stage["valid_json"],
                            retriever,
                            stage["ocr_text"],
                        )
                        for bk, bv in backfill_additions.items():
                            stage["valid_json"][bk] = bv
                            stage["input_contexts"].setdefault(
                                bk,
                                {"filled_by": "", "question": "Derived from OCR header pattern", "page_type": ""},
                            )
                            row = retriever.row_by_key.get(bk)
                            stage["cdm_contexts"].setdefault(
                                bk,
                                str(row.desc if row is not None else "").strip(),
                            )
                            stage["raw_json"].setdefault(
                                bk,
                                {
                                    "CDM_Context": stage["cdm_contexts"].get(bk, ""),
                                    "value": bv,
                                    "input_context": stage["input_contexts"][bk],
                                },
                            )
                        for bk, meta in backfill_rejected.items():
                            stage["rejected_fields"].setdefault(bk, meta)

                        if not stage["raw_json"] and not stage["valid_json"]:
                            ctx["page_errors"].append(
                                {
                                    "image": str(stage["image_name"]),
                                    "error_type": "EmptyBundleMap",
                                    "error": "No valid MAP payload from split agents",
                                }
                            )
                            continue

                        ctx["page_results"].append(
                            PageResult(
                                image_name=str(stage["image_name"]),
                                ocr_text=str(stage["ocr_text"]),
                                raw_json=stage["raw_json"],
                                valid_json=stage["valid_json"],
                                input_contexts=stage.get("input_contexts", {}),
                                cdm_contexts=stage.get("cdm_contexts", {}),
                                rejected_fields=stage["rejected_fields"],
                            )
                        )

    results: List[Dict[str, Any]] = []
    for pdir in patient_dirs:
        ctx = contexts.get(
            pdir.name,
            {"started": time.perf_counter(), "duplicates": [], "page_errors": [], "page_results": []},
        )
        elapsed = time.perf_counter() - float(ctx["started"])
        res = build_patient_result(
            patient_name=pdir.name,
            page_results=ctx["page_results"],
            duplicates=ctx["duplicates"],
            page_errors=ctx["page_errors"],
            output_columns=output_columns,
            save_intermediate=save_intermediate,
            out_dir=output_dir,
            elapsed_s=elapsed,
        )
        results.append(res)
    return results


async def run_pipeline(
    input_root: Path,
    cdm_csv: Path,
    example_csv: Path,
    output_dir: Path,
    use_batch_api: bool,
    batch_model: str,
    batch_poll_interval_sec: int,
    batch_timeout_sec: int,
    batch_image_max_side: int,
    batch_ocr_retry_rounds: int,
    batch_retry_pause_sec: float,
    map_bundle_size: int,
    use_split_map_agents: bool,
    map_agent_count: int,
    concurrency: int,
    patient_concurrency: int,
    request_delay_sec: float,
    top_k: int,
    near_dup_hamming: int,
    debug: bool,
    log_filename: str,
    save_intermediate: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir=output_dir, debug=debug, log_filename=log_filename)
    REQUEST_THROTTLE.configure(request_delay_sec)
    t_run0 = time.perf_counter()

    # Column order: match example.csv exactly
    example_df = pd.read_csv(example_csv)
    output_columns = list(example_df.columns)

    patient_dirs = iter_patient_folders(input_root)
    logger.info("Found %d patient folders", len(patient_dirs))
    logger.info(
        (
            "Run config: use_batch_api=%s, patient_concurrency=%d, page_concurrency=%d, top_k=%d (ignored in full-CDM mode), "
            "near_dup_hamming=%d, save_intermediate=%s, batch_retry_pause_sec=%.1f, map_bundle_size=%d, "
            "split_map_agents=%s, map_agent_count=%d, request_delay_sec=%.2f"
        ),
        use_batch_api,
        patient_concurrency,
        concurrency,
        top_k,
        near_dup_hamming,
        save_intermediate,
        batch_retry_pause_sec,
        map_bundle_size,
        use_split_map_agents,
        max(1, int(map_agent_count)),
        request_delay_sec,
    )

    retriever = CDMRetriever(cdm_csv)
    map_agents = build_map_agent_specs(retriever, map_agent_count) if use_split_map_agents else []
    resolver_llm: Optional[ChatGoogleGenerativeAI] = None
    try:
        resolver_llm = build_gemini()
    except Exception as e:
        logger.warning("Conflict resolver LLM is unavailable. Conflicting keys will remain unresolved. (%s)", e)

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
            overrides, decisions = await resolve_conflicts_with_llm(
                llm=resolver_llm,
                retriever=retriever,
                patient_name=str(res.get("patient", "")),
                conflicts=conflicts,
            )
            if overrides:
                merged_like = dict(res.get("merged") or {})
                for k, v in overrides.items():
                    if k in output_columns:
                        merged_like[k] = v
                res["merged"] = merged_like
                res["row"] = build_output_row(merged_like, output_columns)
            res["conflict_resolution"] = decisions
            logger.info(
                "Patient %s conflict resolution: conflict_keys=%d overrides=%d",
                str(res.get("patient", "")),
                len(conflicts),
                len(overrides),
            )
        except Exception as e:
            logger.warning("Conflict resolver failed for %s: %s", str(res.get("patient", "")), e)

    results: List[Dict[str, Any]] = []
    patient_failures = 0
    if use_batch_api:
        logger.info(
            "Batch mode note: --concurrency and --patient_concurrency are ignored in batch execution.",
        )
        chosen_batch_model = batch_model.strip() or os.getenv("GEMINI_BATCH_MODEL", os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
        logger.info(
            "Batch API mode enabled: model=%s, poll_interval_sec=%d, timeout_sec=%d, image_max_side=%d, "
            "ocr_retry_rounds=%d, map_bundle_size=%d, split_map_agents=%s, map_agent_count=%d",
            chosen_batch_model,
            batch_poll_interval_sec,
            batch_timeout_sec,
            batch_image_max_side,
            batch_ocr_retry_rounds,
            map_bundle_size,
            use_split_map_agents,
            max(1, int(map_agent_count)),
        )
        results = process_patients_with_batch_api(
            input_root=input_root,
            retriever=retriever,
            map_agents=map_agents,
            output_columns=output_columns,
            output_dir=output_dir,
            map_bundle_size=map_bundle_size,
            top_k=top_k,
            near_dup_hamming=near_dup_hamming,
            batch_model=chosen_batch_model,
            batch_poll_interval_sec=batch_poll_interval_sec,
            batch_timeout_sec=batch_timeout_sec,
            batch_image_max_side=batch_image_max_side,
            batch_ocr_retry_rounds=batch_ocr_retry_rounds,
            batch_retry_pause_sec=batch_retry_pause_sec,
            save_intermediate=save_intermediate,
        )
        for res in results:
            await _maybe_resolve_conflicts(res)
            write_patient_outputs(output_dir=output_dir, patient_name=str(res["patient"]), res=res, output_columns=output_columns)
    else:
        llm = resolver_llm if resolver_llm is not None else build_gemini()

        async def _process_patient_slot(idx: int, pdir: Path) -> Dict[str, Any]:
            logger.info("Processing patient %d/%d: %s", idx, len(patient_dirs), pdir.name)
            res = await process_one_patient(
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
            return res

        patient_tasks = [_process_patient_slot(idx, pdir) for idx, pdir in enumerate(patient_dirs, start=1)]
        patient_outputs = await gather_with_concurrency(patient_concurrency, patient_tasks)

        for pdir, out in zip(patient_dirs, patient_outputs):
            if isinstance(out, Exception):
                patient_failures += 1
                logger.error("Failed processing patient folder %s: %s", pdir, out)
                continue

            res = out
            results.append(res)
            await _maybe_resolve_conflicts(res)
            write_patient_outputs(output_dir=output_dir, patient_name=pdir.name, res=res, output_columns=output_columns)

    # Combined CSV
    rows = [r["row"] for r in results if r.get("row") is not None]
    if rows:
        df_all = pd.DataFrame(rows, columns=output_columns)
        df_all.to_csv(output_dir / "all_patients.csv", index=False)
        logger.info("Wrote %d rows to %s", len(rows), output_dir / "all_patients.csv")
    else:
        logger.warning("No patient rows produced.")

    total_page_errors = sum(len(r.get("page_errors", [])) for r in results)
    total_conflicts = sum(len(r.get("conflicts", {})) for r in results)
    total_duplicates = sum(len(r.get("duplicates", [])) for r in results)
    elapsed = time.perf_counter() - t_run0
    logger.info(
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


def main():
    load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True, help="Root directory containing one folder per patient")
    ap.add_argument("--cdm_csv", type=str, required=True, help="Path to CDM definition CSV (e.g., cdm_revised.csv)")
    ap.add_argument("--example_csv", type=str, required=True, help="Path to example.csv (column order template)")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for CSV files")
    ap.add_argument("--use_batch_api", action="store_true", help="Use Gemini Batch API (two batch jobs: OCR then mapping)")
    ap.add_argument("--batch_model", type=str, default="", help="Gemini model for batch mode (default: GEMINI_BATCH_MODEL or GEMINI_MODEL)")
    ap.add_argument("--batch_poll_interval_sec", type=int, default=15, help="Polling interval for batch job status")
    ap.add_argument("--batch_timeout_sec", type=int, default=7200, help="Timeout for each batch job in seconds")
    ap.add_argument("--batch_image_max_side", type=int, default=2338, help="Max image side length for batch OCR requests")
    ap.add_argument("--batch_ocr_retry_rounds", type=int, default=2, help="Number of extra OCR batch retries for unresolved/empty OCR pages")
    ap.add_argument(
        "--batch_retry_pause_sec",
        type=float,
        default=10.0,
        help="Pause before each OCR retry batch submission (seconds)",
    )
    ap.add_argument(
        "--map_bundle_size",
        type=int,
        default=1,
        help="Number of OCRed images to merge into one MAP input unit",
    )
    ap.add_argument(
        "--disable_split_map_agents",
        action="store_true",
        help="Disable split map agents and use a single full-CDM map agent",
    )
    ap.add_argument(
        "--map_agent_count",
        type=int,
        default=6,
        help="Number of equal CDM slices when split map agents are enabled",
    )
    ap.add_argument("--patient_concurrency", type=int, default=1, help="Number of patient folders processed in parallel")
    ap.add_argument("--concurrency", type=int, default=3, help="Parallelism for per-image Gemini runs")
    ap.add_argument(
        "--request_delay_sec",
        type=float,
        default=0.0,
        help="Minimum delay between each live LLM request (seconds); 0 disables pacing",
    )
    ap.add_argument("--top_k", type=int, default=220, help="Legacy option (ignored in full-CDM prompt mode)")
    ap.add_argument("--near_dup_hamming", type=int, default=6, help="Perceptual hash distance threshold for near-duplicate page filtering")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    ap.add_argument("--log_filename", type=str, default="pipeline.log", help="Log filename under <output_dir>/logs")
    ap.add_argument("--save_intermediate", action="store_true", help="Save per-image OCR text and JSON outputs")
    args = ap.parse_args()

    asyncio.run(
        run_pipeline(
            input_root=Path(args.input_root),
            cdm_csv=Path(args.cdm_csv),
            example_csv=Path(args.example_csv),
            output_dir=Path(args.output_dir),
            use_batch_api=args.use_batch_api,
            batch_model=args.batch_model,
            batch_poll_interval_sec=args.batch_poll_interval_sec,
            batch_timeout_sec=args.batch_timeout_sec,
            batch_image_max_side=args.batch_image_max_side,
            batch_ocr_retry_rounds=args.batch_ocr_retry_rounds,
            batch_retry_pause_sec=args.batch_retry_pause_sec,
            map_bundle_size=max(1, int(args.map_bundle_size)),
            use_split_map_agents=(not args.disable_split_map_agents),
            map_agent_count=max(1, int(args.map_agent_count)),
            patient_concurrency=args.patient_concurrency,
            concurrency=args.concurrency,
            request_delay_sec=max(0.0, float(args.request_delay_sec)),
            top_k=args.top_k,
            near_dup_hamming=args.near_dup_hamming,
            debug=args.debug,
            log_filename=args.log_filename,
            save_intermediate=args.save_intermediate,
        )
    )


if __name__ == "__main__":
    main()
