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
CDM_LAST_INCLUDED_KEY = "PSG_M_08_Wake"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

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
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


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


def _normalize_numeric_value(x: float) -> Any:
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return round(x, 4)


def _is_pure_numeric_string(s: str) -> bool:
    s = s.strip().replace(",", "")
    return re.fullmatch(r"-?\d+(?:\.\d+)?", s) is not None


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


def validate_extracted_json(obj: Dict[str, Any], retriever: "CDMRetriever") -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    valid: Dict[str, Any] = {}
    rejected: Dict[str, Dict[str, Any]] = {}

    for k, v in obj.items():
        key = str(k).strip()
        if key not in retriever.row_by_key:
            rejected[key] = {"value": v, "reason": "unknown_key"}
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
    new_votes = 0
    old_votes = 0

    for base in PSQI_BASE_GROUPS:
        for unit in ("HH", "MM"):
            old_k = f"{base}_{unit}"
            wk_k = f"{base}_{unit}_week"
            fr_k = f"{base}_{unit}_free"

            old_p = old_k in row and not _is_missing_value(row.get(old_k))
            wk_p = wk_k in row and not _is_missing_value(row.get(wk_k))
            fr_p = fr_k in row and not _is_missing_value(row.get(fr_k))

            if (wk_p or fr_p) and not old_p:
                new_votes += 1
            elif old_p and not (wk_p or fr_p):
                old_votes += 1
            elif old_p and wk_p and fr_p:
                if (_norm_cmp(row.get(wk_k)) != _norm_cmp(row.get(fr_k))) or (_norm_cmp(row.get(old_k)) != _norm_cmp(row.get(wk_k))):
                    new_votes += 1
                else:
                    old_votes += 1

    use_new_format = new_votes > old_votes

    for base in PSQI_BASE_GROUPS:
        for unit in ("HH", "MM"):
            old_k = f"{base}_{unit}"
            wk_k = f"{base}_{unit}_week"
            fr_k = f"{base}_{unit}_free"

            old_p = old_k in row and not _is_missing_value(row.get(old_k))
            wk_p = wk_k in row and not _is_missing_value(row.get(wk_k))
            fr_p = fr_k in row and not _is_missing_value(row.get(fr_k))

            if use_new_format:
                if old_p:
                    if not wk_p:
                        row[wk_k] = row.get(old_k)
                    if not fr_p:
                        row[fr_k] = row.get(old_k)
                if old_k in row:
                    row[old_k] = None
            else:
                if not old_p:
                    if wk_p:
                        row[old_k] = row.get(wk_k)
                    elif fr_p:
                        row[old_k] = row.get(fr_k)
                if wk_k in row:
                    row[wk_k] = None
                if fr_k in row:
                    row[fr_k] = None

            target_keys = [old_k, wk_k, fr_k]
            for tk in target_keys:
                if tk not in row or _is_missing_value(row.get(tk)):
                    continue
                iv = _coerce_int(row.get(tk))
                if iv is None:
                    continue
                if unit == "MM":
                    if iv == 60:
                        iv = 0
                    row[tk] = iv if 0 <= iv <= 59 else None
                elif unit == "HH":
                    if base in {"PSQI_01_BedIn", "PSQI_03_BedOut"}:
                        row[tk] = _normalize_psqi_clock_hour(base, iv)
                    else:
                        row[tk] = iv


def apply_phx_default_rules(row: Dict[str, Any]) -> None:
    phx_cols = [k for k in row.keys() if k.startswith("PHx_")]
    if not phx_cols:
        return

    any_answered = any(not _is_missing_value(row.get(k)) for k in phx_cols)
    if not any_answered:
        return

    for k in phx_cols:
        if _is_missing_value(row.get(k)):
            row[k] = 0
            continue
        iv = _coerce_int(row.get(k))
        if iv is not None:
            row[k] = iv


def merge_page_results(page_results: List["PageResult"]) -> Tuple[Dict[str, Any], Dict[str, List[Any]], Dict[str, List[Dict[str, Any]]]]:
    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pr in page_results:
        for k, v in pr.valid_json.items():
            by_key[k].append({"image": pr.image_name, "value": v})

    merged: Dict[str, Any] = {}
    conflicts: Dict[str, List[Any]] = {}
    provenance: Dict[str, List[Dict[str, Any]]] = {}

    for k, entries in by_key.items():
        provenance[k] = entries

        tokens = [_value_token(e["value"]) for e in entries]
        counts = Counter(tokens)
        first_idx: Dict[str, int] = {}
        for idx, tk in enumerate(tokens):
            first_idx.setdefault(tk, idx)
        winner = max(counts.keys(), key=lambda tk: (counts[tk], -first_idx[tk]))

        for e in entries:
            if _value_token(e["value"]) == winner:
                merged[k] = e["value"]
                break

        unique_values: List[Any] = []
        seen_tokens = set()
        for e in entries:
            tk = _value_token(e["value"])
            if tk in seen_tokens:
                continue
            seen_tokens.add(tk)
            unique_values.append(e["value"])
        if len(unique_values) > 1:
            conflicts[k] = unique_values

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


# -----------------------------
# CDM TF-IDF retriever (local RAG)
# -----------------------------
@dataclass
class CDMRow:
    key: str
    desc: str
    format_range: str
    options: Dict[str, str]  # code -> label


@dataclass
class PageResult:
    image_name: str
    ocr_text: str
    raw_json: Dict[str, Any]
    valid_json: Dict[str, Any]
    rejected_fields: Dict[str, Dict[str, Any]]


class CDMRetriever:
    """
    Local retrieval over CDM rows using TF-IDF char-ngrams (good for Korean/English mix).
    Expects cdm.csv columns:
      - 'csv key'
      - '설명'
      - 'Format/Range'
      - option columns named like '0','1','2',...
    """
    def __init__(self, cdm_csv_path: Path):
        self.cdm_df = pd.read_csv(cdm_csv_path)

        option_cols = [c for c in self.cdm_df.columns if re.fullmatch(r"\d+", str(c))]

        self.rows: List[CDMRow] = []
        self._texts: List[str] = []

        cutoff_found = False
        for _, r in self.cdm_df.iterrows():
            key = str(r.get("csv key", "")).strip()
            if not key or key.lower() == "nan":
                continue

            desc_v = r.get("설명", "")
            fr_v = r.get("Format/Range", "")
            desc = "" if pd.isna(desc_v) else str(desc_v).strip()
            fr = "" if pd.isna(fr_v) else str(fr_v).strip()

            opts: Dict[str, str] = {}
            for c in option_cols:
                val = r.get(c)
                if pd.isna(val):
                    continue
                label = str(val).strip()
                if label:
                    opts[str(c)] = label

            row = CDMRow(key=key, desc=desc, format_range=fr, options=opts)
            self.rows.append(row)

            opt_str = " | ".join([f"{code}:{label}" for code, label in sorted(opts.items(), key=lambda x: int(x[0]))])
            self._texts.append(f"KEY={key}\nDESC={desc}\nFORMAT={fr}\nOPTIONS={opt_str}")

            if key == CDM_LAST_INCLUDED_KEY:
                cutoff_found = True
                break

        if not cutoff_found:
            logger.warning(
                "CDM cutoff key %s was not found. Using all available CDM rows (%d).",
                CDM_LAST_INCLUDED_KEY,
                len(self.rows),
            )
        else:
            logger.info(
                "CDM rows truncated at %s. Using first %d rows.",
                CDM_LAST_INCLUDED_KEY,
                len(self.rows),
            )

        self.key_set = {r.key for r in self.rows}
        self.row_by_key = {r.key: r for r in self.rows}
        self.rows_by_prefix: Dict[str, List[CDMRow]] = defaultdict(list)
        for row in self.rows:
            prefix = row.key.split("_", 1)[0]
            self.rows_by_prefix[prefix].append(row)

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self._texts)
        self._full_cdm_prompt_block = self._build_full_cdm_prompt_block()

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
        if compact:
            block = (
                f"- {row.key} | desc={_clip_prompt_text(row.desc, 80)} | "
                f"format={_clip_prompt_text(row.format_range, 40)} | "
                f"options={_clip_prompt_text(opt_str, 180)}\n"
            )
        else:
            block = (
                f"- {row.key}\n"
                f"  desc: {_clip_prompt_text(row.desc, 220)}\n"
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


OCR_SYSTEM = """You are an OCR/transcription assistant for sleep clinic questionnaire documents (Korean/English mixed).
Task:
- Transcribe ALL visible printed and handwritten text faithfully in natural reading order.
- Preserve question/item numbers, option numbers/symbols, labels, handwritten values, and units.
- For checkboxes / radio buttons / circled options / check marks / X marks / underlined choices: explicitly state which option is selected.
- A visible mark attached to an option counts as a selected answer. Do not treat an unmarked printed option as selected.
- If options include numeric codes or scales (e.g., 0/1/2/3 or 1-7), include the selected number and label when visible.
- For tables or scale rows, keep each row label and make the chosen value explicit.
- If a handwritten answer is written on a blank line or next to a question, place it inline with that item.
- Prefer explicit visual marks over inferred meaning from nearby text.
- If more than one option is visibly marked, write `(multiple marked: ...)`.
- If an answer field is blank, write `(not filled)`.
- If text or a selected option is not confidently readable, write `(unclear)` or `(unclear selection)` instead of guessing.
- You do NOT need to preserve exact table layout/shape, but keep question text, options, and selected answers clearly associated.
- It is acceptable to add a short clarification such as `Selected: ③ 같다` or `Selected: 아니오` when that makes the chosen answer explicit.
Output: plain text only (no JSON, no markdown, no explanations)."""

OCR_USER_PROMPT = (
    "Please transcribe this questionnaire image. Pay special attention to circled, checked, "
    "or otherwise selected answers, handwritten entries, and scale choices. Do not guess "
    "uncertain marks."
)


MAP_SYSTEM = """You are a clinical data extraction assistant.
You will be given:
1) OCR text from one patient (may include multiple questionnaire pages)
2) Candidate CDM fields (keys) with descriptions/ranges/options

Return ONE flat JSON mapping CDM keys -> extracted values.
Rules:
- Use ONLY keys from the candidate list.
- Maximize recall: if OCR clearly provides a value and there is a matching candidate key, include it.
- Do NOT invent values. If absent, not filled, unclear, or contradictory, omit the key.
- If options are coded (0/1/2/3...), output the numeric code only.
- For Yes/No type options, map to the coded option using the option labels.
- For numeric fields, output numbers without unit text.
- Dates must follow CDM format/range (typically YYYYMMDD).
- If CDM separates HH/MM, output numeric HH and MM keys separately.
- Keep identifiers as-is when textual (e.g., PSG_No, Database_ID).
- Do not translate or transliterate person names; copy the script exactly as it appears in OCR.
- Prefer explicitly selected/circled/checked answers over inferred narrative statements.
- Do not output null, empty strings, arrays, comments, or explanations.
- Output JSON object only.
"""


MAP_RECALL_SYSTEM = """You are a clinical data extraction assistant performing a second-pass recall step.
You will be given:
1) OCR text
2) Candidate CDM fields
3) Existing extracted JSON

Task:
- Return ONLY additional key-value pairs that are clearly supported by OCR text.
- Do NOT repeat keys already present in existing JSON.
- Use ONLY keys from the candidate list.
- Do not translate or transliterate person names; preserve OCR script.
- Follow coded options/range/date rules exactly.
- If uncertain, omit the key.
Output JSON object only.
"""


# -----------------------------
# Gemini model builder
# -----------------------------
def build_gemini() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")
    model = os.getenv("GEMINI_MODEL", "gemini-3.0-vision")  # set to your real Gemini 3 model name
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
) -> Dict[str, Any]:
    user = build_map_user_prompt(ocr_text, candidates_block)
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
) -> Dict[str, Any]:
    user = build_map_recall_user_prompt(ocr_text, candidates_block, existing_json)
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


def build_map_user_prompt(ocr_text: str, candidates_block: str) -> str:
    return f"""OCR TEXT:
\"\"\"{ocr_text[:12000]}\"\"\"

CANDIDATE CDM FIELDS (use ONLY these keys):
{candidates_block}

Extract as many clearly supported keys as possible.
Return ONE JSON object only."""


def build_map_recall_user_prompt(ocr_text: str, candidates_block: str, existing_json: Dict[str, Any]) -> str:
    return f"""OCR TEXT:
\"\"\"{ocr_text[:12000]}\"\"\"

EXISTING JSON (do not repeat these keys):
{json.dumps(existing_json, ensure_ascii=False)}

CANDIDATE CDM FIELDS (use ONLY these keys):
{candidates_block}

Return ONLY additional key-value pairs as ONE JSON object."""


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


def build_map_batch_request(ocr_text: str, candidates_block: str) -> Dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": build_map_user_prompt(ocr_text, candidates_block)}]}],
        "systemInstruction": {"parts": [{"text": MAP_SYSTEM}]},
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }


def build_map_recall_batch_request(ocr_text: str, candidates_block: str, existing_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_map_recall_user_prompt(ocr_text, candidates_block, existing_json)}],
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
    return "\n".join([t for t in texts if t]).strip()


def build_patient_ocr_bundle(ocr_items: List[Tuple[str, str]]) -> str:
    parts: List[str] = []
    for image_name, text in ocr_items:
        parts.append(f"[PAGE: {image_name}]\n{text}")
    return "\n\n".join(parts).strip()


# -----------------------------
# Main per-image: fixed two-stage flow
# -----------------------------
async def image_to_cdm_json(
    llm: ChatGoogleGenerativeAI,
    retriever: CDMRetriever,
    image_path: Path,
    top_k: int = 80,
) -> PageResult:
    t0 = time.perf_counter()
    logger.debug("Page start: image=%s", image_path.name)

    t_ocr0 = time.perf_counter()
    ocr_text = await gemini_ocr(llm, image_path)
    t_ocr = time.perf_counter() - t_ocr0

    # Full-CDM mode: include all CDM rows in mapping prompt.
    candidates_block = retriever.full_cdm_prompt_block()

    t_map0 = time.perf_counter()
    raw_obj = await gemini_map_to_json(llm, ocr_text, candidates_block)
    valid_obj, rejected_fields = validate_extracted_json(raw_obj, retriever)

    backfill_additions, backfill_rejected = apply_core_backfill(valid_obj, retriever, ocr_text)
    for k, v in backfill_additions.items():
        valid_obj[k] = v
        raw_obj.setdefault(k, v)
    if backfill_rejected:
        for k, meta in backfill_rejected.items():
            rejected_fields.setdefault(k, meta)

    recall_added_keys = 0
    if should_run_recall_pass(ocr_text, valid_obj):
        recall_block = retriever.full_cdm_prompt_block()
        add_raw = await gemini_map_additional_json(
            llm=llm,
            ocr_text=ocr_text,
            candidates_block=recall_block,
            existing_json=valid_obj,
        )
        add_valid, add_rejected = validate_extracted_json(add_raw, retriever)
        for k, v in add_valid.items():
            if k not in valid_obj:
                valid_obj[k] = v
                recall_added_keys += 1
        for k, meta in add_rejected.items():
            rejected_fields.setdefault(k, meta)
        for k, v in add_raw.items():
            raw_obj.setdefault(k, v)

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
        len(backfill_additions),
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
    apply_phx_default_rules(row)

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
            (out_dir / "intermediate" / patient_name / f"{stem}.json").write_text(
                json.dumps(pr.valid_json, ensure_ascii=False, indent=2),
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
        "conflicts": conflicts,
        "provenance": provenance,
        "duplicates": duplicates,
        "validation_rejections": validation_rejections,
        "page_errors": page_errors,
    }


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


# -----------------------------
# Patient processing
# -----------------------------
async def process_one_patient(
    patient_dir: Path,
    llm: ChatGoogleGenerativeAI,
    retriever: CDMRetriever,
    output_columns: List[str],
    concurrency: int,
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
        }

    logger.info("Patient %s: %d images", patient_dir.name, len(images))
    logger.debug(
        "Patient %s images: %s",
        patient_dir.name,
        ", ".join([img.name for img in images]),
    )

    # Stage 1: OCR all pages first.
    ocr_tasks = [gemini_ocr(llm=llm, image_path=img) for img in images]
    ocr_outputs = await gather_with_concurrency(concurrency, ocr_tasks)

    ocr_items: List[Tuple[str, str]] = []
    page_errors: List[Dict[str, str]] = []
    for img, out in zip(images, ocr_outputs):
        if isinstance(out, Exception):
            logger.warning("Page failed: %s/%s (%s)", patient_dir.name, img.name, out)
            page_errors.append({"image": img.name, "error_type": type(out).__name__, "error": str(out)})
            continue
        txt = str(out).strip()
        if not txt:
            logger.warning("Page failed: %s/%s (Empty OCR text)", patient_dir.name, img.name)
            page_errors.append({"image": img.name, "error_type": "EmptyOCR", "error": "Empty OCR text"})
            continue
        ocr_items.append((img.name, txt))

        if save_intermediate:
            inter_dir = out_dir / "intermediate" / patient_dir.name
            inter_dir.mkdir(parents=True, exist_ok=True)
            (inter_dir / f"{Path(img.name).stem}.txt").write_text(txt, encoding="utf-8")

    page_results: List[PageResult] = []
    if ocr_items:
        # Stage 2: one MAP call for the entire patient bundle.
        full_ocr_text = build_patient_ocr_bundle(ocr_items)
        full_candidates_block = retriever.full_cdm_prompt_block()
        try:
            raw_obj = await gemini_map_to_json(llm=llm, ocr_text=full_ocr_text, candidates_block=full_candidates_block)
            valid_obj, rejected_fields = validate_extracted_json(raw_obj, retriever)

            backfill_additions, backfill_rejected = apply_core_backfill(valid_obj, retriever, full_ocr_text)
            for bk, bv in backfill_additions.items():
                valid_obj[bk] = bv
                raw_obj.setdefault(bk, bv)
            for bk, meta in backfill_rejected.items():
                rejected_fields.setdefault(bk, meta)

            page_results.append(
                PageResult(
                    image_name="__patient_combined__",
                    ocr_text=full_ocr_text,
                    raw_json=raw_obj,
                    valid_json=valid_obj,
                    rejected_fields=rejected_fields,
                )
            )
        except Exception as e:
            logger.warning("Patient MAP failed: %s (%s)", patient_dir.name, e)
            page_errors.append({"image": "__patient_combined__", "error_type": type(e).__name__, "error": str(e)})

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
    output_columns: List[str],
    output_dir: Path,
    top_k: int,
    near_dup_hamming: int,
    batch_model: str,
    batch_poll_interval_sec: int,
    batch_timeout_sec: int,
    batch_image_max_side: int,
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
            }
            continue

        logger.info("Patient %s: %d images", pdir.name, len(images))
        contexts[pdir.name] = {
            "started": started,
            "duplicates": duplicates,
            "page_errors": page_errors,
            "page_results": [],
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
            for p in ocr_plan:
                key = p["key"]
                ctx = contexts[p["patient"]]
                rec = ocr_records.get(key)
                if rec is None:
                    ctx["page_errors"].append({"image": p["image_path"].name, "error_type": "MissingResult", "error": "Missing OCR batch result"})
                    continue
                err = batch_record_error_message(rec)
                if err:
                    ctx["page_errors"].append({"image": p["image_path"].name, "error_type": "BatchError", "error": err})
                    continue
                ocr_text = batch_record_response_text(rec)
                if not ocr_text:
                    ctx["page_errors"].append({"image": p["image_path"].name, "error_type": "EmptyOCR", "error": "Empty OCR text from batch response"})
                    continue
                ocr_text_by_key[key] = ocr_text

                if save_intermediate:
                    inter_dir = output_dir / "intermediate" / p["patient"]
                    inter_dir.mkdir(parents=True, exist_ok=True)
                    (inter_dir / f"{Path(p['image_path'].name).stem}.txt").write_text(ocr_text, encoding="utf-8")

            map_plan = [p for p in ocr_plan if p["key"] in ocr_text_by_key]
            if map_plan:
                full_candidates_block = retriever.full_cdm_prompt_block()
                patient_ocr_items: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
                for p in map_plan:
                    patient_ocr_items[p["patient"]].append((p["image_path"].name, ocr_text_by_key[p["key"]]))

                map_requests_path = tmp_dir / f"map_requests_{uuid4().hex}.jsonl"
                with map_requests_path.open("w", encoding="utf-8") as f:
                    for patient_name in sorted(patient_ocr_items.keys()):
                        bundle = build_patient_ocr_bundle(patient_ocr_items[patient_name])
                        req = build_map_batch_request(bundle, full_candidates_block)
                        f.write(json.dumps({"key": patient_name, "request": req}, ensure_ascii=False) + "\n")

                map_file_mb = map_requests_path.stat().st_size / (1024 * 1024)
                logger.info(
                    "Submitting MAP batch with %d requests (input_jsonl=%.2f MB)",
                    len(patient_ocr_items),
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

                for patient_name in sorted(patient_ocr_items.keys()):
                    ctx = contexts[patient_name]
                    rec = map_records.get(patient_name)
                    if rec is None:
                        ctx["page_errors"].append(
                            {
                                "image": "__patient_combined__",
                                "error_type": "MissingResult",
                                "error": "Missing MAP batch result",
                            }
                        )
                        continue
                    err = batch_record_error_message(rec)
                    if err:
                        ctx["page_errors"].append(
                            {
                                "image": "__patient_combined__",
                                "error_type": "BatchError",
                                "error": err,
                            }
                        )
                        continue

                    raw_text = batch_record_response_text(rec)
                    if not raw_text:
                        ctx["page_errors"].append(
                            {
                                "image": "__patient_combined__",
                                "error_type": "EmptyMapOutput",
                                "error": "Empty MAP response text",
                            }
                        )
                        continue
                    try:
                        raw_json = safe_extract_json(raw_text)
                    except Exception as e:
                        ctx["page_errors"].append(
                            {
                                "image": "__patient_combined__",
                                "error_type": type(e).__name__,
                                "error": str(e),
                            }
                        )
                        continue

                    valid_json, rejected_fields = validate_extracted_json(raw_json, retriever)
                    patient_ocr_bundle = build_patient_ocr_bundle(patient_ocr_items[patient_name])
                    backfill_additions, backfill_rejected = apply_core_backfill(valid_json, retriever, patient_ocr_bundle)
                    for bk, bv in backfill_additions.items():
                        valid_json[bk] = bv
                        raw_json.setdefault(bk, bv)
                    for bk, meta in backfill_rejected.items():
                        rejected_fields.setdefault(bk, meta)
                    ctx["page_results"].append(
                        PageResult(
                            image_name="__patient_combined__",
                            ocr_text=patient_ocr_bundle,
                            raw_json=raw_json,
                            valid_json=valid_json,
                            rejected_fields=rejected_fields,
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
    concurrency: int,
    patient_concurrency: int,
    top_k: int,
    near_dup_hamming: int,
    debug: bool,
    log_filename: str,
    save_intermediate: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir=output_dir, debug=debug, log_filename=log_filename)
    t_run0 = time.perf_counter()

    # Column order: match example.csv exactly
    example_df = pd.read_csv(example_csv)
    output_columns = list(example_df.columns)

    patient_dirs = iter_patient_folders(input_root)
    logger.info("Found %d patient folders", len(patient_dirs))
    logger.info(
        (
            "Run config: use_batch_api=%s, patient_concurrency=%d, ocr_page_concurrency=%d, top_k=%d (ignored in patient-level MAP mode), "
            "near_dup_hamming=%d, save_intermediate=%s"
        ),
        use_batch_api,
        patient_concurrency,
        concurrency,
        top_k,
        near_dup_hamming,
        save_intermediate,
    )

    retriever = CDMRetriever(cdm_csv)

    results: List[Dict[str, Any]] = []
    patient_failures = 0
    if use_batch_api:
        logger.info(
            "Batch mode note: --concurrency and --patient_concurrency are ignored in batch execution.",
        )
        chosen_batch_model = batch_model.strip() or os.getenv("GEMINI_BATCH_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        logger.info(
            "Batch API mode enabled: model=%s, poll_interval_sec=%d, timeout_sec=%d, image_max_side=%d",
            chosen_batch_model,
            batch_poll_interval_sec,
            batch_timeout_sec,
            batch_image_max_side,
        )
        results = process_patients_with_batch_api(
            input_root=input_root,
            retriever=retriever,
            output_columns=output_columns,
            output_dir=output_dir,
            top_k=top_k,
            near_dup_hamming=near_dup_hamming,
            batch_model=chosen_batch_model,
            batch_poll_interval_sec=batch_poll_interval_sec,
            batch_timeout_sec=batch_timeout_sec,
            batch_image_max_side=batch_image_max_side,
            save_intermediate=save_intermediate,
        )
        for res in results:
            write_patient_outputs(output_dir=output_dir, patient_name=str(res["patient"]), res=res, output_columns=output_columns)
    else:
        llm = build_gemini()

        async def _process_patient_slot(idx: int, pdir: Path) -> Dict[str, Any]:
            logger.info("Processing patient %d/%d: %s", idx, len(patient_dirs), pdir.name)
            res = await process_one_patient(
                patient_dir=pdir,
                llm=llm,
                retriever=retriever,
                output_columns=output_columns,
                concurrency=concurrency,
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
    ap.add_argument("--cdm_csv", type=str, required=True, help="Path to cdm.csv")
    ap.add_argument("--example_csv", type=str, required=True, help="Path to example.csv (column order template)")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for CSV files")
    ap.add_argument("--use_batch_api", action="store_true", help="Use Gemini Batch API (OCR per page, then one MAP request per patient)")
    ap.add_argument("--batch_model", type=str, default="", help="Gemini model for batch mode (default: GEMINI_BATCH_MODEL or GEMINI_MODEL)")
    ap.add_argument("--batch_poll_interval_sec", type=int, default=15, help="Polling interval for batch job status")
    ap.add_argument("--batch_timeout_sec", type=int, default=7200, help="Timeout for each batch job in seconds")
    ap.add_argument("--batch_image_max_side", type=int, default=1600, help="Max image side length for batch OCR requests")
    ap.add_argument("--patient_concurrency", type=int, default=1, help="Number of patient folders processed in parallel")
    ap.add_argument("--concurrency", type=int, default=3, help="Parallelism for per-image OCR runs in non-batch mode")
    ap.add_argument("--top_k", type=int, default=220, help="Legacy option (ignored in patient-level MAP mode)")
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
            patient_concurrency=args.patient_concurrency,
            concurrency=args.concurrency,
            top_k=args.top_k,
            near_dup_hamming=args.near_dup_hamming,
            debug=args.debug,
            log_filename=args.log_filename,
            save_intermediate=args.save_intermediate,
        )
    )


if __name__ == "__main__":
    main()
