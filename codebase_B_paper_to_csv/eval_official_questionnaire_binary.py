from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_openai_map_module() -> Any:
    path = Path(__file__).with_name("109_ocr_map_patient_openai.py")
    spec = importlib.util.spec_from_file_location("ocr_map_patient_openai_109", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OFFICIAL_FAMILIES = ["MQ", "SSS", "ESS", "FSS", "BQ", "ISI", "RLS", "IRLS", "PSQI", "BDI", "QOL"]
RULE_PATTERNS = {
    "MQ": [r"아침 질문 사항"],
    "SSS": [r"Stanford Sleepiness Scale", r"Stanford sleepiness scale"],
    "ESS": [r"The Epworth Sleepiness Scale", r"Epworth Sleepiness Scale"],
    "FSS": [r"Fatigue Severity Scale", r"피로 정도에 대한 설문"],
    "BQ": [r"Berlin Questionnaire"],
    "ISI": [r"Insomnia Severity Index", r"불면증에 관한 설문", r"불면증에 관한 질문"],
    "RLS": [r"하지불안증후군/주기성사지운동증후군", r"Restless Legs Syndromes? and PLMS questions"],
    "IRLS": [r"하지불안증후군에 대한 설문"],
    "PSQI": [r"수면의 질 지수", r"Pittsburgh Sleep Quality Index", r"PITTSBURGH SLEEP QUALITY INDEX"],
    "BDI": [r"우울증에 관한 설문", r"Beck Depression Inventory", r"Beck depression inventory"],
    "QOL": [r"삶의 질 척도", r"WHOQOL-BREF"],
}
MULTIPAGE_FAMILIES = {"PSQI", "BDI", "QOL"}
GENERIC_BREAK_PATTERNS = [
    r"생활 습관",
    r"병력과 가족력",
    r"수면 습관",
    r"수면에 관한 설문지",
    r"수면다원검사 설문지",
    r"POLYSOMNOGRAPHY\s*\|\s*QUESTIONNAIRE",
    r"SLEEP QUESTIONNAIRE",
    r"SLEEP - WAKE QUESTIONNAIRE",
    r"Living habit",
]
PSQI_PAGE2_CUE_PATTERNS = [
    r"During the past month,\s*how would you rate your sleep quality overall\?",
    r"rate your sleep quality overall",
    r"지난 한달 동안,\s*당신의 전반적인 수면의 질은 어떠하였습니까",
]

LLM_SYSTEM = """You classify a single OCR page as either:
1. part or whole of an OFFICIAL questionnaire page
2. NOT an official questionnaire page

Official questionnaire families allowed:
- MQ: morning-after questionnaire titled '아침 질문 사항'
- SSS: Stanford Sleepiness Scale
- ESS: Epworth Sleepiness Scale
- FSS: Fatigue Severity Scale / 피로 정도에 대한 설문
- BQ: Berlin Questionnaire
- ISI: Insomnia Severity Index / 불면증에 관한 설문
- RLS: Restless Legs Syndromes and PLMS questions
- IRLS: 하지불안증후군에 대한 설문 / Restless Legs Syndrome Rating Scale
- PSQI: Pittsburgh Sleep Quality Index / 수면의 질 지수
- BDI: Beck Depression Inventory / 우울증에 관한 설문
- QOL: WHOQOL-BREF / 삶의 질 척도

Important:
- A page may still be official even if the title is missing, if the page content is clearly a continuation of one of the official questionnaires.
- Generic hospital intake forms, sleep history forms, living habit pages, family history pages, and sleep-wake questionnaires are NOT official questionnaires.

Return strict JSON only:
{"official":0 or 1,"family":"MQ|SSS|ESS|FSS|BQ|ISI|RLS|IRLS|PSQI|BDI|QOL|NON","reason":"short reason"}"""


@dataclass
class PageRow:
    patient: str
    bundle: str
    route: str
    source_image: str
    ocr_txt: str
    manual_official: int
    manual_family: str
    snippet: str


def load_labels(path: Path) -> list[PageRow]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                PageRow(
                    patient=row["patient"],
                    bundle=row["bundle"],
                    route=row["route"],
                    source_image=row["source_image"],
                    ocr_txt=row["ocr_txt"],
                    manual_official=int(row["manual_official"]),
                    manual_family=row["manual_family"],
                    snippet=row["snippet"],
                )
            )
        return rows


def bundle_sort_key(bundle: str) -> int:
    m = re.match(r"bundle_(\d+)", bundle)
    return int(m.group(1)) if m else 999999


def base_rule_family(text: str) -> str:
    for family, patterns in RULE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                return family
    return "NON"


def looks_like_generic_break(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in GENERIC_BREAK_PATTERNS)


def looks_like_psqi_page2(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in PSQI_PAGE2_CUE_PATTERNS)


def connected_rule_predictions(rows: list[PageRow]) -> dict[str, str]:
    preds: dict[str, str] = {}
    grouped: dict[str, list[PageRow]] = defaultdict(list)
    for row in rows:
        grouped[row.patient].append(row)

    for patient, patient_rows in grouped.items():
        active_family = "NON"
        for row in sorted(patient_rows, key=lambda r: bundle_sort_key(r.bundle)):
            text = row.snippet
            family = base_rule_family(text)
            if family != "NON":
                preds[row.bundle] = family
                active_family = family if family in MULTIPAGE_FAMILIES else "NON"
                continue
            if looks_like_generic_break(text):
                active_family = "NON"
                preds[row.bundle] = "NON"
                continue
            if active_family in MULTIPAGE_FAMILIES:
                if active_family == "PSQI" and not looks_like_psqi_page2(text):
                    active_family = "NON"
                    preds[row.bundle] = "NON"
                    continue
                preds[row.bundle] = active_family
                continue
            preds[row.bundle] = "NON"
    return preds


def metrics(true_official: list[int], pred_official: list[int]) -> dict[str, Any]:
    tp = sum(1 for t, p in zip(true_official, pred_official) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_official, pred_official) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_official, pred_official) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_official, pred_official) if t == 1 and p == 0)
    total = len(true_official)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "total": total,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


async def llm_predict(rows: list[PageRow], source_run: Path, model: str) -> dict[str, dict[str, Any]]:
    mod = _load_openai_map_module()
    llm = mod.RemoteOpenAIResponsesTextAgent(
        model_id=model,
        max_new_tokens=120,
        temperature=0.0,
        top_p=1.0,
        max_inflight=8,
        timeout_sec=120.0,
        max_retries=3,
        api_key_env="OPENAI_API_KEY",
    )

    async def one(row: PageRow) -> tuple[str, dict[str, Any]]:
        ocr_path = source_run / row.patient / "ocr_pages" / row.ocr_txt
        text = ocr_path.read_text(encoding="utf-8", errors="ignore")
        user = f"""Classify this single OCR page.\n\nRoute (for context only): {row.route}\nOCR page text:\n\"\"\"\n{text}\n\"\"\""""
        raw = await llm.atext(LLM_SYSTEM, user)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            obj = json.loads(m.group(0)) if m else {"official": 0, "family": "NON", "reason": "invalid_json"}
        fam = str(obj.get("family", "NON")).strip().upper()
        if fam not in OFFICIAL_FAMILIES:
            fam = "NON"
        off = 1 if str(obj.get("official", "0")).strip() in {"1", "true", "True"} and fam != "NON" else 0
        return row.bundle, {"official": off, "family": fam, "reason": str(obj.get("reason", "")).strip()}

    pairs = await asyncio.gather(*(one(row) for row in rows))
    return dict(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="official_questionnaire_labels_psqi_wrong_patients_20260318.csv")
    ap.add_argument(
        "--source-run",
        default="out_patient01to10_liveocr_gpt54_route_gpt54_map_gpt51multiroute_resolve_gpt54_resumable_20260317",
    )
    ap.add_argument("--out-dir", default="out_official_questionnaire_binary_test_20260318")
    ap.add_argument("--llm-model", default="gpt-5.1")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case1 = {row.bundle: base_rule_family(row.snippet) for row in labels}
    case2 = connected_rule_predictions(labels)
    case3 = asyncio.run(llm_predict(labels, Path(args.source_run), args.llm_model))

    records = []
    for row in labels:
        c1_family = case1[row.bundle]
        c2_family = case2[row.bundle]
        c3 = case3[row.bundle]
        records.append(
            {
                "patient": row.patient,
                "bundle": row.bundle,
                "route": row.route,
                "manual_official": row.manual_official,
                "manual_family": row.manual_family,
                "case1_official": 1 if c1_family != "NON" else 0,
                "case1_family": c1_family,
                "case2_official": 1 if c2_family != "NON" else 0,
                "case2_family": c2_family,
                "case3_official": c3["official"],
                "case3_family": c3["family"],
                "case3_reason": c3["reason"],
            }
        )

    summary = {}
    for case in ["case1", "case2", "case3"]:
        true = [r["manual_official"] for r in records]
        pred = [r[f"{case}_official"] for r in records]
        summary[case] = metrics(true, pred)

    with (out_dir / "page_level_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
