from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_MODULE_PATH = SCRIPT_DIR / "103_paper_to_cdm_SA.py"


def _load_pipeline_module() -> Any:
    module_name = "paper_to_cdm_sa_103"
    spec = importlib.util.spec_from_file_location(module_name, PIPELINE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load pipeline module from {PIPELINE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PIPELINE_MOD = _load_pipeline_module()
PATIENT_MAP_CATEGORIES: Tuple[str, ...] = tuple(getattr(PIPELINE_MOD, "PATIENT_MAP_CATEGORIES", ()))
MAP_CATEGORY_RULE_PATTERNS: Dict[str, Tuple[str, ...]] = dict(getattr(PIPELINE_MOD, "MAP_CATEGORY_RULE_PATTERNS", {}))


ITEM_CUE_PATTERNS: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "psqi": (
        (r"지난 한달 동안[, ]+몇 시에 잠자리에 들었", 1),
        (r"지난 한달 동안[, ]+밤마다 잠드는데 얼마나 오래 걸렸", 1),
        (r"지난 한달 동안[, ]+아침에 몇 시에 일어났", 1),
        (r"지난 한달 동안[, ]+실제로 잠잔 시간", 1),
    ),
    "sss": (
        (r"당신은 지금 얼마나 졸립다고 생각하십니까", 1),
        (r"최상의 상태는 아니지만", 1),
    ),
    "ess": (
        (r"앉아서 책을 읽을 때", 1),
        (r"텔레비전을 볼 때", 1),
        (r"극장이나 회의석상", 1),
        (r"버스나 택시", 1),
        (r"누군가에게 말을 하고 있을 때", 1),
        (r"점심식사 후 조용히 앉아 있을 때", 1),
    ),
    "fss": (
        (r"피로하면 의욕이 없어진다", 1),
        (r"운동을 하면 피곤해진다", 1),
        (r"쉽게 피곤해진다", 1),
        (r"피로 때문에 신체활동이 감소된다", 1),
    ),
    "berlin": (
        (r"berlin questionnaire", 1),
        (r"코골", 1),
        (r"고혈압", 1),
    ),
    "isi": (
        (r"잠들기 어렵", 1),
        (r"잠을 유지하기 어렵", 1),
        (r"너무 일찍 잠에서 깹", 1),
    ),
    "rls": (
        (r"하지불안", 1),
        (r"restless legs", 1),
        (r"다리를 움직이", 1),
    ),
    "rbd": (
        (r"렘수면행동장애", 1),
        (r"rbdsq", 1),
        (r"수면 중 이상행동", 1),
        (r"꿈을 많이 꾸십니까", 1),
        (r"고약한 잠버릇", 1),
        (r"꿈에서의 행동", 1),
        (r"잠자는 동안 몸부림", 1),
        (r"수면 중 소리를 지르", 1),
        (r"팔을 휘두르", 1),
        (r"신경계 질환이 있다", 1),
    ),
    "phq": (
        (r"patient health questionnaire", 1),
        (r"\bphq(?:-?9)?\b", 1),
        (r"기분이 가라앉거나 우울하거나 희망이 없", 1),
        (r"평소 하던 일에 대한 흥미가 없어지거나 즐거움을 느끼지 못", 1),
        (r"차라리 죽는 것이 낫겠다고", 1),
    ),
    "bdi": (
        (r"beck depression inventory", 1),
        (r"우울증에 관한 설문", 1),
        (r"슬프", 1),
    ),
    "qol": (
        (r"whoqol", 1),
        (r"삶의 질 척도", 1),
        (r"quality of life", 1),
    ),
}


@dataclass
class ValidationRecord:
    patient: str
    category: str
    present_in_split: bool
    chars: int
    exact_line_coverage: str
    provenance_pass: bool
    evidence_in_merged: bool
    structure_pass: bool
    overall_pass: bool
    reasons: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_nonempty_lines(text: str) -> Iterable[str]:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line:
            yield line


def _looks_informative_line(line: str) -> bool:
    if len(line) >= 8:
        return True
    if re.search(r"[A-Za-z가-힣]", line) and len(line) >= 4:
        return True
    return False


def _find_missing_verbatim_lines(category_text: str, merged_text: str) -> Tuple[int, int, List[str]]:
    informative = 0
    found = 0
    missing: List[str] = []
    for line in _iter_nonempty_lines(category_text):
        if not _looks_informative_line(line):
            continue
        informative += 1
        if line in merged_text:
            found += 1
        else:
            missing.append(line[:160])
    return informative, found, missing


def _category_has_evidence_in_merged(category: str, merged_text: str) -> bool:
    patterns = MAP_CATEGORY_RULE_PATTERNS.get(category, ())
    title_hits = 0
    for pattern in patterns:
        if re.search(pattern, merged_text, flags=re.I):
            title_hits += 1
    if title_hits:
        return True
    cue_specs = ITEM_CUE_PATTERNS.get(category, ())
    cue_hits = 0
    for pattern, _min_hits in cue_specs:
        if re.search(pattern, merged_text, flags=re.I):
            cue_hits += 1
    if category == "berlin":
        return cue_hits >= 2
    if category == "rbd":
        return cue_hits >= 2
    if category == "phq":
        return cue_hits >= 2
    if cue_hits:
        return True
    return False


def _check_category_structure(category: str, category_text: str, merged_text: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    text = str(category_text or "").strip()
    if not text:
        if _category_has_evidence_in_merged(category, merged_text):
            return False, ["missing_category_text_despite_merged_evidence"]
        return True, []

    if "[SOURCE_IMAGE:" in text:
        reasons.append("contains_source_image_marker")

    if category == "basic_questionnaire":
        if len(text) < 80:
            reasons.append("too_short_for_basic_questionnaire")
        return not reasons, reasons

    if category == "psg":
        psg_patterns = MAP_CATEGORY_RULE_PATTERNS.get(category, ())
        has_psg_marker = any(re.search(pattern, text, flags=re.I) for pattern in psg_patterns)
        if not has_psg_marker:
            reasons.append("missing_psg_marker")
        if len(text) < 120:
            reasons.append("too_short_for_psg")
        return not reasons, reasons

    title_patterns = MAP_CATEGORY_RULE_PATTERNS.get(category, ())
    cue_specs = ITEM_CUE_PATTERNS.get(category, ())
    hit_count = 0
    if cue_specs:
        for pattern, _min_hits in cue_specs:
            if re.search(pattern, text, flags=re.I):
                hit_count += 1
        if hit_count == 0:
            reasons.append("missing_category_item_cues")

    has_title_or_primary_pattern = bool(
        title_patterns and any(re.search(pattern, text, flags=re.I) for pattern in title_patterns)
    )
    if category == "rbd":
        if not has_title_or_primary_pattern and hit_count < 2:
            reasons.append("missing_category_title_or_primary_pattern")
    else:
        if title_patterns and not has_title_or_primary_pattern:
            reasons.append("missing_category_title_or_primary_pattern")

    if len(text) <= 3:
        reasons.append("collapsed_to_scalar_or_tiny_text")

    return not reasons, reasons


def validate_patient_split(patient_dir: Path) -> List[ValidationRecord]:
    merged_paths = sorted(patient_dir.glob("*_ocr_merged.txt"))
    if not merged_paths:
        raise FileNotFoundError(f"No merged OCR text found in {patient_dir}")
    merged_text = merged_paths[0].read_text(encoding="utf-8")

    split_payload = _read_json(patient_dir / "category_split_result.json")
    records = split_payload if isinstance(split_payload, list) else split_payload.get("category_records", [])
    by_category: Dict[str, Dict[str, Any]] = {str(r.get("category") or ""): r for r in records}

    out: List[ValidationRecord] = []
    for category in PATIENT_MAP_CATEGORIES:
        record = by_category.get(category, {})
        text = str(record.get("merged_text") or "")
        present = bool(text.strip())
        informative_total, informative_found, missing_lines = _find_missing_verbatim_lines(text, merged_text)
        provenance_pass = not missing_lines and "[SOURCE_IMAGE:" not in text
        evidence_in_merged = _category_has_evidence_in_merged(category, merged_text)
        structure_pass, structure_reasons = _check_category_structure(category, text, merged_text)

        reasons: List[str] = []
        if missing_lines:
            reasons.append("contains_nonverbatim_lines")
            reasons.extend([f"missing_line:{line}" for line in missing_lines[:3]])
        reasons.extend(structure_reasons)
        if not present and evidence_in_merged:
            reasons.append("empty_despite_merged_evidence")

        overall_pass = provenance_pass and structure_pass and not (not present and evidence_in_merged)
        coverage = f"{informative_found}/{informative_total}" if informative_total else "0/0"
        out.append(
            ValidationRecord(
                patient=patient_dir.name,
                category=category,
                present_in_split=present,
                chars=len(text),
                exact_line_coverage=coverage,
                provenance_pass=provenance_pass,
                evidence_in_merged=evidence_in_merged,
                structure_pass=structure_pass,
                overall_pass=overall_pass,
                reasons=" | ".join(reasons),
            )
        )
    return out


def write_reports(run_root: Path, records: Sequence[ValidationRecord], output_dir: Optional[Path]) -> Path:
    outdir = output_dir or (run_root / "split_validation_20260324")
    outdir.mkdir(parents=True, exist_ok=True)

    detail_path = outdir / "split_validation_detail.csv"
    with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "category",
                "present_in_split",
                "chars",
                "exact_line_coverage",
                "provenance_pass",
                "evidence_in_merged",
                "structure_pass",
                "overall_pass",
                "reasons",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)

    summary_rows: List[Dict[str, Any]] = []
    for category in PATIENT_MAP_CATEGORIES:
        rows = [r for r in records if r.category == category]
        passed = sum(1 for r in rows if r.overall_pass)
        present = sum(1 for r in rows if r.present_in_split)
        evidence = sum(1 for r in rows if r.evidence_in_merged)
        summary_rows.append(
            {
                "category": category,
                "patients_total": len(rows),
                "split_present": present,
                "merged_evidence": evidence,
                "validation_pass": passed,
                "validation_fail": len(rows) - passed,
                "pass_rate": f"{(passed / len(rows) * 100):.2f}%" if rows else "",
            }
        )

    summary_path = outdir / "split_validation_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "run_root": str(run_root),
        "patients_checked": sorted({r.patient for r in records}),
        "categories_checked": list(PATIENT_MAP_CATEGORIES),
        "overall_pass_count": sum(1 for r in records if r.overall_pass),
        "overall_fail_count": sum(1 for r in records if not r.overall_pass),
        "summary_csv": str(summary_path),
        "detail_csv": str(detail_path),
    }
    (outdir / "split_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outdir


def discover_patient_dirs(run_root: Path) -> List[Path]:
    return sorted(
        [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("Patient_")],
        key=lambda p: p.name,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate category split outputs using pure-Python checks only.")
    ap.add_argument("--run_root", type=Path, required=True, help="Run root containing Patient_* directories")
    ap.add_argument("--output_dir", type=Path, default=None, help="Where to write validation CSV/JSON files")
    args = ap.parse_args()

    patient_dirs = discover_patient_dirs(args.run_root)
    if not patient_dirs:
        raise SystemExit(f"No Patient_* directories found under {args.run_root}")

    all_records: List[ValidationRecord] = []
    for patient_dir in patient_dirs:
        all_records.extend(validate_patient_split(patient_dir))

    outdir = write_reports(args.run_root, all_records, args.output_dir)
    print(f"Validation written to: {outdir}")


if __name__ == "__main__":
    main()
