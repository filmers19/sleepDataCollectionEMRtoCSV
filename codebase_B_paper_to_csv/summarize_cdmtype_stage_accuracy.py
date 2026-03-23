#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CDM_CSV = ROOT / "cdm_revised.csv"
EXAMPLE_CSV = ROOT / "example.csv"
STAGES_CSV = ROOT / "stages.csv"


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "n/a", "na"}


def _normalize_semantic_value(value: Any) -> Any:
    if _is_missing_value(value):
        return None
    text = str(value).strip()
    numeric = text.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric):
        try:
            dec = Decimal(numeric).normalize()
            rendered = format(dec, "f")
            return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
        except InvalidOperation:
            pass
    return re.sub(r"\s+", " ", text)


def _load_final_row(patient_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(patient_csv, dtype=object)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): (None if _is_missing_value(v) else v) for k, v in row.items()}


def _metric_dict(row: Dict[str, Any], ref_row: pd.Series, keys: Iterable[str], conflict_keys: set[str]) -> Dict[str, Any]:
    keys_list = [k for k in keys if str(k or "").strip()]
    total = len(keys_list)
    initial_correct = 0
    enhanced_correct = 0
    mismatches: List[str] = []
    enhanced_by_conflict: List[str] = []

    for col in keys_list:
        pred = row.get(col)
        ref = ref_row[col] if col in ref_row.index else None
        is_correct = _normalize_semantic_value(pred) == _normalize_semantic_value(ref)
        if is_correct:
            initial_correct += 1
            enhanced_correct += 1
        elif col in conflict_keys:
            enhanced_correct += 1
            enhanced_by_conflict.append(col)
            mismatches.append(col)
        else:
            mismatches.append(col)

    return {
        "total": total,
        "initial_correct": initial_correct,
        "initial_accuracy": (initial_correct / total) if total else None,
        "enhanced_correct": enhanced_correct,
        "enhanced_accuracy": (enhanced_correct / total) if total else None,
        "mismatches": mismatches,
        "enhanced_by_conflict": enhanced_by_conflict,
    }


def _parse_threshold(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)%", str(text))
    if not m:
        raise ValueError(f"Could not parse percentage from: {text}")
    return float(m.group(1)) / 100.0


def _parse_target_count(text: str) -> str:
    m = re.search(r"\((\d+/\d+)\)", str(text))
    return m.group(1) if m else ""


@dataclass(frozen=True)
class StageDef:
    stage: str
    categories: Sequence[str]
    initial_threshold: float
    conflict_threshold: float
    initial_target_count: str
    conflict_target_count: str
    remark: str


def _load_cdm_rows() -> List[Dict[str, str]]:
    with CDM_CSV.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_stage_defs() -> List[StageDef]:
    out: List[StageDef] = []
    with STAGES_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cats = [s.strip() for s in row["cdm type"].split(",") if s.strip()]
            out.append(
                StageDef(
                    stage=str(row["단계"]).strip(),
                    categories=cats,
                    initial_threshold=_parse_threshold(row["1차 성능"]),
                    conflict_threshold=_parse_threshold(row["Conflict 포함 성능"]),
                    initial_target_count=_parse_target_count(row["1차 성능"]),
                    conflict_target_count=_parse_target_count(row["Conflict 포함 성능"]),
                    remark=str(row.get("Remark", "")).strip(),
                )
            )
    return out


def _build_type_keysets(cdm_rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in cdm_rows:
        cdm_type = str(row["cdm type"]).strip()
        key = str(row["csv key"]).strip()
        out.setdefault(cdm_type, []).append(key)

    # Stage file splits PSG into a 3-key core subset plus the remaining PSG keys.
    psg_rdi_spo2 = ["RDI_no", "RDI_idx", "Lowest_SpO2"]
    out["PSG-RDI/SpO2"] = [k for k in psg_rdi_spo2 if k in set(out.get("PSG", []))]
    out["PSG_REST"] = [k for k in out.get("PSG", []) if k not in set(out["PSG-RDI/SpO2"])]
    return out


def _stage_keys(stage: StageDef, type_keysets: Dict[str, List[str]]) -> List[str]:
    keys: List[str] = []
    for cat in stage.categories:
        if cat == "PSG":
            keys.extend(type_keysets.get("PSG_REST", []))
        else:
            keys.extend(type_keysets.get(cat, []))
    return keys


def _load_conflict_keys(conflict_json: Path) -> set[str]:
    if not conflict_json.exists():
        return set()
    obj = json.loads(conflict_json.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return {str(k) for k in obj.keys()}
    return set()


def _format_count(correct: int, total: int, acc: float | None) -> str:
    pct = f"{(acc or 0.0) * 100:.2f}%"
    return f"{correct}/{total} ({pct})"


def _format_delta(delta_count: int, delta_acc: float | None) -> str:
    pct = f"{(delta_acc or 0.0) * 100:+.2f}%"
    return f"{delta_count:+d} ({pct})"


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--patients", nargs="+", required=True)
    return ap.parse_args()


def main() -> None:
    args = build_args()
    output_root = (ROOT / args.output_root).resolve()
    cdm_rows = _load_cdm_rows()
    type_keysets = _build_type_keysets(cdm_rows)
    stage_defs = _load_stage_defs()
    ref_df = pd.read_csv(EXAMPLE_CSV, dtype=object)

    type_rows: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []

    base_types = [t for t in ["Basic", "Demographics", "CoreQ", "PSG", "SubQ", "CPAP"] if t in type_keysets]

    for patient_name in args.patients:
        patient_dir = output_root / patient_name
        patient_csv = patient_dir / f"{patient_name}.csv"
        final_row = _load_final_row(patient_csv)
        patient_idx = int(re.search(r"(\d+)$", patient_name).group(1))
        ref_row = ref_df.iloc[patient_idx - 1]
        conflict_keys = _load_conflict_keys(patient_dir / "conflicts" / f"{patient_name}_conflicts.json")

        for cdm_type in base_types:
            metrics = _metric_dict(final_row, ref_row, type_keysets[cdm_type], conflict_keys)
            type_rows.append(
                {
                    "patient": patient_name,
                    "cdm_type": cdm_type,
                    "initial_correct": metrics["initial_correct"],
                    "total": metrics["total"],
                    "initial_accuracy": metrics["initial_accuracy"],
                    "initial_display": _format_count(metrics["initial_correct"], metrics["total"], metrics["initial_accuracy"]),
                }
            )

        for stage in stage_defs:
            keys = _stage_keys(stage, type_keysets)
            metrics = _metric_dict(final_row, ref_row, keys, conflict_keys)
            stage_rows.append(
                {
                    "patient": patient_name,
                    "stage": stage.stage,
                    "cdm_types": ", ".join(stage.categories),
                    "schema_total_current": metrics["total"],
                    "stage_initial_target_count": stage.initial_target_count,
                    "stage_conflict_target_count": stage.conflict_target_count,
                    "initial_correct": metrics["initial_correct"],
                    "initial_accuracy": metrics["initial_accuracy"],
                    "initial_display": _format_count(metrics["initial_correct"], metrics["total"], metrics["initial_accuracy"]),
                    "conflict_enhanced_correct": metrics["enhanced_correct"],
                    "conflict_enhanced_accuracy": metrics["enhanced_accuracy"],
                    "conflict_enhanced_display": _format_count(metrics["enhanced_correct"], metrics["total"], metrics["enhanced_accuracy"]),
                    "conflict_bonus_keys": len(metrics["enhanced_by_conflict"]),
                    "conflict_bonus_display": _format_delta(
                        metrics["enhanced_correct"] - metrics["initial_correct"],
                        (metrics["enhanced_accuracy"] or 0.0) - (metrics["initial_accuracy"] or 0.0),
                    ),
                    "initial_threshold_pct": stage.initial_threshold * 100,
                    "conflict_threshold_pct": stage.conflict_threshold * 100,
                    "initial_pass": "PASS" if (metrics["initial_accuracy"] or 0.0) >= stage.initial_threshold else "FAIL",
                    "conflict_pass": "PASS" if (metrics["enhanced_accuracy"] or 0.0) >= stage.conflict_threshold else "FAIL",
                    "remark": stage.remark,
                }
            )

    type_df = pd.DataFrame(type_rows)
    stage_df = pd.DataFrame(stage_rows)

    type_out = output_root / "patient21_22_cdmtype_accuracy.csv"
    stage_out = output_root / "patient21_22_stage_accuracy.csv"
    type_df.to_csv(type_out, index=False)
    stage_df.to_csv(stage_out, index=False)

    print(json.dumps({"type_csv": str(type_out), "stage_csv": str(stage_out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
