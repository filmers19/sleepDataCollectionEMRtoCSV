#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPE103_PATH = ROOT / "codebase_B_paper_to_csv" / "103_paper_to_cdm_SA.py"
CDM_CSV = ROOT / "cdm_revised.csv"
EXAMPLE_CSV = ROOT / "example.csv"


def _load_pipe103():
    spec = importlib.util.spec_from_file_location("pipe103_finalrow_route_summary", PIPE103_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {PIPE103_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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
            from decimal import Decimal, InvalidOperation

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


def _metric_dict(row: Dict[str, Any], ref_row: pd.Series, keys: Iterable[str]) -> Dict[str, Any]:
    keys_list = [k for k in keys if str(k or "").strip()]
    total = len(keys_list)
    out = {
        "total_columns": total,
        "semantic_matches": 0,
        "semantic_accuracy": None,
        "mismatches": [],
    }
    if total == 0:
        return out

    for col in keys_list:
        pred = row.get(col)
        ref = ref_row[col] if col in ref_row.index else None
        pred_norm = _normalize_semantic_value(pred)
        ref_norm = _normalize_semantic_value(ref)
        if pred_norm == ref_norm:
            out["semantic_matches"] += 1
        else:
            out["mismatches"].append(
                {
                    "field": col,
                    "predicted": None if _is_missing_value(pred) else pred,
                    "expected": None if _is_missing_value(ref) else ref,
                }
            )

    out["semantic_accuracy"] = out["semantic_matches"] / max(1, total)
    return out


@dataclass(frozen=True)
class RouteBucket:
    label: str
    route_name: str
    official_questionnaire: Optional[bool] = None


ROUTE_BUCKETS: List[RouteBucket] = [
    RouteBucket("psg_signal_acc", "map_route_polysomnography_signals"),
    RouteBucket("psg_report_general_acc", "map_route_psg_report_general"),
    RouteBucket("psg_report_extensive_acc", "map_route_psg_report_extensive"),
    RouteBucket("cpap_psg_report_general_acc", "map_route_cpap_psg_report_general"),
    RouteBucket("cpap_psg_report_extensive_acc", "map_route_cpap_psg_report_extensive"),
    RouteBucket("morning_questionnaire_acc", "map_route_morning_questionnaire"),
    RouteBucket("night_questionnaire_type_a_acc", "map_route_night_questionnaire", True),
    RouteBucket("night_questionnaire_type_b_acc", "map_route_night_questionnaire", False),
]


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize per-patient route accuracies from final resolved rows")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--example_csv", default=str(EXAMPLE_CSV))
    ap.add_argument("--patient_start", type=int, default=1)
    ap.add_argument("--patient_end", type=int, default=21)
    return ap.parse_args()


def main() -> None:
    args = build_args()
    output_root = (ROOT / args.output_root).resolve()
    example_csv = Path(args.example_csv).resolve()
    pipe103 = _load_pipe103()
    retriever = pipe103.CDMRetriever(CDM_CSV)
    output_columns = [str(c) for c in pd.read_csv(example_csv, nrows=0).columns.tolist()]
    ref_df = pd.read_csv(example_csv, dtype=object)

    wide_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    for patient_idx in range(int(args.patient_start), int(args.patient_end) + 1):
        patient_name = f"Patient_{patient_idx:02d}"
        patient_dir = output_root / patient_name
        patient_csv = patient_dir / f"{patient_name}.csv"
        if not patient_csv.exists():
            continue

        final_row = _load_final_row(patient_csv)
        ref_row = ref_df.iloc[patient_idx - 1]
        overall = _metric_dict(final_row, ref_row, output_columns)

        wide_row: Dict[str, Any] = {
            "patient": patient_name,
            "overall_acc": overall["semantic_accuracy"],
            "overall_matches": overall["semantic_matches"],
            "overall_total_keys": overall["total_columns"],
            "overall_mismatches": len(overall["mismatches"]),
        }

        for bucket in ROUTE_BUCKETS:
            allowed_keys = [
                row.key
                for row in retriever.route_rows(
                    bucket.route_name,
                    official_questionnaire=bucket.official_questionnaire,
                )
            ]
            metrics = _metric_dict(final_row, ref_row, allowed_keys)
            wide_row[bucket.label] = metrics["semantic_accuracy"]
            wide_row[f"{bucket.label}_matches"] = metrics["semantic_matches"]
            wide_row[f"{bucket.label}_total_keys"] = metrics["total_columns"]
            wide_row[f"{bucket.label}_mismatches"] = len(metrics["mismatches"])
            long_rows.append(
                {
                    "patient": patient_name,
                    "route_bucket": bucket.label,
                    "semantic_accuracy": metrics["semantic_accuracy"],
                    "semantic_matches": metrics["semantic_matches"],
                    "total_columns": metrics["total_columns"],
                    "mismatches": len(metrics["mismatches"]),
                }
            )

        wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    wide_path = output_root / "route_accuracy_finalrow_by_patient.csv"
    long_path = output_root / "route_accuracy_finalrow_long.csv"
    aggregate_path = output_root / "route_accuracy_finalrow_aggregate.json"

    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)

    aggregate: Dict[str, Any] = {"patients": int(len(wide_df)), "routes": {}}
    if not long_df.empty:
        for bucket in ROUTE_BUCKETS:
            sub = long_df[long_df["route_bucket"] == bucket.label]
            aggregate["routes"][bucket.label] = {
                "avg_semantic_accuracy": float(sub["semantic_accuracy"].dropna().mean())
                if sub["semantic_accuracy"].notna().any()
                else None,
                "avg_matches": float(sub["semantic_matches"].mean()) if len(sub) else None,
                "avg_total_keys": float(sub["total_columns"].mean()) if len(sub) else None,
                "avg_mismatches": float(sub["mismatches"].mean()) if len(sub) else None,
            }

    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "wide_csv": str(wide_path),
                "long_csv": str(long_path),
                "aggregate_json": str(aggregate_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
