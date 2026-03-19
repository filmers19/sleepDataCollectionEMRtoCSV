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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPE103_PATH = ROOT / "codebase_B_paper_to_csv" / "103_paper_to_cdm_SA.py"
CDM_CSV = ROOT / "cdm_revised.csv"
EXAMPLE_CSV = ROOT / "example.csv"


def _load_pipe103():
    spec = importlib.util.spec_from_file_location("pipe103_for_route_summary", PIPE103_PATH)
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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _metric_dict(row: Dict[str, Any], ref_row: pd.Series, keys: Iterable[str]) -> Dict[str, Any]:
    keys_list = [k for k in keys if str(k or "").strip()]
    total = len(keys_list)
    out = {
        "total_columns": total,
        "semantic_matches": 0,
        "semantic_accuracy": None,
        "reference_non_null": 0,
        "predicted_non_null": 0,
        "correct_non_null": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "wrong_value_fields": 0,
        "precision": None,
        "recall": None,
        "f1": None,
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
        pred_present = pred_norm is not None
        ref_present = ref_norm is not None
        if pred_present:
            out["predicted_non_null"] += 1
        if ref_present:
            out["reference_non_null"] += 1
        if pred_present and ref_present and pred_norm == ref_norm:
            out["correct_non_null"] += 1
        elif pred_present and not ref_present:
            out["false_positives"] += 1
        elif ref_present and not pred_present:
            out["false_negatives"] += 1
        elif pred_present and ref_present:
            out["wrong_value_fields"] += 1

    out["semantic_accuracy"] = out["semantic_matches"] / max(1, total)
    out["precision"] = out["correct_non_null"] / max(1, out["predicted_non_null"])
    out["recall"] = out["correct_non_null"] / max(1, out["reference_non_null"])
    denom = out["precision"] + out["recall"]
    out["f1"] = (2 * out["precision"] * out["recall"] / denom) if denom > 0 else 0.0
    return out


def _load_final_row(patient_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(patient_csv, dtype=object)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): (None if _is_missing_value(v) else v) for k, v in row.items()}


def _build_page_result(pipe103: Any, bundle_prefix: Path, allowed_keys: set[str]) -> Optional[Any]:
    valid_path = bundle_prefix.with_suffix(".valid.json")
    contexts_path = bundle_prefix.with_suffix(".contexts.json")
    if not valid_path.exists() or not contexts_path.exists():
        return None
    valid_all = _load_json(valid_path)
    contexts_all = _load_json(contexts_path)
    valid_json: Dict[str, Any] = {}
    cdm_contexts: Dict[str, str] = {}
    input_contexts: Dict[str, Dict[str, Any]] = {}
    for key, value in valid_all.items():
        if key not in allowed_keys:
            continue
        valid_json[key] = value
        ctx = contexts_all.get(key) or {}
        cdm_contexts[key] = str(ctx.get("CDM_Context") or "").strip()
        input_contexts[key] = ctx.get("input_context") or {}
    if not valid_json:
        return None
    return pipe103.PageResult(
        image_name=bundle_prefix.name + ".txt",
        ocr_text="",
        raw_json={},
        valid_json=valid_json,
        input_contexts=input_contexts,
        cdm_contexts=cdm_contexts,
        rejected_fields={},
    )


def _route_row_for_bucket(
    pipe103: Any,
    retriever: Any,
    patient_dir: Path,
    bucket: RouteBucket,
    output_columns: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    allowed_rows = retriever.route_rows(bucket.route_name, official_questionnaire=bucket.official_questionnaire)
    allowed_keys = {row.key for row in allowed_rows}
    page_results: List[Any] = []

    map_pages_dir = patient_dir / "map_pages"
    for meta_path in sorted(map_pages_dir.glob("*.meta.json")):
        meta = _load_json(meta_path)
        if str(meta.get("map_route") or "") != bucket.route_name:
            continue
        bundle_prefix = meta_path.with_suffix("")
        pr = _build_page_result(pipe103=pipe103, bundle_prefix=bundle_prefix, allowed_keys=allowed_keys)
        if pr is not None:
            page_results.append(pr)

    if not page_results:
        empty = {c: None for c in output_columns}
        aux = {"pages_used": 0, "raw_keys_after_merge": 0, "final_keys_after_merge": 0, "conflict_keys": 0}
        return empty, empty, aux

    merged, conflicts, _provenance = pipe103.merge_page_results(page_results)
    raw_row = pipe103.build_output_row(dict(merged), output_columns)
    overrides, _decisions, pending, _vote_df = pipe103.resolve_conflicts_by_majority_vote(conflicts)
    merged.update(overrides)
    # Leave unresolved route-local ties blank so the route metric stays deterministic.
    for key in pending:
        merged.pop(key, None)
    final_row = pipe103.build_output_row(merged, output_columns)
    aux = {
        "pages_used": len(page_results),
        "raw_keys_after_merge": len([k for k, v in raw_row.items() if not _is_missing_value(v)]),
        "final_keys_after_merge": len([k for k, v in final_row.items() if not _is_missing_value(v)]),
        "conflict_keys": len(conflicts),
    }
    return raw_row, final_row, aux


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize per-patient route-level accuracies")
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

        wide_row: Dict[str, Any] = {"patient": patient_name}
        overall = _metric_dict(final_row, ref_row, output_columns)
        wide_row["overall_acc"] = overall["semantic_accuracy"]

        for bucket in ROUTE_BUCKETS:
            raw_route_row, final_route_row, aux = _route_row_for_bucket(
                pipe103=pipe103,
                retriever=retriever,
                patient_dir=patient_dir,
                bucket=bucket,
                output_columns=output_columns,
            )
            allowed_keys = [row.key for row in retriever.route_rows(bucket.route_name, official_questionnaire=bucket.official_questionnaire)]
            raw_metrics = _metric_dict(raw_route_row, ref_row, allowed_keys)
            final_metrics = _metric_dict(final_route_row, ref_row, allowed_keys)
            wide_row[f"{bucket.label}_raw"] = raw_metrics["semantic_accuracy"]
            wide_row[f"{bucket.label}_final"] = final_metrics["semantic_accuracy"]
            wide_row[f"{bucket.label}_pages"] = aux["pages_used"]
            wide_row[f"{bucket.label}_keys"] = final_metrics["total_columns"]
            wide_row[f"{bucket.label}_raw_mismatches"] = len(raw_metrics["mismatches"])
            wide_row[f"{bucket.label}_final_mismatches"] = len(final_metrics["mismatches"])
            long_rows.append(
                {
                    "patient": patient_name,
                    "route_bucket": bucket.label,
                    "raw_semantic_accuracy": raw_metrics["semantic_accuracy"],
                    "raw_precision": raw_metrics["precision"],
                    "raw_recall": raw_metrics["recall"],
                    "raw_f1": raw_metrics["f1"],
                    "raw_mismatches": len(raw_metrics["mismatches"]),
                    "final_semantic_accuracy": final_metrics["semantic_accuracy"],
                    "final_precision": final_metrics["precision"],
                    "final_recall": final_metrics["recall"],
                    "final_f1": final_metrics["f1"],
                    "final_mismatches": len(final_metrics["mismatches"]),
                    "total_columns": final_metrics["total_columns"],
                    "pages_used": aux["pages_used"],
                    "raw_keys_after_merge": aux["raw_keys_after_merge"],
                    "final_keys_after_merge": aux["final_keys_after_merge"],
                    "conflict_keys": aux["conflict_keys"],
                }
            )

        wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    wide_path = output_root / "route_accuracy_by_patient.csv"
    long_path = output_root / "route_accuracy_long.csv"
    aggregate_path = output_root / "route_accuracy_aggregate.json"

    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)

    aggregate: Dict[str, Any] = {"patients": int(len(wide_df)), "routes": {}}
    if not long_df.empty:
        for bucket in ROUTE_BUCKETS:
            sub = long_df[long_df["route_bucket"] == bucket.label]
            aggregate["routes"][bucket.label] = {
                "avg_raw_semantic_accuracy": float(sub["raw_semantic_accuracy"].dropna().mean()) if sub["raw_semantic_accuracy"].notna().any() else None,
                "avg_raw_precision": float(sub["raw_precision"].dropna().mean()) if sub["raw_precision"].notna().any() else None,
                "avg_raw_recall": float(sub["raw_recall"].dropna().mean()) if sub["raw_recall"].notna().any() else None,
                "avg_raw_f1": float(sub["raw_f1"].dropna().mean()) if sub["raw_f1"].notna().any() else None,
                "avg_raw_mismatches": float(sub["raw_mismatches"].mean()) if len(sub) else None,
                "avg_final_semantic_accuracy": float(sub["final_semantic_accuracy"].dropna().mean()) if sub["final_semantic_accuracy"].notna().any() else None,
                "avg_final_precision": float(sub["final_precision"].dropna().mean()) if sub["final_precision"].notna().any() else None,
                "avg_final_recall": float(sub["final_recall"].dropna().mean()) if sub["final_recall"].notna().any() else None,
                "avg_final_f1": float(sub["final_f1"].dropna().mean()) if sub["final_f1"].notna().any() else None,
                "avg_final_mismatches": float(sub["final_mismatches"].mean()) if len(sub) else None,
            }

    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wide_csv": str(wide_path), "long_csv": str(long_path), "aggregate_json": str(aggregate_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
