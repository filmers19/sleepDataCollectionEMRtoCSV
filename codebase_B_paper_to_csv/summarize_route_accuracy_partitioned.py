#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPE103_PATH = ROOT / "codebase_B_paper_to_csv" / "103_paper_to_cdm_SA.py"
CDM_CSV = ROOT / "cdm_revised.csv"
EXAMPLE_CSV = ROOT / "example.csv"

ROUTE_BUCKET_ORDER = [
    "psg_signal",
    "psg_report_general",
    "psg_report_extensive",
    "cpap_psg_report_general",
    "cpap_psg_report_extensive",
    "morning_questionnaire",
    "night_questionnaire_type_a",
    "night_questionnaire_type_b",
]

ROUTE_LABEL_TO_MAP_ROUTE = {
    "psg_signal": "map_route_polysomnography_signals",
    "psg_report_general": "map_route_psg_report_general",
    "psg_report_extensive": "map_route_psg_report_extensive",
    "cpap_psg_report_general": "map_route_cpap_psg_report_general",
    "cpap_psg_report_extensive": "map_route_cpap_psg_report_extensive",
    "morning_questionnaire": "map_route_morning_questionnaire",
    "night_questionnaire_type_a": "map_route_night_questionnaire",
    "night_questionnaire_type_b": "map_route_night_questionnaire",
}


def _load_pipe103():
    spec = importlib.util.spec_from_file_location("pipe103_partitioned_route_summary", PIPE103_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {PIPE103_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return df.iloc[0].to_dict()


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


def _bucket_from_map_route(route: str, official_questionnaire: Optional[bool]) -> Optional[str]:
    route = str(route or "").strip()
    if route == "map_route_polysomnography_signals":
        return "psg_signal"
    if route == "map_route_psg_report_general":
        return "psg_report_general"
    if route == "map_route_psg_report_extensive":
        return "psg_report_extensive"
    if route == "map_route_cpap_psg_report_general":
        return "cpap_psg_report_general"
    if route == "map_route_cpap_psg_report_extensive":
        return "cpap_psg_report_extensive"
    if route == "map_route_morning_questionnaire":
        return "morning_questionnaire"
    if route == "map_route_night_questionnaire":
        return "night_questionnaire_type_a" if bool(official_questionnaire) else "night_questionnaire_type_b"
    return None


def _official_questionnaire_from_text(pipe103: Any, text: str) -> Optional[bool]:
    route = pipe103.normalize_map_route_name("map_route_night_questionnaire")
    seq = pipe103.classify_official_questionnaire_sequence([{"route_name": route, "ocr_text": text}])
    info = seq.get("bundle_0001") or {}
    if "official_questionnaire" in info:
        return bool(info.get("official_questionnaire"))
    return None


def _build_patient_meta_maps(pipe103: Any, patient_dir: Path) -> tuple[Dict[str, str], Dict[str, bool], Dict[str, int]]:
    image_to_route: Dict[str, str] = {}
    image_to_official: Dict[str, bool] = {}
    route_counts: Counter[str] = Counter()
    ocr_dir = patient_dir / "ocr_pages"
    map_dir = patient_dir / "map_pages"
    text_cache: Dict[str, str] = {}
    for txt_path in ocr_dir.glob("*.txt"):
        text_cache[txt_path.name] = txt_path.read_text(encoding="utf-8", errors="ignore")
    for meta_path in map_dir.glob("*.meta.json"):
        meta = _load_json(meta_path)
        route = str(meta.get("map_route") or "").strip()
        if not route:
            continue
        route_counts[route] += 1
        for img in meta.get("source_images") or []:
            image_to_route[str(img)] = route
            if route == "map_route_night_questionnaire":
                txt_name = Path(str(img)).with_suffix(".txt").name
                official = _official_questionnaire_from_text(pipe103, text_cache.get(txt_name, ""))
                if official is not None:
                    image_to_official[str(img)] = bool(official)
    return image_to_route, image_to_official, dict(route_counts)


def _extract_source_bucket_from_resolution(
    field: str,
    resolution: Dict[str, Any],
    image_to_route: Dict[str, str],
    image_to_official: Dict[str, bool],
) -> Optional[str]:
    info = resolution.get(field) or {}
    source_image = str(info.get("source_image") or "").strip()
    if not source_image:
        return None
    image_name = Path(source_image).with_suffix(".jpg").name
    route = image_to_route.get(image_name)
    if not route:
        return None
    official = image_to_official.get(image_name)
    return _bucket_from_map_route(route, official)


def _extract_source_bucket_from_provenance(
    field: str,
    provenance: Dict[str, Any],
    image_to_route: Dict[str, str],
    image_to_official: Dict[str, bool],
) -> Optional[str]:
    entries = provenance.get(field) or []
    buckets: List[str] = []
    for entry in entries:
        image_name = Path(str(entry.get("image") or "")).with_suffix(".jpg").name
        route = image_to_route.get(image_name)
        if not route:
            route = str(((entry.get("input_context") or {}).get("page_type")) or "").strip()
        if not route:
            continue
        official = image_to_official.get(image_name)
        bucket = _bucket_from_map_route(route, official)
        if bucket:
            buckets.append(bucket)
    if not buckets:
        return None
    return Counter(buckets).most_common(1)[0][0]


def _build_membership_sets(pipe103: Any, retriever: Any) -> Dict[str, set[str]]:
    sets: Dict[str, set[str]] = {}
    sets["psg_signal"] = {row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_PSG_SIGNALS)}
    sets["psg_report_general"] = {row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_PSG_REPORT_GENERAL)}
    sets["psg_report_extensive"] = {row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_PSG_REPORT_EXTENSIVE)}
    sets["cpap_psg_report_general"] = {row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_CPAP_PSG_REPORT_GENERAL)}
    sets["cpap_psg_report_extensive"] = {row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE)}
    sets["morning_questionnaire"] = {row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_MORNING_QUESTIONNAIRE)}
    sets["night_questionnaire_type_a"] = {
        row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_NIGHT_QUESTIONNAIRE, official_questionnaire=True)
    }
    sets["night_questionnaire_type_b"] = {
        row.key for row in retriever.route_rows(pipe103.MAP_ROUTE_NIGHT_QUESTIONNAIRE, official_questionnaire=False)
    }
    return sets


def _fallback_bucket_for_field(
    field: str,
    memberships: Dict[str, set[str]],
    route_counts: Dict[str, int],
) -> str:
    if field in memberships["night_questionnaire_type_a"]:
        return "night_questionnaire_type_a"
    if field in memberships["night_questionnaire_type_b"]:
        return "night_questionnaire_type_b"
    if field in memberships["morning_questionnaire"]:
        return "morning_questionnaire"

    cpap_present = route_counts.get("map_route_cpap_psg_report_extensive", 0) or route_counts.get("map_route_cpap_psg_report_general", 0)
    psg_present = route_counts.get("map_route_psg_report_extensive", 0) or route_counts.get("map_route_psg_report_general", 0)

    if field in memberships["cpap_psg_report_extensive"] and field not in memberships["psg_report_extensive"]:
        if route_counts.get("map_route_cpap_psg_report_extensive", 0):
            return "cpap_psg_report_extensive"
        return "cpap_psg_report_general"

    if field in memberships["psg_report_extensive"] or field in memberships["psg_report_general"]:
        if cpap_present and not psg_present:
            if route_counts.get("map_route_cpap_psg_report_extensive", 0):
                return "cpap_psg_report_extensive"
            return "cpap_psg_report_general"
        if route_counts.get("map_route_psg_report_extensive", 0):
            return "psg_report_extensive"
        if route_counts.get("map_route_psg_report_general", 0):
            return "psg_report_general"
        if route_counts.get("map_route_cpap_psg_report_extensive", 0):
            return "cpap_psg_report_extensive"
        if route_counts.get("map_route_cpap_psg_report_general", 0):
            return "cpap_psg_report_general"

    if field in memberships["psg_signal"]:
        return "psg_signal"

    # Last resort: choose the dominant report route or night B.
    precedence = [
        "map_route_cpap_psg_report_extensive",
        "map_route_cpap_psg_report_general",
        "map_route_psg_report_extensive",
        "map_route_psg_report_general",
        "map_route_morning_questionnaire",
        "map_route_night_questionnaire",
        "map_route_polysomnography_signals",
    ]
    for route in precedence:
        if route_counts.get(route, 0):
            bucket = _bucket_from_map_route(route, False)
            if bucket:
                return bucket
    return "night_questionnaire_type_b"


def _assign_field_to_bucket(
    field: str,
    provenance: Dict[str, Any],
    resolution: Dict[str, Any],
    image_to_route: Dict[str, str],
    image_to_official: Dict[str, bool],
    memberships: Dict[str, set[str]],
    route_counts: Dict[str, int],
) -> str:
    bucket = _extract_source_bucket_from_resolution(field, resolution, image_to_route, image_to_official)
    if bucket:
        return bucket
    bucket = _extract_source_bucket_from_provenance(field, provenance, image_to_route, image_to_official)
    if bucket:
        return bucket
    return _fallback_bucket_for_field(field, memberships, route_counts)


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Partition final-row columns into disjoint route buckets")
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
    memberships = _build_membership_sets(pipe103, retriever)

    ref_df = pd.read_csv(example_csv, dtype=object)
    output_columns = list(ref_df.columns)
    wide_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []

    for patient_idx in range(int(args.patient_start), int(args.patient_end) + 1):
        patient_name = f"Patient_{patient_idx:02d}"
        patient_dir = output_root / patient_name
        patient_csv = patient_dir / f"{patient_name}.csv"
        if not patient_csv.exists():
            continue

        final_row = _load_final_row(patient_csv)
        ref_row = ref_df.iloc[patient_idx - 1]
        provenance = _load_json(patient_dir / "provenance" / f"{patient_name}_provenance.json") if (patient_dir / "provenance" / f"{patient_name}_provenance.json").exists() else {}
        resolution = _load_json(patient_dir / "conflict_resolution" / f"{patient_name}_resolution.json") if (patient_dir / "conflict_resolution" / f"{patient_name}_resolution.json").exists() else {}
        image_to_route, image_to_official, route_counts = _build_patient_meta_maps(pipe103, patient_dir)

        field_to_bucket: Dict[str, str] = {}
        for field in output_columns:
            field_to_bucket[field] = _assign_field_to_bucket(
                field=field,
                provenance=provenance,
                resolution=resolution,
                image_to_route=image_to_route,
                image_to_official=image_to_official,
                memberships=memberships,
                route_counts=route_counts,
            )

        wide_row: Dict[str, Any] = {"patient": patient_name}
        overall = _metric_dict(final_row, ref_row, output_columns)
        wide_row["overall_acc"] = overall["semantic_accuracy"]
        wide_row["overall_mismatches"] = len(overall["mismatches"])

        for bucket in ROUTE_BUCKET_ORDER:
            keys = [field for field, assigned in field_to_bucket.items() if assigned == bucket]
            metrics = _metric_dict(final_row, ref_row, keys)
            wide_row[f"{bucket}_acc"] = metrics["semantic_accuracy"]
            wide_row[f"{bucket}_mismatches"] = len(metrics["mismatches"])
            wide_row[f"{bucket}_keys"] = metrics["total_columns"]
            assignment_rows.append(
                {
                    "patient": patient_name,
                    "route_bucket": bucket,
                    "semantic_accuracy": metrics["semantic_accuracy"],
                    "mismatches": len(metrics["mismatches"]),
                    "total_columns": metrics["total_columns"],
                }
            )

        wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(assignment_rows)
    wide_path = output_root / "route_accuracy_partitioned_by_patient.csv"
    long_path = output_root / "route_accuracy_partitioned_long.csv"
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    print(json.dumps({"wide_csv": str(wide_path), "long_csv": str(long_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
