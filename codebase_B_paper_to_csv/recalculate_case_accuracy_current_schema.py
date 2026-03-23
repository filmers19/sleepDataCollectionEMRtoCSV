#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CDM_CSV = ROOT / "cdm_revised.csv"
EXAMPLE_CSV = ROOT / "example.csv"
STAGES_CSV = ROOT / "stages.csv"

RUNS: List[Tuple[str, Path, range]] = [
    (
        "21patients",
        ROOT / "out_patient01to21_liveocr_gpt54_route_gpt54_map_gpt51multiroute_resolve_gpt54_fullrerun_20260319",
        range(1, 22),
    ),
    (
        "22patients",
        ROOT / "out_patient01to22_preprocessed_liveocr_gpt54_route_gpt54_map_gpt51multiroute_resolve_gpt54_20260321",
        range(1, 23),
    ),
]


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full_cpap_family",
        action="store_true",
        help="Use the full repeated CPAP pressure family from example.csv headers "
        "(Pressure_05~29 and Pr05~29_*) instead of only CPAP-labeled keys in cdm_revised.csv.",
    )
    return ap.parse_args()


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        return math.isnan(v)
    t = str(v).strip()
    return t == "" or t.lower() in {"nan", "none", "null", "n/a", "na"}


def _norm(v: Any) -> Any:
    if _is_missing(v):
        return None
    text = str(v).strip()
    numeric = text.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric):
        try:
            dec = Decimal(numeric).normalize()
            rendered = format(dec, "f")
            return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
        except InvalidOperation:
            pass
    return re.sub(r"\s+", " ", text)


def _pct(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def _disp(n: int, d: int) -> str:
    return f"{n}/{d} ({_pct(n, d) * 100:.2f}%)"


def _load_keysets(full_cpap_family: bool = False) -> Dict[str, List[str]]:
    with CDM_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, List[str]] = {}
    for row in rows:
        out.setdefault(row["cdm type"], []).append(row["csv key"])
    psg_rdi_spo2 = ["RDI_no", "RDI_idx", "Lowest_SpO2"]
    out["PSG-RDI/SpO2"] = [k for k in psg_rdi_spo2 if k in set(out.get("PSG", []))]
    out["PSG_REST"] = [k for k in out.get("PSG", []) if k not in set(out["PSG-RDI/SpO2"])]
    if full_cpap_family:
        with EXAMPLE_CSV.open(newline="", encoding="utf-8-sig") as f:
            headers = next(csv.reader(f))
        cpap_full = [
            h
            for h in headers
            if re.fullmatch(r"Pressure_\d{2}", h) or re.fullmatch(r"Pr\d{2}_.+", h)
        ]
        out["CPAP"] = cpap_full
    return out


def _load_stages() -> List[Dict[str, str]]:
    with STAGES_CSV.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _parse_stage_pct(s: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)%", s)
    if not m:
        raise ValueError(f"Could not parse percentage from {s}")
    return float(m.group(1)) / 100.0


def _metric(final_row: Dict[str, Any], ref_row: Dict[str, Any], keys: List[str], conflict_keys: set[str]) -> Dict[str, Any]:
    init = 0
    enh = 0
    for k in keys:
        ok = _norm(final_row.get(k)) == _norm(ref_row.get(k))
        if ok:
            init += 1
            enh += 1
        elif k in conflict_keys:
            enh += 1
    total = len(keys)
    return {
        "initial_correct": init,
        "enhanced_correct": enh,
        "total": total,
        "initial_display": _disp(init, total),
        "enhanced_display": _disp(enh, total),
    }


def _stage_keys(stage_types: str, keysets: Dict[str, List[str]]) -> List[str]:
    keys: List[str] = []
    for cat in [s.strip() for s in stage_types.split(",") if s.strip()]:
        if cat == "PSG":
            keys.extend(keysets["PSG_REST"])
        else:
            keys.extend(keysets.get(cat, []))
    return keys


def _write_md(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    rows = [cols] + df.astype(str).values.tolist()
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(cols))]
    lines = []
    lines.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_args()
    keysets = _load_keysets(full_cpap_family=args.full_cpap_family)
    stages = _load_stages()
    ref_df = pd.read_csv(EXAMPLE_CSV, dtype=object)

    cdm_cols = ["Basic", "Demographics", "CoreQ", "SubQ", "PSG-RDI/SpO2", "PSG", "CPAP"]

    manifest: Dict[str, Dict[str, str]] = {}

    for tag, run_root, patient_range in RUNS:
        patient_cdm_rows: List[Dict[str, Any]] = []
        patient_stage_rows: List[Dict[str, Any]] = []
        stage_totals: Dict[str, Dict[str, Any]] = {
            st["단계"]: {
                "Stage": st["단계"],
                "cdm_types": st["cdm type"],
                "current_schema_total_per_patient": len(_stage_keys(st["cdm type"], keysets)),
                "initial_correct": 0,
                "enhanced_correct": 0,
                "total": 0,
                "initial_threshold": _parse_stage_pct(st["1차 성능"]),
                "enhanced_threshold": _parse_stage_pct(st["Conflict 포함 성능"]),
            }
            for st in stages
        }

        for idx in patient_range:
            patient = f"Patient_{idx:02d}"
            patient_csv = run_root / patient / f"{patient}.csv"
            with patient_csv.open(newline="", encoding="utf-8-sig") as f:
                final_row = next(csv.DictReader(f))
            ref_row = ref_df.iloc[idx - 1].to_dict()
            conflict_path = run_root / patient / "conflicts" / f"{patient}_conflicts.json"
            conflict_keys = set(json.loads(conflict_path.read_text(encoding="utf-8")).keys()) if conflict_path.exists() else set()

            cdm_row: Dict[str, Any] = {"Patient_no": patient}
            for cdm_type in cdm_cols:
                keys = keysets["PSG_REST"] if cdm_type == "PSG" else keysets[cdm_type]
                m = _metric(final_row, ref_row, keys, conflict_keys)
                cdm_row[cdm_type] = m["initial_display"]
            patient_cdm_rows.append(cdm_row)

            stage_row: Dict[str, Any] = {"Patient_no": patient}
            for st in stages:
                sid = st["단계"]
                keys = _stage_keys(st["cdm type"], keysets)
                m = _metric(final_row, ref_row, keys, conflict_keys)
                stage_row[f"S{sid}_initial"] = m["initial_display"]
                stage_row[f"S{sid}_conflict_enhanced"] = m["enhanced_display"]
                init_pass = _pct(m["initial_correct"], m["total"]) >= stage_totals[sid]["initial_threshold"]
                enh_pass = _pct(m["enhanced_correct"], m["total"]) >= stage_totals[sid]["enhanced_threshold"]
                stage_row[f"S{sid}_pass_fail"] = f"{'PASS' if init_pass else 'FAIL'} / {'PASS' if enh_pass else 'FAIL'}"

                stage_totals[sid]["initial_correct"] += m["initial_correct"]
                stage_totals[sid]["enhanced_correct"] += m["enhanced_correct"]
                stage_totals[sid]["total"] += m["total"]
            patient_stage_rows.append(stage_row)

        stage_total_rows: List[Dict[str, Any]] = []
        for sid in ["1", "2", "3", "4"]:
            rec = stage_totals[sid]
            init_pass = _pct(rec["initial_correct"], rec["total"]) >= rec["initial_threshold"]
            enh_pass = _pct(rec["enhanced_correct"], rec["total"]) >= rec["enhanced_threshold"]
            stage_total_rows.append(
                {
                    "Stage": sid,
                    "cdm_types": rec["cdm_types"],
                    "current_schema_total_per_patient": rec["current_schema_total_per_patient"],
                    "total_initial_acc": _disp(rec["initial_correct"], rec["total"]),
                    "total_conflict_enhanced_acc": _disp(rec["enhanced_correct"], rec["total"]),
                    "pass_fail": f"{'PASS' if init_pass else 'FAIL'} / {'PASS' if enh_pass else 'FAIL'}",
                }
            )

        patient_cdm_df = pd.DataFrame(patient_cdm_rows)
        patient_stage_df = pd.DataFrame(patient_stage_rows)
        stage_total_df = pd.DataFrame(stage_total_rows)

        suffix = "current_schema_full_cpap" if args.full_cpap_family else "current_schema"
        out1 = run_root / f"patient_cdmtype_accuracy_table_{tag}_{suffix}.csv"
        out2 = run_root / f"patient_stage_accuracy_table_{tag}_{suffix}.csv"
        out3 = run_root / f"stage_total_accuracy_table_{tag}_{suffix}.csv"
        patient_cdm_df.to_csv(out1, index=False)
        patient_stage_df.to_csv(out2, index=False)
        stage_total_df.to_csv(out3, index=False)
        _write_md(patient_cdm_df, out1.with_suffix(".md"))
        _write_md(patient_stage_df, out2.with_suffix(".md"))
        _write_md(stage_total_df, out3.with_suffix(".md"))

        manifest[tag] = {
            "patient_cdmtype_csv": str(out1),
            "patient_stage_csv": str(out2),
            "stage_total_csv": str(out3),
        }

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
