#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_PATH = ROOT / "codebase_B_paper_to_csv" / "111_unified_ocr_map_pipeline.py"


def _patient_name(i: int) -> str:
    return f"Patient_{i:02d}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_patient_complete(summary_path: Path, pipeline_mode: str) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = _load_json(summary_path)
    except Exception:
        return False
    ocr_ok = int(summary.get("ocr_ok") or 0)
    if pipeline_mode == "ocr_only":
        return ocr_ok > 0
    map_ok = int(summary.get("map_ok") or 0)
    return ocr_ok > 0 and map_ok > 0


def _find_fatal_patient_error(patient_dir: Path) -> str:
    errors_dir = patient_dir / "errors"
    if not errors_dir.exists():
        return ""
    for error_file in sorted(errors_dir.glob("*_errors.json")):
        try:
            errors = _load_json(error_file)
        except Exception:
            continue
        if not isinstance(errors, list):
            continue
        for item in errors:
            message = str((item or {}).get("error") or "")
            lowered = message.lower()
            if (
                "missing api key" in lowered
                or "invalid_api_key" in lowered
                or ("status=401" in lowered and "openai" in lowered)
            ):
                return message
    return ""


def refresh_summary(output_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for i in range(1, 100):
        pat = _patient_name(i)
        patient_dir = output_root / pat
        summary_path = patient_dir / "ocr_map_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        row: Dict[str, Any] = {
            "patient": pat,
            "images_total": summary.get("images_total"),
            "ocr_ok": summary.get("ocr_ok"),
            "map_ok": summary.get("map_ok"),
            "semantic_accuracy": summary.get("evaluation_semantic_accuracy"),
            "precision": summary.get("evaluation_precision"),
            "recall": summary.get("evaluation_recall"),
            "f1": summary.get("evaluation_f1"),
            "request_count": summary.get("openai_usage_request_count"),
            "input_tokens": summary.get("openai_usage_input_tokens"),
            "output_tokens": summary.get("openai_usage_output_tokens"),
            "total_tokens": summary.get("openai_usage_total_tokens"),
            "elapsed_seconds": summary.get("total_elapsed_seconds"),
        }

        eval_path = patient_dir / "evaluation.json"
        if eval_path.exists():
            evaluation = _load_json(eval_path)
            row["evaluation_status"] = evaluation.get("evaluation_status")
            row["evaluation_skip_reason"] = evaluation.get("skip_reason")
            if str(evaluation.get("evaluation_status") or "completed") == "completed":
                row["mismatches"] = len(evaluation.get("mismatches") or [])

        conflicts_path = patient_dir / "conflicts" / f"{pat}_conflicts.json"
        if conflicts_path.exists():
            conflicts = _load_json(conflicts_path)
            row["conflict_keys"] = len(conflicts)

        resolution_path = patient_dir / "conflict_resolution" / f"{pat}_resolution.json"
        if resolution_path.exists():
            resolution = _load_json(resolution_path)
            vals = list((resolution or {}).values())
            row["resolved_by_code"] = sum(1 for v in vals if str(v.get("resolver_mode") or "") == "code_majority")
            row["resolved_by_llm_batch"] = sum(1 for v in vals if str(v.get("resolver_mode") or "") == "llm_batch")
            row["resolved_by_llm_single"] = sum(1 for v in vals if str(v.get("resolver_mode") or "") == "llm_single")

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "summary_by_patient.csv", index=False)

    aggregate = {
        "patients_completed": int(len(summary_df)),
        "patients_with_completed_evaluation": int((summary_df["evaluation_status"] == "completed").sum()) if "evaluation_status" in summary_df else 0,
        "patients_with_skipped_evaluation": int((summary_df["evaluation_status"] == "skipped").sum()) if "evaluation_status" in summary_df else 0,
        "images_total": int(summary_df["images_total"].fillna(0).sum()) if "images_total" in summary_df else 0,
        "avg_semantic_accuracy": float(summary_df["semantic_accuracy"].dropna().mean()) if "semantic_accuracy" in summary_df and summary_df["semantic_accuracy"].notna().any() else None,
        "avg_precision": float(summary_df["precision"].dropna().mean()) if "precision" in summary_df and summary_df["precision"].notna().any() else None,
        "avg_recall": float(summary_df["recall"].dropna().mean()) if "recall" in summary_df and summary_df["recall"].notna().any() else None,
        "avg_f1": float(summary_df["f1"].dropna().mean()) if "f1" in summary_df and summary_df["f1"].notna().any() else None,
        "total_mismatches": int(summary_df["mismatches"].fillna(0).sum()) if "mismatches" in summary_df else 0,
        "total_conflict_keys": int(summary_df["conflict_keys"].fillna(0).sum()) if "conflict_keys" in summary_df else 0,
        "total_resolved_by_code": int(summary_df["resolved_by_code"].fillna(0).sum()) if "resolved_by_code" in summary_df else 0,
        "total_resolved_by_llm_batch": int(summary_df["resolved_by_llm_batch"].fillna(0).sum()) if "resolved_by_llm_batch" in summary_df else 0,
        "total_resolved_by_llm_single": int(summary_df["resolved_by_llm_single"].fillna(0).sum()) if "resolved_by_llm_single" in summary_df else 0,
        "total_requests": int(summary_df["request_count"].fillna(0).sum()) if "request_count" in summary_df else 0,
        "total_input_tokens": int(summary_df["input_tokens"].fillna(0).sum()) if "input_tokens" in summary_df else 0,
        "total_output_tokens": int(summary_df["output_tokens"].fillna(0).sum()) if "output_tokens" in summary_df else 0,
        "total_tokens": int(summary_df["total_tokens"].fillna(0).sum()) if "total_tokens" in summary_df else 0,
        "total_elapsed_seconds": float(summary_df["elapsed_seconds"].fillna(0).sum()) if "elapsed_seconds" in summary_df else 0.0,
    }
    (output_root / "summary_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggregate


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resumable live OCR/router/map/resolve batch runner")
    ap.add_argument("--input_root", type=str, default="paper_patients")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--patient_start", type=int, default=1)
    ap.add_argument("--patient_end", type=int, default=10)
    ap.add_argument("--ocr_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--route_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--map_model_id", type=str, default="gpt-5.1")
    ap.add_argument("--resolver_model_id", type=str, default="gpt-5.4")
    ap.add_argument("--map_agent_count", type=int, default=1)
    ap.add_argument("--map_agent_count_night", type=int, default=5)
    ap.add_argument("--map_agent_count_morning", type=int, default=1)
    ap.add_argument("--map_agent_count_psg", type=int, default=2)
    ap.add_argument("--map_agent_count_cpap", type=int, default=2)
    ap.add_argument("--ocr_concurrency", type=int, default=2)
    ap.add_argument("--map_concurrency", type=int, default=2)
    ap.add_argument("--pipeline_mode", type=str, default="ocr_map_resolve")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = build_args()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "input_root": args.input_root,
        "output_root": str(output_root),
        "patient_start": args.patient_start,
        "patient_end": args.patient_end,
        "ocr_model_id": args.ocr_model_id,
        "route_model_id": args.route_model_id,
        "map_model_id": args.map_model_id,
        "resolver_model_id": args.resolver_model_id,
        "map_agent_count": args.map_agent_count,
        "map_agent_count_night": args.map_agent_count_night,
        "map_agent_count_morning": args.map_agent_count_morning,
        "map_agent_count_psg": args.map_agent_count_psg,
        "map_agent_count_cpap": args.map_agent_count_cpap,
        "ocr_concurrency": args.ocr_concurrency,
        "map_concurrency": args.map_concurrency,
        "pipeline_mode": args.pipeline_mode,
        "started_at_epoch": time.time(),
    }
    (output_root / "batch_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for patient_idx in range(int(args.patient_start), int(args.patient_end) + 1):
        patient_name = _patient_name(patient_idx)
        patient_dir = output_root / patient_name
        summary_path = patient_dir / "ocr_map_summary.json"
        if args.resume and _is_patient_complete(summary_path, args.pipeline_mode):
            print(f"SKIP {patient_name} (already complete)", flush=True)
            continue
        if args.resume and summary_path.exists():
            print(f"RETRY {patient_name} (found incomplete prior run)", flush=True)

        patient_dir.mkdir(parents=True, exist_ok=True)
        print(f"RUN {patient_name}", flush=True)
        cmd = [
            sys.executable,
            str(UNIFIED_PATH),
            "--input_root", args.input_root,
            "--patient_name", patient_name,
            "--output_dir", str(patient_dir),
            "--pipeline_mode", args.pipeline_mode,
            "--ocr_model_id", args.ocr_model_id,
            "--route_model_id", args.route_model_id,
            "--map_model_id", args.map_model_id,
            "--resolver_model_id", args.resolver_model_id,
            "--map_agent_count", str(args.map_agent_count),
            "--map_agent_count_night", str(args.map_agent_count_night),
            "--map_agent_count_morning", str(args.map_agent_count_morning),
            "--map_agent_count_psg", str(args.map_agent_count_psg),
            "--map_agent_count_cpap", str(args.map_agent_count_cpap),
            "--ocr_concurrency", str(args.ocr_concurrency),
            "--map_concurrency", str(args.map_concurrency),
            "--eval_reference_index", str(patient_idx),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        if not summary_path.exists():
            raise RuntimeError(f"Run for {patient_name} finished without {summary_path.name}")
        fatal_error = _find_fatal_patient_error(patient_dir)
        if fatal_error and not _is_patient_complete(summary_path, args.pipeline_mode):
            raise RuntimeError(f"Fatal error while processing {patient_name}: {fatal_error}")
        aggregate = refresh_summary(output_root)
        print(
            f"DONE {patient_name} | patients_completed={aggregate['patients_completed']} "
            f"avg_semantic_accuracy={aggregate['avg_semantic_accuracy']}",
            flush=True,
        )

    aggregate = refresh_summary(output_root)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
