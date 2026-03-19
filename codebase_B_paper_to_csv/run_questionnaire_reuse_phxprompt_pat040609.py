#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PIPELINE_PATH = ROOT / "codebase_B_paper_to_csv" / "103_paper_to_cdm_SA.py"
UNIFIED_PATH = ROOT / "codebase_B_paper_to_csv" / "111_unified_ocr_map_pipeline.py"
SOURCE_ROOT = ROOT / "out_patient01to10_liveocr_gpt54_route_gpt54_map_gpt51multiroute_resolve_gpt54_resumable_20260317"
CDM_CSV = ROOT / "cdm_revised.csv"
EXAMPLE_CSV = ROOT / "example.csv"
OUTPUT_ROOT = ROOT / "out_questionnaire_reuse_phxprompt_pat040609_20260319"
TARGET_PATIENTS = ["Patient_04", "Patient_06", "Patient_09"]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _patient_reference_index(patient_name: str) -> int:
    return int(str(patient_name).split("_")[-1])


def _questionnaire_keys(pipeline_mod: Any, retriever: Any) -> List[str]:
    keys: Set[str] = set()
    for route in (
        pipeline_mod.MAP_ROUTE_NIGHT_QUESTIONNAIRE,
        pipeline_mod.MAP_ROUTE_MORNING_QUESTIONNAIRE,
    ):
        for row in retriever.route_rows(route):
            if row.key in getattr(pipeline_mod, "CORE_ALWAYS_KEYS", set()):
                continue
            keys.add(row.key)
    return sorted(keys)


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _evaluate_subset(row: Dict[str, Any], ref_row: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    total = len(list(keys))
    semantic_matches = 0
    correct_non_null = 0
    pred_non_null = 0
    ref_non_null = 0
    mismatches: List[Dict[str, str]] = []

    for key in keys:
        pred = _norm(row.get(key))
        gold = _norm(ref_row.get(key))
        if pred == gold:
            semantic_matches += 1
        else:
            mismatches.append({"key": key, "pred": pred, "gold": gold})
        if pred:
            pred_non_null += 1
        if gold:
            ref_non_null += 1
        if pred and gold and pred == gold:
            correct_non_null += 1

    semantic_accuracy = (semantic_matches / total) if total else 0.0
    precision = (correct_non_null / pred_non_null) if pred_non_null else 0.0
    recall = (correct_non_null / ref_non_null) if ref_non_null else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
    return {
        "key_count": total,
        "semantic_accuracy": semantic_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mismatches": mismatches,
    }


def _load_page_results_for_patient(source_dir: Path, pipeline_mod: Any) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
    page_results: List[Any] = []
    meta_by_bundle: Dict[str, Dict[str, Any]] = {}
    for meta_path in sorted((source_dir / "map_pages").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bundle = str(meta.get("bundle") or meta_path.stem)
        stem = Path(bundle).stem
        raw_path = source_dir / "map_pages" / f"{stem}.raw.json"
        valid_path = source_dir / "map_pages" / f"{stem}.valid.json"
        contexts_path = source_dir / "map_pages" / f"{stem}.contexts.json"
        rejected_path = source_dir / "map_pages" / f"{stem}.rejected.json"
        source_images = list(meta.get("source_images") or [])
        if len(source_images) != 1:
            raise RuntimeError(f"Expected exactly one source image in {meta_path}, got {source_images}")
        source_image = source_images[0]
        ocr_text = (source_dir / "ocr_pages" / f"{Path(source_image).stem}.txt").read_text(encoding="utf-8")
        raw_obj = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
        valid_obj = json.loads(valid_path.read_text(encoding="utf-8")) if valid_path.exists() else {}
        ctx_obj = json.loads(contexts_path.read_text(encoding="utf-8")) if contexts_path.exists() else {}
        rejected_obj = json.loads(rejected_path.read_text(encoding="utf-8")) if rejected_path.exists() else {}
        input_contexts = {k: (ctx_obj.get(k, {}) or {}).get("input_context", {}) for k in valid_obj.keys()}
        cdm_contexts = {k: str((ctx_obj.get(k, {}) or {}).get("CDM_Context", "")).strip() for k in valid_obj.keys()}
        page_results.append(
            pipeline_mod.PageResult(
                image_name=bundle,
                ocr_text=ocr_text,
                raw_json=raw_obj,
                valid_json=valid_obj,
                input_contexts=input_contexts,
                cdm_contexts=cdm_contexts,
                rejected_fields=rejected_obj,
            )
        )
        meta_by_bundle[bundle] = meta
    page_results.sort(key=lambda x: x.image_name)
    return page_results, meta_by_bundle


def _copy_ocr_files(source_dir: Path, patient_out: Path) -> None:
    ocr_dst = patient_out / "ocr_pages"
    ocr_dst.mkdir(parents=True, exist_ok=True)
    for src in sorted((source_dir / "ocr_pages").glob("*")):
        if src.is_file():
            shutil.copy2(src, ocr_dst / src.name)


def _write_page_artifacts(
    *,
    patient_out: Path,
    bundle_name: str,
    raw_obj: Dict[str, Any],
    valid_obj: Dict[str, Any],
    valid_contexts: Dict[str, Dict[str, Any]],
    valid_cdm_contexts: Dict[str, str],
    rejected_fields: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    map_page_dir = patient_out / "map_pages"
    map_page_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(bundle_name).stem
    (map_page_dir / f"{stem}.raw.json").write_text(json.dumps(raw_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    (map_page_dir / f"{stem}.valid.json").write_text(json.dumps(valid_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    (map_page_dir / f"{stem}.contexts.json").write_text(
        json.dumps(
            {
                k: {
                    "CDM_Context": valid_cdm_contexts.get(k, ""),
                    "input_context": valid_contexts.get(k, {}),
                }
                for k in valid_obj.keys()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if rejected_fields:
        (map_page_dir / f"{stem}.rejected.json").write_text(
            json.dumps(rejected_fields, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (map_page_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


async def amain() -> None:
    pipeline_mod = _load_module(PIPELINE_PATH, "paper_to_cdm_sa_questionnaire_phxprompt")
    unified_mod = _load_module(UNIFIED_PATH, "unified_runner_questionnaire_phxprompt")

    if callable(getattr(pipeline_mod, "load_env", None)):
        pipeline_mod.load_env()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    retriever = pipeline_mod.CDMRetriever(CDM_CSV)
    output_columns = list(pd.read_csv(EXAMPLE_CSV, nrows=0).columns)
    questionnaire_keys = _questionnaire_keys(pipeline_mod, retriever)
    reference_rows = _read_csv_rows(EXAMPLE_CSV)

    map_backends = [
        unified_mod.build_text_backend(
            model_id="gpt-5.1",
            max_new_tokens=4096,
            temperature=0.0,
            top_p=0.95,
            max_inflight=5,
            timeout_sec=180.0,
            max_retries=3,
            openai_api_key_env="OPENAI_API_KEY",
            gemini_api_key_env="GOOGLE_API_KEY",
            dtype="bfloat16",
            attn_implementation="sdpa",
            disable_trust_remote_code=False,
            rate_limit_overrides={},
            rate_limit_window_sec=60.0,
            rate_limit_margin=0.9,
        )
        for _ in range(5)
    ]
    resolver_backend = unified_mod.build_text_backend(
        model_id="gpt-5.4",
        max_new_tokens=2048,
        temperature=0.0,
        top_p=0.95,
        max_inflight=5,
        timeout_sec=180.0,
        max_retries=3,
        openai_api_key_env="OPENAI_API_KEY",
        gemini_api_key_env="GOOGLE_API_KEY",
        dtype="bfloat16",
        attn_implementation="sdpa",
        disable_trust_remote_code=False,
        rate_limit_overrides={},
        rate_limit_window_sec=60.0,
        rate_limit_margin=0.9,
    )

    started = time.perf_counter()
    patient_summaries: List[Dict[str, Any]] = []

    try:
        for patient_name in TARGET_PATIENTS:
            print(f"RUN {patient_name}", flush=True)
            patient_started = time.perf_counter()
            source_dir = SOURCE_ROOT / patient_name
            patient_out = OUTPUT_ROOT / patient_name
            patient_out.mkdir(parents=True, exist_ok=True)
            _copy_ocr_files(source_dir, patient_out)

            old_page_results, meta_by_bundle = _load_page_results_for_patient(source_dir, pipeline_mod)
            target_pages = []
            for old_pr in old_page_results:
                meta = meta_by_bundle[old_pr.image_name]
                route_name = str(meta.get("map_route") or pipeline_mod.DEFAULT_MAP_ROUTE)
                if route_name in {
                    str(pipeline_mod.MAP_ROUTE_NIGHT_QUESTIONNAIRE),
                    str(pipeline_mod.MAP_ROUTE_MORNING_QUESTIONNAIRE),
                }:
                    target_pages.append(
                        {
                            "bundle_name": old_pr.image_name,
                            "ocr_text": old_pr.ocr_text,
                            "route_name": route_name,
                            "meta": meta,
                        }
                    )

            official_by_bundle = pipeline_mod.classify_official_questionnaire_sequence(
                [
                    {
                        "bundle_name": item["bundle_name"],
                        "route_name": item["route_name"],
                        "ocr_text": item["ocr_text"],
                    }
                    for item in target_pages
                ]
            )

            new_page_results: List[Any] = []
            for item in target_pages:
                bundle_name = item["bundle_name"]
                route_name = item["route_name"]
                ocr_text = item["ocr_text"]
                meta = dict(item["meta"])
                official_info = official_by_bundle.get(bundle_name) or {
                    "official_questionnaire": False,
                    "official_family": "NON",
                }

                raw_obj, valid_obj, valid_contexts, valid_cdm_contexts, rejected_fields = await unified_mod.map_ocr_text_multi_agent(
                    llm=map_backends,
                    pipeline_mod=pipeline_mod,
                    retriever=retriever,
                    ocr_text=ocr_text,
                    map_agent_count=1,
                    map_agent_count_by_route={
                        str(pipeline_mod.MAP_ROUTE_NIGHT_QUESTIONNAIRE): 5,
                        str(pipeline_mod.MAP_ROUTE_MORNING_QUESTIONNAIRE): 1,
                        str(pipeline_mod.MAP_ROUTE_PSG_REPORT_GENERAL): 1,
                        str(pipeline_mod.MAP_ROUTE_PSG_REPORT_EXTENSIVE): 1,
                        str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_GENERAL): 1,
                        str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE): 1,
                        str(pipeline_mod.MAP_ROUTE_PSG_SIGNALS): 1,
                    },
                    route_name=route_name,
                    official_questionnaire=bool(official_info.get("official_questionnaire")),
                    official_family=str(official_info.get("official_family") or "NON"),
                    json_retry_attempts=3,
                    enable_recall=True,
                )

                meta.update(
                    {
                        "ok": True,
                        "reused_previous_ocr": True,
                        "reused_previous_route": True,
                        "questionnaire_only_rerun": True,
                        "map_model_id": "gpt-5.1",
                        "night_agent_count": 5,
                        "morning_agent_count": 1,
                        "official_questionnaire": bool(official_info.get("official_questionnaire")),
                        "official_questionnaire_family": str(official_info.get("official_family") or "NON"),
                        "cdm_csv": str(CDM_CSV.resolve()),
                    }
                )

                new_page_results.append(
                    pipeline_mod.PageResult(
                        image_name=bundle_name,
                        ocr_text=ocr_text,
                        raw_json=raw_obj,
                        valid_json=valid_obj,
                        input_contexts=valid_contexts,
                        cdm_contexts=valid_cdm_contexts,
                        rejected_fields=rejected_fields,
                    )
                )
                _write_page_artifacts(
                    patient_out=patient_out,
                    bundle_name=bundle_name,
                    raw_obj=raw_obj,
                    valid_obj=valid_obj,
                    valid_contexts=valid_contexts,
                    valid_cdm_contexts=valid_cdm_contexts,
                    rejected_fields=rejected_fields,
                    meta=meta,
                )

            new_page_results.sort(key=lambda x: x.image_name)
            patient_res = pipeline_mod.build_patient_result(
                patient_name=patient_name,
                page_results=new_page_results,
                duplicates=[],
                page_errors=[],
                output_columns=output_columns,
                save_intermediate=False,
                out_dir=patient_out,
                elapsed_s=(time.perf_counter() - patient_started),
            )

            if patient_res.get("row") is not None and patient_res.get("conflicts"):
                overrides, decisions = await unified_mod.resolve_conflicts(
                    llm=resolver_backend,
                    pipeline_mod=pipeline_mod,
                    retriever=retriever,
                    patient_name=patient_name,
                    conflicts=patient_res.get("conflicts") or {},
                    json_retry_attempts=2,
                )
                if overrides:
                    merged_like = dict(patient_res.get("merged") or {})
                    for key, value in overrides.items():
                        if key in output_columns:
                            merged_like[key] = value
                    patient_res["merged"] = merged_like
                    patient_res["row"] = pipeline_mod.build_output_row(merged_like, output_columns)
                patient_res["conflict_resolution"] = decisions

            pipeline_mod.write_patient_outputs(
                output_dir=patient_out,
                patient_name=patient_name,
                res=patient_res,
                output_columns=output_columns,
            )

            ref_idx = _patient_reference_index(patient_name) - 1
            ref_row = reference_rows[ref_idx]
            new_row = patient_res.get("row") or {}
            baseline_row = _read_csv_rows(source_dir / f"{patient_name}.csv")[0]
            new_eval = _evaluate_subset(new_row, ref_row, questionnaire_keys)
            baseline_eval = _evaluate_subset(baseline_row, ref_row, questionnaire_keys)

            mismatch_details = {
                "new_mismatches": new_eval["mismatches"],
                "baseline_mismatches": baseline_eval["mismatches"],
            }
            (patient_out / "questionnaire_eval_details.json").write_text(
                json.dumps(mismatch_details, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            patient_summary = {
                "patient_name": patient_name,
                "questionnaire_key_count": new_eval["key_count"],
                "rerun_pages": len(target_pages),
                "conflicts": len(patient_res.get("conflicts") or {}),
                "resolver_agent_count": sum(
                    1
                    for value in (patient_res.get("conflict_resolution") or {}).values()
                    if str(value.get("resolver_mode") or "").startswith("llm_")
                ),
                "elapsed_seconds": time.perf_counter() - patient_started,
                "new_semantic_accuracy": new_eval["semantic_accuracy"],
                "new_precision": new_eval["precision"],
                "new_recall": new_eval["recall"],
                "new_f1": new_eval["f1"],
                "new_mismatches": len(new_eval["mismatches"]),
                "baseline_semantic_accuracy": baseline_eval["semantic_accuracy"],
                "baseline_precision": baseline_eval["precision"],
                "baseline_recall": baseline_eval["recall"],
                "baseline_f1": baseline_eval["f1"],
                "baseline_mismatches": len(baseline_eval["mismatches"]),
                "gain_vs_baseline": new_eval["semantic_accuracy"] - baseline_eval["semantic_accuracy"],
                "mismatch_delta_vs_baseline": len(baseline_eval["mismatches"]) - len(new_eval["mismatches"]),
            }
            patient_summaries.append(patient_summary)
            print(json.dumps(patient_summary, ensure_ascii=False), flush=True)

        pd.DataFrame(patient_summaries).to_csv(OUTPUT_ROOT / "summary_by_patient.csv", index=False)

        df = pd.DataFrame(patient_summaries)
        aggregate = {
            "output_root": str(OUTPUT_ROOT),
            "source_root": str(SOURCE_ROOT),
            "target_patients": TARGET_PATIENTS,
            "questionnaire_key_count": len(questionnaire_keys),
            "avg_new_semantic_accuracy": float(df["new_semantic_accuracy"].mean()) if not df.empty else None,
            "avg_baseline_semantic_accuracy": float(df["baseline_semantic_accuracy"].mean()) if not df.empty else None,
            "avg_gain_vs_baseline": float(df["gain_vs_baseline"].mean()) if not df.empty else None,
            "total_new_mismatches": int(df["new_mismatches"].sum()) if not df.empty else 0,
            "total_baseline_mismatches": int(df["baseline_mismatches"].sum()) if not df.empty else 0,
            "total_mismatch_delta_vs_baseline": int(df["mismatch_delta_vs_baseline"].sum()) if not df.empty else 0,
            "elapsed_seconds": time.perf_counter() - started,
            "map_usage_summary": unified_mod.summarize_openai_usage(*map_backends),
            "resolver_usage_summary": unified_mod.summarize_openai_usage(resolver_backend),
        }
        (OUTPUT_ROOT / "summary_aggregate.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    finally:
        close_fn = getattr(resolver_backend, "close", None)
        if callable(close_fn):
            maybe = close_fn()
            if asyncio.iscoroutine(maybe):
                await maybe
        for backend in map_backends:
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                maybe = close_fn()
                if asyncio.iscoroutine(maybe):
                    await maybe


if __name__ == "__main__":
    asyncio.run(amain())
