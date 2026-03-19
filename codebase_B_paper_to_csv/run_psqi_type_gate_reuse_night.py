#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "codebase_B_paper_to_csv" / "103_paper_to_cdm_SA.py"
UNIFIED_PATH = ROOT / "codebase_B_paper_to_csv" / "111_unified_ocr_map_pipeline.py"
SOURCE_ROOT = ROOT / "out_patient01to10_liveocr_gpt54_route_gpt54_map_gpt51multiroute_resolve_gpt54_resumable_20260317"
CDM_CSV = ROOT / "cdm_revised_types.csv"
EXAMPLE_CSV = ROOT / "example.csv"
PSQI_0104_PREFIXES = ("PSQI_01_", "PSQI_02_", "PSQI_03_", "PSQI_04_")


def _load_module(path: Path, name: str) -> Any:
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_target_patients() -> List[str]:
    run_csv = SOURCE_ROOT / "all_patients.csv"
    with run_csv.open(encoding="utf-8-sig", newline="") as f:
        run_rows = list(csv.DictReader(f))
    with EXAMPLE_CSV.open(encoding="utf-8-sig", newline="") as f:
        ref_rows = list(csv.DictReader(f))
    keys = [k for k in run_rows[0].keys() if any(k.startswith(p) for p in PSQI_0104_PREFIXES)]
    out: List[str] = []
    for idx, (run_row, ref_row) in enumerate(zip(run_rows, ref_rows), start=1):
        if any((run_row.get(k) or "").strip() != (ref_row.get(k) or "").strip() for k in keys):
            out.append(f"Patient_{idx:02d}")
    return out


def _patient_reference_index(patient_name: str) -> int:
    return int(str(patient_name).split("_")[-1])


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


def _psqi_0104_mismatch_summary(row: Dict[str, Any], patient_name: str) -> Dict[str, Any]:
    ref_idx = _patient_reference_index(patient_name)
    with EXAMPLE_CSV.open(encoding="utf-8-sig", newline="") as f:
        ref_rows = list(csv.DictReader(f))
    ref_row = ref_rows[ref_idx - 1]
    mismatches: List[Dict[str, str]] = []
    for key in row.keys():
        if not any(key.startswith(p) for p in PSQI_0104_PREFIXES):
            continue
        pred = str(row.get(key) or "").strip()
        gold = str(ref_row.get(key) or "").strip()
        if pred != gold:
            mismatches.append({"key": key, "pred": pred, "gold": gold})
    return {"psqi_0104_mismatches": len(mismatches), "psqi_0104_mismatch_details": mismatches}


async def amain() -> None:
    pipeline_mod = _load_module(PIPELINE_PATH, "paper_to_cdm_sa_psqi_type")
    unified_mod = _load_module(UNIFIED_PATH, "unified_runner_psqi_type")

    if callable(getattr(pipeline_mod, "load_env", None)):
        pipeline_mod.load_env()

    target_patients = _find_target_patients()
    output_root = ROOT / f"out_psqi_typeA_night_reuse_20260318"
    output_root.mkdir(parents=True, exist_ok=True)

    retriever = pipeline_mod.CDMRetriever(CDM_CSV)
    output_columns = list(pd.read_csv(EXAMPLE_CSV, nrows=0).columns)

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

    patient_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for patient_name in target_patients:
            patient_started = time.perf_counter()
            source_dir = SOURCE_ROOT / patient_name
            patient_out = output_root / patient_name
            patient_out.mkdir(parents=True, exist_ok=True)
            _copy_ocr_files(source_dir, patient_out)
            old_page_results, meta_by_bundle = _load_page_results_for_patient(source_dir, pipeline_mod)
            new_page_results: List[Any] = []
            rerun_night_bundles = 0

            for old_pr in old_page_results:
                meta = dict(meta_by_bundle[old_pr.image_name])
                route_name = str(meta.get("map_route") or pipeline_mod.DEFAULT_MAP_ROUTE)
                if route_name == str(pipeline_mod.MAP_ROUTE_NIGHT_QUESTIONNAIRE):
                    rerun_night_bundles += 1
                    raw_obj, valid_obj, valid_contexts, valid_cdm_contexts, rejected_fields = await unified_mod.map_ocr_text_multi_agent(
                        llm=map_backends,
                        pipeline_mod=pipeline_mod,
                        retriever=retriever,
                        ocr_text=old_pr.ocr_text,
                        map_agent_count=1,
                        map_agent_count_by_route={
                            str(pipeline_mod.MAP_ROUTE_NIGHT_QUESTIONNAIRE): 5,
                            str(pipeline_mod.MAP_ROUTE_MORNING_QUESTIONNAIRE): 1,
                            str(pipeline_mod.MAP_ROUTE_PSG_REPORT_GENERAL): 1,
                            str(pipeline_mod.MAP_ROUTE_PSG_REPORT_EXTENSIVE): 1,
                            str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_GENERAL): 1,
                            str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE): 1,
                        },
                        route_name=route_name,
                        json_retry_attempts=3,
                        enable_recall=True,
                    )
                    meta.update(
                        {
                            "ok": True,
                            "rerun_night_with_type_gate": True,
                            "map_model_id": "gpt-5.1",
                            "night_agent_count": 5,
                            "cdm_csv": str(CDM_CSV.resolve()),
                        }
                    )
                    pr = pipeline_mod.PageResult(
                        image_name=old_pr.image_name,
                        ocr_text=old_pr.ocr_text,
                        raw_json=raw_obj,
                        valid_json=valid_obj,
                        input_contexts=valid_contexts,
                        cdm_contexts=valid_cdm_contexts,
                        rejected_fields=rejected_fields,
                    )
                else:
                    meta.update({"reused_previous_map": True})
                    pr = old_pr
                    raw_obj = old_pr.raw_json
                    valid_obj = old_pr.valid_json
                    valid_contexts = old_pr.input_contexts
                    valid_cdm_contexts = old_pr.cdm_contexts
                    rejected_fields = old_pr.rejected_fields

                _write_page_artifacts(
                    patient_out=patient_out,
                    bundle_name=pr.image_name,
                    raw_obj=raw_obj,
                    valid_obj=valid_obj,
                    valid_contexts=valid_contexts,
                    valid_cdm_contexts=valid_cdm_contexts,
                    rejected_fields=rejected_fields,
                    meta=meta,
                )
                new_page_results.append(pr)

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

            evaluation = unified_mod.evaluate_against_reference(
                output_dir=patient_out,
                patient_name=patient_name,
                row=patient_res.get("row") or {},
                example_csv=EXAMPLE_CSV,
                reference_name="",
                reference_index=_patient_reference_index(patient_name),
            )
            row = patient_res.get("row") or {}
            psqi_extra = _psqi_0104_mismatch_summary(row, patient_name)
            patient_summary = {
                "patient_name": patient_name,
                "rerun_night_bundles": rerun_night_bundles,
                "conflicts": len(patient_res.get("conflicts") or {}),
                "resolver_agent_count": sum(
                    1
                    for v in (patient_res.get("conflict_resolution") or {}).values()
                    if str(v.get("resolver_mode") or "").startswith("llm_")
                ),
                "elapsed_seconds": time.perf_counter() - patient_started,
                "semantic_accuracy": evaluation.get("semantic_accuracy"),
                "precision": evaluation.get("precision"),
                "recall": evaluation.get("recall"),
                "f1": evaluation.get("f1"),
                "mismatches": len(evaluation.get("mismatches") or []),
                **psqi_extra,
            }
            patient_summaries.append(patient_summary)
            all_rows.append(row)

        unified_mod.refresh_combined_patient_csv(output_dir=output_root, output_columns=output_columns)
        summary = {
            "output_root": str(output_root),
            "source_root": str(SOURCE_ROOT),
            "cdm_csv": str(CDM_CSV),
            "patients": patient_summaries,
            "elapsed_seconds": time.perf_counter() - started,
            "map_usage_summary": unified_mod.summarize_openai_usage(*map_backends),
            "resolver_usage_summary": unified_mod.summarize_openai_usage(resolver_backend),
        }
        (output_root / "summary_by_patient.csv").write_text(
            pd.DataFrame(patient_summaries).to_csv(index=False),
            encoding="utf-8",
        )
        (output_root / "summary_aggregate.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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
