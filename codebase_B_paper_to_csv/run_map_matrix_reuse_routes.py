#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "codebase_B_paper_to_csv" / "103_paper_to_cdm_SA.py"
UNIFIED_PATH = ROOT / "codebase_B_paper_to_csv" / "111_unified_ocr_map_pipeline.py"
CDM_CSV = ROOT / "cdm_revised.csv"
EXAMPLE_CSV = ROOT / "example.csv"

DEFAULT_PATIENT_SOURCES = {
    "Patient_01": ROOT / "out_first5_liveocr_gpt54_route_gpt54_map_gpt54_resolve_codefirst_20260315" / "Patient_01",
    "Patient_10": ROOT / "out_patient10_liveocr_gpt54_route_gpt54_map_gpt54_resolve_codefirst_20260316",
}

DEFAULT_MATRIX: List[Tuple[str, int]] = (
    [("gpt-5.4", n) for n in (2, 3, 4, 5)]
    + [("gpt-5.2", n) for n in (2, 3, 4, 5)]
    + [("gpt-5.1", n) for n in (2, 3, 4, 5)]
    + [("gpt-5-mini", n) for n in (5, 8)]
)


def _load_module(path: Path, name: str) -> Any:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_route_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    route_by_image: Dict[str, Dict[str, Any]] = {}
    for meta_path in sorted((run_dir / "map_pages").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_images = list(meta.get("source_images") or [])
        if len(source_images) != 1:
            raise RuntimeError(f"Expected 1 source image in {meta_path}, got {source_images}")
        route_by_image[source_images[0]] = meta
    return route_by_image


def _ordered_images_from_plan(run_dir: Path) -> List[str]:
    plan = json.loads((run_dir / "unified_plan.json").read_text(encoding="utf-8"))
    return list(plan.get("selected_images") or [])


def _summarize_eval(eval_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not eval_obj:
        return {
            "semantic_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "mismatches": None,
        }
    return {
        "semantic_accuracy": eval_obj.get("semantic_accuracy"),
        "precision": eval_obj.get("precision"),
        "recall": eval_obj.get("recall"),
        "f1": eval_obj.get("f1"),
        "mismatches": len(eval_obj.get("mismatches") or []),
    }


async def run_combo(
    *,
    unified_mod: Any,
    pipeline_mod: Any,
    model_id: str,
    night_agents: int,
    output_root: Path,
    patient_sources: Dict[str, Path],
    map_concurrency: int,
    request_timeout_sec: float,
    max_retries: int,
    enable_recall: bool,
) -> Dict[str, Any]:
    combo_slug = f"{model_id.replace('.', '').replace('-', '')}_night{night_agents}"
    combo_dir = output_root / combo_slug
    combo_dir.mkdir(parents=True, exist_ok=True)

    if callable(getattr(pipeline_mod, "load_env", None)):
        pipeline_mod.load_env()

    retriever = pipeline_mod.CDMRetriever(CDM_CSV)
    output_columns = list(pd.read_csv(EXAMPLE_CSV, nrows=0).columns)

    map_backends = [
        unified_mod.build_text_backend(
            model_id=model_id,
            max_new_tokens=4096,
            temperature=0.0,
            top_p=0.95,
            max_inflight=map_concurrency,
            timeout_sec=request_timeout_sec,
            max_retries=max_retries,
            openai_api_key_env="OPENAI_API_KEY",
            gemini_api_key_env="GOOGLE_API_KEY",
            dtype="bfloat16",
            attn_implementation="sdpa",
            disable_trust_remote_code=False,
            rate_limit_overrides={},
            rate_limit_window_sec=60.0,
            rate_limit_margin=0.9,
        )
        for _ in range(night_agents)
    ]
    resolver_backend = unified_mod.build_text_backend(
        model_id="gpt-5.4",
        max_new_tokens=2048,
        temperature=0.0,
        top_p=0.95,
        max_inflight=map_concurrency,
        timeout_sec=request_timeout_sec,
        max_retries=max_retries,
        openai_api_key_env="OPENAI_API_KEY",
        gemini_api_key_env="GOOGLE_API_KEY",
        dtype="bfloat16",
        attn_implementation="sdpa",
        disable_trust_remote_code=False,
        rate_limit_overrides={},
        rate_limit_window_sec=60.0,
        rate_limit_margin=0.9,
    )

    combo_started = time.perf_counter()
    patient_summaries: List[Dict[str, Any]] = []

    try:
        for patient_name, source_dir in patient_sources.items():
            patient_started = time.perf_counter()
            patient_out = combo_dir / patient_name
            ocr_page_dir = patient_out / "ocr_pages"
            map_page_dir = patient_out / "map_pages"
            ocr_page_dir.mkdir(parents=True, exist_ok=True)
            map_page_dir.mkdir(parents=True, exist_ok=True)

            ordered_images = _ordered_images_from_plan(source_dir)
            route_meta_by_image = _load_route_map(source_dir)
            page_results: List[Any] = []
            page_errors: List[Dict[str, str]] = []

            for idx, image_name in enumerate(ordered_images, start=1):
                stem = Path(image_name).stem
                ocr_txt_path = source_dir / "ocr_pages" / f"{stem}.txt"
                ocr_meta_src = source_dir / "ocr_pages" / f"{stem}.meta.json"
                if not ocr_txt_path.exists():
                    raise FileNotFoundError(f"Missing OCR text: {ocr_txt_path}")
                if image_name not in route_meta_by_image:
                    raise KeyError(f"Missing saved route meta for {patient_name}/{image_name}")

                merged_text = ocr_txt_path.read_text(encoding="utf-8")
                route_meta = route_meta_by_image[image_name]
                route_name = str(route_meta.get("map_route") or pipeline_mod.DEFAULT_MAP_ROUTE)

                raw_obj, valid_obj, valid_contexts, valid_cdm_contexts, rejected_fields = await unified_mod.map_ocr_text_multi_agent(
                    llm=map_backends,
                    pipeline_mod=pipeline_mod,
                    retriever=retriever,
                    ocr_text=merged_text,
                    map_agent_count=1,
                    map_agent_count_by_route={
                        str(pipeline_mod.MAP_ROUTE_NIGHT_QUESTIONNAIRE): night_agents,
                        str(pipeline_mod.MAP_ROUTE_MORNING_QUESTIONNAIRE): 1,
                        str(pipeline_mod.MAP_ROUTE_PSG_REPORT_GENERAL): 1,
                        str(pipeline_mod.MAP_ROUTE_PSG_REPORT_EXTENSIVE): 1,
                        str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_GENERAL): 1,
                        str(pipeline_mod.MAP_ROUTE_CPAP_PSG_REPORT_EXTENSIVE): 1,
                    },
                    route_name=route_name,
                    json_retry_attempts=3,
                    enable_recall=enable_recall,
                )

                bundle_name = pipeline_mod.make_bundle_image_name(idx, [image_name])
                page_results.append(
                    pipeline_mod.PageResult(
                        image_name=bundle_name,
                        ocr_text=merged_text,
                        raw_json=raw_obj,
                        valid_json=valid_obj,
                        input_contexts=valid_contexts,
                        cdm_contexts=valid_cdm_contexts,
                        rejected_fields=rejected_fields,
                    )
                )

                (ocr_page_dir / f"{stem}.txt").write_text(merged_text, encoding="utf-8")
                if ocr_meta_src.exists():
                    (ocr_page_dir / f"{stem}.meta.json").write_text(
                        ocr_meta_src.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                (map_page_dir / f"{Path(bundle_name).stem}.raw.json").write_text(
                    json.dumps(raw_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (map_page_dir / f"{Path(bundle_name).stem}.valid.json").write_text(
                    json.dumps(valid_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (map_page_dir / f"{Path(bundle_name).stem}.contexts.json").write_text(
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
                    (map_page_dir / f"{Path(bundle_name).stem}.rejected.json").write_text(
                        json.dumps(rejected_fields, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                map_meta = {
                    "bundle": bundle_name,
                    "source_images": [image_name],
                    "map_route": route_name,
                    "map_route_confidence": route_meta.get("map_route_confidence"),
                    "map_route_report_score": route_meta.get("map_route_report_score"),
                    "map_route_extensive_report_score": route_meta.get("map_route_extensive_report_score"),
                    "map_route_morning_score": route_meta.get("map_route_morning_score"),
                    "map_route_night_score": route_meta.get("map_route_night_score"),
                    "map_route_reason": route_meta.get("map_route_reason"),
                    "ok": True,
                    "elapsed_seconds": None,
                    "valid_keys": len(valid_obj),
                    "error": "",
                    "reused_route": True,
                    "reused_route_from": str((source_dir / "map_pages" / f"{Path(bundle_name).stem}.meta.json").resolve()),
                    "map_model_id": model_id,
                    "night_agent_count": night_agents,
                }
                (map_page_dir / f"{Path(bundle_name).stem}.meta.json").write_text(
                    json.dumps(map_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            page_results.sort(key=lambda x: x.image_name)
            patient_res = pipeline_mod.build_patient_result(
                patient_name=patient_name,
                page_results=page_results,
                duplicates=[],
                page_errors=page_errors,
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

            reference_index = 1 if patient_name == "Patient_01" else 10
            evaluation = unified_mod.evaluate_against_reference(
                output_dir=patient_out,
                patient_name=patient_name,
                row=patient_res.get("row") or {},
                example_csv=EXAMPLE_CSV,
                reference_name="",
                reference_index=reference_index,
            )
            patient_summary = {
                "patient_name": patient_name,
                "source_dir": str(source_dir),
                "output_dir": str(patient_out),
                "pages": len(ordered_images),
                "conflicts": len(patient_res.get("conflicts") or {}),
                "resolver_agent_count": sum(
                    1
                    for v in (patient_res.get("conflict_resolution") or {}).values()
                    if str(v.get("resolver_mode") or "").startswith("llm_")
                ),
                "elapsed_seconds": time.perf_counter() - patient_started,
                **_summarize_eval(evaluation),
            }
            patient_summaries.append(patient_summary)

        unified_mod.refresh_combined_patient_csv(output_dir=combo_dir, output_columns=output_columns)
        map_usage = unified_mod.summarize_openai_usage(*map_backends)
        resolver_usage = unified_mod.summarize_openai_usage(resolver_backend)
        result = {
            "model_id": model_id,
            "night_agents": night_agents,
            "combo_dir": str(combo_dir),
            "elapsed_seconds": time.perf_counter() - combo_started,
            "patients": patient_summaries,
            "map_usage_summary": map_usage,
            "resolver_usage_summary": resolver_usage,
        }
        (combo_dir / "combo_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
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


async def amain(args: argparse.Namespace) -> None:
    pipeline_mod = _load_module(PIPELINE_PATH, "paper_to_cdm_sa_matrix")
    unified_mod = _load_module(UNIFIED_PATH, "unified_runner_matrix")

    if not CDM_CSV.exists():
        raise FileNotFoundError(f"Missing CDM CSV: {CDM_CSV}")
    if not EXAMPLE_CSV.exists():
        raise FileNotFoundError(f"Missing example.csv: {EXAMPLE_CSV}")

    patient_sources = {k: v for k, v in DEFAULT_PATIENT_SOURCES.items()}
    for patient_name, src in patient_sources.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source run for {patient_name}: {src}")

    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    matrix = list(DEFAULT_MATRIX)
    if args.pairs:
        wanted: List[Tuple[str, int]] = []
        for item in args.pairs.split(","):
            token = str(item).strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid --pairs item: {token}. Expected model:count")
            model_id, count_raw = token.split(":", 1)
            wanted.append((model_id.strip(), int(count_raw.strip())))
        matrix = [pair for pair in matrix if pair in wanted]
        missing = [pair for pair in wanted if pair not in matrix]
        if missing:
            raise ValueError(f"--pairs requested unknown combinations: {missing}")
    all_results: List[Dict[str, Any]] = []
    for idx, (model_id, night_agents) in enumerate(matrix, start=1):
        combo_slug = f"{model_id.replace('.', '').replace('-', '')}_night{night_agents}"
        combo_dir = output_root / combo_slug
        combo_summary_path = combo_dir / "combo_summary.json"
        if combo_summary_path.exists() and not args.force:
            result = json.loads(combo_summary_path.read_text(encoding="utf-8"))
            all_results.append(result)
            print(
                json.dumps(
                    {
                        "skipped_existing": True,
                        "model_id": model_id,
                        "night_agents": night_agents,
                        "combo_dir": str(combo_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        started = time.perf_counter()
        print(f"[{idx}/{len(matrix)}] model={model_id} night_agents={night_agents}", flush=True)
        try:
            result = await run_combo(
                unified_mod=unified_mod,
                pipeline_mod=pipeline_mod,
                model_id=model_id,
                night_agents=night_agents,
                output_root=output_root,
                patient_sources=patient_sources,
                map_concurrency=args.map_concurrency,
                request_timeout_sec=args.request_timeout_sec,
                max_retries=args.max_retries,
                enable_recall=args.enable_recall,
            )
        except Exception as exc:
            result = {
                "model_id": model_id,
                "night_agents": night_agents,
                "combo_dir": str(combo_dir),
                "elapsed_seconds": time.perf_counter() - started,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "patients": [],
            }
            combo_dir.mkdir(parents=True, exist_ok=True)
            combo_summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "model_id": model_id,
                        "night_agents": night_agents,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        all_results.append(result)
        print(
            json.dumps(
                {
                    "model_id": model_id,
                    "night_agents": night_agents,
                    "status": result.get("status", "ok"),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "patients": [
                        {
                            "patient_name": p["patient_name"],
                            "semantic_accuracy": p["semantic_accuracy"],
                            "mismatches": p["mismatches"],
                            "conflicts": p["conflicts"],
                            "resolver_agent_count": p["resolver_agent_count"],
                        }
                        for p in result.get("patients", [])
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    (output_root / "matrix_summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: List[Dict[str, Any]] = []
    for result in all_results:
        if result.get("status") == "failed":
            rows.append(
                {
                    "model_id": result["model_id"],
                    "night_agents": result["night_agents"],
                    "patient_name": "",
                    "semantic_accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "mismatches": None,
                    "conflicts": None,
                    "resolver_agent_count": None,
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "output_dir": result.get("combo_dir"),
                    "status": "failed",
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                }
            )
            continue
        for patient in result["patients"]:
            rows.append(
                {
                    "model_id": result["model_id"],
                    "night_agents": result["night_agents"],
                    "patient_name": patient["patient_name"],
                    "semantic_accuracy": patient["semantic_accuracy"],
                    "precision": patient["precision"],
                    "recall": patient["recall"],
                    "f1": patient["f1"],
                    "mismatches": patient["mismatches"],
                    "conflicts": patient["conflicts"],
                    "resolver_agent_count": patient["resolver_agent_count"],
                    "elapsed_seconds": patient["elapsed_seconds"],
                    "output_dir": patient["output_dir"],
                    "status": result.get("status", "ok"),
                    "error_type": "",
                    "error": "",
                }
            )
    pd.DataFrame(rows).to_csv(output_root / "matrix_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run map-only model/count matrix with reused OCR text and saved routes.")
    ap.add_argument(
        "--output_root",
        default="out_patient01_patient10_map_matrix_reuseocr_reuseroute_20260317",
        help="Output root directory, relative to repo root by default.",
    )
    ap.add_argument("--map_concurrency", type=int, default=1)
    ap.add_argument("--request_timeout_sec", type=float, default=180.0)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--enable_recall", action="store_true", default=True)
    ap.add_argument("--pairs", default="", help="Optional subset like gpt-5.4:2,gpt-5-mini:5")
    ap.add_argument("--force", action="store_true", help="Re-run combinations even if combo_summary.json already exists.")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
