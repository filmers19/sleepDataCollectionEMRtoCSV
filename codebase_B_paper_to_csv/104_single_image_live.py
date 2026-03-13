from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_script_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_pipeline_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("paper_to_cdm_sa", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    # Required for dataclass/type resolution on Python 3.13 during dynamic import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


async def run_one(args: argparse.Namespace) -> None:
    module_path = resolve_script_path(args.pipeline_script)
    mod = load_pipeline_module(module_path)

    mod.load_env()
    os.environ["GEMINI_MODEL"] = args.model
    mod.REQUEST_THROTTLE.configure(args.request_delay_sec)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mod.configure_logging(output_dir=output_dir, debug=args.debug, log_filename=args.log_filename)

    image_path = resolve_repo_path(args.image_path)
    cdm_csv = resolve_repo_path(args.cdm_csv)
    example_csv = resolve_repo_path(args.example_csv)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not cdm_csv.exists():
        raise FileNotFoundError(f"CDM CSV not found: {cdm_csv}")
    if not example_csv.exists():
        raise FileNotFoundError(f"example.csv not found: {example_csv}")

    retriever = mod.CDMRetriever(cdm_csv)
    map_agents = [] if args.disable_split_map_agents else mod.build_map_agent_specs(retriever)
    llm = mod.build_gemini()

    page_result = await mod.image_to_cdm_json(
        llm=llm,
        retriever=retriever,
        image_path=image_path,
        map_agents=map_agents,
        top_k=args.top_k,
    )

    output_columns = list(pd.read_csv(example_csv, nrows=0).columns)
    row = mod.build_output_row(page_result.valid_json, output_columns)

    stem = image_path.stem
    (output_dir / f"{stem}.ocr.txt").write_text(page_result.ocr_text or "", encoding="utf-8")
    (output_dir / f"{stem}.raw.json").write_text(
        json.dumps(page_result.raw_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{stem}.valid.json").write_text(
        json.dumps(page_result.valid_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{stem}.contexts.json").write_text(
        json.dumps(page_result.input_contexts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{stem}.rejected.json").write_text(
        json.dumps(page_result.rejected_fields, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{stem}.row.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([row]).to_csv(output_dir / f"{stem}.row.csv", index=False, encoding="utf-8-sig")

    print(f"Done. Outputs saved under: {output_dir}")
    print(f"- {stem}.ocr.txt")
    print(f"- {stem}.raw.json")
    print(f"- {stem}.valid.json")
    print(f"- {stem}.contexts.json")
    print(f"- {stem}.rejected.json")
    print(f"- {stem}.row.json")
    print(f"- {stem}.row.csv")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pipeline_script",
        type=str,
        default="103_paper_to_cdm_SA.py",
        help="Path to the main SA pipeline script",
    )
    ap.add_argument(
        "--image_path",
        type=str,
        default="paper_patients/Patient_04/S20260210204833505232_JUYOUNG.KIM1102_email_0045.jpg",
        help="Single image path to test",
    )
    ap.add_argument("--cdm_csv", type=str, default="cdm_revised.csv", help="CDM CSV path")
    ap.add_argument("--example_csv", type=str, default="example.csv", help="example.csv path")
    ap.add_argument("--output_dir", type=str, default="out_single_live", help="Output directory")
    ap.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model for live OCR/MAP",
    )
    ap.add_argument(
        "--disable_split_map_agents",
        action="store_true",
        help="Use single full-CDM map agent instead of split map agents",
    )
    ap.add_argument("--top_k", type=int, default=220, help="Legacy pass-through (currently not used in full-CDM mode)")
    ap.add_argument("--request_delay_sec", type=float, default=0.0, help="Delay between live LLM requests")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    ap.add_argument("--log_filename", type=str, default="single_image.log", help="Log file name in output_dir/logs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_one(args))


if __name__ == "__main__":
    main()
