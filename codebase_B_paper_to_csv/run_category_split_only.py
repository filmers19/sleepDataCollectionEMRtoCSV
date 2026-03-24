from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_patient_dirs(run_root: Path, patients: Sequence[str]) -> List[Path]:
    wanted = {p.strip() for p in patients if p.strip()}
    dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("Patient_")], key=lambda p: p.name)
    if wanted:
        dirs = [p for p in dirs if p.name in wanted]
    return dirs


def read_ocr_pairs(patient_dir: Path) -> List[Tuple[str, str]]:
    ocr_dir = patient_dir / "ocr_pages"
    pairs: List[Tuple[str, str]] = []
    for txt_path in sorted(ocr_dir.glob("*.txt")):
        pairs.append((txt_path.stem + ".jpg", txt_path.read_text(encoding="utf-8")))
    if not pairs:
        raise FileNotFoundError(f"No OCR page texts found under {ocr_dir}")
    return pairs


async def run_split_for_patient(
    patient_name: str,
    image_name_text_pairs: Sequence[Tuple[str, str]],
    output_dir: Path,
    pipeline_mod: Any,
    unified_mod: Any,
    llm: Any,
    max_attempts: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    map_page_dir = output_dir / "map_pages"
    map_page_dir.mkdir(parents=True, exist_ok=True)

    merged = pipeline_mod.merge_ocr_text_blocks(list(image_name_text_pairs))
    (output_dir / f"{patient_name}_ocr_merged.txt").write_text(merged, encoding="utf-8")
    numbered = pipeline_mod.build_numbered_merged_ocr_text(merged)
    (output_dir / f"{patient_name}_ocr_merged_numbered.txt").write_text(numbered, encoding="utf-8")

    category_records = await unified_mod.split_patient_ocr_categories(
        llm=llm,
        pipeline_mod=pipeline_mod,
        image_name_text_pairs=image_name_text_pairs,
        max_attempts=max_attempts,
    )
    (output_dir / "category_split_result.json").write_text(
        json.dumps(category_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for record in category_records:
        category = str(record["category"])
        (map_page_dir / f"category__{category}.txt").write_text(str(record["merged_text"]), encoding="utf-8")


async def amain(args: argparse.Namespace) -> None:
    unified_mod = load_module(SCRIPT_DIR / "111_unified_ocr_map_pipeline.py", "split_only_unified_111")
    pipeline_mod = load_module(SCRIPT_DIR / "103_paper_to_cdm_SA.py", "split_only_pipeline_103")

    backend = unified_mod.build_text_backend(
        model_id=args.model_id,
        max_new_tokens=4000,
        temperature=0.0,
        top_p=1.0,
        max_inflight=max(1, int(args.max_inflight)),
        timeout_sec=float(args.timeout_sec),
        max_retries=max(0, int(args.max_retries)),
        openai_api_key_env=str(args.openai_api_key_env),
        gemini_api_key_env=str(args.gemini_api_key_env),
        dtype="auto",
        attn_implementation="sdpa",
        disable_trust_remote_code=False,
        rate_limit_overrides=None,
    )

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    patient_dirs = discover_patient_dirs(input_root, args.patients)
    if not patient_dirs:
        raise SystemExit(f"No matching Patient_* directories found under {input_root}")

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_id": args.model_id,
        "patients": [p.name for p in patient_dirs],
        "mode": "category_split_only",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        pairs = read_ocr_pairs(patient_dir)
        await run_split_for_patient(
            patient_name=patient_name,
            image_name_text_pairs=pairs,
            output_dir=output_root / patient_name,
            pipeline_mod=pipeline_mod,
            unified_mod=unified_mod,
            llm=backend,
            max_attempts=max(1, int(args.max_attempts)),
        )
        print(f"DONE {patient_name}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run category split only on existing OCR page texts.")
    ap.add_argument("--input_root", type=str, required=True, help="Existing run root containing Patient_*/ocr_pages/*.txt")
    ap.add_argument("--output_root", type=str, required=True, help="Where split-only outputs will be written")
    ap.add_argument("--patients", type=str, default="", help="Comma-separated patient names, e.g. Patient_03,Patient_12")
    ap.add_argument("--model_id", type=str, default="gpt-5.4")
    ap.add_argument("--max_attempts", type=int, default=2)
    ap.add_argument("--max_inflight", type=int, default=2)
    ap.add_argument("--timeout_sec", type=float, default=120.0)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--gemini_api_key_env", type=str, default="GOOGLE_API_KEY")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.patients.strip():
        args.patients = [p.strip() for p in str(args.patients).split(",") if p.strip()]
    else:
        args.patients = []
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
