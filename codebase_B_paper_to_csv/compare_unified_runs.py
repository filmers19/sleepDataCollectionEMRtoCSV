from __future__ import annotations

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_files(folder: Path) -> Dict[str, str]:
    if not folder.exists():
        return {}
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.txt"))}


def read_json_files(folder: Path, suffix: str) -> Dict[str, Dict[str, Any]]:
    if not folder.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(folder.glob(f"*{suffix}")):
        out[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def read_single_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def compare_ocr(dir_a: Path, dir_b: Path) -> Dict[str, Any]:
    files_a = read_text_files(dir_a / "ocr_pages")
    files_b = read_text_files(dir_b / "ocr_pages")
    names = sorted(set(files_a) | set(files_b))
    identical = 0
    compared = 0
    ratios: List[float] = []
    changed: List[Dict[str, Any]] = []
    for name in names:
        a = files_a.get(name)
        b = files_b.get(name)
        if a is None or b is None:
            changed.append({"file": name, "status": "missing_in_one_side"})
            continue
        compared += 1
        ratio = SequenceMatcher(None, a, b).ratio()
        ratios.append(ratio)
        if a == b:
            identical += 1
        else:
            changed.append(
                {
                    "file": name,
                    "status": "different",
                    "similarity": round(ratio, 4),
                    "chars_a": len(a),
                    "chars_b": len(b),
                }
            )
    return {
        "files_a": len(files_a),
        "files_b": len(files_b),
        "compared": compared,
        "identical": identical,
        "different": len(changed),
        "avg_similarity": (sum(ratios) / len(ratios)) if ratios else None,
        "examples": changed[:20],
    }


def compare_route(dir_a: Path, dir_b: Path) -> Dict[str, Any]:
    metas_a = read_json_files(dir_a / "map_pages", ".meta.json")
    metas_b = read_json_files(dir_b / "map_pages", ".meta.json")
    names = sorted(set(metas_a) | set(metas_b))
    same = 0
    diffs: List[Dict[str, Any]] = []
    for name in names:
        a = metas_a.get(name)
        b = metas_b.get(name)
        if not a or not b:
            diffs.append({"file": name, "status": "missing_in_one_side"})
            continue
        route_a = a.get("map_route")
        route_b = b.get("map_route")
        if route_a == route_b:
            same += 1
        else:
            diffs.append({"file": name, "route_a": route_a, "route_b": route_b})
    return {
        "files_a": len(metas_a),
        "files_b": len(metas_b),
        "same_route": same,
        "different_route": len(diffs),
        "examples": diffs[:20],
    }


def compare_map(dir_a: Path, dir_b: Path) -> Dict[str, Any]:
    vals_a = read_json_files(dir_a / "map_pages", ".valid.json")
    vals_b = read_json_files(dir_b / "map_pages", ".valid.json")
    names = sorted(set(vals_a) | set(vals_b))
    same_pages = 0
    page_diffs: List[Dict[str, Any]] = []
    total_shared_keys = 0
    total_same_values = 0
    for name in names:
        a = vals_a.get(name)
        b = vals_b.get(name)
        if a is None or b is None:
            page_diffs.append({"file": name, "status": "missing_in_one_side"})
            continue
        if a == b:
            same_pages += 1
            total_shared_keys += len(a)
            total_same_values += len(a)
            continue
        keys_a = set(a)
        keys_b = set(b)
        shared = keys_a & keys_b
        same_vals = sum(1 for key in shared if a.get(key) == b.get(key))
        total_shared_keys += len(shared)
        total_same_values += same_vals
        page_diffs.append(
            {
                "file": name,
                "keys_a": len(keys_a),
                "keys_b": len(keys_b),
                "shared_keys": len(shared),
                "same_values": same_vals,
            }
        )
    return {
        "files_a": len(vals_a),
        "files_b": len(vals_b),
        "same_pages": same_pages,
        "different_pages": len(page_diffs),
        "shared_key_value_match_rate": (total_same_values / total_shared_keys) if total_shared_keys else None,
        "examples": page_diffs[:20],
    }


def compare_final(dir_a: Path, dir_b: Path) -> Dict[str, Any]:
    csv_a = read_single_csv(next(iter(sorted(dir_a.glob("Patient_10.csv"))), dir_a / "Patient_10.csv"))
    csv_b = read_single_csv(next(iter(sorted(dir_b.glob("Patient_10.csv"))), dir_b / "Patient_10.csv"))
    keys = sorted(set(csv_a) | set(csv_b))
    same = 0
    diffs: List[Dict[str, Any]] = []
    for key in keys:
        va = csv_a.get(key)
        vb = csv_b.get(key)
        if va == vb:
            same += 1
        else:
            diffs.append({"field": key, "a": va, "b": vb})
    return {
        "fields_a": len(csv_a),
        "fields_b": len(csv_b),
        "same_fields": same,
        "different_fields": len(diffs),
        "examples": diffs[:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()
    report = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "summaries": {
            "ocr": compare_ocr(run_a, run_b),
            "route": compare_route(run_a, run_b),
            "map": compare_map(run_a, run_b),
            "final": compare_final(run_a, run_b),
        },
        "ocr_map_summary_a": load_json(run_a / "ocr_map_summary.json"),
        "ocr_map_summary_b": load_json(run_b / "ocr_map_summary.json"),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
