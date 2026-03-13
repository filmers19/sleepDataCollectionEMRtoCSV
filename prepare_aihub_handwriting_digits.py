from __future__ import annotations

import argparse
import io
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from PIL import Image


NUMERIC_RE = re.compile(r"^[0-9]+$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export digit-focused crop manifests from the AI-Hub large handwriting OCR dataset."
    )
    ap.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("대용량손글씨OCR데이터") / "01.데이터",
        help="Root directory containing 1.Training and 2.Validation.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data") / "qwen35_digit_ft" / "aihub_handwriting_digits",
    )
    ap.add_argument("--pad_px", type=int, default=24, help="Context padding around each bbox crop.")
    ap.add_argument("--min_text_len", type=int, default=1)
    ap.add_argument("--max_text_len", type=int, default=4)
    ap.add_argument("--train_limit", type=int, default=0, help="Max exported train crops. 0 means all.")
    ap.add_argument("--val_limit", type=int, default=0, help="Max exported val crops. 0 means all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def ensure_clean_output(path: Path, force: bool) -> None:
    manifest_train = path / "train" / "train.jsonl"
    manifest_val = path / "val" / "val.jsonl"
    if force and path.exists():
        for child in sorted(path.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
        for child in sorted(path.glob("**/*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
    if manifest_train.exists() or manifest_val.exists():
        raise RuntimeError(f"{path} already contains manifests. Use --force to overwrite.")
    (path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (path / "val" / "images").mkdir(parents=True, exist_ok=True)


def build_source_index(zip_paths: Iterable[Path]) -> Tuple[Dict[str, Tuple[Path, str]], List[str]]:
    index: Dict[str, Tuple[Path, str]] = {}
    bad_archives: List[str] = []
    for zip_path in zip_paths:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".png", ".jpg", ".jpeg")):
                        index[name] = (zip_path, name)
        except zipfile.BadZipFile:
            bad_archives.append(str(zip_path))
    return index, bad_archives


def iter_label_entries(label_zip: Path) -> Iterator[Tuple[str, dict]]:
    with zipfile.ZipFile(label_zip) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            data = json.loads(zf.read(name).decode("utf-8"))
            yield name, data


def numeric_bbox_records(
    label_entries: Iterable[Tuple[str, dict]], min_len: int, max_len: int
) -> List[dict]:
    rows: List[dict] = []
    for label_name, payload in label_entries:
        image_ext = payload.get("Images", {}).get("type", "png")
        source_name = label_name
        if source_name.startswith("라벨/"):
            source_name = source_name[len("라벨/") :]
        source_name = str(Path(source_name).with_suffix(f".{image_ext}"))
        for bbox in payload.get("bbox", []):
            text = str(bbox.get("data", "")).strip()
            if not text or not NUMERIC_RE.fullmatch(text):
                continue
            if len(text) < min_len or len(text) > max_len:
                continue
            rows.append(
                {
                    "label_name": label_name,
                    "source_name": source_name,
                    "text": text,
                    "bbox_id": int(bbox.get("id", 0)),
                    "x": [int(v) for v in bbox.get("x", [])],
                    "y": [int(v) for v in bbox.get("y", [])],
                }
            )
    return rows


def clamp_crop_box(xs: List[int], ys: List[int], width: int, height: int, pad_px: int) -> Tuple[int, int, int, int]:
    left = max(0, min(xs) - pad_px)
    top = max(0, min(ys) - pad_px)
    right = min(width, max(xs) + pad_px)
    bottom = min(height, max(ys) + pad_px)
    return left, top, right, bottom


def extract_one_image(source_index: Dict[str, Tuple[Path, str]], source_name: str) -> Image.Image:
    zip_path, entry_name = source_index[source_name]
    with zipfile.ZipFile(zip_path) as zf:
        payload = zf.read(entry_name)
    img = Image.open(io.BytesIO(payload)).convert("RGB")
    return img


def export_split(
    split_name: str,
    rows: List[dict],
    out_root: Path,
    source_index: Dict[str, Tuple[Path, str]],
    pad_px: int,
    limit: int,
) -> int:
    manifest_path = out_root / split_name / f"{split_name}.jsonl"
    img_dir = out_root / split_name / "images"
    exported = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, row in enumerate(rows):
            if limit > 0 and exported >= limit:
                break
            if row["source_name"] not in source_index:
                continue
            image = extract_one_image(source_index, row["source_name"])
            left, top, right, bottom = clamp_crop_box(
                row["x"], row["y"], image.width, image.height, pad_px=pad_px
            )
            crop = image.crop((left, top, right, bottom))
            rel_name = f"{split_name}_{exported:07d}_{row['text']}_bbox{row['bbox_id']}.png"
            out_path = img_dir / rel_name
            crop.save(out_path)
            record = {
                "image": f"images/{rel_name}",
                "prompt": "Transcribe the handwritten digits exactly. Output only the digits.",
                "target_text": row["text"],
                "source_name": row["source_name"],
                "bbox_id": row["bbox_id"],
                "crop_box": [left, top, right, bottom],
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
    return exported


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    ensure_clean_output(output_dir, force=bool(args.force))

    train_source_zips = sorted((dataset_root / "1.Training" / "원천데이터").glob("*.zip"))
    val_source_zips = sorted((dataset_root / "2.Validation" / "원천데이터").glob("*.zip"))
    train_label_zip = dataset_root / "1.Training" / "라벨링데이터" / "TL.zip"
    val_label_zip = dataset_root / "2.Validation" / "라벨링데이터" / "VL.zip"

    if not train_source_zips or not val_source_zips or not train_label_zip.exists() or not val_label_zip.exists():
        raise RuntimeError("Could not find expected AI-Hub zip files under dataset_root.")

    train_index, bad_train_archives = build_source_index(train_source_zips)
    val_index, bad_val_archives = build_source_index(val_source_zips)

    train_rows = numeric_bbox_records(
        iter_label_entries(train_label_zip),
        min_len=int(args.min_text_len),
        max_len=int(args.max_text_len),
    )
    val_rows = numeric_bbox_records(
        iter_label_entries(val_label_zip),
        min_len=int(args.min_text_len),
        max_len=int(args.max_text_len),
    )

    random.shuffle(train_rows)
    random.shuffle(val_rows)

    train_count = export_split(
        "train",
        train_rows,
        output_dir,
        train_index,
        pad_px=int(args.pad_px),
        limit=int(args.train_limit),
    )
    val_count = export_split(
        "val",
        val_rows,
        output_dir,
        val_index,
        pad_px=int(args.pad_px),
        limit=int(args.val_limit),
    )

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "train_candidates": len(train_rows),
        "val_candidates": len(val_rows),
        "train_exported": train_count,
        "val_exported": val_count,
        "pad_px": int(args.pad_px),
        "min_text_len": int(args.min_text_len),
        "max_text_len": int(args.max_text_len),
        "bad_train_archives": bad_train_archives,
        "bad_val_archives": bad_val_archives,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
