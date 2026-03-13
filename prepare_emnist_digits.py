from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageOps
from torchvision.datasets import EMNIST


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare EMNIST Digits as image/text OCR crops.")
    ap.add_argument("--output_dir", type=str, default="data/qwen35_digit_ft/emnist_digits")
    ap.add_argument("--train_limit", type=int, default=50000)
    ap.add_argument("--val_limit", type=int, default=5000)
    ap.add_argument("--canvas_size", type=int, default=224)
    ap.add_argument("--digit_size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def ensure_output_dir(path: Path, force: bool) -> None:
    train_manifest = path / "train" / "train.jsonl"
    val_manifest = path / "val" / "val.jsonl"
    if train_manifest.exists() and val_manifest.exists() and not force:
        raise RuntimeError(
            f"{path} already contains prepared manifests. Use --force to rebuild, or reuse the existing dataset."
        )
    path.mkdir(parents=True, exist_ok=True)


def render_digit(image: Image.Image, canvas_size: int, digit_size: int) -> Image.Image:
    # EMNIST digits are grayscale; convert to a white RGB canvas with a large centered digit.
    image = ImageOps.invert(image.convert("L"))
    image = image.resize((digit_size, digit_size), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (canvas_size, canvas_size), color="white")
    x = (canvas_size - digit_size) // 2
    y = (canvas_size - digit_size) // 2
    canvas.paste(Image.merge("RGB", (image, image, image)), (x, y))
    return canvas


def write_split(
    split_name: str,
    rows: Iterable[Tuple[Image.Image, int]],
    out_root: Path,
    limit: int,
    canvas_size: int,
    digit_size: int,
) -> Path:
    split_dir = out_root / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"{split_name}.jsonl"

    count = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        for idx, (image, label) in enumerate(rows):
            if count >= limit:
                break
            rendered = render_digit(image, canvas_size=canvas_size, digit_size=digit_size)
            rel_path = Path("images") / f"{split_name}_{idx:06d}.png"
            rendered.save(split_dir / rel_path)
            sample = {
                "image": str(rel_path),
                "target_text": str(int(label)),
                "prompt": "Transcribe the handwritten digit exactly. Output only the digit.",
                "split": split_name,
                "source": "EMNIST/digits",
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return manifest_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_root = Path(args.output_dir).resolve()
    ensure_output_dir(out_root, force=args.force)

    raw_root = out_root / "_raw"
    train_ds = EMNIST(root=str(raw_root), split="digits", train=True, download=True)
    test_ds = EMNIST(root=str(raw_root), split="digits", train=False, download=True)

    train_manifest = write_split(
        "train",
        train_ds,
        out_root=out_root,
        limit=max(1, int(args.train_limit)),
        canvas_size=max(32, int(args.canvas_size)),
        digit_size=max(16, int(args.digit_size)),
    )
    val_manifest = write_split(
        "val",
        test_ds,
        out_root=out_root,
        limit=max(1, int(args.val_limit)),
        canvas_size=max(32, int(args.canvas_size)),
        digit_size=max(16, int(args.digit_size)),
    )

    summary = {
        "dataset": "EMNIST/digits",
        "output_dir": str(out_root),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "train_limit": int(args.train_limit),
        "val_limit": int(args.val_limit),
        "canvas_size": int(args.canvas_size),
        "digit_size": int(args.digit_size),
        "seed": int(args.seed),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
