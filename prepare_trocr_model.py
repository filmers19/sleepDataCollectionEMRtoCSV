from __future__ import annotations

import argparse
import time
import unicodedata
from typing import Any, Tuple

from PIL import Image


def resolve_device_and_dtype(torch_mod: Any, dtype_name: str) -> Tuple[Any, Any]:
    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    alias = str(dtype_name or "auto").strip().lower()

    if alias in {"", "auto"}:
        if device.type == "cuda":
            bf16_supported = bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)())
            return device, (torch_mod.bfloat16 if bf16_supported else torch_mod.float16)
        return device, torch_mod.float32

    if alias in {"bf16", "bfloat16"}:
        if device.type != "cuda":
            return device, torch_mod.float32
        return device, torch_mod.bfloat16
    if alias in {"fp16", "float16", "half"}:
        if device.type != "cuda":
            return device, torch_mod.float32
        return device, torch_mod.float16
    if alias in {"fp32", "float32"}:
        return device, torch_mod.float32
    raise ValueError(f"Unsupported --dtype value: {dtype_name}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download and warm up a Hugging Face TrOCR model.")
    ap.add_argument("--model_id", type=str, default="ddobokki/ko-trocr")
    ap.add_argument("--dtype", type=str, default="auto", help="auto|bfloat16|float16|float32")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    return ap.parse_args()


def materialize_decoder_positional_weights(model: Any, device: Any, infer_dtype: Any) -> None:
    try:
        pos_mod = model.decoder.model.decoder.embed_positions
    except AttributeError:
        return

    weights = getattr(pos_mod, "weights", None)
    if weights is None:
        return
    if str(getattr(weights, "device", "")) == "meta":
        weights = pos_mod.get_embedding(weights.size(0), pos_mod.embedding_dim, pos_mod.padding_idx)
    pos_mod.weights = weights.to(device=device, dtype=infer_dtype)


def main() -> None:
    args = parse_args()

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device, infer_dtype = resolve_device_and_dtype(torch, args.dtype)
    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = infer_dtype

    started = time.perf_counter()
    print(f"Loading processor: {args.model_id}")
    processor = TrOCRProcessor.from_pretrained(args.model_id, use_fast=False)
    print(f"Loading model: {args.model_id}")
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id, **model_kwargs)
    model = model.to(device=device, dtype=infer_dtype).eval()
    materialize_decoder_positional_weights(model=model, device=device, infer_dtype=infer_dtype)
    load_s = time.perf_counter() - started

    blank = Image.new("RGB", (512, 512), color="white")
    pixel_values = processor(images=blank, return_tensors="pt").pixel_values.to(device=device, dtype=infer_dtype)
    with torch.inference_mode():
        ids = model.generate(pixel_values=pixel_values, max_new_tokens=max(1, int(args.max_new_tokens)))
    text = unicodedata.normalize("NFC", processor.batch_decode(ids, skip_special_tokens=True)[0])

    print(f"Ready: device={device} dtype={infer_dtype} load_seconds={load_s:.1f}")
    print(f"Warmup decode: {text!r}")


if __name__ == "__main__":
    main()
