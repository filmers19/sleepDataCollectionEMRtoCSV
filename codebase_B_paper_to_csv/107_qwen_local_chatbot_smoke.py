from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Dict, List


def resolve_dtype(torch_mod: Any, dtype_name: str) -> Any:
    alias = (dtype_name or "auto").strip().lower()
    if alias in {"", "auto"}:
        return "auto"
    if alias in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if alias in {"fp16", "float16", "half"}:
        return torch_mod.float16
    if alias in {"fp32", "float32"}:
        return torch_mod.float32
    raise ValueError(f"Unsupported --dtype value: {dtype_name}")


def pick_model_cls(transformers_mod: Any, model_id: str) -> Any:
    model_id_l = model_id.lower()

    cls_qwen3_moe = getattr(transformers_mod, "Qwen3VLMoeForConditionalGeneration", None)
    cls_qwen3_dense = getattr(transformers_mod, "Qwen3VLForConditionalGeneration", None)
    cls_qwen25 = getattr(transformers_mod, "Qwen2_5_VLForConditionalGeneration", None)

    if "qwen3-vl" in model_id_l and cls_qwen3_moe is not None and ("a3b" in model_id_l or "moe" in model_id_l):
        return cls_qwen3_moe
    if "qwen3-vl" in model_id_l and cls_qwen3_dense is not None:
        return cls_qwen3_dense
    if "qwen2.5-vl" in model_id_l and cls_qwen25 is not None:
        return cls_qwen25

    for auto_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        auto_cls = getattr(transformers_mod, auto_name, None)
        if auto_cls is not None:
            return auto_cls

    raise RuntimeError(
        "No compatible transformers class found for Qwen VL models. "
        "Please install a newer transformers build."
    )


def print_gpu_info(torch_mod: Any) -> None:
    print(f"torch.cuda.is_available(): {torch_mod.cuda.is_available()}")
    if not torch_mod.cuda.is_available():
        print("No CUDA device detected. Model will run on CPU.")
        return
    print(f"CUDA device count: {torch_mod.cuda.device_count()}")
    for idx in range(torch_mod.cuda.device_count()):
        name = torch_mod.cuda.get_device_name(idx)
        total_gb = torch_mod.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"  - cuda:{idx}: {name} ({total_gb:.1f} GiB)")


def _to_cuda_device_str(dev: Any) -> str:
    if isinstance(dev, int):
        return f"cuda:{dev}"
    dev_s = str(dev).strip().lower()
    if dev_s.isdigit():
        return f"cuda:{dev_s}"
    if dev_s.startswith("cuda"):
        return dev_s
    return ""


def summarize_model_devices(model: Any) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        counts = Counter(str(v) for v in device_map.values())
        print("hf_device_map summary:")
        for dev, cnt in sorted(counts.items(), key=lambda x: x[0]):
            print(f"  - {dev}: {cnt} module(s)")
        return any(bool(_to_cuda_device_str(dev)) for dev in device_map.values())

    try:
        first_device = next(model.parameters()).device
        print(f"Model parameter device: {first_device}")
        return str(first_device).startswith("cuda")
    except StopIteration:
        print("Could not determine model parameter device.")
        return False


def choose_compute_device(model: Any, torch_mod: Any) -> Any:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        for dev in device_map.values():
            cuda_dev = _to_cuda_device_str(dev)
            if cuda_dev:
                return torch_mod.device(cuda_dev)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def generate_reply(
    model: Any,
    processor: Any,
    torch_mod: Any,
    history: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    model_inputs = processor.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = choose_compute_device(model, torch_mod)
    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(device)
    else:
        model_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(model_inputs).items()}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": float(temperature) > 0.0,
    }
    if float(temperature) > 0.0:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = max(0.01, min(1.0, float(top_p)))

    with torch_mod.inference_mode():
        generated_ids = model.generate(**model_inputs, **gen_kwargs)

    prompt_len = int(model_inputs["input_ids"].shape[1])
    trimmed = generated_ids[:, prompt_len:]
    text_list = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (text_list[0] if text_list else "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple local Qwen chatbot smoke test (GPU load + chat).")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="auto|bfloat16|float16|float32")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention backend (e.g. sdpa, flash_attention_2, eager)",
    )
    parser.add_argument("--disable_trust_remote_code", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--system_prompt", type=str, default="You are a concise and helpful assistant.")
    parser.add_argument("--max_turns", type=int, default=20, help="Keep only the most recent N user+assistant turns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoProcessor

    print_gpu_info(torch)
    print(f"Loading model: {args.model_id}")

    model_cls = pick_model_cls(transformers, args.model_id)
    dtype_value = resolve_dtype(torch, args.dtype)

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype_value,
        "device_map": "auto",
        "trust_remote_code": (not args.disable_trust_remote_code),
    }
    if args.attn_implementation.strip():
        model_kwargs["attn_implementation"] = args.attn_implementation.strip()

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=(not args.disable_trust_remote_code),
    )
    try:
        model = model_cls.from_pretrained(args.model_id, **model_kwargs)
    except ImportError as exc:
        requested_attn = args.attn_implementation.strip().lower()
        if requested_attn == "flash_attention_2" and "flash_attn" in str(exc).lower():
            print("flash_attn package not found. Retrying with --attn_implementation=sdpa.")
            model_kwargs["attn_implementation"] = "sdpa"
            model = model_cls.from_pretrained(args.model_id, **model_kwargs)
        else:
            raise
    model.eval()
    dt = time.time() - t0
    print(f"Model loaded in {dt:.1f}s")

    uses_gpu = summarize_model_devices(model)
    if uses_gpu:
        print("GPU placement check: PASS (at least part of model is on CUDA).")
    else:
        print("GPU placement check: FAIL (no CUDA modules detected).")

    history: List[Dict[str, Any]] = []
    if args.system_prompt.strip():
        history.append({"role": "system", "content": [{"type": "text", "text": args.system_prompt.strip()}]})

    print("\nRunning warm-up generation...")
    warmup_user = "Reply with exactly: READY"
    history.append({"role": "user", "content": [{"type": "text", "text": warmup_user}]})
    warmup_reply = generate_reply(
        model=model,
        processor=processor,
        torch_mod=torch,
        history=history,
        max_new_tokens=min(32, args.max_new_tokens),
        temperature=0.0,
        top_p=args.top_p,
    )
    history.append({"role": "assistant", "content": [{"type": "text", "text": warmup_reply}]})
    print(f"Warm-up reply: {warmup_reply}")

    print("\nChat started. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        reply = generate_reply(
            model=model,
            processor=processor,
            torch_mod=torch,
            history=history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})

        keep_turns = max(1, int(args.max_turns))
        non_system = [m for m in history if m["role"] != "system"]
        if len(non_system) > keep_turns * 2:
            trimmed = non_system[-(keep_turns * 2) :]
            history = [m for m in history if m["role"] == "system"] + trimmed


if __name__ == "__main__":
    main()
