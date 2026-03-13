from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


logger = logging.getLogger("train_qwen_digit_lora")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA finetuning for Qwen OCR on handwritten digits.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--val_manifest", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="outputs/qwen35_digit_lora")
    ap.add_argument("--logging_dir", type=str, default="logs/tensorboard/qwen35_digit_lora")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_train_samples", type=int, default=10000)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names, or 'all-linear'.",
    )
    ap.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing to save VRAM at the cost of speed.",
    )
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_remote_code", action="store_true")
    return ap.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


class DigitManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, max_samples: int = 0) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.rows = load_jsonl(self.manifest_path, limit=max_samples)
        if not self.rows:
            raise RuntimeError(f"No samples found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image_path = (self.root / row["image"]).resolve()
        return {
            "image_path": str(image_path),
            "prompt": str(row.get("prompt") or "Transcribe the handwritten digit exactly. Output only the digit."),
            "target_text": str(row["target_text"]),
        }


@dataclass
class QwenDigitCollator:
    processor: Any
    enable_thinking: bool = False

    def _apply_chat_template(self, messages: List[Dict[str, Any]], tokenize: bool) -> Any:
        kwargs: Dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": True if tokenize else False,
            "return_dict": True if tokenize else False,
            "return_tensors": "pt" if tokenize else None,
        }
        if self.enable_thinking:
            return self.processor.apply_chat_template(messages, **kwargs)
        try:
            return self.processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.processor.apply_chat_template(messages, **kwargs)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []
        mm_token_types: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        extra_tensors: Dict[str, List[torch.Tensor]] = {}

        for feat in features:
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": feat["image_path"]},
                        {"type": "text", "text": feat["prompt"]},
                    ],
                }
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": feat["target_text"]}]}
            ]

            prompt_inputs = self._apply_chat_template(prompt_messages, tokenize=True)
            full_inputs = self.processor.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            input_ids = full_inputs["input_ids"].squeeze(0)
            attention_mask = full_inputs["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            input_ids_list.append(input_ids)
            attention_masks.append(attention_mask)
            labels_list.append(labels)
            if "mm_token_type_ids" in full_inputs:
                mm_token_types.append(full_inputs["mm_token_type_ids"].squeeze(0))

            for key, value in dict(full_inputs).items():
                if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
                    continue
                if torch.is_tensor(value):
                    extra_tensors.setdefault(key, []).append(value)

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
        batch: Dict[str, Any] = {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(attention_masks, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
        }
        if mm_token_types:
            batch["mm_token_type_ids"] = pad_sequence(mm_token_types, batch_first=True, padding_value=0)

        for key, tensors in extra_tensors.items():
            batch[key] = torch.cat(tensors, dim=0)
        return batch


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    train_manifest = Path(args.train_manifest).resolve()
    val_manifest = Path(args.val_manifest).resolve() if args.val_manifest else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.min_pixels > 0:
        processor_kwargs["min_pixels"] = int(args.min_pixels)
    if args.max_pixels > 0:
        processor_kwargs["max_pixels"] = int(args.max_pixels)

    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)

    torch_dtype = resolve_dtype(args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_target_modules: str | list[str]
    if args.lora_target_modules.strip() == "all-linear":
        lora_target_modules = "all-linear"
    else:
        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    peft_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_ds = DigitManifestDataset(train_manifest, max_samples=max(0, int(args.max_train_samples)))
    eval_ds = None
    if val_manifest and val_manifest.exists():
        eval_ds = DigitManifestDataset(val_manifest, max_samples=max(0, int(args.max_val_samples)))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(Path(args.logging_dir).resolve()),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        bf16=(args.dtype == "bfloat16" and torch.cuda.is_available()),
        fp16=(args.dtype == "float16" and torch.cuda.is_available()),
        save_steps=int(args.save_steps),
        logging_steps=int(args.logging_steps),
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=int(args.dataloader_num_workers),
        seed=int(args.seed),
        eval_strategy="no",
        do_train=True,
        do_eval=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=QwenDigitCollator(processor=processor, enable_thinking=False),
    )

    train_result = trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_ds)
    metrics["eval_samples"] = len(eval_ds) if eval_ds is not None else 0
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Training complete: %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
