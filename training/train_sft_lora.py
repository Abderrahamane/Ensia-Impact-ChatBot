"""Train a LoRA adapter for ENSIA assistant style using Hugging Face + TRL."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA SFT adapter for ENSIA assistant")
    parser.add_argument("--train-file", type=Path, required=True, help="Path to SFT JSONL data")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base instruct model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lora-ensia-assistant"),
        help="Output adapter directory",
    )
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of train epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Limit training rows (0 = all)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    return parser.parse_args()


def _load_jsonl(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and (i + 1) >= max_samples:
                break
    return rows


def _render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").strip().capitalize()
            content = msg.get("content", "").strip()
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)


def _to_dataset(rows: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    rendered: list[dict[str, str]] = []
    for row in rows:
        messages = row.get("messages", [])
        if not isinstance(messages, list) or not messages:
            continue
        rendered.append({"text": _render_chat(tokenizer, messages)})
    if not rendered:
        raise ValueError("No valid training rows with 'messages' were found.")
    return Dataset.from_list(rendered)


def _infer_lora_target_modules(model: Any) -> list[str]:
    # Cover common attention projection names across model families.
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "c_attn",
        "c_proj",
        "query_key_value",
        "Wqkv",
    ]
    names = {name for name, _ in model.named_modules()}
    found = [c for c in candidates if any(n.endswith(c) for n in names)]
    return found or ["q_proj", "k_proj", "v_proj", "o_proj"]


def _build_sft_trainer(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    peft_config: LoraConfig,
    training_args: TrainingArguments,
    max_seq_len: int,
) -> SFTTrainer:
    # TRL changed constructor names across versions; we adapt dynamically.
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters

    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": training_args,
    }

    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_len

    return SFTTrainer(**kwargs)


def main() -> None:
    args = parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")

    try:
        tokenizer: Any = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    
    if tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    use_4bit = (not args.no_4bit) and torch.cuda.is_available()
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    rows = _load_jsonl(args.train_file, args.max_train_samples)
    dataset = _to_dataset(rows, tokenizer)

    target_modules = _infer_lora_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
    )

    trainer = _build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
        training_args=training_args,
        max_seq_len=args.max_seq_len,
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    meta = {
        "base_model": args.base_model,
        "train_file": str(args.train_file),
        "rows_loaded": len(rows),
        "rows_used": len(dataset),
        "max_seq_len": args.max_seq_len,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
        "use_4bit": use_4bit,
    }
    (args.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Training finished. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

