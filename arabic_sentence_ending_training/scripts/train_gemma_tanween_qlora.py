"""Cloud/Colab QLoRA fine-tuning for Gemma tanween correction.

Run this on Colab, Kaggle, RunPod, or any GPU machine. It is intentionally not
run on the local ETIB laptop by default.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch


def normalize_messages_for_gemma(messages):
    """Gemma chat templates do not support a separate system role."""
    system_parts = []
    normalized = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            if system_parts:
                content = "\n\n".join(system_parts + [content])
                system_parts = []
            normalized.append({"role": "user", "content": content})
        elif role == "assistant":
            normalized.append({"role": "assistant", "content": content})
    if system_parts:
        normalized.insert(0, {"role": "user", "content": "\n\n".join(system_parts)})
    return normalized


def format_chat(example, tokenizer):
    messages = normalize_messages_for_gemma(example["messages"])
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model-id", default="google/gemma-2-2b-it")
    parser.add_argument("--output-dir", default="gemma-tanween-qlora-adapter")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    dataset = dataset.map(lambda ex: {"text": format_chat(ex, tokenizer)}, remove_columns=dataset.column_names)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_strategy="epoch",
        bf16=False,
        fp16=False,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
