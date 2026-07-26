"""Annotate Cohere ASR JSONL examples with a trained Gemma LoRA adapter.

Run this on Colab/GPU after training the adapter. It is the continuation of the
expert proposal: use the successful Gemma correction model to label more ASR
transcripts with tashkeel/tanween.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def prompt(asr_text: str) -> str:
    return (
        "You restore Arabic tashkeel and final tanween from ASR text.\n\n"
        "Correct this Arabic ASR transcript for Modern Standard Arabic. "
        "Return only the same sentence with full tashkeel and final tanween.\n\n"
        f"ASR transcript:\n{asr_text}"
    )


def generate(model, tokenizer, asr_text: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt(asr_text)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return generated.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="google/gemma-2-2b-it")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as dst:
        for i, line in enumerate(src):
            if args.limit and i >= args.limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            asr = record.get("cohere_transcript") or record.get("asr_transcript") or ""
            if not asr:
                continue
            prediction = generate(model, tokenizer, asr, args.max_new_tokens)
            record["gemma_diacritized_text"] = prediction
            record["gemma_annotation_model"] = {
                "base_model": args.base_model,
                "adapter_dir": args.adapter_dir,
                "purpose": "automatic tashkeel/tanween annotation from ASR text",
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(json.dumps({"id": record.get("id"), "asr": asr, "gemma": prediction}, ensure_ascii=False))

    print(json.dumps({"output": str(output_path), "examples": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
