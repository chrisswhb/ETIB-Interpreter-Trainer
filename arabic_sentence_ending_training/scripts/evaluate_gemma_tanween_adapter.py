"""Evaluate the Gemma tanween LoRA adapter on held-out JSONL examples.

This script is designed for Colab/GPU first. Local CPU inference with Gemma
2B can be slow, but the same command works if dependencies and memory permit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ARABIC_DIACRITICS_RE = re.compile(r"[\u064b-\u0652]")


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text)


def final_mark(word: str) -> str:
    for ch in reversed(word):
        if ch in "\u064b\u064c\u064d\u064e\u064f\u0650\u0652":
            return {
                "\u064e": "fatha",
                "\u064f": "damma",
                "\u0650": "kasra",
                "\u064b": "tanwin_fath",
                "\u064c": "tanwin_damm",
                "\u064d": "tanwin_kasr",
                "\u0652": "none",
            }.get(ch, "none")
        if "\u0600" <= ch <= "\u06ff":
            break
    return "none"


def ending_accuracy(pred: str, gold: str) -> dict:
    pred_words = pred.split()
    gold_words = gold.split()
    total = 0
    correct = 0
    tanween_total = 0
    tanween_correct = 0
    for pred_word, gold_word in zip(pred_words, gold_words):
        if strip_diacritics(pred_word) != strip_diacritics(gold_word):
            continue
        pred_end = final_mark(pred_word)
        gold_end = final_mark(gold_word)
        total += 1
        correct += int(pred_end == gold_end)
        if gold_end.startswith("tanwin"):
            tanween_total += 1
            tanween_correct += int(pred_end == gold_end)
    return {
        "matched_words": total,
        "ending_correct": correct,
        "tanween_words": tanween_total,
        "tanween_correct": tanween_correct,
    }


def prompt(asr: str) -> str:
    return (
        "You restore Arabic tashkeel and final tanween from ASR text.\n\n"
        "Correct this Arabic ASR transcript for Modern Standard Arabic. "
        "Return only the same sentence with full tashkeel and final tanween.\n\n"
        f"ASR transcript:\n{asr}"
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
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="google/gemma-2-2b-it")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", default="gemma_tanween_eval_results.json")
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

    rows = []
    totals = {"matched_words": 0, "ending_correct": 0, "tanween_words": 0, "tanween_correct": 0}
    with Path(args.test_jsonl).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            item = json.loads(line)
            asr = item.get("cohere_transcript") or item.get("asr_transcript") or ""
            gold = item.get("gold_diacritized_text") or ""
            pred = generate(model, tokenizer, asr, args.max_new_tokens)
            metrics = ending_accuracy(pred, gold)
            for key in totals:
                totals[key] += metrics[key]
            rows.append({"id": item.get("id"), "asr": asr, "gold": gold, "prediction": pred, "metrics": metrics})
            print(json.dumps(rows[-1], ensure_ascii=False))

    summary = {
        **totals,
        "ending_accuracy": totals["ending_correct"] / totals["matched_words"] if totals["matched_words"] else 0,
        "tanween_accuracy": totals["tanween_correct"] / totals["tanween_words"] if totals["tanween_words"] else 0,
    }
    payload = {"summary": summary, "rows": rows}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output": args.output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
