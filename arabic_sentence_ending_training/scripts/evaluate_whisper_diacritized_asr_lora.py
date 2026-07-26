"""Evaluate a diacritized Arabic Whisper LoRA ASR model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DIACRITICS_RE = re.compile(r"[\u064b-\u0652\u0670]")


def strip_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub("", text)


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
    totals = {"matched_words": 0, "ending_correct": 0, "tanween_words": 0, "tanween_correct": 0}
    for pred_word, gold_word in zip(pred_words, gold_words):
        if strip_diacritics(pred_word) != strip_diacritics(gold_word):
            continue
        pred_end = final_mark(pred_word)
        gold_end = final_mark(gold_word)
        totals["matched_words"] += 1
        totals["ending_correct"] += int(pred_end == gold_end)
        if gold_end.startswith("tanwin"):
            totals["tanween_words"] += 1
            totals["tanween_correct"] += int(pred_end == gold_end)
    return totals


def cer(pred: str, gold: str) -> float:
    a, b = pred, gold
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(b))


def load_audio(path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        factor = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // factor, sr // factor).astype("float32")
    return audio.astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument("--data-root")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="whisper_diacritized_asr_eval.json")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.adapter_dir, language="Arabic", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    data_root = Path(args.data_root) if args.data_root else None
    rows = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        path = Path(item["audio_path"])
        if data_root and not path.is_absolute():
            path = data_root / path
        if path.exists():
            rows.append({**item, "audio_path": str(path)})
        if args.limit and len(rows) >= args.limit:
            break

    totals = {"matched_words": 0, "ending_correct": 0, "tanween_words": 0, "tanween_correct": 0}
    row_results = []
    cers = []
    for row in rows:
        audio = load_audio(row["audio_path"], processor.feature_extractor.sampling_rate)
        inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=225)
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        gold = row["text"]
        metrics = ending_accuracy(pred, gold)
        for key in totals:
            totals[key] += metrics[key]
        cers.append(cer(pred, gold))
        payload = {"id": row.get("id"), "prediction": pred, "gold": gold, "metrics": metrics}
        row_results.append(payload)
        print(json.dumps(payload, ensure_ascii=False))

    summary = {
        **totals,
        "ending_accuracy": totals["ending_correct"] / totals["matched_words"] if totals["matched_words"] else 0,
        "tanween_accuracy": totals["tanween_correct"] / totals["tanween_words"] if totals["tanween_words"] else 0,
        "cer": float(np.mean(cers)) if cers else 0,
        "examples": len(rows),
    }
    Path(args.output).write_text(
        json.dumps({"summary": summary, "rows": row_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "output": args.output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
