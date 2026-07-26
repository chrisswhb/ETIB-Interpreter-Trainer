"""Fine-tune Whisper to output diacritized Arabic with LoRA.

This is the final research step from the expert proposal: use audio plus
diacritized/tanween labels to fine-tune an ASR model itself, instead of only
post-correcting ASR text with Gemma.

Run on Colab/Kaggle/RunPod GPU, not on the laptop CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


def load_rows(manifest: Path, data_root: Path | None, limit: int = 0) -> list[dict]:
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        audio_path = Path(item["audio_path"])
        if data_root and not audio_path.is_absolute():
            audio_path = data_root / audio_path
        if not audio_path.exists():
            continue
        rows.append({"audio_path": str(audio_path), "text": item["text"], "id": item.get("id")})
        if limit and len(rows) >= limit:
            break
    return rows


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


def prepare_example(example: dict, feature_extractor, tokenizer, sampling_rate: int) -> dict:
    audio = load_audio(example["audio_path"], sampling_rate)
    features = feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
    labels = tokenizer(example["text"]).input_ids
    return {"input_features": features, "labels": labels}


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] and torch.all(labels[:, 0] == self.processor.tokenizer.bos_token_id).item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--model-id", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="whisper-ar-diacritized-lora")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    data_root = Path(args.data_root) if args.data_root else None
    rows = load_rows(manifest, data_root, args.limit)
    if len(rows) < 20:
        raise SystemExit(f"Need at least 20 audio/text rows, found {len(rows)}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language="Arabic", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_id, language="Arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = Dataset.from_list(rows)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    sampling_rate = feature_extractor.sampling_rate
    dataset = dataset.map(
        lambda item: prepare_example(item, feature_extractor, tokenizer, sampling_rate),
        remove_columns=dataset["train"].column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
