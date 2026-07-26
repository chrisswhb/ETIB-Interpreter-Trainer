# ASR Fine-Tuning and Live Alignment Status

## Completed Now

### 1. ASR fine-tuning preparation

The ASR fine-tuning dataset manifest exists:

```text
arabic_sentence_ending_training/data/gemma_tanween/asr_finetune_gold_manifest.jsonl
```

It contains:

```text
1065 labelled audio/text pairs
```

The full audio set is about:

```text
936 MB
```

### 2. Whisper LoRA ASR fine-tuning scripts

Added:

```text
arabic_sentence_ending_training/scripts/train_whisper_diacritized_asr_lora.py
arabic_sentence_ending_training/scripts/evaluate_whisper_diacritized_asr_lora.py
```

These fine-tune Whisper to output Arabic text with diacritics/tanween directly.
This is the concrete ASR fine-tuning step from the expert proposal.

### 3. Upload bundles

Tools bundle:

```text
arabic_sentence_ending_training/models/upload_ready_v5/whisper_asr_finetune_tools.zip
```

Small 50-example smoke-test audio bundle:

```text
arabic_sentence_ending_training/models/upload_ready_v5/asr_finetune_bundle_50.zip
```

The full 1065-example audio bundle was not created automatically because it is
large. It can be created with:

```powershell
python arabic_sentence_ending_training/scripts/package_asr_finetune_bundle.py --skip 15 --limit 1050 --out-dir arabic_sentence_ending_training/models/asr_finetune_bundle_full --zip
```

### 4. Live forced-alignment integration

The trained endpoint now tries alignment in this order:

```text
1. externally supplied forced-alignment word spans
2. Groq/Whisper word timestamps if configured
3. local wav2vec2 CTC word timestamps
4. reference-weighted energy segmentation fallback
```

This means the frontend/backend no longer has only rough segmentation as the
automatic option. The local CTC path is a real automatic word-timing attempt,
although still weaker than a dedicated MFA Arabic aligner.

## Colab Commands for ASR Fine-Tuning Smoke Test

Upload:

```text
whisper_asr_finetune_tools.zip
asr_finetune_bundle_50.zip
```

Then run:

```bash
!unzip -o whisper_asr_finetune_tools.zip
!unzip -o asr_finetune_bundle_50.zip -d asr50
!pip install -U transformers datasets peft accelerate soundfile scipy
```

Train:

```bash
!python train_whisper_diacritized_asr_lora.py \
  --manifest asr50/manifest.jsonl \
  --data-root asr50 \
  --model-id openai/whisper-small \
  --output-dir whisper-ar-diacritized-lora \
  --epochs 5
```

Evaluate:

```bash
!python evaluate_whisper_diacritized_asr_lora.py \
  --manifest asr50/manifest.jsonl \
  --data-root asr50 \
  --adapter-dir whisper-ar-diacritized-lora \
  --base-model openai/whisper-small \
  --limit 20 \
  --output whisper_asr_eval.json
```

## Important Reality Check

The successful Gemma result is already excellent:

```text
ending_accuracy: 97.34%
tanween_accuracy: 94.21%
```

ASR fine-tuning may or may not beat Gemma with only 1065 examples. It is the
expert's future path, but it usually needs more audio than text correction.

For live pronunciation-error detection, the strongest next upgrade is still a
dedicated forced aligner such as MFA with an Arabic acoustic model/dictionary.
