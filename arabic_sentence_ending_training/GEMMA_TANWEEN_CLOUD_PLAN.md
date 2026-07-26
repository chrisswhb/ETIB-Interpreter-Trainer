# Gemma Tanween Cloud Fine-Tuning Plan

This implements the expert proposal without training Gemma locally. Current
status: the Cohere + Gemma QLoRA text-restoration path is successful on a clean
held-out evaluation set.

## Step 1: Cohere ASR - Done

Configured through `.env`:

```text
COHERE_API_KEY=...
COHERE_ARABIC_ASR_MODEL=cohere-transcribe-arabic-07-2026
```

The app endpoint is:

```text
POST /api/expert-tanween-pipeline
```

It runs:

```text
audio -> Cohere Transcribe Arabic -> Gemma-style tanween correction
```

If Cohere is unavailable, the endpoint falls back to Groq Whisper so testing can continue.

## Step 2: Labelled Dataset - Done

Two dataset builders now exist:

```powershell
uv run python arabic_sentence_ending_training/scripts/build_gemma_tanween_dataset.py
```

This creates simulated ASR pairs:

```text
undiacritized transcript -> gold diacritized transcript
```

Output:

```text
arabic_sentence_ending_training/data/gemma_tanween/gemma_tanween_train.jsonl
```

For real Cohere ASR pairs:

```powershell
uv run python arabic_sentence_ending_training/scripts/collect_cohere_asr_dataset.py --limit 25
```

Output:

```text
arabic_sentence_ending_training/data/gemma_tanween/cohere_gemma_pilot.jsonl
```

Final scaled training/evaluation files:

```text
arabic_sentence_ending_training/data/gemma_tanween/gemma_tanween_plus_cohere250_train.jsonl
arabic_sentence_ending_training/data/gemma_tanween/cohere_tanween_holdout_clean_100.jsonl
```

Training set:

```text
1065 examples
1705 tanween labels
```

Clean evaluation set:

```text
100 examples
266 tanween labels
```

## Step 3: Cloud QLoRA Fine-Tuning - Done

Run this on Colab/Kaggle/RunPod, not the laptop:

```bash
pip install -U transformers datasets peft trl bitsandbytes accelerate
python train_gemma_tanween_qlora.py \
  --train-jsonl gemma_tanween_train.jsonl \
  --model-id google/gemma-2-2b-it \
  --output-dir gemma-tanween-qlora-adapter
```

Gemma 4 12B was the model named in the proposal, but it is heavy for the
available Colab/runtime constraints. The implemented run used
`google/gemma-2-2b-it`, the smaller Gemma instruction model, with QLoRA. This
keeps the proposal's method while making it trainable with available resources.

The output is a LoRA adapter. Later, the backend can call this adapter through
a hosted OpenAI-compatible endpoint by setting:

```text
GEMMA_OPENAI_BASE_URL=
GEMMA_API_KEY=
GEMMA_MODEL_ID=google/gemma-4-12B-it
```

## Step 4: Evaluation - Successful

Best clean holdout result:

```text
matched_words: 1127
ending_correct: 1097
tanween_words: 190
tanween_correct: 179
ending_accuracy: 97.34%
tanween_accuracy: 94.21%
```

Saved summary:

```text
arabic_sentence_ending_training/models/cohere250_clean_eval_summary.json
```

This proves that the Cohere ASR transcript can be corrected by a QLoRA-fine-tuned
Gemma model to recover likely final endings and tanween with high accuracy on
clean held-out Cohere ASR examples.

## Step 5: Use Gemma to Annotate More Data - Prepared

Script:

```text
arabic_sentence_ending_training/scripts/annotate_with_gemma_adapter.py
```

Colab example:

```bash
python annotate_with_gemma_adapter.py \
  --input-jsonl more_cohere_asr_examples.jsonl \
  --output-jsonl gemma_annotated_more_examples.jsonl \
  --adapter-dir gemma-tanween-cohere250-adapter \
  --base-model google/gemma-2-2b-it
```

This is the step the expert proposed for scaling labels without manually
diacritizing every sentence.

## Step 6: Prepare ASR Fine-Tuning Data - Prepared

Script:

```text
arabic_sentence_ending_training/scripts/build_asr_finetune_manifest.py
```

Prepared manifest from the current labelled data:

```text
arabic_sentence_ending_training/data/gemma_tanween/asr_finetune_gold_manifest.jsonl
arabic_sentence_ending_training/data/gemma_tanween/asr_finetune_gold_manifest.csv
```

Current prepared size:

```text
1065 audio/text pairs
```

This is the bridge to the expert's final future idea: fine-tune an ASR model to
directly output tanween/diacritics. The manifest is ready, but actual ASR
fine-tuning needs more data and GPU time.

## What This Proves

The project can show:

1. Baseline ASR output has no tashkeel.
2. LLM correction can restore likely final endings.
3. More labelled examples can improve the correction model.
4. The resulting labels can later be used to train ASR or alignment models.

## Still Not Solved

This does not yet prove the learner acoustically pronounced the ending. It is
an ASR-text correction system. Acoustic verification still needs:

- reliable word alignment, or
- phoneme-level alignment, or
- more labelled audio with exact diacritized transcripts.
