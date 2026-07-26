# ETIB Arabic Sentence Ending Training

This workspace is for the real ETIB Arabic task:

```text
full spoken Arabic sentence -> each reference word -> final haraka/tanween label
```

The previous `arabic_ending_training` model is an isolated-word prototype. It is useful, but full ETIB exercises need sentence-level word segments.

## Data Format

All training scripts expect a CSV manifest with this header:

```csv
utterance_id,audio_path,reference_text,word_index,word,expected_ending,start_ms,end_ms,speaker,source
```

Labels:

```text
fatha
damma
kasra
none
tanwin_fath
tanwin_damm
tanwin_kasr
ignore
```

`ignore` rows are kept for traceability but skipped during training.

## Recommended Pipeline

1. Put dataset audio under `data/raw/<dataset_name>/`.
2. Create or import a sentence manifest:

```powershell
venv\Scripts\python.exe arabic_sentence_ending_training\scripts\prepare_simple_manifest.py --input data\my_sentences.csv --out arabic_sentence_ending_training\data\manifests\manual_sentence_manifest.csv
```

3. Extract wav2vec2 features from each word segment:

```powershell
venv\Scripts\python.exe arabic_sentence_ending_training\scripts\extract_sentence_word_features.py --manifest data\manifests\manual_sentence_manifest.csv --out data\features\sentence_word_features.npz --meta data\features\sentence_word_features_meta.json
```

4. Train and evaluate with grouped splitting by utterance:

```powershell
venv\Scripts\python.exe arabic_sentence_ending_training\scripts\train_sentence_classifier.py --features data\features\sentence_word_features.npz --model-out models\sentence_ending_classifier.joblib --report-out models\sentence_training_report.json
```

## Why Grouped Evaluation Matters

The test split must not contain words from the same sentence used in training. Otherwise the accuracy looks high but fails in the frontend. This workspace uses utterance-level grouping by default.

## Dataset Strategy

Use datasets in this order:

1. Arabic Speech Corpus: best first target because it includes alignment information.
2. ArVoice: useful because it has diacritized text, but may need word alignment.
3. Quran-MD: useful for pretraining only; recitation style is different from ETIB speech.
4. Your ETIB-style recordings: best final adaptation set, even if small.

