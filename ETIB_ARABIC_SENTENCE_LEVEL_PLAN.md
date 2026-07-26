# ETIB Arabic Sentence-Level Plan

## Decision

The Arabic module should be a reference-guided i'rab pronunciation assessment system, not a general Arabic ASR system.

The system knows the target Arabic sentence, extracts the expected final ending from each diacritized reference word, then checks whether the student's spoken ending matches.

## Final Architecture

```text
Fully diacritized reference sentence
        |
        v
Expected final-ending extractor
        |
        v
Student sentence recording
        |
        v
Word alignment / word segmentation
        |
        v
Final 300-650 ms acoustic feature extraction
        |
        v
wav2vec2 embeddings + trained classifier
        |
        v
Per-word comparison: expected vs detected
        |
        v
Frontend feedback for each Arabic word
```

## Why The Previous Model Was Not Enough

The first trained model used isolated words. It reached useful isolated-word accuracy, but full sentences are a different acoustic problem. In a sentence, Arabic word endings are shorter, connected to the next word, and sometimes weakened by speaking speed.

Therefore, the next model must be trained from word segments inside full sentences.

## New Workspace Added

```text
arabic_sentence_ending_training/
```

Important scripts:

```text
scripts/prepare_simple_manifest.py
scripts/prepare_textgrid_manifest.py
scripts/extract_sentence_word_features.py
scripts/train_sentence_classifier.py
```

Expected manifest:

```csv
utterance_id,audio_path,reference_text,word_index,word,expected_ending,start_ms,end_ms,speaker,source
```

## Dataset Plan

Use external datasets first, then optionally adapt with ETIB recordings.

1. Arabic Speech Corpus
   - Best first dataset because it has alignment information.
   - Use `prepare_textgrid_manifest.py` if TextGrid files are available.

2. ArVoice
   - Useful because it has diacritized Arabic text.
   - It may need word alignment before training.

3. Quran-MD
   - Useful for extra word-level audio.
   - Use carefully because Quran recitation style is different from ETIB spoken Arabic.

4. ETIB-style validation
   - Even if training uses public datasets, the project still needs a small ETIB-style test set to prove the frontend works.

## Training Commands

After a word-level manifest exists:

```powershell
venv\Scripts\python.exe arabic_sentence_ending_training\scripts\extract_sentence_word_features.py --manifest data\manifests\textgrid_sentence_manifest.csv --out data\features\sentence_word_features.npz --meta data\features\sentence_word_features_meta.json
```

Then train:

```powershell
venv\Scripts\python.exe arabic_sentence_ending_training\scripts\train_sentence_classifier.py --features data\features\sentence_word_features.npz --model-out models\sentence_ending_classifier.joblib --report-out models\sentence_training_report.json
```

## What Is Still Needed

The code structure is ready, but the actual dataset files are still needed:

```text
audio files
word timing/alignment files
diacritized Arabic text
```

Once those are present, the scripts can produce a real sentence-level classifier.

