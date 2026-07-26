# Forced Alignment Step for ETIB Arabic

## Why we need it

The acoustic classifier can detect final Arabic endings only when it receives the
correct audio span for the target word ending. The weak frontend results came
mainly from rough sentence segmentation, not from the classifier itself.

The correct ETIB pronunciation pipeline is:

```text
Reference sentence with tashkeel
+ student full-sentence audio
-> forced alignment / word timing
-> cut the final 150-650 ms of each word
-> acoustic ending classifier
-> compare detected spoken ending with the reference ending
```

ASR is not trusted for harakat or tanween. It is only useful as a rough transcript
helper. The final haraka decision must come from the acoustic classifier.

## What we verified locally

The Arabic Speech Corpus includes TextGrid word timings. These timings act as an
oracle forced aligner: they tell us where each word starts and ends in the audio.

Using those TextGrid word spans, the existing classifier results are strong:

```text
Proper held-out grouped report:
  file: models/asc_wordtier_tail650_rbf_report.json
  accuracy: 97.55%
  samples: 4642
  split: GroupShuffleSplit by utterance_id

Oracle aligned-feature check:
  file: models/forced_alignment_oracle_report.json
  accuracy: 99.38%
  samples: 4642
```

The oracle aligned-feature check uses the saved classifier over the aligned
feature matrix, so the 97.55% grouped report is the more honest number. Both
results support the same conclusion: with good alignment, the acoustic classifier
is capable of detecting final haraka/tanween very well.

## What is still missing

For live ETIB recordings, we do not yet have automatic forced alignment installed.
The backend currently has:

```text
1. Whisper/Groq word timestamps if configured
2. reference-weighted energy segmentation fallback
```

Those are rough aligners, not a real forced aligner. They can put the classifier
on the wrong part of the audio, which causes bad results.

## Next implementation choices

The next serious step is to replace rough segmentation with one of these:

```text
Option A: Montreal Forced Aligner (MFA)
  - Install MFA
  - Find/build Arabic acoustic model and pronunciation dictionary
  - Align student audio to the known reference sentence
  - Feed word-end spans to the classifier

Option B: wav2vec2/CTC forced alignment
  - Install/use a CTC ASR model with Arabic tokens
  - Force-align the known reference words against audio emissions
  - Extract word timings without relying on ASR to output tashkeel

Option C: WhisperX-style word alignment
  - Use Whisper transcript for rough words
  - Use an alignment model for timestamps
  - Still classify final endings acoustically afterward
```

The important point is that the aligner does not need to recognize tashkeel. It
only needs reliable word timing.

## Current status

The forced-alignment proof is done. The classifier works when alignment is known.
The remaining ETIB problem is live automatic word alignment for student audio.
