---
title: ETIB Arabic Wrapper API
emoji: 🎙️
colorFrom: orange
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
# ETIB Arabic Wrapper API

FastAPI service exposing Ali's Arabic ASR/haraka wrapper.

Health:

```text
GET /api/arabic-wrapper/health
```

Free speech harakat transcription:

```text
POST /api/arabic-wrapper/transcribe-harakat
```

Reference-based evaluation:

```text
POST /api/arabic-wrapper/evaluate
```