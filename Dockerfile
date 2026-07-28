FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    ETIB_WAV2VEC2_MODEL_ID=jonatasgrosman/wav2vec2-large-xlsr-53-arabic \
    PRELOAD_TRAINED_DETECTOR=0 \
    PRELOAD_FULL_MODELS=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scikit-learn joblib

COPY . .

EXPOSE 7860
CMD ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
