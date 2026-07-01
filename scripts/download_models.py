r"""
Download ETIB speech models once into a persistent local folder.

Default destination:
    C:\Users\<you>\Desktop\etib_models

Override with:
    set ETIB_MODEL_DIR=C:\some\folder
"""

from __future__ import annotations

import os
from pathlib import Path


MODEL_ROOT = Path(
    os.environ.get(
        "ETIB_MODEL_DIR",
        Path.home() / "Desktop" / "etib_models",
    )
)

WAV2VEC2_MODELS = [
    ("jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "wav2vec2_jonatasgrosman_arabic"),
    ("elgeish/wav2vec2-large-xlsr-53-arabic", "wav2vec2_elgeish_arabic"),
    ("kmfoda/wav2vec2-large-xlsr-arabic", "wav2vec2_kmfoda_arabic"),
]


def download_wav2vec2_models() -> None:
    from huggingface_hub import snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for repo_id, folder_name in WAV2VEC2_MODELS:
        target = MODEL_ROOT / folder_name
        if any((target / name).exists() for name in ("pytorch_model.bin", "model.safetensors", "flax_model.msgpack")):
            print(f"Already have usable weights -> {target}")
            continue
        print(f"Downloading {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "pytorch_model.bin",
                "model.safetensors",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "config.json",
            ],
        )


def download_silero_vad() -> None:
    import torch

    target = MODEL_ROOT / "silero-vad"
    if target.exists():
        print(f"Silero VAD already exists -> {target}")
        return

    print(f"Downloading snakers4/silero-vad -> {target}")
    torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        source="github",
        trust_repo=True,
        force_reload=False,
    )

    hub_dir = Path(torch.hub.get_dir())
    candidates = sorted(hub_dir.glob("snakers4_silero-vad*"), key=lambda p: p.stat().st_mtime)
    if candidates:
        import shutil

        shutil.copytree(candidates[-1], target)
        print(f"Copied Silero VAD hub repo -> {target}")
    else:
        print("Silero VAD was cached by torch.hub, but the repo folder was not found.")


def main() -> None:
    print(f"ETIB_MODEL_DIR={MODEL_ROOT}")
    download_wav2vec2_models()
    download_silero_vad()
    print("Done. Start the app normally; it will load these local model folders first.")


if __name__ == "__main__":
    main()
