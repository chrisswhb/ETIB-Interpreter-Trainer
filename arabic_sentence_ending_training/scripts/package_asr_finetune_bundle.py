"""Package ASR fine-tuning audio + manifest for Colab.

The full audio set is large, so this script can create smaller chunks:

python package_asr_finetune_bundle.py --limit 100 --zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="arabic_sentence_ending_training/data/gemma_tanween/asr_finetune_gold_manifest.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="arabic_sentence_ending_training/models/asr_finetune_bundle",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    out_dir = resolve_path(args.out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    rows = rows[args.skip :]
    if args.limit:
        rows = rows[: args.limit]

    packaged = []
    missing = []
    for i, row in enumerate(rows, start=1):
        src = resolve_path(row["audio_path"])
        if not src.exists():
            missing.append(row["audio_path"])
            continue
        suffix = src.suffix.lower() or ".wav"
        dst_name = f"audio_{i:05d}{suffix}"
        dst = audio_dir / dst_name
        shutil.copy2(src, dst)
        packaged.append({**row, "audio_path": f"audio/{dst_name}"})

    out_manifest = out_dir / "manifest.jsonl"
    with out_manifest.open("w", encoding="utf-8", newline="\n") as f:
        for row in packaged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    zip_path = None
    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in out_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(out_dir))

    size = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "zip": str(zip_path) if zip_path else None,
                "examples": len(packaged),
                "missing": len(missing),
                "size_mb": round(size / 1024 / 1024, 2),
                "manifest": str(out_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
