import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy ETIB Arabic wrapper to a Hugging Face Docker Space.")
    parser.add_argument("--space-id", required=True, help="Example: Ali-Ali/etib-arabic-wrapper")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--copy-local-env", action="store_true", help="Copy COHERE/GROQ/API secrets from local .env into Space secrets")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Missing Hugging Face token. Set HF_TOKEN or pass --token.")

    repo_dir = Path(args.repo_dir).resolve()
    api = HfApi(token=args.token)

    create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="docker",
        token=args.token,
        exist_ok=True,
        private=False,
    )

    if args.copy_local_env:
        env_values = read_env(repo_dir / ".env")
        for key in ["COHERE_API_KEY", "GROQ_API_KEY", "GEMMA_API_KEY", "GEMMA_OPENAI_BASE_URL", "ARABIC_WRAPPER_API_KEY"]:
            value = env_values.get(key) or os.environ.get(key)
            if value:
                api.add_space_secret(repo_id=args.space_id, key=key, value=value)
        api.add_space_variable(repo_id=args.space_id, key="ETIB_WAV2VEC2_MODEL_ID", value=os.environ.get("ETIB_WAV2VEC2_MODEL_ID", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"))

    upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(repo_dir),
        token=args.token,
        ignore_patterns=[
            ".git/*",
            ".env",
            "*.log",
            ".tools/*",
            "venv/*",
            "backend/venv/*",
            "data/recordings/*",
            "data/word_recordings/*",
            "*.zip",
            "*.docx",
            "*.html",
        ],
    )

    print(f"https://huggingface.co/spaces/{args.space_id}")
    owner, name = args.space_id.split("/", 1)
    print(f"https://{owner}-{name}.hf.space")


if __name__ == "__main__":
    main()
