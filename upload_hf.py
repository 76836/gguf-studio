#!/usr/bin/env python3
"""
Upload Akari example GGUFs to Hugging Face.

Usage:
  export HF_TOKEN=hf_xxxxxxxx
  python3 upload_hf.py --repo YOUR_USER/akari-tiny-gguf

Or:
  python3 upload_hf.py --repo YOUR_USER/akari-tiny-gguf --token hf_xxx
"""
import argparse
import os
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. username/akari-tiny-gguf")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    token = args.token
    if not token:
        print("No token. Set HF_TOKEN or pass --token hf_...", file=sys.stderr)
        sys.exit(1)
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    private = not args.public
    api = HfApi(token=token)
    create_repo(args.repo, token=token, private=private, exist_ok=True, repo_type="model")

    examples = Path(__file__).resolve().parent / "examples"
    files = [
        "akari-tiny-q4_0.gguf",
        "akari-tiny-q8_0.gguf",
        "akari-tiny-f16.gguf",
        "akari-tiny-q2_0.gguf",
        "README.md",
    ]
    for name in files:
        path = examples / name
        if not path.exists():
            print("skip missing", name)
            continue
        print("upload", name, path.stat().st_size)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo,
            token=token,
            repo_type="model",
        )
    print("done:", f"https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
