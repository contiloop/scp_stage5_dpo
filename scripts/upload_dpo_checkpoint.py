#!/usr/bin/env python3
"""Upload a trained DPO checkpoint to a HuggingFace model repo.

Default source is `artifacts/dpo/final` (set by train_dpo.py). Use
`--src` to push a different checkpoint (e.g. one of the mid-training
`artifacts/dpo/checkpoint-*`).

No fallbacks: missing files, missing required HF files, or auth failure
all surface as exceptions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("artifacts/dpo/final"),
                    help="local directory to push (default: artifacts/dpo/final)")
    ap.add_argument("--dest-repo", required=True,
                    help="destination HF model repo id, e.g. alwaysgood/qwen35_sft_023_dpo")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message", default=None)
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[error] src dir not found: {args.src}", file=sys.stderr)
        sys.exit(2)

    # sharded safetensors fallback (very large models)
    if not (args.src / "model.safetensors").exists() \
       and not (args.src / "model.safetensors.index.json").exists():
        print(f"[error] no model.safetensors[.index.json] under {args.src}",
              file=sys.stderr)
        sys.exit(2)

    missing = [n for n in REQUIRED_FILES
               if not (args.src / n).exists() and n != "model.safetensors"]
    # explicit safetensors handled above; require everything else
    for n in REQUIRED_FILES:
        if n == "model.safetensors":
            continue
        if not (args.src / n).exists():
            print(f"[error] missing required file: {args.src/n}", file=sys.stderr)
            sys.exit(2)

    for p in args.src.iterdir():
        print(f"[file] {p.name}  {p.stat().st_size:,} bytes")

    api = HfApi()
    msg = args.commit_message or f"upload DPO checkpoint from {args.src}"
    print(f"[upload] create_repo (exist_ok=True)  private={args.private}")
    api.create_repo(args.dest_repo, repo_type="model",
                    private=args.private, exist_ok=True)
    print(f"[upload] upload_folder {args.src} -> {args.dest_repo}")
    api.upload_folder(
        repo_id=args.dest_repo,
        repo_type="model",
        folder_path=str(args.src),
        commit_message=msg,
    )
    print(f"[done] https://huggingface.co/{args.dest_repo}")


if __name__ == "__main__":
    main()
