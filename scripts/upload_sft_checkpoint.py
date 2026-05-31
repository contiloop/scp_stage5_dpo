#!/usr/bin/env python3
"""Mirror an SFT checkpoint from the runs dataset repo into a standalone model repo.

The SFT run artifacts live under
  hf://datasets/<runs_repo>/<runs_root>/<run_id>/subsets/<subset>/train_final/full_weight_model/

Loading them via `AutoModelForCausalLM.from_pretrained` is awkward because
that path is inside a *dataset* repo. This script downloads just that
directory and re-uploads it as a *model* repo so downstream code can do
`from_pretrained("<model_repo_id>")` directly.

No fallbacks: every required arg must be passed explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-repo", required=True,
                    help="HF dataset repo holding the SFT run, e.g. alwaysgood/scp-stage4-sft-v2-runs")
    ap.add_argument("--run-id", required=True,
                    help="run id under <runs-root>/, e.g. sft_v2_c4sel_from014")
    ap.add_argument("--subset", required=True,
                    help="subset directory name, e.g. subset_023")
    ap.add_argument("--runs-root", default="artifacts/runs",
                    help="path prefix inside the dataset repo (default: artifacts/runs)")
    ap.add_argument("--ckpt-subpath", default="train_final/full_weight_model",
                    help="path inside the subset dir holding HF model files")
    ap.add_argument("--dest-repo", required=True,
                    help="destination HF model repo id, e.g. alwaysgood/qwen35_sft_023")
    ap.add_argument("--local-dir", type=Path, default=Path("artifacts/sft_ckpt"),
                    help="local staging dir for the download")
    ap.add_argument("--private", action="store_true",
                    help="create dest model repo as private")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    inner = f"{args.runs_root}/{args.run_id}/subsets/{args.subset}/{args.ckpt_subpath}"
    print(f"[plan] dataset={args.runs_repo}  subpath={inner}")
    print(f"[plan] local staging={args.local_dir}")
    print(f"[plan] dest model repo={args.dest_repo} (private={args.private})")

    if not args.skip_download:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] snapshot_download from dataset repo ...")
        snap_dir = snapshot_download(
            repo_id=args.runs_repo,
            repo_type="dataset",
            allow_patterns=[f"{inner}/*"],
            local_dir=str(args.local_dir),
        )
        print(f"[download] snapshot dir: {snap_dir}")

    model_dir = args.local_dir / inner
    missing = [n for n in REQUIRED_FILES if not (model_dir / n).exists()]
    if missing:
        print(f"[error] missing required files in {model_dir}: {missing}", file=sys.stderr)
        sys.exit(2)
    for n in REQUIRED_FILES:
        size = (model_dir / n).stat().st_size
        print(f"[ok] {n}  {size:,} bytes")

    if args.skip_upload:
        print(f"[done] staged at {model_dir} (upload skipped)")
        return

    api = HfApi()
    print(f"[upload] create_repo (exist_ok=True)")
    api.create_repo(args.dest_repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"[upload] upload_folder -> {args.dest_repo}")
    api.upload_folder(
        repo_id=args.dest_repo,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=f"import {args.run_id}/{args.subset}/{args.ckpt_subpath}",
    )
    print(f"[done] https://huggingface.co/{args.dest_repo}")


if __name__ == "__main__":
    main()
