#!/usr/bin/env python3
"""Try to recover a previously computed reference-logp dataset from the
HF datasets cache and copy it into our explicit precompute cache.

The HF `datasets` library auto-caches `.map()` outputs to
~/.cache/huggingface/datasets/ keyed by an input + function fingerprint.
TRL's DPOTrainer ref-logp precompute uses `.map()`, so the augmented
dataset (with reference_chosen_logps + reference_rejected_logps columns)
*may* still be on disk from a previous run even though our explicit
caching code wasn't in place yet.

This script:
  1. Walks the HF datasets cache.
  2. Loads each arrow dataset, checks for the ref-logp columns.
  3. Reports candidates (size, columns, row count).
  4. With --apply <key>, copies the chosen pair into our cache location
     so the next train_dpo.py run hits it.

No fallbacks: if a candidate is malformed, it is skipped with a warning;
no silent guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REF_COLS = {"reference_chosen_logps", "reference_rejected_logps"}


def scan_hf_cache(cache_root: Path) -> list[dict]:
    from datasets import load_from_disk
    hits = []
    if not cache_root.exists():
        return hits
    for entry in cache_root.rglob("dataset_info.json"):
        ds_dir = entry.parent
        try:
            info = json.loads(entry.read_text(encoding="utf-8"))
        except Exception:
            continue
        feats = info.get("features") or {}
        cols = set(feats.keys()) if isinstance(feats, dict) else set()
        if not REF_COLS.issubset(cols):
            continue
        try:
            ds = load_from_disk(str(ds_dir))
            rows = len(ds)
        except Exception as exc:
            print(f"[skip] {ds_dir}: {exc}", file=sys.stderr)
            continue
        size_bytes = sum(p.stat().st_size for p in ds_dir.glob("*.arrow"))
        hits.append({
            "path": str(ds_dir),
            "rows": rows,
            "columns": sorted(cols),
            "arrow_bytes": size_bytes,
        })
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                    default=Path(os.environ.get(
                        "HF_DATASETS_CACHE",
                        Path.home() / ".cache" / "huggingface" / "datasets")))
    ap.add_argument("--apply-train", type=Path, default=None,
                    help="path of a scanned candidate to use as the train split")
    ap.add_argument("--apply-eval", type=Path, default=None,
                    help="path of a scanned candidate to use as the eval split")
    ap.add_argument("--dst", type=Path, default=None,
                    help="destination cache dir (must already be the key used "
                         "by train_dpo.py; copy meta.json from a freshly-built "
                         "cache or compute by hand)")
    args = ap.parse_args()

    print(f"[scan] HF datasets cache: {args.cache_root}")
    hits = scan_hf_cache(args.cache_root)
    if not hits:
        print("[scan] no datasets with reference_*_logps columns found.")
        print("       The first-run precompute is most likely gone.")
        print("       Run `python scripts/train_dpo.py --config configs/dpo.yaml "
              "--precompute-only` to repopulate (~2h on A100 80GB).")
        return

    print(f"[scan] {len(hits)} candidate(s):")
    for i, h in enumerate(hits):
        print(f"  [{i}] rows={h['rows']:>6}  arrow={h['arrow_bytes']:>12,}b")
        print(f"       cols={h['columns']}")
        print(f"       path={h['path']}")

    if args.apply_train and args.apply_eval and args.dst:
        if not args.apply_train.exists() or not args.apply_eval.exists():
            print("[error] --apply-train / --apply-eval paths do not exist",
                  file=sys.stderr)
            sys.exit(2)
        args.dst.mkdir(parents=True, exist_ok=True)
        train_dst = args.dst / "train"
        eval_dst = args.dst / "eval"
        if train_dst.exists() or eval_dst.exists():
            print(f"[error] destination already has train/ or eval/ — "
                  f"refuse to overwrite", file=sys.stderr)
            sys.exit(3)
        print(f"[copy] {args.apply_train} -> {train_dst}")
        shutil.copytree(str(args.apply_train), str(train_dst))
        print(f"[copy] {args.apply_eval}  -> {eval_dst}")
        shutil.copytree(str(args.apply_eval), str(eval_dst))
        meta = {"recovered_from_hf_cache": True,
                "train_src": str(args.apply_train),
                "eval_src": str(args.apply_eval)}
        (args.dst / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] recovered to {args.dst}")
    else:
        print()
        print("To recover, pick a train and eval candidate from the list above,")
        print("then run with --apply-train <path> --apply-eval <path> --dst "
              "artifacts/dpo/precompute_cache/<key_from_a_fresh_train_dpo_log>")


if __name__ == "__main__":
    main()
