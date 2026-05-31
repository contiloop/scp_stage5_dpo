#!/usr/bin/env python3
"""Build DPO training data from scp-stage4 preference pairs.

Source: HF dataset `alwaysgood/scp-stage4-run-main-001`, file `preference_pairs.jsonl`.

Filter:
  - error_type == "none"        (drop technical failures / filtered rows)
  - teacher_label in {minor_edit, major_edit}
  - source, gold, student all non-empty
  - gold != student              (otherwise chosen == rejected)

Output schema (one JSON object per line):
  {"prompt": <str>, "chosen": <str>, "rejected": <str>,
   "row_id": <str>, "teacher_label": <str>}

Prompt format mirrors scp_stage4_sft_v2 `sft` template so DPO sees the same
input shape as supervised fine-tuning:

  ### Instruction:
  Translate the English source into Korean.

  ### Source:
  {source}

  ### Response:
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

HF_URL = (
    "https://huggingface.co/datasets/alwaysgood/scp-stage4-run-main-001/"
    "resolve/main/preference_pairs.jsonl?download=true"
)

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Translate the English source into Korean.\n\n"
    "### Source:\n{source}\n\n"
    "### Response:\n"
)

KEEP_LABELS_DEFAULT = {"minor_edit", "major_edit"}
KEEP_LABELS_MAJOR_ONLY = {"major_edit"}


def download(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst} already exists ({dst.stat().st_size:,} bytes)")
        return
    print(f"[download] {HF_URL} -> {dst}")
    urllib.request.urlretrieve(HF_URL, dst)
    print(f"[done] {dst.stat().st_size:,} bytes")


def build(src: Path, out_dir: Path, val_ratio: float, seed: int,
          keep_labels: set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    counts = {"total": 0, "kept": 0}
    reasons: dict[str, int] = {}

    def drop(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts["total"] += 1
            r = json.loads(line)
            if r.get("error_type") != "none":
                drop(f"error_type={r.get('error_type')}")
                continue
            label = r.get("teacher_label")
            if label not in keep_labels:
                drop(f"label={label}")
                continue
            source = (r.get("source") or "").strip()
            gold = (r.get("gold") or "").strip()
            student = (r.get("student") or "").strip()
            if not source or not gold or not student:
                drop("empty_field")
                continue
            if gold == student:
                drop("gold==student")
                continue
            rows.append(
                {
                    "prompt": PROMPT_TEMPLATE.format(source=source),
                    "chosen": gold,
                    "rejected": student,
                    "row_id": r.get("row_id"),
                    "teacher_label": label,
                }
            )
            counts["kept"] += 1

    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = int(len(rows) * val_ratio)
    val, train = rows[:n_val], rows[n_val:]

    train_path = out_dir / "dpo_train.jsonl"
    val_path = out_dir / "dpo_val.jsonl"
    for path, chunk in [(train_path, train), (val_path, val)]:
        with path.open("w", encoding="utf-8") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_dist = {l: sum(1 for r in rows if r["teacher_label"] == l) for l in keep_labels}
    stats = {
        "input_rows": counts["total"],
        "kept_rows": counts["kept"],
        "train_rows": len(train),
        "val_rows": len(val),
        "kept_labels": sorted(keep_labels),
        "label_distribution": label_dist,
        "drop_reasons": reasons,
        "seed": seed,
        "val_ratio": val_ratio,
    }
    (out_dir / "dpo_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/preference_pairs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--major-only", action="store_true",
                    help="keep only teacher_label=major_edit (drops minor_edit)")
    args = ap.parse_args()

    if not args.skip_download:
        download(args.raw)
    if not args.raw.exists():
        print(f"[error] {args.raw} not found", file=sys.stderr)
        sys.exit(1)
    keep_labels = KEEP_LABELS_MAJOR_ONLY if args.major_only else KEEP_LABELS_DEFAULT
    print(f"[filter] keep_labels = {sorted(keep_labels)}")
    build(args.raw, args.out, args.val_ratio, args.seed, keep_labels)


if __name__ == "__main__":
    main()
