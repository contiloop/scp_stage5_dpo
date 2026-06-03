#!/usr/bin/env python3
"""Pretty-print OOD eval predictions for quick eyeballing.

Usage:
  python scripts/view_ood.py artifacts/dpo/ood_eval/ood_predictions_final.jsonl
  python scripts/view_ood.py ... --n 20            # first 20
  python scripts/view_ood.py ... --grep finance    # rows containing 'finance' in source
  python scripts/view_ood.py ... --diff            # only rows where hyp != ref
  python scripts/view_ood.py ... --json            # raw json dump (one per line)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--grep", type=str, default=None,
                    help="filter rows containing this substring in source")
    ap.add_argument("--diff", action="store_true",
                    help="only show rows where hypothesis != reference")
    ap.add_argument("--truncated", action="store_true",
                    help="only show rows that were truncated")
    ap.add_argument("--json", action="store_true",
                    help="output raw jsonl instead of pretty format")
    ap.add_argument("--sort", choices=["xcomet", "bleu", "chrf", "idx"],
                    default="idx",
                    help="sort by per-sentence metric (default: idx = file order)")
    ap.add_argument("--worst", action="store_true",
                    help="ascending (worst first); default with --sort is "
                         "descending (best first). Combine with --sort.")
    args = ap.parse_args()

    if not args.file.exists():
        print(f"not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    # Load everything (file is small, ~few MB) so we can sort + filter freely.
    rows: list[dict] = []
    with args.file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    n_truncated = sum(1 for r in rows if r.get("truncated"))
    n_empty = sum(1 for r in rows if not r.get("hypothesis"))

    # Filters
    pool = rows
    if args.truncated:
        pool = [r for r in pool if r.get("truncated")]
    if args.grep:
        pool = [r for r in pool if args.grep in (r.get("source") or "")]
    if args.diff:
        pool = [r for r in pool if r.get("hypothesis") != r.get("reference")]

    # Sort
    key_map = {"xcomet": "xcomet", "bleu": "sentence_bleu",
               "chrf": "sentence_chrf", "idx": "idx"}
    sort_key = key_map[args.sort]
    if sort_key != "idx":
        # None values go to the end regardless of direction
        def sort_fn(r):
            v = r.get(sort_key)
            if v is None:
                return (1, 0.0)
            return (0, v if args.worst else -v)
        pool = sorted(pool, key=sort_fn)
    else:
        # idx order, --worst flips
        pool = sorted(pool, key=lambda r: r.get("idx", 0),
                      reverse=args.worst)

    shown = 0
    for row in pool[: args.n]:
        shown += 1
        if args.json:
            print(json.dumps(row, ensure_ascii=False))
        else:
            scores = []
            if row.get("xcomet") is not None:
                scores.append(f"xcomet={row['xcomet']:.4f}")
            if row.get("sentence_bleu") is not None:
                scores.append(f"BLEU={row['sentence_bleu']:.2f}")
            if row.get("sentence_chrf") is not None:
                scores.append(f"chrF={row['sentence_chrf']:.2f}")
            score_str = "  ".join(scores) if scores else ""
            print("=" * 80)
            print(f"# idx={row.get('idx', '?')}  truncated={row.get('truncated', False)}  {score_str}")
            print(f"[SOURCE]    {row.get('source', '')}")
            print(f"[REFERENCE] {row.get('reference', '')}")
            print(f"[HYPOTHESIS]{row.get('hypothesis', '')}")
            print()

    print("=" * 80, file=sys.stderr)
    print(f"total rows: {total}  truncated: {n_truncated}  "
          f"empty_hyp: {n_empty}  filtered_pool: {len(pool)}  shown: {shown}",
          file=sys.stderr)
    if any(sort_key == "xcomet" for _ in [0]) and not all(r.get("xcomet") is not None for r in rows):
        n_with = sum(1 for r in rows if r.get("xcomet") is not None)
        print(f"note: {total - n_with} rows have no xcomet score", file=sys.stderr)


if __name__ == "__main__":
    main()
