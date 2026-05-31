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
    args = ap.parse_args()

    if not args.file.exists():
        print(f"not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    shown = 0
    total = 0
    n_truncated = 0
    n_empty = 0
    with args.file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if row.get("truncated"):
                n_truncated += 1
            if not row.get("hypothesis"):
                n_empty += 1
            if args.truncated and not row.get("truncated"):
                continue
            if args.grep and args.grep not in row.get("source", ""):
                continue
            if args.diff and row.get("hypothesis") == row.get("reference"):
                continue
            if shown >= args.n:
                continue
            shown += 1
            if args.json:
                print(json.dumps(row, ensure_ascii=False))
            else:
                print("=" * 80)
                print(f"# row {total}  truncated={row.get('truncated', False)}")
                print(f"[SOURCE]    {row.get('source', '')}")
                print(f"[REFERENCE] {row.get('reference', '')}")
                print(f"[HYPOTHESIS]{row.get('hypothesis', '')}")
                print()

    print("=" * 80, file=sys.stderr)
    print(f"total rows: {total}  truncated: {n_truncated}  "
          f"empty_hyp: {n_empty}  shown: {shown}", file=sys.stderr)


if __name__ == "__main__":
    main()
