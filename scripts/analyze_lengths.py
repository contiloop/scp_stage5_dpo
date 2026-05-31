#!/usr/bin/env python3
"""Tokenize DPO data with the Qwen 3.5 tokenizer and report length issues.

Checks each row against the configured budgets:
  - prompt tokens > max_prompt_length   -> DPO left-truncates the prompt
  - prompt+chosen tokens > max_length   -> chosen response right-truncated
  - prompt+rejected tokens > max_length -> rejected response right-truncated

Reports percentiles and the worst offenders. No fallbacks.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml
from transformers import AutoTokenizer


def analyze(jsonl_path: Path, tok, max_length: int, max_prompt_length: int) -> dict:
    p_lens, c_lens, r_lens = [], [], []
    flag_prompt_over = []
    flag_chosen_over = []
    flag_rejected_over = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pi = tok(row["prompt"], add_special_tokens=False)["input_ids"]
            ci = tok(row["chosen"], add_special_tokens=False)["input_ids"]
            ri = tok(row["rejected"], add_special_tokens=False)["input_ids"]
            lp, lc, lr = len(pi), len(ci), len(ri)
            p_lens.append(lp); c_lens.append(lc); r_lens.append(lr)
            if lp > max_prompt_length:
                flag_prompt_over.append((i, lp, row.get("row_id")))
            if lp + lc > max_length:
                flag_chosen_over.append((i, lp, lc, lp + lc, row.get("row_id")))
            if lp + lr > max_length:
                flag_rejected_over.append((i, lp, lr, lp + lr, row.get("row_id")))

    def pct(xs):
        xs = sorted(xs)
        n = len(xs)
        if n == 0:
            return {}
        def q(p): return xs[min(n - 1, max(0, int(round(p * (n - 1)))))]
        return {
            "n": n, "min": xs[0], "p50": q(0.50), "p90": q(0.90),
            "p95": q(0.95), "p99": q(0.99), "p99.9": q(0.999), "max": xs[-1],
            "mean": round(statistics.mean(xs), 1),
        }

    return {
        "file": str(jsonl_path),
        "rows": len(p_lens),
        "prompt_tokens": pct(p_lens),
        "chosen_tokens": pct(c_lens),
        "rejected_tokens": pct(r_lens),
        "prompt+chosen": pct([a + b for a, b in zip(p_lens, c_lens)]),
        "prompt+rejected": pct([a + b for a, b in zip(p_lens, r_lens)]),
        "budgets": {"max_length": max_length, "max_prompt_length": max_prompt_length},
        "violations": {
            "prompt_over_max_prompt_length": {
                "count": len(flag_prompt_over),
                "pct": round(100 * len(flag_prompt_over) / max(1, len(p_lens)), 3),
                "worst": sorted(flag_prompt_over, key=lambda x: -x[1])[:5],
            },
            "prompt_plus_chosen_over_max_length": {
                "count": len(flag_chosen_over),
                "pct": round(100 * len(flag_chosen_over) / max(1, len(p_lens)), 3),
                "worst": sorted(flag_chosen_over, key=lambda x: -x[3])[:5],
            },
            "prompt_plus_rejected_over_max_length": {
                "count": len(flag_rejected_over),
                "pct": round(100 * len(flag_rejected_over) / max(1, len(p_lens)), 3),
                "worst": sorted(flag_rejected_over, key=lambda x: -x[3])[:5],
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    ap.add_argument("--tokenizer", type=str, default="alwaysgood/qwen35-it")
    ap.add_argument("--out", type=Path, default=Path("data/processed/length_report.json"))
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m = cfg["model"]
    max_length = int(m["max_length"])
    max_prompt_length = int(m["max_prompt_length"])

    print(f"[tok] loading {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[tok] vocab_size={tok.vocab_size}")
    print(f"[budgets] max_length={max_length}  max_prompt_length={max_prompt_length}")

    reports = []
    for p in args.files:
        print(f"[scan] {p}")
        reports.append(analyze(p, tok, max_length, max_prompt_length))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
