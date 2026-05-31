#!/usr/bin/env python3
"""xcomet scorer — runs inside the QE isolation venv ($COMET_PYTHON).

Reads a predictions jsonl with fields {source, reference, hypothesis, truncated}
and writes a metrics json with xcomet_mean, xcomet_n, xcomet_skipped_empty.

Kept separate from `eval_ood.py` so the heavy COMET deps (unbabel-comet)
live in the QE venv and don't poison the training env, matching the
stage4 `set-real-env` pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--gpus", type=int, required=True)
    args = ap.parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)

    data = []
    skipped = 0
    with args.predictions.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            hyp = r.get("hypothesis") or ""
            if not hyp:
                skipped += 1
                continue
            data.append({"src": r["source"], "mt": hyp, "ref": r["reference"]})

    from comet import download_model, load_from_checkpoint
    print(f"[xcomet] loading {args.model_name}", file=sys.stderr)
    ckpt = download_model(args.model_name)
    comet_model = load_from_checkpoint(ckpt)

    if data:
        result = comet_model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
        out = {
            "xcomet_mean": float(result["system_score"]),
            "xcomet_n": len(data),
            "xcomet_skipped_empty": skipped,
            "xcomet_model": args.model_name,
        }
    else:
        out = {"xcomet_mean": None, "xcomet_n": 0,
               "xcomet_skipped_empty": skipped, "xcomet_model": args.model_name}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
