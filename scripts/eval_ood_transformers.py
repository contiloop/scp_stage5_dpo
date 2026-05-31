#!/usr/bin/env python3
"""Fallback OOD generation using transformers `.generate()` instead of vLLM.

Use when vLLM fails to load the saved DPO checkpoint (e.g. because the
config declares a VLM architecture but the weights are text-only LM
without the expected `language_model.` prefix).

Outputs the same `ood_predictions_<tag>.jsonl` shape as scripts/eval_ood.py
so the xcomet/BLEU/chrF flow can pick up from there:

  python scripts/eval_ood_transformers.py --config configs/dpo.yaml \
      --model artifacts/dpo/final --tag final
  # then score:
  $COMET_PYTHON scripts/eval_ood_xcomet.py \
      --predictions artifacts/dpo/ood_eval/ood_predictions_final.jsonl \
      --out artifacts/dpo/ood_eval/ood_xcomet_final.json \
      --model-name Unbabel/wmt22-comet-da --batch-size 8 --gpus 1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Translate the English source into Korean.\n\n"
    "### Source:\n{source}\n\n"
    "### Response:\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--tag", type=str, default="final")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ood = cfg["eval_ood"]
    gen = ood["generation"]
    out_dir = Path(ood["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = int(gen["max_new_tokens"])
    max_length = int(cfg["model"]["max_length"])
    max_input_len = max_length - max_new_tokens

    sources, refs = [], []
    with Path(ood["test_csv"]).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row[ood["source_column"]])
            refs.append(row[ood["reference_column"]])
    print(f"[ood] {len(sources)} rows")

    print(f"[hf] loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    model.eval()

    prompts = [PROMPT_TEMPLATE.format(source=s) for s in sources]
    hyps: list[str] = [""] * len(sources)
    truncated = [False] * len(sources)

    t0 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=False)
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        keep = [j for j, l in enumerate(input_lens) if l <= max_input_len]
        for j, l in enumerate(input_lens):
            if l > max_input_len:
                truncated[i + j] = True
        if not keep:
            continue
        sub = tok([batch[j] for j in keep], return_tensors="pt", padding=True, truncation=False)
        sub = {k: v.to("cuda") for k, v in sub.items()}
        with torch.inference_mode():
            out = model.generate(
                **sub,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        gen_only = out[:, sub["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        for local_j, full_j in enumerate(keep):
            hyps[i + full_j] = texts[local_j].strip()
        done = i + len(batch)
        print(f"[gen] {done}/{len(prompts)}")

    gen_secs = time.time() - t0
    print(f"[gen] done in {gen_secs:.1f}s")

    preds_path = out_dir / f"ood_predictions_{args.tag}.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for src, ref, hyp, tr in zip(sources, refs, hyps, truncated):
            f.write(json.dumps({"source": src, "reference": ref,
                                "hypothesis": hyp, "truncated": tr},
                               ensure_ascii=False) + "\n")
    print(f"[ood] -> {preds_path}")


if __name__ == "__main__":
    main()
