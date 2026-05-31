#!/usr/bin/env python3
"""Remap a triple-wrapped Unsloth+TRL DPO checkpoint into the layout
vLLM expects for the Qwen3.5 VLM architecture.

Background:
  Unsloth's FastLanguageModel.from_pretrained on a Qwen3.5 VLM puts the
  LM inside a `.model.language_model` attribute. TRL's DPOTrainer wraps
  it once more, then HF Trainer.save_model serializes the resulting
  state_dict verbatim. The result is that every LM weight ends up with
  a `model.language_model.language_model.language_model.X` prefix,
  while vLLM's Qwen3_5ForConditionalGeneration loader expects
  `language_model.model.X`.

This script:
  1. Reads <src>/model.safetensors
  2. Renames `model.language_model.language_model.language_model.X`
     → `language_model.model.X`
  3. Writes <dst>/model.safetensors with the new names
  4. Copies config.json, tokenizer, generation_config, etc. unchanged

No fallbacks: any key that does not match the expected wrap pattern
raises and aborts. Run again with the printed offenders if you need to
add a special case.

Usage:
  python scripts/remap_dpo_checkpoint.py \
      --src artifacts/dpo/final --dst artifacts/dpo/final_vllm
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


WRAP_PREFIX = "model.language_model.language_model.language_model."
TARGET_PREFIX = "language_model.model."


def remap_keys(keys: list[str]) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    bad: list[str] = []
    for k in keys:
        if k.startswith(WRAP_PREFIX):
            mapping[k] = TARGET_PREFIX + k[len(WRAP_PREFIX):]
        elif k == "lm_head.weight" or k.startswith("lm_head."):
            mapping[k] = k
        else:
            bad.append(k)
    return mapping, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    src_safetensors = args.src / "model.safetensors"
    if not src_safetensors.exists():
        # sharded?
        idx = args.src / "model.safetensors.index.json"
        if idx.exists():
            print("[error] sharded safetensors not yet supported", file=sys.stderr)
        else:
            print(f"[error] not found: {src_safetensors}", file=sys.stderr)
        sys.exit(2)

    args.dst.mkdir(parents=True, exist_ok=True)

    with safe_open(str(src_safetensors), framework="pt") as f:
        keys = list(f.keys())

    print(f"[scan] {len(keys)} tensors in {src_safetensors}")
    mapping, bad = remap_keys(keys)
    if bad:
        print(f"[error] {len(bad)} keys do not match expected wrap pattern:",
              file=sys.stderr)
        for k in bad[:20]:
            print(f"        {k}", file=sys.stderr)
        sys.exit(3)

    print(f"[map] renaming {len(mapping)} tensors")
    print(f"[map] sample: {next(iter(mapping.items()))}")

    # Load + remap. Loads the full state dict into memory; for a ~9 GB
    # text-only Qwen3.5 LM this is fine on a workstation/instance.
    new_state: dict = {}
    with safe_open(str(src_safetensors), framework="pt") as f:
        for k in keys:
            new_state[mapping[k]] = f.get_tensor(k)

    dst_safetensors = args.dst / "model.safetensors"
    save_file(new_state, str(dst_safetensors), metadata={"format": "pt"})
    print(f"[write] {dst_safetensors}  ({dst_safetensors.stat().st_size:,} bytes)")

    # Copy remaining files unchanged
    for name in ("config.json", "generation_config.json",
                 "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json",
                 "chat_template.json", "preprocessor_config.json"):
        p = args.src / name
        if p.exists():
            shutil.copy2(p, args.dst / name)
            print(f"[copy] {name}")

    print(f"[done] {args.dst}")


if __name__ == "__main__":
    main()
