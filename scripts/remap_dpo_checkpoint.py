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
import os
import shutil
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


# Three rename rules, applied in order (first match wins):
#   LM (text):
#     model.language_model.language_model.language_model.X  ->  language_model.model.X
#   Vision tower (untouched by DPO, but still wrapped once by Unsloth's outer .model):
#     model.language_model.visual.X                          ->  visual.X
#   lm_head (rare; usually tied to embed_tokens so absent):
#     lm_head.X                                              ->  lm_head.X
RENAME_RULES = (
    ("model.language_model.language_model.language_model.", "language_model.model."),
    ("model.language_model.visual.",                        "visual."),
    ("lm_head.",                                            "lm_head."),
)


def remap_keys(keys: list[str]) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    bad: list[str] = []
    for k in keys:
        for src, dst in RENAME_RULES:
            if k.startswith(src):
                mapping[k] = dst + k[len(src):]
                break
        else:
            bad.append(k)
    return mapping, bad


def _remap_safetensors_file(src: Path, dst: Path) -> dict[str, int]:
    """Read safetensors at src, remap key names, write to dst. Returns stats."""
    with safe_open(str(src), framework="pt") as f:
        keys = list(f.keys())
    mapping, bad = remap_keys(keys)
    if bad:
        raise RuntimeError(
            f"{len(bad)} keys do not match expected wrap pattern; "
            f"first offenders: {bad[:5]}"
        )
    new_state: dict = {}
    with safe_open(str(src), framework="pt") as f:
        for k in keys:
            new_state[mapping[k]] = f.get_tensor(k)
    save_file(new_state, str(dst), metadata={"format": "pt"})
    return {"renamed": len(mapping), "src_bytes": src.stat().st_size,
            "dst_bytes": dst.stat().st_size}


def remap_directory(src_dir: Path, dst_dir: Path) -> None:
    """Remap weights in src_dir and write a complete model dir to dst_dir.
    Copies config / tokenizer files unchanged."""
    src_safetensors = src_dir / "model.safetensors"
    if not src_safetensors.exists():
        idx = src_dir / "model.safetensors.index.json"
        if idx.exists():
            raise NotImplementedError("sharded safetensors not yet supported")
        raise FileNotFoundError(src_safetensors)
    dst_dir.mkdir(parents=True, exist_ok=True)
    stats = _remap_safetensors_file(src_safetensors, dst_dir / "model.safetensors")
    print(f"[remap] {stats['renamed']} tensors  "
          f"src={stats['src_bytes']:,}b  dst={stats['dst_bytes']:,}b")
    for name in ("config.json", "generation_config.json",
                 "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json",
                 "chat_template.json", "preprocessor_config.json"):
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, dst_dir / name)


def remap_inplace(target_dir: Path) -> None:
    """Remap weights in target_dir, writing back to the same model.safetensors.
    Uses a temp file + atomic rename so a crash mid-write doesn't corrupt the
    original. Other files (config, tokenizer) are untouched."""
    src = target_dir / "model.safetensors"
    if not src.exists():
        idx = target_dir / "model.safetensors.index.json"
        if idx.exists():
            raise NotImplementedError("sharded safetensors not yet supported")
        raise FileNotFoundError(src)
    with tempfile.NamedTemporaryFile(
        dir=target_dir, prefix="model.safetensors.", suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        stats = _remap_safetensors_file(src, tmp_path)
        os.replace(tmp_path, src)
        print(f"[remap-inplace] {stats['renamed']} tensors rewritten in {src}")
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path,
                    help="if omitted, remap is performed in-place on --src")
    args = ap.parse_args()

    if args.dst is None:
        remap_inplace(args.src)
    else:
        remap_directory(args.src, args.dst)
    print(f"[done] {args.dst or args.src}")


if __name__ == "__main__":
    main()
