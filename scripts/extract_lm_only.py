#!/usr/bin/env python3
"""Extract the text-only language model from a Qwen3.5 VLM checkpoint and
save it as a standalone Qwen3ForCausalLM-style HF model directory.

Why: llama.cpp's convert_hf_to_gguf.py expects the language model only;
the vision tower weights and the VLM wrapper class are not useful for
text-only translation inference (LM Studio etc).

Input layout (after remap_dpo_checkpoint.py / vLLM-ready):
  language_model.model.X
  visual.X                  (dropped)
  lm_head.X                 (kept)
Output layout (Qwen3ForCausalLM):
  model.X
  lm_head.X

No fallbacks: missing inputs raise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LM_PREFIX = "language_model.model."
VISION_PREFIX = "visual."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="source HF model dir (post-remap, vLLM-ready)")
    ap.add_argument("--dst", type=Path, required=True,
                    help="destination dir for the text-only model")
    args = ap.parse_args()

    src_safetensors = args.src / "model.safetensors"
    if not src_safetensors.exists():
        print(f"[error] {src_safetensors} not found "
              f"(only single-file safetensors supported)", file=sys.stderr)
        sys.exit(2)

    args.dst.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1) Filter weights: keep LM and lm_head, drop visual.*
    # ----------------------------------------------------------------
    new_state: dict = {}
    kept_lm = kept_head = dropped_vision = other = 0
    with safe_open(str(src_safetensors), framework="pt") as f:
        for k in f.keys():
            if k.startswith(LM_PREFIX):
                # language_model.model.X -> model.X
                new_state["model." + k[len(LM_PREFIX):]] = f.get_tensor(k)
                kept_lm += 1
            elif k.startswith("lm_head."):
                new_state[k] = f.get_tensor(k)
                kept_head += 1
            elif k.startswith(VISION_PREFIX):
                dropped_vision += 1
            else:
                # Unknown — refuse to silently drop. Likely a remap pattern issue.
                other += 1
                print(f"[warn] unknown key (kept as-is): {k}", file=sys.stderr)
                new_state[k] = f.get_tensor(k)

    print(f"[strip] kept LM={kept_lm}  kept lm_head={kept_head}  "
          f"dropped vision={dropped_vision}  unknown_kept={other}")

    dst_safetensors = args.dst / "model.safetensors"
    save_file(new_state, str(dst_safetensors), metadata={"format": "pt"})
    print(f"[write] {dst_safetensors}  ({dst_safetensors.stat().st_size:,} bytes)")

    # ----------------------------------------------------------------
    # 2) Rewrite config.json: VLM -> text-only LM
    # ----------------------------------------------------------------
    cfg_path = args.src / "config.json"
    if not cfg_path.exists():
        print(f"[error] {cfg_path} missing", file=sys.stderr)
        sys.exit(2)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Replace VLM architecture with text LM. Qwen3-style LM class name is
    # typically Qwen3ForCausalLM; for hybrid variants it may differ. We
    # leave the rest of config intact (hidden_size, num_layers, vocab, etc.)
    # but drop vision-related sub-configs which would confuse downstream loaders.
    old_arch = cfg.get("architectures")
    cfg["architectures"] = ["Qwen3ForCausalLM"]
    cfg.pop("vision_config", None)
    cfg.pop("visual", None)
    cfg.pop("multimodal_config", None)
    cfg["model_type"] = cfg.get("text_config", {}).get("model_type") or "qwen3"

    # If the original config had a nested text_config, hoist its fields up
    text_cfg = cfg.pop("text_config", None)
    if isinstance(text_cfg, dict):
        for k, v in text_cfg.items():
            cfg.setdefault(k, v)

    (args.dst / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[config] architectures: {old_arch} -> {cfg['architectures']}")
    print(f"[config] model_type: {cfg['model_type']}")

    # ----------------------------------------------------------------
    # 3) Copy tokenizer / generation_config unchanged
    # ----------------------------------------------------------------
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json",
                 "chat_template.json", "generation_config.json"):
        p = args.src / name
        if p.exists():
            shutil.copy2(p, args.dst / name)
            print(f"[copy] {name}")

    print(f"[done] {args.dst}")


if __name__ == "__main__":
    main()
