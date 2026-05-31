#!/usr/bin/env python3
"""Unsloth-based DPO trainer for scp stage5.

Mirrors Unsloth's Qwen3 DPO recipe:
  https://docs.unsloth.ai/  (Qwen DPO notebook)

Design rules:
  - Full-weight fine-tune (no LoRA), as requested.
  - All hyperparameters and paths come from configs/dpo.yaml (or its overrides).
  - No silent fallbacks: required keys must exist; required files must exist.

Usage:
  python scripts/train_dpo.py --config configs/dpo.yaml
  python scripts/train_dpo.py --config configs/dpo.yaml --model alwaysgood/other_repo
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# --- Unsloth MUST be imported before transformers/trl for its patches to apply.
# We use FastLanguageModel only for efficient model loading; the trainer itself
# is vanilla TRL DPOTrainer. PatchDPOTrainer() is deliberately NOT called: its
# compiled cache (unsloth_compiled_cache/UnslothDPOTrainer.py) misroutes the
# transformers 5.x TokenizersBackend into the vision processor path and crashes
# with `TokenizersBackend has no attribute tokenizer`. Full-weight DPO doesn't
# need the LoRA-oriented PatchDPOTrainer optimizations anyway.
import unsloth  # noqa: F401  (side-effect import; keep before transformers/trl)
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from transformers import set_seed
from trl import DPOConfig, DPOTrainer


def _req(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"missing required config key '{ctx}.{key}'")
    return d[key]


def load_config(path: Path, model_override: str | None) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"{path} must parse to a dict")
    if model_override:
        cfg.setdefault("model", {})["name_or_path"] = model_override
    return cfg


def build_model_and_tokenizer(model_cfg: dict):
    name = _req(model_cfg, "name_or_path", "model")
    max_length = int(_req(model_cfg, "max_length", "model"))
    dtype_name = _req(model_cfg, "dtype", "model")
    if dtype_name != "bf16":
        raise ValueError(f"only dtype=bf16 supported, got {dtype_name!r}")
    dtype = torch.bfloat16

    attn_impl = _req(model_cfg, "attention_impl", "model")
    padding_side = _req(model_cfg, "padding_side", "model")
    trust_remote_code = bool(_req(model_cfg, "trust_remote_code", "model"))

    print(f"[load] FastLanguageModel.from_pretrained({name!r})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_length,
        dtype=dtype,
        load_in_4bit=False,
        full_finetuning=True,         # full-weight training
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
    )
    tokenizer.padding_side = padding_side
    return model, tokenizer


def load_dpo_dataset(data_cfg: dict):
    train_path = Path(_req(data_cfg, "train_path", "data"))
    eval_path = Path(_req(data_cfg, "eval_path", "data"))
    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"eval file not found: {eval_path}")

    p = _req(data_cfg, "prompt_field", "data")
    c = _req(data_cfg, "chosen_field", "data")
    r = _req(data_cfg, "rejected_field", "data")

    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "eval": str(eval_path)},
    )

    needed = {p, c, r}
    for split in ("train", "eval"):
        cols = set(ds[split].column_names)
        missing = needed - cols
        if missing:
            raise KeyError(f"{split} split missing fields {missing}; has {sorted(cols)}")

    def _rename(row):
        return {"prompt": row[p], "chosen": row[c], "rejected": row[r]}

    ds = ds.map(_rename, remove_columns=[c for c in ds["train"].column_names if c not in {"prompt", "chosen", "rejected"}])
    print(f"[data] train={len(ds['train'])}  eval={len(ds['eval'])}")
    return ds["train"], ds["eval"]


def build_dpo_config(cfg: dict) -> DPOConfig:
    t = _req(cfg, "train", "")
    m = cfg["model"]
    d = cfg["dpo"]

    output_dir = _req(t, "output_dir", "train")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_to = _req(t, "report_to", "train")

    return DPOConfig(
        output_dir=output_dir,
        num_train_epochs=float(_req(t, "num_train_epochs", "train")),
        learning_rate=float(_req(t, "learning_rate", "train")),
        weight_decay=float(_req(t, "weight_decay", "train")),
        warmup_ratio=float(_req(t, "warmup_ratio", "train")),
        lr_scheduler_type=_req(t, "lr_scheduler_type", "train"),
        optim=_req(t, "optim", "train"),
        max_grad_norm=float(_req(t, "max_grad_norm", "train")),
        per_device_train_batch_size=int(_req(t, "per_device_train_batch_size", "train")),
        per_device_eval_batch_size=int(_req(t, "per_device_eval_batch_size", "train")),
        gradient_accumulation_steps=int(_req(t, "gradient_accumulation_steps", "train")),
        gradient_checkpointing=bool(_req(t, "gradient_checkpointing", "train")),
        bf16=bool(_req(t, "bf16", "train")),
        seed=int(_req(t, "seed", "train")),
        logging_steps=int(_req(t, "logging_steps", "train")),
        eval_strategy=_req(t, "eval_strategy", "train"),
        eval_steps=int(_req(t, "eval_steps", "train")),
        save_strategy=_req(t, "save_strategy", "train"),
        save_steps=int(_req(t, "save_steps", "train")),
        save_total_limit=int(_req(t, "save_total_limit", "train")),
        report_to=report_to,
        run_name=_req(t, "run_name", "train"),
        # DPO-specific
        beta=float(_req(d, "beta", "dpo")),
        loss_type=_req(d, "loss_type", "dpo"),
        label_smoothing=float(_req(d, "label_smoothing", "dpo")),
        reference_free=bool(_req(d, "reference_free", "dpo")),
        precompute_ref_log_probs=bool(_req(d, "precompute_ref_log_probs", "dpo")),
        max_length=int(_req(m, "max_length", "model")),
        max_prompt_length=int(_req(m, "max_prompt_length", "model")),
        remove_unused_columns=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=str, default=None,
                    help="override model.name_or_path")
    args = ap.parse_args()

    cfg = load_config(args.config, args.model)
    print(json.dumps({"effective_config": cfg}, ensure_ascii=False, indent=2))

    wb = cfg.get("wandb") or {}
    if wb.get("project"):
        os.environ.setdefault("WANDB_PROJECT", wb["project"])
    if wb.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", wb["entity"])

    set_seed(int(cfg["train"]["seed"]))

    model, tokenizer = build_model_and_tokenizer(cfg["model"])
    train_ds, eval_ds = load_dpo_dataset(cfg["data"])
    dpo_args = build_dpo_config(cfg)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                  # DPOTrainer will clone for full FT
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"[train] starting DPO  steps_per_epoch~={len(train_ds) // (dpo_args.per_device_train_batch_size * dpo_args.gradient_accumulation_steps)}")
    trainer.train()

    final_dir = Path(dpo_args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] saved to {final_dir}")

    # OOD eval after training (spawn fresh process so COMET/policy model don't share state)
    ood_cfg = cfg.get("eval_ood") or {}
    if ood_cfg.get("enabled"):
        when = ood_cfg.get("when")
        if when != "end":
            raise ValueError(f"eval_ood.when={when!r} not implemented; only 'end' supported")
        eval_script = Path(__file__).parent / "eval_ood.py"
        cmd = [sys.executable, str(eval_script),
               "--config", str(args.config),
               "--model", str(final_dir),
               "--tag", "final"]
        env = os.environ.copy()
        # propagate wandb run linkage so eval logs into the same run
        try:
            import wandb
            run = getattr(wandb, "run", None)
            if run is not None:
                env["WANDB_RUN_ID"] = run.id
                env["WANDB_PROJECT"] = run.project
                wandb.finish()
        except Exception as exc:
            print(f"[wandb] could not capture run id for eval: {exc}", file=sys.stderr)
        print(f"[ood] launching: {' '.join(cmd)}")
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            # Training already saved successfully; OOD failure shouldn't
            # mask that. Surface clearly but don't raise.
            print(f"[ood][WARN] eval_ood.py exited with rc={rc}; "
                  f"training artifacts are intact at {final_dir}",
                  file=sys.stderr)
    else:
        print("[ood] eval_ood.enabled=false; skipping")


if __name__ == "__main__":
    main()
