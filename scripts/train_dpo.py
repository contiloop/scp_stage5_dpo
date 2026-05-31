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
import hashlib
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

# --- Force text routing in DPOTrainer._prepare_dataset.
# Unsloth misclassifies Qwen 3.5 (text-only LM) as a VLM
# (see warning "VLM processor fallback returned None for model_type=qwen3_5"),
# which makes `self.is_vision_model = True` and routes preprocessing through
# dpo_trainer_vision_process_row. That path then does
# `processing_class.tokenizer` and crashes on the transformers 5.x
# TokenizersBackend (which has no `.tokenizer` attribute).
# Forcing is_vision_model=False before the dataset map sends both train
# and eval splits through the correct text `tokenize_row` path.
_orig_prepare_dataset = DPOTrainer._prepare_dataset

def _prepare_dataset_text_only(self, *args, **kwargs):
    self.is_vision_model = False
    return _orig_prepare_dataset(self, *args, **kwargs)

DPOTrainer._prepare_dataset = _prepare_dataset_text_only


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
    ap.add_argument("--precompute-only", action="store_true",
                    help="run reference-logp precompute and save the cache, "
                         "then exit before trainer.train() is called. Use this "
                         "to populate the cache ahead of time (e.g. in parallel "
                         "with OOD eval review) so the next real training run "
                         "starts immediately.")
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

    # --- Reference log-prob precompute cache.
    # The precomputed ref logps depend only on (reference model + data + tokenizer);
    # they are invariant to lr / beta / grad_clip / epochs / etc. Save them after
    # the first run and reuse on subsequent runs so we don't pay the ~2h precompute
    # cost again for every hyperparameter sweep.
    #
    # Cache key: sha1 of (model id, train+eval file paths and their mtime/size).
    # Change the model or the data file content and you get a cache miss.
    cache_root = Path(cfg["train"]["output_dir"]) / "precompute_cache"
    train_path = Path(cfg["data"]["train_path"])
    eval_path = Path(cfg["data"]["eval_path"])
    cache_seed = "|".join([
        str(cfg["model"]["name_or_path"]),
        f"{train_path}:{train_path.stat().st_size}:{int(train_path.stat().st_mtime)}",
        f"{eval_path}:{eval_path.stat().st_size}:{int(eval_path.stat().st_mtime)}",
        f"max_length={cfg['model']['max_length']}",
        f"max_prompt_length={cfg['model']['max_prompt_length']}",
    ])
    cache_key = hashlib.sha1(cache_seed.encode("utf-8")).hexdigest()[:12]
    cache_dir = cache_root / cache_key
    cache_train = cache_dir / "train"
    cache_eval = cache_dir / "eval"
    cache_meta = cache_dir / "meta.json"

    used_cache = False
    if cache_train.exists() and cache_eval.exists() and cache_meta.exists():
        from datasets import load_from_disk
        print(f"[precompute] cache HIT at {cache_dir}")
        print(f"[precompute] loading augmented datasets (includes ref logps)")
        train_ds = load_from_disk(str(cache_train))
        eval_ds = load_from_disk(str(cache_eval))
        used_cache = True
    else:
        print(f"[precompute] cache MISS at {cache_dir} — will precompute "
              f"and save (~2h on A100 80GB for 20k pairs)")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                  # DPOTrainer will clone for full FT
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # If we just computed the ref logps, persist the augmented datasets for reuse.
    # TRL DPOTrainer.__init__ runs the precompute pass and stores the augmented
    # dataset on the trainer instance, so by the time we reach here it's ready.
    if not used_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[precompute] saving augmented datasets -> {cache_dir}")
            trainer.train_dataset.save_to_disk(str(cache_train))
            trainer.eval_dataset.save_to_disk(str(cache_eval))
            cache_meta.write_text(json.dumps({
                "model": cfg["model"]["name_or_path"],
                "train_path": str(train_path),
                "eval_path": str(eval_path),
                "max_length": cfg["model"]["max_length"],
                "max_prompt_length": cfg["model"]["max_prompt_length"],
                "cache_key": cache_key,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[precompute] cache saved; future runs with same "
                  f"model+data will skip the precompute step")
        except Exception as exc:
            # Caching is an optimization, never a correctness requirement.
            print(f"[precompute][WARN] failed to save cache: {exc}", file=sys.stderr)

    if args.precompute_only:
        print(f"[precompute-only] cache populated; exiting before trainer.train()")
        return

    print(f"[train] starting DPO  steps_per_epoch~={len(train_ds) // (dpo_args.per_device_train_batch_size * dpo_args.gradient_accumulation_steps)}")
    trainer.train()

    final_dir = Path(dpo_args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[save] raw checkpoint -> {final_dir}")

    # Unsloth + TRL DPOTrainer wrap the Qwen3.5 LM three deep
    # (model.language_model.language_model.language_model.X) and leave the
    # vision tower wrapped once (model.language_model.visual.X). vLLM's
    # Qwen3_5ForConditionalGeneration loader needs
    # language_model.model.X and visual.X. Rewrite the safetensors file
    # in place right after save so downstream tools (vLLM, hub upload)
    # see a clean checkpoint and no separate remap step is needed.
    sys.path.insert(0, str(Path(__file__).parent))
    from remap_dpo_checkpoint import remap_inplace  # noqa: E402
    try:
        remap_inplace(final_dir)
        print(f"[done] saved + remapped to {final_dir}")
    except Exception as exc:
        # Don't lose the run: the raw checkpoint is still on disk and
        # scripts/remap_dpo_checkpoint.py can repair it manually.
        print(f"[remap][WARN] in-place remap failed: {exc}", file=sys.stderr)
        print(f"[remap][WARN] raw checkpoint preserved at {final_dir}; "
              f"recover with: python scripts/remap_dpo_checkpoint.py "
              f"--src {final_dir} --dst {final_dir}_vllm", file=sys.stderr)

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
