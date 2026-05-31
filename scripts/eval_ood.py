#!/usr/bin/env python3
"""OOD evaluation for the scp stage5 DPO policy.

Pipeline:
  1. Load held-out CSV (Source_En, Target_Ko).
  2. Generate Korean translations with vLLM (greedy).
  3. Compute BLEU + chrF (sacrebleu) in this process.
  4. Compute xcomet by subprocessing into the QE venv ($COMET_PYTHON),
     mirroring stage4's split between the training env and the COMET env.
  5. Merge metrics, write json, optionally log to wandb.

Strictly no fallbacks: every required cfg key / file / env var must be present.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Translate the English source into Korean.\n\n"
    "### Source:\n{source}\n\n"
    "### Response:\n"
)


def _req(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"missing required config key '{ctx}.{key}'")
    return d[key]


class _SkipEval(Exception):
    """Raised when OOD eval cannot proceed but the surrounding pipeline should
    continue (e.g. training job should not be marked failed just because eval
    config is degenerate)."""


def load_test_csv(path: Path, src_col: str, ref_col: str) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"OOD test csv not found: {path}")
    sources, refs = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for col in (src_col, ref_col):
            if col not in cols:
                raise KeyError(f"{path} missing column {col!r}; has {cols}")
        for row in reader:
            sources.append(row[src_col])
            refs.append(row[ref_col])
    return sources, refs


def vllm_generate(model_path: str, prompts: list[str], *, max_length: int,
                  max_new_tokens: int, on_truncation: str,
                  tensor_parallel_size: int, dtype: str,
                  gpu_memory_utilization: float) -> tuple[list[str], list[bool]]:
    """Greedy generation with vLLM. Returns (hypotheses, truncated_flags)."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[vllm] loading {model_path} tp={tensor_parallel_size} dtype={dtype} "
          f"max_model_len={max_length}")
    # language_model_only=True: Qwen 3.5 config declares
    # `Qwen3_5ForConditionalGeneration` (VLM). Without this flag vLLM tries to
    # initialize the vision encoder and load an image processor, which doesn't
    # exist in our text-only SFT/DPO checkpoint (no preprocessor_config.json).
    # Mirrors scp_stage4_sft_v2/src/scp_stage4/pipeline/workers/vllm_inference_worker.py.
    llm = LLM(
        model=model_path,
        dtype=dtype,
        max_model_len=max_length,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        language_model_only=True,
    )

    tok = AutoTokenizer.from_pretrained(model_path)
    max_input_len = max_length - max_new_tokens
    if max_input_len <= 0:
        # Soft fail: signal caller to skip eval rather than abort the whole run.
        raise _SkipEval(
            f"max_length({max_length}) <= max_new_tokens({max_new_tokens}); "
            f"no prompt budget. Skipping OOD eval."
        )

    truncated = [False] * len(prompts)
    keep_idx: list[int] = []
    keep_prompts: list[str] = []
    for i, p in enumerate(prompts):
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_input_len:
            if on_truncation == "discard":
                truncated[i] = True
                continue
            elif on_truncation == "keep":
                pass
            else:
                raise ValueError(f"on_truncation={on_truncation!r} not supported")
        keep_idx.append(i)
        keep_prompts.append(p)

    n_drop = sum(truncated)
    print(f"[vllm] generating {len(keep_prompts)} / {len(prompts)} (discarded {n_drop})")

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(keep_prompts, sp)
    hyps: list[str] = [""] * len(prompts)
    # vLLM preserves input order for `generate(list, sampling_params)`
    for j, out in enumerate(outputs):
        full_i = keep_idx[j]
        hyps[full_i] = out.outputs[0].text.strip()
    return hyps, truncated


def run_xcomet_subprocess(predictions_path: Path, out_path: Path, settings: dict) -> dict:
    comet_python = os.environ.get("COMET_PYTHON")
    if not comet_python:
        raise RuntimeError(
            "COMET_PYTHON env var is not set. "
            "Run `eval $(make set-real-env | tail -1)` or set it manually "
            "to the QE venv python (e.g. ~/.venvs/comet/bin/python)."
        )
    if not Path(comet_python).exists():
        raise FileNotFoundError(f"COMET_PYTHON points to missing file: {comet_python}")

    model_name = _req(settings, "model_name", "metric_settings.xcomet")
    batch_size = int(_req(settings, "batch_size", "metric_settings.xcomet"))
    gpus = int(_req(settings, "gpus", "metric_settings.xcomet"))
    script = Path(__file__).parent / "eval_ood_xcomet.py"
    cmd = [
        comet_python, str(script),
        "--predictions", str(predictions_path),
        "--out", str(out_path),
        "--model-name", model_name,
        "--batch-size", str(batch_size),
        "--gpus", str(gpus),
    ]
    print(f"[xcomet] subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True,
                    help="local path or HF id of the model to evaluate")
    ap.add_argument("--tag", type=str, default="final")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"{args.config} must parse to a dict")

    ood = _req(cfg, "eval_ood", "")
    if not bool(_req(ood, "enabled", "eval_ood")):
        print("[ood] disabled; exit")
        return

    when = _req(ood, "when", "eval_ood")
    if when != "end":
        raise ValueError(f"eval_ood.when={when!r} not implemented; only 'end' supported")

    test_csv = Path(_req(ood, "test_csv", "eval_ood"))
    src_col = _req(ood, "source_column", "eval_ood")
    ref_col = _req(ood, "reference_column", "eval_ood")
    out_dir = Path(_req(ood, "output_dir", "eval_ood"))
    metrics = _req(ood, "metrics", "eval_ood")
    metric_settings = _req(ood, "metric_settings", "eval_ood")
    gen_cfg = _req(ood, "generation", "eval_ood")

    max_new_tokens = int(_req(gen_cfg, "max_new_tokens", "eval_ood.generation"))
    on_truncation = _req(gen_cfg, "on_truncation", "eval_ood.generation")
    if bool(_req(gen_cfg, "do_sample", "eval_ood.generation")):
        raise ValueError("eval_ood.generation.do_sample=true not supported (greedy only)")

    vcfg = _req(ood, "vllm", "eval_ood")
    tp = int(_req(vcfg, "tensor_parallel_size", "eval_ood.vllm"))
    vdtype = _req(vcfg, "dtype", "eval_ood.vllm")
    gpu_mem = float(_req(vcfg, "gpu_memory_utilization", "eval_ood.vllm"))

    max_length = int(_req(cfg["model"], "max_length", "model"))

    out_dir.mkdir(parents=True, exist_ok=True)
    sources, refs = load_test_csv(test_csv, src_col, ref_col)
    print(f"[ood] {len(sources)} rows from {test_csv}")
    prompts = [PROMPT_TEMPLATE.format(source=s) for s in sources]

    t0 = time.time()
    try:
        hyps, truncated = vllm_generate(
            args.model, prompts,
            max_length=max_length, max_new_tokens=max_new_tokens,
            on_truncation=on_truncation,
            tensor_parallel_size=tp, dtype=vdtype,
            gpu_memory_utilization=gpu_mem,
        )
    except _SkipEval as exc:
        print(f"[ood][WARN] {exc}", file=sys.stderr)
        skipped = {"tag": args.tag, "model": args.model, "skipped": True,
                   "reason": str(exc)}
        (out_dir / f"ood_metrics_{args.tag}.json").write_text(
            json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    gen_secs = time.time() - t0
    print(f"[gen] done in {gen_secs:.1f}s")

    preds_path = out_dir / f"ood_predictions_{args.tag}.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for src, ref, hyp, tr in zip(sources, refs, hyps, truncated):
            f.write(json.dumps({"source": src, "reference": ref,
                                "hypothesis": hyp, "truncated": tr},
                               ensure_ascii=False) + "\n")
    print(f"[ood] predictions -> {preds_path}")

    merged: dict[str, Any] = {
        "tag": args.tag,
        "model": args.model,
        "n_rows": len(sources),
        "n_truncated": sum(truncated),
        "generation_seconds": gen_secs,
    }

    if "BLEU" in metrics or "chrF" in metrics:
        import sacrebleu
        # Restrict BLEU/chrF to non-empty hypotheses for parity with xcomet
        pairs = [(h, r) for h, r in zip(hyps, refs) if h]
        bleu_hyps = [h for h, _ in pairs]
        bleu_refs = [r for _, r in pairs]
        if "BLEU" in metrics:
            merged["BLEU"] = float(sacrebleu.corpus_bleu(bleu_hyps, [bleu_refs]).score)
        if "chrF" in metrics:
            merged["chrF"] = float(sacrebleu.corpus_chrf(bleu_hyps, [bleu_refs]).score)
        merged["bleu_chrf_n"] = len(pairs)

    if "xcomet" in metrics:
        xcomet_out = out_dir / f"ood_xcomet_{args.tag}.json"
        x_settings = _req(metric_settings, "xcomet", "eval_ood.metric_settings")
        xres = run_xcomet_subprocess(preds_path, xcomet_out, x_settings)
        merged.update(xres)

    metrics_path = out_dir / f"ood_metrics_{args.tag}.json"
    metrics_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, indent=2))

    wb_run_id = os.environ.get("WANDB_RUN_ID")
    wb_project = os.environ.get("WANDB_PROJECT")
    if wb_run_id and wb_project:
        try:
            import wandb
            wandb.init(project=wb_project, id=wb_run_id, resume="allow")
            wandb.log({f"ood/{k}": v for k, v in merged.items()
                       if isinstance(v, (int, float))})
            wandb.finish()
            print(f"[wandb] logged ood/* to run {wb_run_id}")
        except Exception as exc:
            print(f"[wandb] skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
