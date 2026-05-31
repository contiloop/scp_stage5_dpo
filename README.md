# scp stage5 DPO

Unsloth full-weight DPO on top of the stage4 SFT v2 checkpoint.

## Layout

```
configs/dpo.yaml                     # single source of truth (no fallbacks)
scripts/prepare_dpo_data.py          # HF preference_pairs.jsonl -> {prompt,chosen,rejected} jsonl
scripts/upload_sft_checkpoint.py     # mirror SFT ckpt from runs-dataset -> standalone model repo
scripts/train_dpo.py                 # Unsloth DPOTrainer driver
data/processed/dpo_{train,val}.jsonl # built artifacts
artifacts/dpo/                       # training output
```

## 0. Environment

Mirrors `scp_stage4_sft_v2`'s `make set-real-env` exactly (same pinned
torch / transformers 5.5.0 / unsloth / vLLM 0.19.1 / flash-attn 2.8.3 /
COMET isolation venv):

```
# inside the project root
make set-real-env                  # uses system python3.11 directly
# or with a project-local venv:
USE_VENV=1 make set-real-env
```

The QE isolation venv for COMET is created at `~/.venvs/comet`. The
target prints `export COMET_PYTHON=...` at the end — export it before
running OOD eval:

```
export COMET_PYTHON=$HOME/.venvs/comet/bin/python
```

Then login:

```
huggingface-cli login
wandb login
```

## 1. Build DPO data (already run)

```
python scripts/prepare_dpo_data.py
```

Source: `alwaysgood/scp-stage4-run-main-001/preference_pairs.jsonl`.
Filter:
- `error_type == "none"`
- `teacher_label in {minor_edit, major_edit}` (rewrite / no_change / invalid dropped)
- non-empty fields, `gold != student`

Output: 20,191 train / 203 val.

## 2. Upload the SFT checkpoint as a model repo

The SFT checkpoint lives inside a *dataset* repo subpath, which is awkward
for `from_pretrained`. Mirror it once to a model repo:

```
python scripts/upload_sft_checkpoint.py \
  --runs-repo alwaysgood/scp-stage4-sft-v2-runs \
  --run-id    sft_v2_c4sel_from014 \
  --subset    subset_023 \
  --dest-repo alwaysgood/qwen35_sft_023
```

(Use `--private` if you want the model repo private. Use `--skip-upload`
to only stage locally.)

## 3. Train

```
python scripts/train_dpo.py --config configs/dpo.yaml
```

OOD eval runs automatically at the end of training (vLLM greedy generation
on `data/test.csv` → BLEU + chrF in-process, xcomet via `$COMET_PYTHON`
subprocess). You can also run it standalone:

```
make eval-ood
# or
python scripts/eval_ood.py --config configs/dpo.yaml --model artifacts/dpo/final --tag final
```

`COMET_PYTHON` must be exported (see Environment). If unset, eval_ood
raises rather than silently skipping xcomet.

If a wandb run is active when training finishes, the eval subprocess
attaches to the same run (via `WANDB_RUN_ID` / `WANDB_PROJECT`) and logs
metrics under `ood/*`.

To swap the base checkpoint without touching the YAML:

```
python scripts/train_dpo.py --config configs/dpo.yaml \
  --model alwaysgood/some_other_repo
```

## Hyperparameters

Locked per request:

| key                      | value          |
|--------------------------|----------------|
| learning_rate            | 8e-7           |
| beta                     | 0.1            |
| epochs                   | 1              |
| effective_batch_size     | 128            |
| warmup_ratio             | 0.03           |
| bf16                     | true           |
| weight_decay             | 0.0            |
| training mode            | full weight    |

Mirrored from `scp_stage4_sft_v2`:

| key              | value                |
|------------------|----------------------|
| max_length       | 4096                 |
| max_prompt_length| 2048 (≈source 1700 + headroom) |
| attention_impl   | flash_attention_2    |
| padding_side     | left                 |

DPO-specific defaults (in `configs/dpo.yaml`, edit if needed):

| key                       | value     |
|---------------------------|-----------|
| loss_type                 | sigmoid   |
| precompute_ref_log_probs  | true      |
| gradient_checkpointing    | true      |

## Single-GPU sizing (A100 80GB)

- per_device_train_batch_size = 1, gradient_accumulation_steps = 128 → effective 128.
- ~20k pairs / 128 ≈ **158 steps** per epoch.
- `precompute_ref_log_probs=true` runs the reference model once over the data
  then frees it from GPU, so only the policy (~4.5B params) carries the
  optimizer + gradient footprint.
