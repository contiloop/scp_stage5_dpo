#!/usr/bin/env bash
# Overnight pipeline: Phase 1 (sft_023 + apo_zero + major-only)
#                  -> Phase 2 (sft_025 + apo_zero + major-only)
#                  -> Phase 3 (push everything to HF)
#
# Design rules:
#   - Each step logs its own stdout/stderr to logs/overnight/<phase>_<step>.log
#   - A failure in one step is recorded in the summary but does NOT abort
#     subsequent steps (so you wake up to as many results as possible).
#   - The final summary file logs/overnight/SUMMARY.md tells you exactly
#     what succeeded, what failed, and where to look.
#
# Usage from the project root:
#   nohup bash scripts/run_overnight.sh > logs/overnight/driver.log 2>&1 &
#   echo "PID: $!"

set -u                       # unset vars are errors
shopt -s lastpipe

# --- Configuration ---
PY=${PY:-python3.11}
SFT1_REPO=alwaysgood/qwen35_sft_023
SFT2_RUN_ID=sft_v2_equal_bucket_from_024
SFT2_SUBSET=subset_025
SFT2_REPO=alwaysgood/qwen35_sft_025

DPO1_REPO=alwaysgood/qwen35_sft_023_dpo_run3_apo
DPO2_REPO=alwaysgood/qwen35_sft_025_dpo_run3_apo

PRECOMPUTE_REPO=alwaysgood/scp-stage5-precompute
OOD_REPO=alwaysgood/scp-stage5-ood-run3

LOG_DIR=logs/overnight
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.md"

# --- Helpers ---
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

run() {
    # run "<step-id>" command args...
    local step="$1"; shift
    local logfile="$LOG_DIR/${step}.log"
    log "==> $step  (log: $logfile)"
    {
        echo "# $(ts) $step"
        echo "# cmd: $*"
        echo "----"
    } >> "$logfile"
    local t0=$(date +%s)
    if "$@" >> "$logfile" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        log "    OK     $step  (${dt}s)"
        echo "- ✅ **$step** (${dt}s) — \`$logfile\`" >> "$SUMMARY"
        return 0
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        log "    FAIL   $step  rc=$rc  (${dt}s)"
        echo "- ❌ **$step** rc=$rc (${dt}s) — \`$logfile\`" >> "$SUMMARY"
        return $rc
    fi
}

# --- Summary header ---
{
    echo "# overnight summary"
    echo
    echo "started: $(ts)"
    echo "host:    $(hostname)"
    echo "pwd:     $(pwd)"
    echo "py:      $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
    echo
    echo "## steps"
} > "$SUMMARY"

log "overnight pipeline starting"

# ====================================================================
# Phase 0: refresh code + data
# ====================================================================
run p0_git_pull git pull

# Re-prepare data with --major-only every time so even if local data is
# stale we have the right split.
run p0_prepare_data $PY scripts/prepare_dpo_data.py --major-only
run p0_show_data_stats cat data/processed/dpo_stats.json

# Wipe any stale precompute cache. The cache_key includes (model + data
# file size/mtime), so prepare_dpo_data above already invalidates the
# old key automatically; this is belt-and-suspenders.
run p0_clean_artifacts bash -c "rm -rf artifacts/dpo/precompute_cache && rm -rf artifacts/dpo/checkpoint-* && rm -rf artifacts/dpo/final"

# ====================================================================
# Phase 1: SFT1 (alwaysgood/qwen35_sft_023) + apo_zero + major-only
# ====================================================================
log "##### PHASE 1: $SFT1_REPO #####"
{
    echo
    echo "## Phase 1: $SFT1_REPO"
} >> "$SUMMARY"

# train_dpo uses configs/dpo.yaml's model.name_or_path by default (= SFT1).
# eval_ood runs automatically at the end of training.
run p1_train $PY scripts/train_dpo.py --config configs/dpo.yaml

# Stash phase-1 artifacts under a stable name BEFORE phase 2 overwrites them.
run p1_stash_final bash -c "mv artifacts/dpo/final artifacts/dpo/final_run3_sft023"
run p1_stash_ood_metrics bash -c "[ -f artifacts/dpo/ood_eval/ood_metrics_final.json ] && cp artifacts/dpo/ood_eval/ood_metrics_final.json artifacts/dpo/ood_eval/ood_metrics_run3_sft023.json || true"
run p1_stash_ood_preds bash -c "[ -f artifacts/dpo/ood_eval/ood_predictions_final.jsonl ] && cp artifacts/dpo/ood_eval/ood_predictions_final.jsonl artifacts/dpo/ood_eval/ood_predictions_run3_sft023.jsonl || true"

# Push the trained DPO model to HF.
run p1_push_dpo $PY scripts/upload_dpo_checkpoint.py \
    --src artifacts/dpo/final_run3_sft023 \
    --dest-repo "$DPO1_REPO"

# Capture phase-1 metrics into the summary for quick eyeballing.
if [ -f "artifacts/dpo/ood_eval/ood_metrics_run3_sft023.json" ]; then
    {
        echo
        echo "### Phase 1 OOD metrics"
        echo '```json'
        cat artifacts/dpo/ood_eval/ood_metrics_run3_sft023.json
        echo
        echo '```'
    } >> "$SUMMARY"
fi

# ====================================================================
# Phase 2: SFT2 (sft_v2_equal_bucket_from_024/subset_025) + apo_zero
# ====================================================================
log "##### PHASE 2: $SFT2_REPO #####"
{
    echo
    echo "## Phase 2: $SFT2_REPO (from $SFT2_RUN_ID/$SFT2_SUBSET)"
} >> "$SUMMARY"

# Mirror the SFT2 checkpoint from the runs dataset into its own model repo.
run p2_upload_sft $PY scripts/upload_sft_checkpoint.py \
    --runs-repo alwaysgood/scp-stage4-sft-v2-runs \
    --run-id "$SFT2_RUN_ID" \
    --subset "$SFT2_SUBSET" \
    --dest-repo "$SFT2_REPO"

# Train DPO from SFT2. --model overrides configs/dpo.yaml; cache_key
# automatically differs from SFT1's, so this triggers a fresh precompute.
run p2_train $PY scripts/train_dpo.py --config configs/dpo.yaml --model "$SFT2_REPO"

run p2_stash_final bash -c "mv artifacts/dpo/final artifacts/dpo/final_run3_sft025"
run p2_stash_ood_metrics bash -c "[ -f artifacts/dpo/ood_eval/ood_metrics_final.json ] && cp artifacts/dpo/ood_eval/ood_metrics_final.json artifacts/dpo/ood_eval/ood_metrics_run3_sft025.json || true"
run p2_stash_ood_preds bash -c "[ -f artifacts/dpo/ood_eval/ood_predictions_final.jsonl ] && cp artifacts/dpo/ood_eval/ood_predictions_final.jsonl artifacts/dpo/ood_eval/ood_predictions_run3_sft025.jsonl || true"

run p2_push_dpo $PY scripts/upload_dpo_checkpoint.py \
    --src artifacts/dpo/final_run3_sft025 \
    --dest-repo "$DPO2_REPO"

if [ -f "artifacts/dpo/ood_eval/ood_metrics_run3_sft025.json" ]; then
    {
        echo
        echo "### Phase 2 OOD metrics"
        echo '```json'
        cat artifacts/dpo/ood_eval/ood_metrics_run3_sft025.json
        echo
        echo '```'
    } >> "$SUMMARY"
fi

# ====================================================================
# Phase 3: push caches + OOD results so we can drop the instance
# ====================================================================
log "##### PHASE 3: HF push (precompute cache + OOD) #####"
{
    echo
    echo "## Phase 3: HF push"
} >> "$SUMMARY"

# Both SFT1 and SFT2 precompute caches sit side-by-side under
# artifacts/dpo/precompute_cache/<hash>/. Upload the whole tree.
run p3_push_precompute $PY -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$PRECOMPUTE_REPO', repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(
    folder_path='artifacts/dpo/precompute_cache',
    path_in_repo='precompute_cache',
    repo_id='$PRECOMPUTE_REPO',
    repo_type='dataset',
    commit_message='run3 caches (sft_023 + sft_025, apo_zero, major-only)',
)
print('https://huggingface.co/datasets/$PRECOMPUTE_REPO')
"

run p3_push_ood $PY -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$OOD_REPO', repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(
    folder_path='artifacts/dpo/ood_eval',
    repo_id='$OOD_REPO',
    repo_type='dataset',
    commit_message='run3 OOD predictions + metrics (sft_023 + sft_025)',
)
print('https://huggingface.co/datasets/$OOD_REPO')
"

# ====================================================================
# Final summary
# ====================================================================
{
    echo
    echo "## Final comparison"
    echo '| tag | BLEU | chrF | xcomet |'
    echo '|---|---|---|---|'
    for f in artifacts/dpo/ood_eval/ood_metrics_*.json; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .json | sed 's/^ood_metrics_//')
        $PY -c "
import json,sys
try:
    m=json.load(open('$f'))
    print(f\"| {'$name'} | {m.get('BLEU','-'):.2f} | {m.get('chrF','-'):.2f} | {m.get('xcomet_mean','-'):.4f} |\")
except Exception as e:
    print(f\"| {'$name'} | err: {e} | | |\")
" || echo "| $name | parse-error | | |"
    done
    echo
    echo "finished: $(ts)"
} >> "$SUMMARY"

log "overnight pipeline finished. See $SUMMARY"
echo
echo "===================================================="
cat "$SUMMARY"
echo "===================================================="
