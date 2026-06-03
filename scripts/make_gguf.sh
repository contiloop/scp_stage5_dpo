#!/usr/bin/env bash
# Convert a HF DPO checkpoint (Qwen3.5 VLM) into LM Studio-ready GGUFs
# (BF16 + Q4_K_M) and upload to an HF GGUF model repo.
#
# Flow:
#   1. Download the source HF model
#   2. Strip out the vision tower so we have a pure Qwen3 LM
#   3. Clone & build llama.cpp (skipped if already built at LLAMA_CPP_DIR)
#   4. convert_hf_to_gguf.py -> BF16 GGUF
#   5. llama-quantize -> Q4_K_M GGUF
#   6. Upload both files to a new HF model repo
#
# Defaults can be overridden via env vars at the top.
#
# Usage:
#   bash scripts/make_gguf.sh
#   SRC_REPO=alwaysgood/<other> DST_REPO=alwaysgood/<other_v1> bash scripts/make_gguf.sh

set -eu

SRC_REPO=${SRC_REPO:-alwaysgood/scp-stage4-oneshot-forward_dpo_sigmoid}
DST_REPO=${DST_REPO:-alwaysgood/TranslateQwen_v1}
WORK_DIR=${WORK_DIR:-/tmp/translate_qwen_gguf}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$WORK_DIR/llama.cpp}
BASENAME=${BASENAME:-TranslateQwen_v1}

mkdir -p "$WORK_DIR"
echo "[plan] SRC_REPO  = $SRC_REPO"
echo "[plan] DST_REPO  = $DST_REPO"
echo "[plan] WORK_DIR  = $WORK_DIR"
echo "[plan] BASENAME  = $BASENAME"

SRC_DIR="$WORK_DIR/src_hf"
LM_DIR="$WORK_DIR/lm_only_hf"
GGUF_BF16="$WORK_DIR/${BASENAME}-BF16.gguf"
GGUF_Q4="$WORK_DIR/${BASENAME}-Q4_K_M.gguf"

# ====================================================================
# 1) Download source DPO model
# ====================================================================
if [ ! -f "$SRC_DIR/model.safetensors" ]; then
    echo "[1/6] downloading $SRC_REPO -> $SRC_DIR"
    hf download "$SRC_REPO" --local-dir "$SRC_DIR"
else
    echo "[1/6] already have $SRC_DIR/model.safetensors, skip download"
fi

# ====================================================================
# 2) Strip vision tower -> text-only Qwen3 LM
# ====================================================================
if [ ! -f "$LM_DIR/model.safetensors" ]; then
    echo "[2/6] extracting LM only -> $LM_DIR"
    python3.11 scripts/extract_lm_only.py --src "$SRC_DIR" --dst "$LM_DIR"
else
    echo "[2/6] already have $LM_DIR/model.safetensors, skip"
fi

# ====================================================================
# 3) Build llama.cpp
# ====================================================================
if [ ! -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    echo "[3/6] cloning + building llama.cpp at $LLAMA_CPP_DIR"
    mkdir -p "$WORK_DIR"
    [ -d "$LLAMA_CPP_DIR" ] || git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    pushd "$LLAMA_CPP_DIR" >/dev/null
    git pull --ff-only
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j --target llama-quantize
    python3.11 -m pip install --quiet -r requirements/requirements-convert_hf_to_gguf.txt
    popd >/dev/null
else
    echo "[3/6] llama.cpp already built, skip"
fi

# ====================================================================
# 4) Convert to BF16 GGUF
# ====================================================================
if [ ! -f "$GGUF_BF16" ]; then
    echo "[4/6] convert HF -> BF16 GGUF"
    python3.11 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$LM_DIR" \
        --outfile "$GGUF_BF16" \
        --outtype bf16
else
    echo "[4/6] BF16 GGUF exists, skip"
fi
ls -lh "$GGUF_BF16"

# ====================================================================
# 5) Quantize to Q4_K_M
# ====================================================================
if [ ! -f "$GGUF_Q4" ]; then
    echo "[5/6] quantize -> Q4_K_M"
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_BF16" "$GGUF_Q4" Q4_K_M
else
    echo "[5/6] Q4_K_M GGUF exists, skip"
fi
ls -lh "$GGUF_Q4"

# ====================================================================
# 6) Upload to HF model repo
# ====================================================================
echo "[6/6] upload to $DST_REPO"
python3.11 - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$DST_REPO", repo_type="model", exist_ok=True)
# Upload README first so the repo isn't empty looking
readme = """# $BASENAME

GGUF builds of \`$SRC_REPO\` for LM Studio / llama.cpp.

- \`${BASENAME}-BF16.gguf\` — full precision, ~9 GB
- \`${BASENAME}-Q4_K_M.gguf\` — 4-bit quantization, ~2-3 GB

Built from a Qwen 3.5 LM (text-only, vision tower stripped) DPO'd
on stage4 preference pairs (sigmoid, all minor+major edits).

## Usage in LM Studio

1. Download one of the GGUF files
2. Place it under ~/.lmstudio/models/$DST_REPO/
3. Load and chat. Prompt template:
   \`\`\`
   ### Instruction:
   Translate the English source into Korean.

   ### Source:
   <English text>

   ### Response:
   \`\`\`
"""
api.upload_file(path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id="$DST_REPO",
                repo_type="model")

for f in ["$GGUF_BF16", "$GGUF_Q4"]:
    name = f.rsplit("/", 1)[-1]
    print(f"  uploading {name} ...")
    api.upload_file(path_or_fileobj=f, path_in_repo=name,
                    repo_id="$DST_REPO", repo_type="model")
print(f"https://huggingface.co/$DST_REPO")
PY

echo "[done] all uploaded to https://huggingface.co/$DST_REPO"
