SHELL := /bin/sh
PYTHON_VERSION ?= 3.11
PYTHON ?= python$(PYTHON_VERSION)
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PYTHON_TAG := cp$(subst .,,$(PYTHON_VERSION))
USE_VENV ?= 0
REAL_ENV_PY := $(if $(filter 1,$(USE_VENV)),$(VENV_PYTHON),$(PYTHON))
QE_VENV_DIR ?= $(HOME)/.venvs/comet
PYTHONPATH := src

# ------------------------------------------------------------------
# Pinned versions — IDENTICAL to scp_stage4_sft_v2 set-real-env, so the
# DPO stage runs on the same Qwen3.5 / Unsloth / vLLM stack.
# ------------------------------------------------------------------
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu128
PIN_TORCH_VERSION ?= 2.10.0
PIN_TORCHVISION_VERSION ?= 0.25.0
PIN_TORCHAUDIO_VERSION ?= 2.10.0
PIN_TRANSFORMERS_VERSION ?= 5.5.0
PIN_TRL_VERSION ?= 0.24.0
PIN_DATASETS_VERSION ?= 3.4.1
PIN_UNSLOTH_VERSION ?= 2026.5.2
PIN_UNSLOTH_ZOO_VERSION ?= 2026.5.1
PIN_VLLM_VERSION ?= 0.19.1
PIN_HF_HUB_VERSION ?= 1.14.0
PIN_HF_XET_VERSION ?= 1.5.0
PIN_FLASH_ATTN_VERSION ?= 2.8.3
PIN_SETUPTOOLS_SPEC ?= "setuptools>=77.0.3,<81.0.0"
PIN_NUMPY_VERSION ?= 2.2.6
PIN_FLA_CORE_VERSION ?= 0.4.2
PIN_FLASH_LINEAR_ATTENTION_VERSION ?= 0.4.2

FLASH_ATTN_REPO ?= alwaysgood/scp-stage4-wheels
FLASH_ATTN_WHL ?= flash_attn-$(PIN_FLASH_ATTN_VERSION)-$(PYTHON_TAG)-$(PYTHON_TAG)-linux_x86_64.whl
FLASH_ATTN_WHL_SM80 ?= flash_attn-$(PIN_FLASH_ATTN_VERSION)-1sm80-$(PYTHON_TAG)-$(PYTHON_TAG)-linux_x86_64.whl
FLASH_ATTN_WHL_SM120 ?= flash_attn-$(PIN_FLASH_ATTN_VERSION)-1sm120-$(PYTHON_TAG)-$(PYTHON_TAG)-linux_x86_64.whl
FLASH_ATTN_GPU_ARCH ?= auto
SKIP_CAUSAL_CONV1D ?= 0

# ------------------------------------------------------------------
# Stage5 specific
# ------------------------------------------------------------------
CONFIG ?= configs/dpo.yaml
SFT_RUNS_REPO ?= alwaysgood/scp-stage4-sft-v2-runs
SFT_RUN_ID    ?= sft_v2_c4sel_from014
SFT_SUBSET    ?= subset_023
SFT_DEST_REPO ?= alwaysgood/qwen35_sft_023

.PHONY: set-real-env verify-cuda-kernels prepare-data upload-sft train-dpo eval-ood

# ==================================================================
# set-real-env  (mirror of scp_stage4_sft_v2/Makefile::set-real-env)
# ==================================================================
set-real-env:
	@if [ "$(USE_VENV)" = "1" ] && [ ! -x "$(VENV_PYTHON)" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv --python $(PYTHON_VERSION) --seed $(VENV_DIR); \
		else \
			$(PYTHON) -m venv $(VENV_DIR); \
		fi; \
	fi
	@$(REAL_ENV_PY) -c 'import sys; want=tuple(map(int, "$(PYTHON_VERSION)".split(".")[:2])); print("set-real-env: python", sys.version.split()[0]); sys.exit(f"set-real-env requires Python {want[0]}.{want[1]}, got {sys.version.split()[0]}") if sys.version_info[:2] != want else sys.exit(0)'
	@$(REAL_ENV_PY) -m pip install --upgrade pip
	@$(REAL_ENV_PY) -m pip install $(PIN_SETUPTOOLS_SPEC)
	@$(REAL_ENV_PY) -m pip install \
		--index-url $(TORCH_INDEX_URL) \
		"torch==$(PIN_TORCH_VERSION)" \
		"torchvision==$(PIN_TORCHVISION_VERSION)" \
		"torchaudio==$(PIN_TORCHAUDIO_VERSION)"
	@$(REAL_ENV_PY) -m pip install \
		"trl==$(PIN_TRL_VERSION)" \
		"datasets==$(PIN_DATASETS_VERSION)"
	@$(REAL_ENV_PY) -m pip install \
		"unsloth-zoo==$(PIN_UNSLOTH_ZOO_VERSION)" \
		"unsloth==$(PIN_UNSLOTH_VERSION)"
	@$(REAL_ENV_PY) -m pip uninstall -y vllm || true
	@$(REAL_ENV_PY) -m pip install \
		"vllm==$(PIN_VLLM_VERSION)" \
		--extra-index-url $(TORCH_INDEX_URL)
	@$(REAL_ENV_PY) -m pip install --index-url $(TORCH_INDEX_URL) "xformers==0.0.34"
	@$(REAL_ENV_PY) -m pip install \
		tokenizers hydra-core omegaconf \
		openai peft wandb sacrebleu \
		sentencepiece bitsandbytes hf_transfer msgspec tyro torchao ninja \
		pyyaml
	# Intentionally pin transformers 5.5.0 for Qwen3.5 architecture support
	@$(REAL_ENV_PY) -m pip install --no-deps \
		"transformers==$(PIN_TRANSFORMERS_VERSION)" \
		"huggingface_hub>=$(PIN_HF_HUB_VERSION),<2" \
		"hf-xet>=$(PIN_HF_XET_VERSION),<2"
	@$(REAL_ENV_PY) -m pip install --upgrade "numpy==$(PIN_NUMPY_VERSION)"
	# FlashAttention2 wheel selection by GPU arch
	@arch_choice="$(FLASH_ATTN_GPU_ARCH)"; \
	selected_whl="$(FLASH_ATTN_WHL)"; \
	py_tag="$(PYTHON_TAG)"; \
	py_ver="$$( $(REAL_ENV_PY) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' )"; \
	if [ "$$arch_choice" = "auto" ]; then \
		detected="$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk -F'.' 'BEGIN{max=0} {gsub(/[^0-9.]/,""); if ($$1 ~ /^[0-9]+$$/) {minor=$$2; if (minor !~ /^[0-9]+$$/) minor=0; val=($$1*10)+minor; if (val>max) max=val;}} END{if (max>0) printf "sm%d", max;}')"; \
		if [ -n "$$detected" ]; then arch_choice="$$detected"; else arch_choice="default"; fi; \
	fi; \
	case "$$arch_choice" in \
		sm80) selected_whl="$(FLASH_ATTN_WHL_SM80)" ;; \
		sm120) selected_whl="$(FLASH_ATTN_WHL_SM120)" ;; \
		default|'') selected_whl="$(FLASH_ATTN_WHL)" ;; \
		*) echo "  [WARN] unknown FLASH_ATTN_GPU_ARCH=$$arch_choice, using default wheel"; selected_whl="$(FLASH_ATTN_WHL)" ;; \
	esac; \
	echo "  flash_attn target: python=$$py_ver ($$py_tag) arch=$$arch_choice wheel=$$selected_whl"; \
	if $(REAL_ENV_PY) -m pip install \
		"https://huggingface.co/datasets/$(FLASH_ATTN_REPO)/resolve/main/$$selected_whl"; then \
		echo "  flash_attn wheel install ok: $$selected_whl"; \
	else \
		echo "  [ERROR] flash_attn wheel unavailable: $$selected_whl"; \
		exit 1; \
	fi
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip causal_conv1d setup (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(REAL_ENV_PY) bash scripts/ensure_causal_conv1d.sh; \
	fi
	@$(REAL_ENV_PY) -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2>/dev/null \
		|| $(REAL_ENV_PY) -m pip install --no-deps \
			"fla-core==$(PIN_FLA_CORE_VERSION)" \
			"flash-linear-attention==$(PIN_FLASH_LINEAR_ATTENTION_VERSION)"
	@$(REAL_ENV_PY) -m pip install --upgrade "numpy==$(PIN_NUMPY_VERSION)"
	@$(MAKE) verify-cuda-kernels REAL_ENV_PY=$(REAL_ENV_PY) SKIP_CAUSAL_CONV1D=$(SKIP_CAUSAL_CONV1D)
	@$(REAL_ENV_PY) -c 'import sys, torch; print("set-real-env:", sys.executable, "torch", torch.__version__)'
	@echo "set-real-env: setting up QE isolation venv at $(QE_VENV_DIR)..."
	@if [ ! -x "$(QE_VENV_DIR)/bin/python" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv --python $(PYTHON_VERSION) --seed $(QE_VENV_DIR); \
		else \
			$(PYTHON) -m venv --without-pip $(QE_VENV_DIR) && \
			curl -sS https://bootstrap.pypa.io/get-pip.py | $(QE_VENV_DIR)/bin/python; \
		fi; \
	fi
	@$(QE_VENV_DIR)/bin/python -m pip install --upgrade pip setuptools wheel
	@$(QE_VENV_DIR)/bin/pip install \
		torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	@$(QE_VENV_DIR)/bin/pip install --no-deps transformers
	@$(QE_VENV_DIR)/bin/pip install \
		sentencepiece safetensors accelerate huggingface_hub \
		"unbabel-comet>=2.2.7" sacrebleu
	@$(QE_VENV_DIR)/bin/python -c 'import torch; print("set-real-env: QE venv torch", torch.__version__, "cuda", torch.cuda.is_available())'
	@echo "set-real-env: export COMET_PYTHON=$(QE_VENV_DIR)/bin/python"

verify-cuda-kernels:
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip CUDA kernel verification (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(REAL_ENV_PY) bash scripts/verify_cuda_kernels.sh; \
	fi

# ==================================================================
# Stage5 targets
# ==================================================================
prepare-data:
	$(REAL_ENV_PY) scripts/prepare_dpo_data.py

upload-sft:
	$(REAL_ENV_PY) scripts/upload_sft_checkpoint.py \
		--runs-repo $(SFT_RUNS_REPO) \
		--run-id    $(SFT_RUN_ID) \
		--subset    $(SFT_SUBSET) \
		--dest-repo $(SFT_DEST_REPO)

train-dpo:
	$(REAL_ENV_PY) scripts/train_dpo.py --config $(CONFIG)

eval-ood:
	$(REAL_ENV_PY) scripts/eval_ood.py --config $(CONFIG) \
		--model artifacts/dpo/final --tag final
