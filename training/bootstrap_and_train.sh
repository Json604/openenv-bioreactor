#!/bin/bash
# Bootstrap script for HF Jobs `run` (not `uv run`). Use with a
# CUDA-pre-baked image like pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
# so torch.cuda is already wired to the host nvidia driver.
#
# Submission:
#   hf jobs run \
#       --flavor h200 \
#       -s WANDB_API_KEY=<...> \
#       -s HF_TOKEN=<...> \
#       pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime \
#       bash -lc "curl -sL https://github.com/Json604/openenv-bioreactor/raw/main/training/bootstrap_and_train.sh | bash -s -- --num_samples=64 --max_steps=200 ..."
#
# All args after `bash -s --` are passed straight to run_grpo_job.py.
set -euo pipefail

echo "[bootstrap] python: $(python --version 2>&1)  torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "[bootstrap] cuda available pre-install: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Make sure git is available (the pytorch base image is Debian-based)
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -qq -y git
fi

# Install the rest of the training stack. torch is already in the image.
pip install --quiet --no-cache-dir \
    "transformers>=4.46" \
    "peft>=0.11" \
    "trl>=0.10" \
    "bitsandbytes>=0.43" \
    "accelerate>=0.30" \
    "datasets>=2.20" \
    "wandb>=0.17" \
    "huggingface_hub>=0.24" \
    "unsloth" \
    "unsloth_zoo" \
    "scipy>=1.11" \
    "pandas>=2.0" \
    "pydantic>=2.5" \
    "matplotlib>=3.8" \
    "tqdm>=4.66"

echo "[bootstrap] cuda available post-install: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd /tmp
if [ ! -d openenv-bioreactor ]; then
  git clone --depth 1 https://github.com/Json604/openenv-bioreactor.git
fi
cd openenv-bioreactor

exec python training/run_grpo_job.py "$@"
