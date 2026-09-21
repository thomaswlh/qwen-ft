#!/usr/bin/env bash
# Shared environment for qwen-ft scripts.
# Usage: source scripts/env.sh   (from any cwd)
#
# Activate: lxhu_conda (= this miniconda) + qwen-ft (torch 2.10.0+cu128).
# If CUDA init or megatron.bridge import fails, fall back to zymo/verl_env.

if [[ -n "${_QWEN_FT_ENV_LOADED:-}" ]]; then
    return 0 2>/dev/null || true
fi
export _QWEN_FT_ENV_LOADED=1

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="$(cd "${_ENV_DIR}/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-/home/jpzhao_team/lxhu/miniconda3}"
QWEN_FT_ENV="${QWEN_FT_ENV:-${CONDA_BASE}/envs/qwen-ft}"
ZYMO_PY="${ZYMO_PY:-/home/jpzhao_team/zymo/verl_env/bin/python}"

# lxhu_conda + qwen-ft
# shellcheck disable=SC1091
source "${CONDA_BASE}/bin/activate" "${QWEN_FT_ENV}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# GPU 2 currently has the most free memory on this shared box.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VERL_USE_UV="${VERL_USE_UV:-0}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/home/jpzhao_team/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Parallel / feature knobs. Defaults = single-GPU conservative smoke.
export NGPUS="${NGPUS:-1}"
export TP="${TP:-1}"
export PP="${PP:-1}"
export SP="${SP:-1}"
export FULL_RECOMPUTE="${FULL_RECOMPUTE:-0}"
export FLASH="${FLASH:-0}"
export LORA="${LORA:-0}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export STEPS="${STEPS:-}"
export FULL="${FULL:-0}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.15}"

if [[ "${FLASH}" == "1" ]]; then
    export ATTN_BACKEND=flash
else
    export ATTN_BACKEND=unfused
fi
if [[ "${SP}" == "1" ]]; then
    export SP_BOOL=True
else
    export SP_BOOL=False
fi

_pick_python() {
    if [[ -n "${VERL_PYTHON:-}" ]]; then
        echo "${VERL_PYTHON}"
        return
    fi
    local qwen_py="${QWEN_FT_ENV}/bin/python"
    if "${qwen_py}" -c "import torch; assert torch.cuda.is_available(); t=torch.zeros(1, device='cuda'); del t" >/dev/null 2>&1 \
        && "${qwen_py}" -c "import megatron.bridge" >/dev/null 2>&1; then
        echo "${qwen_py}"
        return
    fi
    echo "[env] qwen-ft cannot init CUDA or import megatron.bridge; falling back to ${ZYMO_PY}" >&2
    echo "${ZYMO_PY}"
}

export PYTHON="$(_pick_python)"
export CONFIG_DIR="${ROOT}/config"

echo "[env] ROOT=${ROOT}"
echo "[env] PYTHON=${PYTHON}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NGPUS=${NGPUS} TP=${TP} PP=${PP} SP=${SP} FLASH=${FLASH} LORA=${LORA} RECOMPUTE=${FULL_RECOMPUTE}"
