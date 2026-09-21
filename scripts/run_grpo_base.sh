#!/usr/bin/env bash
# GRPO from official Instruct weights (no SFT). For reward baseline comparison.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)/models/Qwen2.5-0.5B-Instruct}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

CKPT_DIR="${ROOT}/checkpoints/grpo_from_base"
mkdir -p "${CKPT_DIR}"
exec env MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/run_grpo.sh" \
    trainer.experiment_name=base_gsm8k_qwen25_0_5b \
    trainer.default_local_dir="${CKPT_DIR}" \
    "$@"
