#!/usr/bin/env bash
# Megatron SFT on GSM8K for Qwen2.5-0.5B-Instruct.
# Usage: bash scripts/run_sft.sh [hydra key=value ...]
# Env knobs: CUDA_VISIBLE_DEVICES NGPUS TP PP SP FLASH FULL_RECOMPUTE LORA
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
set -x

MODEL_PATH="${MODEL_PATH:-${ROOT}/models/Qwen2.5-0.5B-Instruct}"
TRAIN_FILES="[${ROOT}/data/gsm8k_sft/train.parquet]"
VAL_FILES="[${ROOT}/data/gsm8k_sft/test.parquet]"
CKPT_DIR="${ROOT}/checkpoints/gsm8k_sft"
LOG_DIR="${ROOT}/logs/sft_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ROOT}/output"

EXTRA=()
if [[ "${FULL_RECOMPUTE}" == "1" ]]; then
    EXTRA+=(
        +engine.override_transformer_config.recompute_method=uniform
        +engine.override_transformer_config.recompute_granularity=full
        +engine.override_transformer_config.recompute_num_layers=1
    )
fi
if [[ "${LORA}" == "1" ]]; then
    EXTRA+=(
        model.lora.rank="${LORA_RANK}"
        model.lora.alpha="${LORA_ALPHA}"
    )
fi

set +e
# shellcheck disable=SC2086
"${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NGPUS}" \
    --master_port="${MASTER_PORT:-29571}" \
    -m verl.trainer.sft_trainer \
    --config-path="${CONFIG_DIR}" \
    --config-name=sft_trainer_engine \
    engine=megatron \
    optim=megatron \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=False \
    engine.tensor_model_parallel_size="${TP}" \
    engine.pipeline_model_parallel_size="${PP}" \
    engine.sequence_parallel="${SP_BOOL}" \
    engine.dtype=bfloat16 \
    engine.use_remove_padding=False \
    engine.override_transformer_config.attention_backend="${ATTN_BACKEND}" \
    ++engine.override_transformer_config.masked_softmax_fusion=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    data.use_dynamic_bsz=False \
    data.max_length=1024 \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.ignore_input_ids_mismatch=True \
    data.messages_key=messages \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=False \
    model.use_fused_kernels=False \
    model.enable_gradient_checkpointing=True \
    ++model.override_config.attn_implementation=sdpa \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.01 \
    trainer.project_name=verl_sft_qwen25_0_5b \
    trainer.experiment_name=gsm8k_sft \
    trainer.total_epochs=3 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    trainer.seed=42 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    checkpoint.save_contents='["model", "optimizer", "extra", "hf_model"]' \
    "${EXTRA[@]}" \
    "$@" \
    hydra.run.dir="${LOG_DIR}" \
    2>&1 | tee "${LOG_DIR}/train.log"

status=${PIPESTATUS[0]}
set -e
if [[ "${status}" -ne 0 ]]; then
    echo "SFT failed with exit ${status}, log: ${LOG_DIR}/train.log"
    exit "${status}"
fi

LAST="$(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | sort -V | tail -1 || true)"
HF_DIR=""
if [[ -n "${LAST}" && -d "${LAST}/huggingface" ]]; then
    HF_DIR="${LAST}/huggingface"
elif [[ -n "${LAST}" && -d "${LAST}/model/huggingface" ]]; then
    HF_DIR="${LAST}/model/huggingface"
fi
if [[ -n "${HF_DIR}" ]]; then
    ln -sfn "${HF_DIR}" "${ROOT}/output/latest_hf"
    echo "SFT done, HF weights: ${ROOT}/output/latest_hf -> ${HF_DIR}"
else
    echo "SFT finished but no huggingface/ dir under ${CKPT_DIR}"
    exit 1
fi
