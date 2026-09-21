#!/usr/bin/env bash
# GRPO from SFT weights: Megatron actor/ref + vLLM rollout.
# Usage: STEPS=8 bash scripts/run_grpo.sh [hydra key=value ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
set -x

MODEL_PATH="${MODEL_PATH:-${ROOT}/output/latest_hf}"
TRAIN_FILES="[${ROOT}/data/gsm8k_grpo/train.parquet]"
VAL_FILES="[${ROOT}/data/gsm8k_grpo/test.parquet]"
CKPT_DIR="${ROOT}/checkpoints/grpo_from_sft"
LOG_DIR="${ROOT}/logs/grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Missing SFT weights at ${MODEL_PATH}. Run scripts/run_sft.sh first."
    exit 1
fi

EXTRA=()
if [[ -n "${STEPS}" ]]; then
    EXTRA+=("trainer.total_training_steps=${STEPS}")
fi
if [[ "${FULL_RECOMPUTE}" == "1" ]]; then
    EXTRA+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    )
fi
if [[ "${LORA}" == "1" ]]; then
    EXTRA+=(
        actor_rollout_ref.model.lora.rank="${LORA_RANK}"
        actor_rollout_ref.model.lora.alpha="${LORA_ALPHA}"
    )
fi

set +e
# shellcheck disable=SC2086
"${PYTHON}" -m verl.trainer.main_ppo \
    --config-path="${CONFIG_DIR}" \
    --config-name=ppo_megatron_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.return_raw_chat=True \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${PP}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP}" \
    actor_rollout_ref.actor.megatron.sequence_parallel="${SP_BOOL}" \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend="${ATTN_BACKEND}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_UTIL}" \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${PP}" \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TP}" \
    actor_rollout_ref.ref.megatron.sequence_parallel="${SP_BOOL}" \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend="${ATTN_BACKEND}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.masked_softmax_fusion=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=verl_grpo_qwen25_0_5b \
    trainer.experiment_name=warmstart_gsm8k_qwen25_0_5b \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    model_engine=megatron \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    "${EXTRA[@]}" \
    "$@" \
    hydra.run.dir="${LOG_DIR}" \
    2>&1 | tee "${LOG_DIR}/train.log"

status=${PIPESTATUS[0]}
set -e
if [[ "${status}" -ne 0 ]]; then
    echo "GRPO failed with exit ${status}, log: ${LOG_DIR}/train.log"
    exit "${status}"
fi
echo "GRPO done, log: ${LOG_DIR}/train.log"
