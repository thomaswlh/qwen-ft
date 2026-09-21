# qwen-ft

Qwen2.5-0.5B-Instruct 在 GSM8K 上的两阶段练手项目：

1. **Megatron SFT**：先学会「分步推理 + `#### 答案`」
2. **GRPO**：verl + Megatron actor/ref + vLLM rollout + Ray 单机调度

默认单卡 `TP=PP=1`。TP / PP / SP / full recompute / Flash Attention / LoRA 都用环境变量按需打开。

## 环境

训练用自己的 conda env：`/home/jpzhao_team/lxhu/miniconda3/envs/qwen-ft`。完整安装步骤见 **[docs/环境安装.md](docs/环境安装.md)**。

```bash
source /home/jpzhao_team/lxhu/miniconda3/bin/activate   # 即 lxhu_conda
conda activate qwen-ft
cd /home/jpzhao_team/lxhu/workspace/qwen-ft
source scripts/env.sh
"$PYTHON" scripts/check_env.py
```

`env.sh` 会把 `PYTHON` 指到 `qwen-ft`。只有 CUDA 或 `megatron.bridge` 自检失败时才回退到 `/home/jpzhao_team/zymo/verl_env`。强制指定：`VERL_PYTHON=/path/to/python`。

当前已自检通过的版本（2026-09-22 重装）：

| 包 | 版本 |
|---|---|
| Python | 3.12.14 |
| torch | 2.10.0+cu128 |
| vllm | 0.19.1 |
| megatron-core | 0.18.2 |
| megatron-bridge | 0.5.1 |
| verl | 0.9.0 |
| ray | 2.58.0 |
| transformers | 5.8.1 |
| peft | 0.20.0 |
| datasets | 5.0.1 |
| TransferQueue | 0.1.10 |

本机驱动是 CUDA 12.9，必须用 **cu128** 的 torch，不要装 cu130。`transformers` 锁 **5.8.1**。`flash_attn` / Transformer Engine 没装，默认 `attention_backend=unfused`；`megatron-bridge` 需要打无 TE 补丁，见安装文档。

默认 `CUDA_VISIBLE_DEVICES=2`（共享机上通常更空）。改卡：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft.sh
```

## 一次跑通

```bash
# 共享卡建议先缩小 batch；空卡可以直接用脚本默认值
FULL_RECOMPUTE=1 bash scripts/run_sft.sh trainer.total_epochs=1 \
  data.train_batch_size=8 data.micro_batch_size_per_gpu=1 data.max_length=768
source scripts/env.sh
"$PYTHON" scripts/generate_sample.py --model output/latest_hf
STEPS=8 VLLM_GPU_UTIL=0.10 FULL_RECOMPUTE=1 bash scripts/run_grpo.sh \
  data.train_batch_size=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

准备数据和权重（只需一次；优先走本机 HF 缓存）：

```bash
source scripts/env.sh
"$PYTHON" scripts/download_model.py --out_dir models/Qwen2.5-0.5B-Instruct
"$PYTHON" scripts/prepare_gsm8k.py \
  --sft_out data/gsm8k_sft --grpo_out data/gsm8k_grpo
# 全量 GSM8K：FULL=1 时用 7473/1319
"$PYTHON" scripts/prepare_gsm8k.py --sft_out data/gsm8k_sft --grpo_out data/gsm8k_grpo \
  --train_samples 7473 --test_samples 1319
```

对比「不经过 SFT」的 GRPO：

```bash
STEPS=8 bash scripts/run_grpo_base.sh
```

## 按需 knobs

| 变量 | 默认 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `2` | 选卡 |
| `NGPUS` | `1` | 可见 GPU 数 |
| `TP` / `PP` | `1` / `1` | 张量 / 流水线并行 |
| `SP` | `1` | sequence parallel（TP=1 时几乎没收益） |
| `FULL_RECOMPUTE` | `0` | Megatron full activation recompute |
| `FLASH` | `0` | `attention_backend=flash`（失败请改回 0） |
| `LORA` | `0` | Megatron-Bridge LoRA，`LORA_RANK`/`LORA_ALPHA` 默认 32 |
| `STEPS` | 空 | GRPO 总步数，smoke 用 `8` |
| `VLLM_GPU_UTIL` | `0.15` | 共享卡给 vLLM 的显存比例 |

多卡练习建议先 `NGPUS=2 TP=2`，不要一上来开 PP。

参数点分路径见 [docs/参数说明.md](docs/参数说明.md)。
