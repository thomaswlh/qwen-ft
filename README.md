# qwen-ft

Qwen2.5-0.5B-Instruct 在 GSM8K 上的两阶段练手项目：

1. **Megatron SFT**：先学会「分步推理 + `#### 答案`」
2. **GRPO**：verl + Megatron actor/ref + vLLM rollout + Ray 单机调度

默认单卡 `TP=PP=1`。TP / PP / SP / full recompute / Flash Attention / LoRA 都用环境变量按需打开。

## 环境

完整安装、版本表和自检见 **[docs/环境安装.md](docs/环境安装.md)**。

```bash
# 共享账号先激活本目录 conda，再进 qwen-ft
source /home/jpzhao_team/lxhu/miniconda3/bin/activate   # 即 lxhu_conda
conda activate qwen-ft
```

脚本会自己 `source scripts/env.sh`。本机驱动是 CUDA 12.9，而 `qwen-ft` 里的 torch 是 `2.11.0+cu130`，无法初始化 GPU；`megatron.bridge` 也因 `megatron-core==0.12.3` 缺 FSDP 模块导不进。`env.sh` 检测到后会回退到已跑通的 `/home/jpzhao_team/zymo/verl_env`。可用 `VERL_PYTHON=...` 强制指定解释器。

复现训练环境请对齐那套：**torch 2.10.0+cu128 + vllm 0.19.1 + megatron-core 0.18.2 + megatron-bridge 0.5.1 + verl 0.9.0**，不要装 cu130。

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
