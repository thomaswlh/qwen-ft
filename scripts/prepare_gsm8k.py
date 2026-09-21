"""Write GSM8K parquet for both SFT (messages) and GRPO (prompt + reward).

User instruction is identical across stages so GRPO starts from the SFT
input distribution. Assistant text keeps `#### <number>` for the built-in
verl GSM8K rule reward.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import datasets


INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_solution(solution_str: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert match is not None, f"no #### answer in: {solution_str[:100]}"
    return match.group(1).replace(",", "")


def clean_answer(question: str, answer: str) -> str:
    question = question.strip()
    answer = answer.strip()
    if answer.startswith(question):
        answer = answer[len(question) :].strip()
    answer = re.sub(r"<<[^>]*>>", "", answer)
    answer = re.sub(r" {2,}", " ", answer)
    return answer


LOCAL_GSM8K = Path(
    "/home/jpzhao_team/.cache/huggingface/hub/datasets--openai--gsm8k/"
    "snapshots/740312add88f781978c0658806c59bc2815b9866/main"
)


def load_gsm8k(dataset: str):
    train_pq = LOCAL_GSM8K / "train-00000-of-00001.parquet"
    test_pq = LOCAL_GSM8K / "test-00000-of-00001.parquet"
    if train_pq.exists() and test_pq.exists():
        print(f"loading GSM8K from {LOCAL_GSM8K}")
        return {
            "train": datasets.Dataset.from_parquet(str(train_pq)),
            "test": datasets.Dataset.from_parquet(str(test_pq)),
        }
    try:
        return datasets.load_dataset(dataset, "main")
    except Exception as exc:
        print(f"offline/default load failed ({exc}); retrying with download")
        return datasets.load_dataset(dataset, "main")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_out", required=True)
    ap.add_argument("--grpo_out", required=True)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--train_samples", type=int, default=512)
    ap.add_argument("--test_samples", type=int, default=128)
    args = ap.parse_args()

    sft_out = Path(args.sft_out)
    grpo_out = Path(args.grpo_out)
    sft_out.mkdir(parents=True, exist_ok=True)
    grpo_out.mkdir(parents=True, exist_ok=True)

    ds = load_gsm8k(args.dataset)

    def process(split: str, n: int):
        sft_rows = []
        grpo_rows = []
        for idx, ex in enumerate(ds[split]):
            if idx >= n:
                break
            question = ex["question"]
            answer = clean_answer(question, ex["answer"])
            assert "####" in answer, f"no #### in answer: {answer[:200]}"
            user = question + " " + INSTRUCTION
            sft_rows.append(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
            grpo_rows.append(
                {
                    "data_source": args.dataset,
                    "prompt": [{"role": "user", "content": user}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": extract_solution(ex["answer"])},
                    "extra_info": {"split": split, "index": idx},
                }
            )
        return datasets.Dataset.from_list(sft_rows), datasets.Dataset.from_list(grpo_rows)

    sft_train, grpo_train = process("train", args.train_samples)
    sft_test, grpo_test = process("test", args.test_samples)
    sft_train.to_parquet(str(sft_out / "train.parquet"))
    sft_test.to_parquet(str(sft_out / "test.parquet"))
    grpo_train.to_parquet(str(grpo_out / "train.parquet"))
    grpo_test.to_parquet(str(grpo_out / "test.parquet"))
    print(f"OK SFT  train={args.train_samples} test={args.test_samples} -> {sft_out}")
    print(f"OK GRPO train={args.train_samples} test={args.test_samples} -> {grpo_out}")


if __name__ == "__main__":
    main()
