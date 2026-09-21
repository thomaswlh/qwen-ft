"""Generate one GSM8K-style answer and check for `#### <number>`."""
from __future__ import annotations

import argparse
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_QUESTION = (
    "Janet has 12 apples. She gives 1/3 of them to her brother and then eats 2 herself. "
    "How many apples does Janet have left?"
)
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    messages = [{"role": "user", "content": args.question + " " + INSTRUCTION}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to("cuda:0")
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    print("=== model answer ===")
    print(text)
    if re.search(r"####\s*-?[0-9\.,]+", text):
        print("OK: found #### answer")
        return 0
    print("WARN: no #### <number> in the answer", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
