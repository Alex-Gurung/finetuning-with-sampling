"""Mean token log-probability of each SFT example's response under a model: how on-policy a training set is
(the paper's Figure 3). The prompt is chat-templated as in training; the score averages over response tokens.

    python likelihood.py --model Qwen/Qwen2.5-7B-Instruct --data groot_qwen/sft.parquet \
        --out groot_qwen/likelihood.jsonl
"""

import argparse
import json

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

CHUNK = 64  # prompt logprobs are materialized per token, so score a few examples at a time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="parquet with prompt and response columns")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    llm = LLM(model=args.model, max_model_len=8192, gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()
    rows = pd.read_parquet(args.data)
    examples = []
    for prompt, response in zip(rows["prompt"], rows["response"], strict=True):
        prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        examples.append((prompt_ids, response_ids))

    params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    with open(args.out, "w") as out:
        for start in range(0, len(examples), CHUNK):
            chunk = examples[start : start + CHUNK]
            outputs = llm.generate([TokensPrompt(prompt_token_ids=p + r) for p, r in chunk], params, use_tqdm=False)
            for (prompt_ids, response_ids), output in zip(chunk, outputs, strict=True):
                positions = range(len(prompt_ids), len(prompt_ids) + len(response_ids))
                logprobs = [output.prompt_logprobs[i][response_ids[i - len(prompt_ids)]].logprob for i in positions]
                out.write(json.dumps({"tokens": len(logprobs), "mean_logprob": sum(logprobs) / len(logprobs)}) + "\n")


if __name__ == "__main__":
    main()
