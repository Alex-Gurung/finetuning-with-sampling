"""pass@k on the chemistry test set (the paper's Figure 4). Draws n samples per problem with the eval's settings
(chat template, temperature 0.6, 1,856 tokens, stop at EOS) and grades them with the eval's graders. Writes the
number of correct samples per problem, from which pass@k follows by the unbiased estimator
1 - C(n - c, k) / C(n, k).

    python passk.py --model allenai/Olmo-3-7B-Instruct --n 64 --shard 0 --num_shards 8 --out passk/olmo_0.jsonl
"""

import argparse
import json

from vllm import LLM, SamplingParams

from groot import chem_correct, chem_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open("sci_data/test_data.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    size = (len(rows) + args.num_shards - 1) // args.num_shards
    indices = range(args.shard * size, min((args.shard + 1) * size, len(rows)))

    llm = LLM(model=args.model, gpu_memory_utilization=0.9, max_model_len=8192)
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": chem_question(rows[i])}], tokenize=False, add_generation_prompt=True
        )
        for i in indices
    ]
    params = SamplingParams(n=args.n, temperature=0.6, max_tokens=1856, stop_token_ids=[tokenizer.eos_token_id], seed=0)
    outputs = llm.generate(prompts, params)
    with open(args.out, "w") as out:
        for i, output in zip(indices, outputs, strict=True):
            correct = sum(chem_correct(rows[i], sample.text) for sample in output.outputs)
            out.write(json.dumps({"idx": i, "n": args.n, "correct": correct}) + "\n")


if __name__ == "__main__":
    main()
