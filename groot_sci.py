"""Samples chemistry training solutions with Groot or IID from a vLLM server, grades them, and writes
the correct ones as SFT data.

Groot asks the model for a decision tree of approaches and n paths through it, then solves the
problem once per path with the path as a hidden hint. IID solves the problem n times. Samples are
graded with the repo's graders; correct, finished samples that do not mention their hint become
the SFT set (prompt and response columns, as train_sft.py and the verl SFT trainer expect).

    vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 32768 --data-parallel-size 8
    python groot_sci.py --model Qwen/Qwen2.5-7B-Instruct --method groot --out groot_qwen
"""

import argparse
import asyncio
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from eval_sci import (
    BALANCE_COT,
    BALANCE_PROMPT,
    MCQ_COT,
    MCQ_PROMPT,
    extract_unbalanced_from_question,
)
from grader_utils.sci_grader import grade_answer, parse_answer_gpqa, same_balanced_equation_olmo

PROMPT_DIR = Path(__file__).parent / "groot_prompts"
NUMBER_WORDS = "zero one two three four five six seven eight nine ten".split()
PLACEHOLDER = re.compile(r"\{(PROBLEM|APPROACH|N|N_WORD|N_MINUS_1_WORD)\}")
APPROACH_TAG = re.compile(r"<approach>(.*?)</approach>", re.DOTALL | re.IGNORECASE)
PLANNER_TEMPERATURE = 0.45
SOLVER_TEMPERATURE = 0.85
PLANNER_MAX_TOKENS = 4096
SOLVER_MAX_TOKENS = 1856  # the paper's maximum response length

# A solution containing any of these phrases reveals that it was given an approach.
LEAK_TERMS = [
    "private note",
    "the suggestion",
    "the memo",
    "prior attempt",
    "hidden instruction",
    "previous attempt",
    "the anchor",
    "sampled attempt",
    "as instructed",
    "the guidance",
    "approaches already tried",
    "the strategy note",
    "given approach",
    "the hint",
    "hint structure",
    "the provided approach",
]


def question(row: dict) -> str:
    """The question exactly as eval_sci.py and boost_sci.py build it."""
    if row["type"].startswith("mcq"):
        choices = zip(row["choices"]["label"], row["choices"]["text"], strict=True)
        options = "\n".join(f"{label}) {text}" for label, text in choices)
        return MCQ_PROMPT + row["question"] + "\n\nOptions:\n" + options + MCQ_COT
    return BALANCE_PROMPT + extract_unbalanced_from_question(row["question"]) + BALANCE_COT


def is_correct(row: dict, response: str) -> bool:
    if row["type"].startswith("mcq"):
        return bool(grade_answer(parse_answer_gpqa(response), row["answerKey"]))
    return bool(same_balanced_equation_olmo(response, row["answer"])["same"])


def render(template: str, **fields: str) -> str:
    text = (PROMPT_DIR / f"{template}.txt").read_text()
    return PLACEHOLDER.sub(lambda match: fields[match.group(1)], text)


def chat(args: argparse.Namespace, message: str, temperature: float, max_tokens: int) -> tuple[str, str]:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": message}],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{args.url}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        choice = json.load(response)["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]


async def sample_problem(
    args: argparse.Namespace, limit: asyncio.Semaphore, index: int, row: dict
) -> list[dict]:
    async def ask(message: str, temperature: float, max_tokens: int) -> tuple[str, str]:
        async with limit:
            return await asyncio.to_thread(chat, args, message, temperature, max_tokens)

    prompt = question(row)
    plan, approaches = None, [None] * args.n
    if args.method == "groot":
        planner = render(
            "planner",
            PROBLEM=prompt,
            N=str(args.n),
            N_WORD=NUMBER_WORDS[args.n],
            N_MINUS_1_WORD=NUMBER_WORDS[args.n - 1],
        )
        plan, _ = await ask(planner, PLANNER_TEMPERATURE, PLANNER_MAX_TOKENS)
        approaches = [block.strip() for block in APPROACH_TAG.findall(plan)][: args.n]
    solutions = await asyncio.gather(
        *(
            ask(
                prompt if a is None else render("solver", PROBLEM=prompt, APPROACH=a),
                SOLVER_TEMPERATURE,
                SOLVER_MAX_TOKENS,
            )
            for a in approaches
        )
    )
    return [
        {
            "idx": index,
            "method": args.method,
            "task": row["details"]["task"],
            "approach": approach,
            "plan": plan,
            "prompt": prompt,
            "response": response,
            "finish_reason": finish_reason,
            "correct": is_correct(row, response),
            "leaked": approach is not None and any(t in response.lower() for t in LEAK_TERMS),
        }
        for approach, (response, finish_reason) in zip(approaches, solutions, strict=True)
    ]


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    args.out.mkdir(parents=True, exist_ok=True)
    samples_path = args.out / "samples.jsonl"
    done = set()
    if samples_path.exists():
        with samples_path.open() as lines:
            done = {json.loads(line)["idx"] for line in lines}
    todo = [(i, row) for i, row in enumerate(rows) if i not in done]
    print(f"{len(todo)} problems to sample, {len(done)} already in {samples_path}")

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(args.workers))
    limit = asyncio.Semaphore(args.workers)
    tasks = [asyncio.create_task(sample_problem(args, limit, i, row)) for i, row in todo]
    with samples_path.open("a") as out:
        for finished in asyncio.as_completed(tasks):
            out.write("".join(json.dumps(sample) + "\n" for sample in await finished))
            out.flush()

    with samples_path.open() as lines:
        samples = [json.loads(line) for line in lines]
    keep = [s for s in samples if s["correct"] and not s["leaked"] and s["finish_reason"] == "stop"]
    pd.DataFrame({"prompt": [s["prompt"] for s in keep], "response": [s["response"] for s in keep]}).to_parquet(
        args.out / "sft.parquet"
    )
    solved = len({s["idx"] for s in keep})
    print(
        f"{len(samples)} samples, {sum(s['correct'] for s in samples)} correct, "
        f"{sum(s['leaked'] for s in samples)} leaked; {len(keep)} kept for SFT covering "
        f"{solved}/{len(rows)} problems -> {args.out / 'sft.parquet'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="the model the vLLM server serves")
    parser.add_argument("--method", choices=["groot", "iid"], default="groot")
    parser.add_argument("--n", type=int, default=4, help="samples per problem")
    parser.add_argument("--data", default="sci_data/train_data.jsonl")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default="http://localhost:8000/v1", help="the vLLM server")
    parser.add_argument("--workers", type=int, default=1024, help="requests in flight")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
