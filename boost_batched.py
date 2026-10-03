"""The paper's projection sampling (proj_samp_vLLM in boost_sci.py and boost_math.py), run on many training problems
at once. Each problem follows the original algorithm step for step: the same proposal prompt, sampling parameters,
block size, MCMC steps and acceptance rule. The originals send one model call at a time; here every problem's calls
go to one vLLM engine concurrently, which batches them. Writes the same records as the originals (without the
trace files, which SFT does not use).

    python boost_batched.py --task chem --model Qwen/Qwen2.5-7B-Instruct --idx_file units/u0000.txt \
        --out boost/k10/qwen
"""

import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

from constants import SEQ_PROMPT_MATH, SEQ_PROMPT_SCI
from eval_math import safe_grade
from grader_utils.math_parse_utils import parse_answer
from grader_utils.sci_grader import grade_answer, parse_answer_gpqa, same_balanced_equation_olmo
from groot import chem_question, chem_rows, math_question, math_rows

MAX_NEW_TOKENS = 1856


def chem_grade(row, response):
    """The answer and its correctness, as boost_sci.py records them."""
    if row["type"].startswith("mcq"):
        answer = parse_answer_gpqa(response)
        return answer, int(grade_answer(answer, row["answerKey"]))
    out = same_balanced_equation_olmo(response, row["answer"])
    return out["equation_1"], out["same"]


def math_grade(row, response):
    """The answer and its correctness, as boost_math.py records them."""
    answer = parse_answer(response)
    return answer, safe_grade(answer, parse_answer(row["solution"]))


# task -> (training rows, question, expert solution, proposal prompt, repetition penalty, blocks, answer and grade)
TASKS = {
    "chem": (chem_rows, chem_question, lambda row: row["chosen"], SEQ_PROMPT_SCI, 1.05, 58, chem_grade),
    "math": (math_rows, math_question, lambda row: row["solution"], SEQ_PROMPT_MATH, 1.1, 32, math_grade),
}


class Sampler:
    def __init__(self, engine, tokenizer, prompt, repetition_penalty):
        self.engine, self.tokenizer = engine, tokenizer
        self.prompt, self.repetition_penalty = prompt, repetition_penalty

    def chat(self, text):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

    async def generate(self, ids, params):
        async for output in self.engine.generate(TokensPrompt(prompt_token_ids=ids), params, str(uuid.uuid4())):
            final = output
        return final

    async def propose(self, question, solution, current, ext_len):
        """proposal_vLLM: continue `current` from the expert-conditioned prompt."""
        response = self.tokenizer.decode(current, skip_special_tokens=True)
        context = self.tokenizer.encode(
            self.chat(self.prompt.format(PROBLEM=question, SOLUTION=solution, RESPONSE=response))
        )
        params = SamplingParams(
            max_tokens=ext_len,
            temperature=0.6,
            stop_token_ids=[self.tokenizer.eos_token_id],
            repetition_penalty=self.repetition_penalty,
            logprobs=0,
        )
        out = (await self.generate(context + current, params)).outputs[0]
        new = list(out.token_ids)
        return current + new, [out.logprobs[i][token].logprob for i, token in enumerate(new)]

    async def score(self, question, current):
        """get_logprobs_vLLM: mean log-probability of `current` under the plain question."""
        context = self.tokenizer.encode(self.chat(question))
        out = await self.generate(context + current, SamplingParams(max_tokens=1, prompt_logprobs=0))
        logprobs = [d[current[i]].logprob for i, d in enumerate(out.prompt_logprobs[len(context) :]) if d]
        return sum(logprobs) / len(logprobs) if logprobs else float("-inf")

    async def project(self, question, solution, mcmc_steps, block_num, rng):
        """proj_samp_vLLM, step for step."""
        gen, log_probs_norm = [], []
        jump_size = MAX_NEW_TOKENS // block_num
        for _ in range(block_num):
            gen, lp_norm = await self.propose(question, solution, gen, jump_size)
            log_probs_norm.extend(lp_norm)
            target_cur = await self.score(question, gen)
            for _ in range(mcmc_steps):
                t = len(gen)
                if t == 0:
                    continue
                idx = rng.randint(0, t - 1)
                prop, log_prob_prop = await self.propose(question, solution, gen[:idx], t - idx)
                target_prop = await self.score(question, prop)
                if target_prop > target_cur:
                    gen = prop
                    s = len(prop)
                    if len(log_probs_norm) < s:
                        log_probs_norm.extend([0.0] * (s - len(log_probs_norm)))
                    elif len(log_probs_norm) > s:
                        log_probs_norm = log_probs_norm[:s]
                    log_probs_norm[idx:] = log_prob_prop
                    target_cur = target_prop
            if self.tokenizer.eos_token_id in gen:
                end = gen.index(self.tokenizer.eos_token_id) + 1
                return gen[:end], target_cur, log_probs_norm[:end]
        return gen, target_cur, log_probs_norm


async def run(args):
    load, question_of, solution_of, prompt, repetition_penalty, block_num, grade = TASKS[args.task]
    rows = load()
    todo = [int(line) for line in Path(args.idx_file).read_text().split()]
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"boosted_{Path(args.idx_file).stem}.jsonl"
    if out_path.exists():
        done = {json.loads(line)["idx"] for line in out_path.read_text().splitlines()}
        todo = [i for i in todo if i not in done]
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.9, max_model_len=16384)
    )
    if args.repetition_penalty is not None:
        repetition_penalty = args.repetition_penalty
    sampler = Sampler(engine, AutoTokenizer.from_pretrained(args.model), prompt, repetition_penalty)
    limit = asyncio.Semaphore(args.concurrency)

    async def one(i):
        async with limit:
            question, solution = question_of(rows[i]), solution_of(rows[i])
            start = time.time()
            gen, target, lps = await sampler.project(question, solution, args.mcmc_steps, block_num, random.Random(i + 1_000_003 * args.seed))
            response = sampler.tokenizer.decode(gen, skip_special_tokens=True)
            answer, correct = grade(rows[i], response)
            return {
                "idx": i,
                "prompt": question,
                "solution": solution,
                "gen": response,
                "target_log_prob_cur": target,
                "token_logprobs": lps,
                "runtime_sec": time.time() - start,
                "answer": answer,
                "is_correct": correct,
                "batched": True,
            }

    with out_path.open("a") as out:
        for finished in asyncio.as_completed([one(i) for i in todo]):
            out.write(json.dumps(await finished) + "\n")
            out.flush()
    engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--idx_file", required=True, help="training indices to sample, one per line")
    parser.add_argument("--out", required=True, type=Path, help="directory for boosted_<idx file name>.jsonl")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="override the task's proposal repetition penalty (the paper's: 1.05 chem, 1.1 math)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the MCMC cut points")
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=128, help="problems sampled at once")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
