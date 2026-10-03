import os, json, time
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import random
from tqdm import tqdm
import argparse

import pandas as pd
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


from constants import *

from grader_utils.math_parse_utils import parse_answer, parse_answer_from_tag
from grader_utils.math_grader import grade_answer

def format_prompt(text, model_type, tokenizer):
    if model_type == "base":
        format_str = text
    elif model_type == "chat":
        answer_context = [{"role": "user", "content": text}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
    return format_str

class vLLMAutoregressiveSampler:
    def __init__(self, model_name, model_type):
        # vLLM manages the model and device internally.
        # gpu_memory_utilization handles the VRAM allocation (0.9 is standard).
        self.llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_type = model_type

def get_logprobs_vLLM(p: vLLMAutoregressiveSampler, question, current):
    context_str = format_prompt(question, p.model_type, p.tokenizer)
    context_tokens = p.tokenizer.encode(context_str)

    full_sequence_ids = context_tokens + current
    c = len(context_tokens)

    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)

    output = p.llm.generate(
        TokensPrompt(prompt_token_ids=full_sequence_ids),
        sampling_params=sampling_params,
        use_tqdm=False
    )

    prompt_logprobs = output[0].prompt_logprobs

    relevant_logprobs = prompt_logprobs[c:]

    log_likelihood = []
    for i, logprob_dict in enumerate(relevant_logprobs):
        if logprob_dict:
            # Get the logprob of the actual token used in 'current'
            token_id = current[i]
            log_likelihood.append(logprob_dict[token_id].logprob)

    if not log_likelihood:
        return float("-inf"), 0.0, 0.0

    return sum(log_likelihood)/len(log_likelihood), 0.0, 0.0



def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())




MATH_PROMPT = "You are an expert mathematician. Can you solve the following math problem? "

MATH_COT = "\n\nExplain your reasoning, thinking step by step, and return your final answer within \\boxed{{}}."



def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "math_eval/",  dest = "save_str")
    parser.add_argument("--model_path", action = "store", type = str, required = True, help = "HF model id or local checkpoint directory")
    parser.add_argument("--model_type", action = "store", type = str, default = "chat", choices = ["base", "chat"])
    parser.add_argument("--out_name", action = "store", type = str, default = None, help = "output file name (default: derived from --model_path)")
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--num_batches", action = "store", type = int, default = 1)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument("--max_tokens", action = "store", type = int, default = 1856, help = "generation cap; the paper uses 1856")
    args = parser.parse_args()

    model_str = args.model_path
    out_name = args.out_name or os.path.basename(model_str.rstrip("/"))

    save_str = args.save_str
    os.makedirs(save_str, exist_ok=True)
    out_path = os.path.join(save_str, f"eval_{out_name}_{args.batch_idx}.jsonl")

    test_df = pd.read_parquet("math_data/test.parquet")


    p = vLLMAutoregressiveSampler(model_name=model_str, model_type=args.model_type)

    num_batches = args.num_batches
    all_idxs = [i for i in range(len(test_df))]
    chunk_size = (len(all_idxs) + num_batches - 1) // num_batches  # ceiling division
    start = args.batch_idx * chunk_size
    end = min(start + chunk_size, len(all_idxs))
    batch_idxs = all_idxs[start:end]

    for i in tqdm(batch_idxs):
        s = test_df["question"][i]
        start = s.index("\nUser: ") + len("\nUser: ")
        end = s.rfind("Show your work")
        base_q = s[start:end]


        question = MATH_PROMPT + base_q + MATH_COT
        solution = test_df["answer"][i]


        t0 = time.time()
        format_str = format_prompt(question, p.model_type, p.tokenizer)

        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.6,
            stop_token_ids=[p.tokenizer.eos_token_id],
            logprobs=0 # We need logprobs of generated tokens
        )

        output = p.llm.generate(
            format_str,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        dt = time.time() - t0

        generated_output = output[0].outputs[0]
        new_tokens = list(generated_output.token_ids)

        log_probs_norm = []
        for m, tok_id in enumerate(new_tokens):
            lp_dict = generated_output.logprobs[m]
            lp_obj = lp_dict.get(tok_id, None)
            if lp_obj is None:
                raise RuntimeError(f"Missing generated token_id={tok_id} in logprobs at step i={m}.")
            log_probs_norm.append(float(lp_obj.logprob))


        boosted_rollout = output[0].outputs[0].text

        answer = parse_answer(boosted_rollout)

        golden_answer = parse_answer_from_tag(solution)

        is_correct = safe_grade(answer, golden_answer)


        avg_log_prob = get_logprobs_vLLM(p, question, list(output[0].outputs[0].token_ids))
        rec = {
            "idx": i,
            "prompt": question,
            "solution": solution,
            "gen": boosted_rollout,
            "target_log_prob_cur": float(avg_log_prob[0]),
            "token_logprobs": [float(x) for x in log_probs_norm],
            "runtime_sec": dt,
            "answer": answer,
            "is_correct": is_correct,
        }

        append_jsonl(out_path, rec)
