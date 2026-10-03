import os, json, time
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from tqdm import tqdm
import argparse
import re

from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


from constants import *

from grader_utils.math_parse_utils import parse_answer, parse_answer_from_tag
from grader_utils.math_grader import grade_answer

class vLLMAutoregressiveSampler:
    def __init__(self, model_name, model_type):
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


def load_ood_dataset(dataset_name):
    """Load an OOD dataset and return a list of dicts with 'question' and 'solution' keys.

    'solution' is wrapped in <answer>...</answer> tags so parse_answer_from_tag works uniformly.
    """
    base = "ood_data"
    if dataset_name == "AMC":
        with open(os.path.join(base, "AMC.json")) as f:
            data = json.load(f)
        return [{"question": d["prompt"], "solution": f"<answer>{d['answer']}</answer>"} for d in data]
    elif dataset_name == "GSM8K":
        records = []
        with open(os.path.join(base, "GSM8K.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                # GSM8K answers end with "#### <number>"
                m = re.search(r'####\s*(.+)$', d["answer"])
                ans = m.group(1).strip() if m else d["answer"].strip()
                records.append({"question": d["question"], "solution": f"<answer>{ans}</answer>"})
        return records
    elif dataset_name == "MATH-TTT":
        with open(os.path.join(base, "MATH-TTT.json")) as f:
            data = json.load(f)
        return [{"question": d["prompt"], "solution": f"<answer>{d['answer']}</answer>"} for d in data]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "ood_math/",  dest = "save_str")
    parser.add_argument("--model_str", action = "store", type = str, required = True,
                        help="Model to evaluate: a local checkpoint path or a HF hub id.")
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--num_batches", action = "store", type = int, default = 1)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument("--dataset", action = "store", type = str, required=True, choices=["AMC", "GSM8K", "MATH-TTT"],
                        help="OOD evaluation dataset from ood_data/")
    args = parser.parse_args()

    model_str = args.model_str

    save_str = args.save_str
    os.makedirs(save_str, exist_ok=True)

    model_tag = "_".join(Path(model_str).parts[-3:])
    out_path = os.path.join(save_str, f"eval_{model_tag}_{args.dataset}_{args.batch_idx}_seed{args.seed}.jsonl")

    test_set = load_ood_dataset(args.dataset)

    p = vLLMAutoregressiveSampler(model_name=model_str, model_type="chat")

    chunk_size = (len(test_set) + args.num_batches - 1) // args.num_batches  # ceiling division
    start = args.batch_idx * chunk_size
    end = min(start + chunk_size, len(test_set))

    for i in tqdm(range(start, end)):
        row = test_set[i]
        base_q = row["question"]
        solution = row["solution"]

        question = MATH_PROMPT + base_q + MATH_COT

        t0 = time.time()
        format_str = p.tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            max_tokens=1856,
            temperature=0.6,
            stop_token_ids=[p.tokenizer.eos_token_id],
            logprobs=0,
            seed=args.seed,
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


        boosted_rollout = generated_output.text

        answer = parse_answer(boosted_rollout)

        golden_answer = parse_answer_from_tag(solution)

        is_correct = safe_grade(answer, golden_answer)

        avg_log_prob = get_logprobs_vLLM(p, question, new_tokens)
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
