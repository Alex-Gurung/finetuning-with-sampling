import os, json, time
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from tqdm import tqdm
import argparse

from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


from constants import *

from grader_utils.sci_grader import parse_answer_gpqa, grade_answer, same_balanced_equation_olmo


class vLLMAutoregressiveSampler:
    def __init__(self, model_name, model_type):
        # vLLM manages the model and device internally.
        # gpu_memory_utilization handles the VRAM allocation (0.9 is standard).
        self.llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_type = model_type

def get_logprobs_vLLM(p: vLLMAutoregressiveSampler, question, current):
    """
    Calculates the average log-likelihood of the 'current' sequence given the 'question'.
    """
    # 1. Format the context prompt
    context_str = format_prompt(question, p.model_type, p.tokenizer)
    context_tokens = p.tokenizer.encode(context_str)
    
    # 2. Combine context + current sequence
    full_sequence_ids = context_tokens + current
    c = len(context_tokens)

    # 3. Use vLLM to score the prompt
    # max_tokens=1 because we don't want to generate new text, just score existing.
    # prompt_logprobs=0 returns the logprob of the actual token at each step.
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    
    output = p.llm.generate(
        TokensPrompt(prompt_token_ids=full_sequence_ids),
        sampling_params=sampling_params,
        use_tqdm=False
    )
    
    prompt_logprobs = output[0].prompt_logprobs

    # Logprobs for the 'current' part start at index c.
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



MCQ_PROMPT = "You are an expert chemist. Given a question and four options, think step by step, and then return the correct letter: A, B, C, or D. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}').\n"
MCQ_COT = "\n\nExplain your reasoning, thinking step by step, and return your final answer within \\boxed{{}}."

BALANCE_PROMPT = "You are an expert chemist. Given a chemical equation, please balance the equation and maintain the order of reactants and products as given. Here is a unbalanced chemical equation:\n"
BALANCE_COT = "\nExplain your reasoning, thinking step by step, and return the balanced equation on one line. Keep reactants and products in the SAME ORDER as provided."



def extract_unbalanced_from_question(q: str) -> Optional[str]:
    for line in q.splitlines():
        if ("=" in line) or ("->" in line):
            return line.strip()
    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "sci_results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", type = str, required = True,
                        help="Model to evaluate: a local checkpoint path or a HF hub id.")
    parser.add_argument("--run_name", action = "store", default = None, type = str,
                        help="Subdirectory name under save_str for results. Defaults to a name derived from the model path.")
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--num_batches", action = "store", type = int, default = 1)
    parser.add_argument("--seed", action = "store", type = int, default = 1)
    args = parser.parse_args()

    model_str = args.model

    if args.run_name is not None:
        run_name = args.run_name
    else:
        # Derive a readable run name from the last few path components,
        # e.g. .../ep2_boosted_sci_olmo_2/checkpoints/global_step_210
        #   -> ep2_boosted_sci_olmo_2_checkpoints_global_step_210
        parts = [x for x in model_str.strip("/").split("/") if x]
        run_name = "_".join(parts[-3:]) if len(parts) > 3 else "_".join(parts)

    save_str = os.path.join(args.save_str, run_name)
    os.makedirs(save_str, exist_ok=True)
    out_path = os.path.join(save_str, f"eval_{args.batch_idx}_seed{args.seed}.jsonl")
    print(f"Model: {model_str}")
    print(f"Writing results to: {out_path}")


    with open("sci_data/test_data.jsonl", "r", encoding="utf-8") as f:
        test_set = [json.loads(line) for line in f if line.strip()]

    p = vLLMAutoregressiveSampler(model_name=model_str, model_type="chat")

    chunk_size = (len(test_set) + args.num_batches - 1) // args.num_batches  # ceiling division
    start = args.batch_idx * chunk_size
    end = min(start + chunk_size, len(test_set))


    for i in tqdm(range(start, end)):
        type_q = test_set[i]["type"]

        if type_q.startswith("mcq"):
            opt_lines = []
            for lbl, txt in zip(test_set[i]["choices"].get("label", []), test_set[i]["choices"].get("text", [])):
                opt_lines.append(f"{lbl}) {txt}")

            question = MCQ_PROMPT + test_set[i]["question"] + "\n\nOptions:\n" + "\n".join(opt_lines) + MCQ_COT
            solution = test_set[i]["chosen"]

        else:
            question = BALANCE_PROMPT + extract_unbalanced_from_question(test_set[i]["question"])+BALANCE_COT
            solution = test_set[i]["chosen"]


        t0 = time.time()
        format_str = p.tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            max_tokens=1856,
            temperature=0.6,
            stop_token_ids=[p.tokenizer.eos_token_id],
            logprobs=0, # We need logprobs of generated tokens
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

        if type_q.startswith("mcq"):
            answer = parse_answer_gpqa(boosted_rollout)
            is_correct = int(grade_answer(answer, test_set[i]["answerKey"]))

        else:
            out = same_balanced_equation_olmo(boosted_rollout, test_set[i]["answer"])
            answer = out["equation_1"]
            is_correct = out["same"]

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
