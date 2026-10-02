SEQ_PROMPT_SCI = """You are an expert chemist given a chemistry problem, its solution, and an initial, partial response. Carefully study the solution,
identifying what reasoning or steps are already provided, and then continue the partial response. Ensure your response is logically consistent with the solution and leads to a complete and correct final answer.

Task: 

{PROBLEM}

This is an example for a solution to the problem: 

{SOLUTION}

Here is a partial response: 

{RESPONSE}

Starting with the partial response, continue the response in your own words, including the thinking process. Ensure the final answer exactly matches that of the provided solution.
"""



SEQ_PROMPT_MATH = """You are an expert mathematician given a math problem, its solution, and an initial, partial response. Carefully study the solution,
identifying what reasoning or steps are already provided, and then continue the partial response. Ensure your response is logically consistent with the solution and leads to a complete and correct final answer.

Task: 

{PROBLEM}

This is an example for a solution to the problem: 

{SOLUTION}

Here is a partial response: 

{RESPONSE}

Starting with the partial response, continue the response in your own words, including the thinking process. Ensure the final answer exactly matches that of the provided solution, and present the final answer within \\boxed{{}}.
"""



SEQ_PROMPT_MED = """You are a medical expert given a medical problem, its solution, and an initial, partial response. Carefully study the solution,
identifying what reasoning or steps are already provided, and then continue the partial response. Ensure your response is logically consistent with the solution and leads to a complete and correct final answer.

Task: 

{PROBLEM}

This is an example solution to the question: 

{SOLUTION}

Here is a partial response: 

{RESPONSE}

Starting with the partial response, continue the response in your own words, including the thinking process. Ensure the answer matches the provided solution.
"""


def format_prompt(text, model_type, tokenizer):
    if model_type == "base":
        format_str = text
    elif model_type == "chat":
        answer_context = [{"role": "user", "content": text}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
    return format_str


