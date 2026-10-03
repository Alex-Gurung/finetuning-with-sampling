"""Writes the SFT parquet (prompt and response columns) for a math baseline.

    python sft_data.py expert OUT.parquet             the training set's expert solutions
    python sft_data.py boosted BOOST_DIR OUT.parquet  the traces boost_math.py wrote, one per problem

Boosted traces are kept whether or not they reach the right answer, as in the README's recipe.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from eval_math import MATH_COT, MATH_PROMPT

if sys.argv[1] == "expert":
    train = pd.read_parquet("math_data/train.parquet")
    prompts = [MATH_PROMPT + problem + MATH_COT for problem in train["problem"]]
    responses = list(train["solution"])
else:
    traces = {}
    for path in sorted(Path(sys.argv[2]).glob("boosted_*.jsonl")):
        for line in path.open():
            trace = json.loads(line)
            traces.setdefault(trace["idx"], trace)
    prompts = [traces[i]["prompt"] for i in sorted(traces)]
    responses = [traces[i]["gen"] for i in sorted(traces)]
pd.DataFrame({"prompt": prompts, "response": responses}).to_parquet(sys.argv[-1])
print(f"{len(prompts)} examples -> {sys.argv[-1]}")
