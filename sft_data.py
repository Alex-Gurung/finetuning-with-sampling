"""Writes the SFT parquet (prompt and response columns) for a baseline.

    python sft_data.py expert TASK OUT.parquet        the training set's expert traces: GPT-5's for chem, MATH's for math
    python sft_data.py boosted BOOST_DIR OUT.parquet  the traces boost_math.py or boost_sci.py wrote, one per problem
    python sft_data.py correct BOOST_DIR OUT.parquet  the same, keeping only traces that reach the right answer

`boosted` keeps every trace whether or not it is correct, as in the README's recipe.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from groot import TASKS

EXPERT_FIELD = {"chem": "chosen", "math": "solution"}

if sys.argv[1] == "expert":
    task = sys.argv[2]
    load, question = TASKS[task][:2]
    rows = load()
    prompts = [question(row) for row in rows]
    responses = [row[EXPERT_FIELD[task]] for row in rows]
else:
    traces = {}
    for path in sorted(Path(sys.argv[2]).glob("boosted_*.jsonl")):
        for line in path.open():
            trace = json.loads(line)
            traces.setdefault(trace["idx"], trace)
    if sys.argv[1] == "correct":
        traces = {i: t for i, t in traces.items() if t["is_correct"]}
    prompts = [traces[i]["prompt"] for i in sorted(traces)]
    responses = [traces[i]["gen"] for i in sorted(traces)]
pd.DataFrame({"prompt": prompts, "response": responses}).to_parquet(sys.argv[-1])
print(f"{len(prompts)} examples -> {sys.argv[-1]}")
