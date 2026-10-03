# Finetuning with Sampling


### [Paper](https://arxiv.org/abs/2610.02140) | [Project Page](https://aakaran.github.io/finetuning_with_sampling/)

[![rws](chem_teaser.png)](chem_teaser.png)


This repo contains the official PyTorch implementation of Finetuning with Sampling.
> [**Finetuning with Sampling: SFT Learns Better Than You Think**](https://arxiv.org/abs/2610.02140)<br>
> [Aayush Karan](https://aakaran.github.io/), [Sitan Chen](https://sitanchen.com/), [Yilun Du](https://yilundu.github.io/)
> <br>Harvard<br>



## Setup

Run the following script to setup environment.

```bash
git clone https://github.com/aakaran/finetuning-with-sampling.git
cd finetuning-with-sampling
conda env create -f environment.yml
conda activate mcmc
```


## Sampling

The main directory contains slurm scripts to run projection sampling for chemistry (```boost_sci.py```), whose training data .json is included in sci_data, as well as math (```boost_math.py```), whose training data .parquet is in math_data. 

To run projection sampling on chemistry with 15 shard parallelism:
```bash
sbatch boost_sci.sh
```
The output is several .jsonl files (based on the shard number) that store the boosted traces, original solution, correctness, etc. Accompanying files are trace_boosted .jsonl files, which track which MCMC candidates actually get accepted throughout the entire MCMC process. These can be used to obtain teacher distillation logits for the full boosted traces if needed. For standard SFT, the trace_boosted .jsonl files are not needed, so the logic can be disabled since the trace_boosted file outputs are memory intensive.


## Training

For SFT training, we refer to the setup in https://github.com/yongliang-wu/DFT. For chemistry and math, use ```--learning_rate=5e-5```, ```--num_train_epochs=2```, and ```--batch_size=16```. A sample SFT launch script is provided for reference (in ```utils/fsdp_utils.py``` in the DFT codebase, may need to modify the file to convert ```fsdp_transformer_layer_cls_to_wrap``` to a list when the instance is a set for indexing). The output of the sampling process should be one large .jsonl file that contains (among other variables) all prompts and generated rollouts. These can be readily converted into .parquet files that rename these keys to ```"prompt"``` and ```"response"``` for SFT.


## Evaluation

To evaluate trained checkpoints on say chemistry, include your hf model checkpoints in the ```MODEL_PATHS``` of ```eval_sci.sh```. Depending on the number of checkpoints, run 
```bash
sbatch --array=0-N eval_sci.sh
```
The output .jsonl grading file stores correctness per evaluation task and can be directly parsed to obtain final accuracy. Similar commands for ```eval_math.sh``` and ```eval_ood_math.sh``` hold. For MMLU and GPQA, we refer to lm-evaluation-harness (https://github.com/eleutherai/lm-evaluation-harness).

## Rejection-sampling fine-tuning with Groot (this fork)

`groot.py` samples training solutions from a vLLM server and keeps the correct ones as SFT data, for the chemistry
or the math task (`--task chem` or `--task math`). With `--method iid` it solves each training problem four times.
With `--method groot` the model first writes a decision tree of approaches and four paths through it, then solves the
problem once per path with the path as a hidden hint. `--method vs` (verbalized sampling) asks for four approaches
with their probabilities in place of the tree and solves once per approach the same way. `--method acg`
(approach-conditioned generation) has the model turn the training set's expert solution into one approach and solves
`--n` times with it as the hidden hint; `--planner` picks another planner prompt, such as `chem_acg_scaffold_planner`. A sample is kept when it is
correct, finishes within 1,856 tokens and does not mention its hint. Prompts are in `groot_prompts/`.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --generation-config vllm --data-parallel-size 8
python groot.py --task chem --model Qwen/Qwen2.5-7B-Instruct --method groot --out data/qwen7b_groot
torchrun --nproc_per_node 8 train_sft.py --data data/qwen7b_groot/sft.parquet \
    --model Qwen/Qwen2.5-7B-Instruct --out checkpoints/qwen7b_groot
```

`train_sft.py` is full-parameter SFT with the settings of the verl run above: lr 5e-5, 2 epochs and batch 16
by default, and `--lr`, `--epochs`, `--batch` for the paper's sweep. `boost_batched.py` is the paper's projection
sampling run on many problems at once, with the same algorithm. `sft_data.py` writes the SFT sets for the paper's
baselines: `expert` from the training set's expert traces, `boosted` from the traces that `boost_sci.py`,
`boost_math.py` or `boost_batched.py` write (one per problem, correct or not), and `correct` from the same traces,
keeping only those that reach the right answer. `likelihood.py` scores an SFT set under a model (the paper's
Figure 3), and `passk.py` counts correct samples per chemistry test problem for pass@k (Figure 4).

Changes to the original scripts: the eval scripts import the graders from `grader_utils` and define
`format_prompt`, and the math evals honour `--model_type`; `grader_utils/math_normalize.py` is added from PRM800K.
`boost_sci.py` and `boost_math.py` take `--gpu_memory_utilization` and `--max_model_len` so several shards can
share a GPU, and `--idx_file` to process a list of training problems. `boost_sci.py` also takes `--resume` and an
Olmo-3-7B-Instruct option, and `boost_math.py` a Qwen3-4B-Instruct-2507 option.
