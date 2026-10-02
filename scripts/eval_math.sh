#!/bin/bash
#SBATCH --job-name=eval_math
#SBATCH -t 0-23:59                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores (MB)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0                  # One task per model path in MODEL_PATHS below: 0-(N-1) for N paths


module load python/3.12.8-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME={HUGGING_FACE_HOME}
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

export PYTHONPATH="$PYTHONPATH:{/path/to/finetuning-with-sampling}"
export HF_TOKEN={HF_TOKEN}

source activate mcmc
cd /path/to/finetuning-with-sampling

# Models to evaluate (local checkpoint directories or HF hub ids); one SLURM array task per entry.
MODEL_PATHS=(
    "{/path/to/checkpoint}"
)

MODEL=${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}
echo "Evaluating model: ${MODEL}"

python eval_math.py \
  --model_path="${MODEL}" \
  --seed=0
