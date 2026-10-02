#!/bin/bash
#SBATCH --job-name=eval_math_ood
#SBATCH -t 0-23:59                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores (MB)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0                # One task per seed


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

# OOD dataset: AMC, GSM8K, or MATH-TTT (override with e.g. `DATASET=GSM8K sbatch eval_math_ood.sh`)
DATASET=${DATASET:-AMC}

# Model to evaluate (local checkpoint directory or HF hub id)
MODEL_STR="{/path/to/checkpoint}"

SEED=$(( SLURM_ARRAY_TASK_ID ))
echo "Evaluating model: ${MODEL_STR} on ${DATASET} with SEED=${SEED}"

python eval_math_ood.py \
  --model_str="${MODEL_STR}" \
  --dataset="${DATASET}" \
  --seed="${SEED}"
