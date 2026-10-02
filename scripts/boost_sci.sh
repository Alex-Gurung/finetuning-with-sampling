#!/bin/bash
#SBATCH --job-name=boost_sci
#SBATCH -t 0-23:59                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores (MB)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0-14

BATCH_IDX=$(( SLURM_ARRAY_TASK_ID ))


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

echo "Running shard BATCH_IDX=${BATCH_IDX}"

python boost_sci.py \
  --batch_idx="${BATCH_IDX}" \
  --model=qwen 
