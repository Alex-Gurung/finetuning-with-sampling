#!/bin/bash
#SBATCH --job-name=mcmc_boost
#SBATCH -t 1-23:59
#SBATCH --mem=250G
#SBATCH --export=ALL
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --array=0

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cmake/3.31.6-fasrc01
module load cudnn

source ~/.bashrc
conda activate DFT

cd /path/to/verl

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_TOKEN={HF_TOKEN}

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LDS=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="$LDS:$LD_LIBRARY_PATH"
export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR={/path/to/torch_extensions}
export DS_BUILD_OPS=1 DS_BUILD_CPU_ADAM=1 MAX_JOBS=1

export HF_HOME={HUGGING_FACE_HOME}
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

export TRITON_CACHE_DIR={/path/to/triton_cache}


export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=300

export WANDB_DIR={/path/to/wandb}
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

export WANDB_PROJECT=data-boost
export WANDB_MODE=online        # or offline
export WANDB_ENTITY={WANDB_ENTITY}
export WANDB_API_KEY={WANDB_API_KEY}

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo


ROOT_DIR=$(pwd)

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs

data_basis=chem_boosted
train_data=data/${data_basis}.parquet
lr=5e-5
epochs=2
max_length=2048
batch_size=16
lr_warmup_steps=10
lr_scheduler=cosine
optimizer=AdamW
model=Qwen/Qwen2.5-7B-Instruct
clip_grad=1.0
date=$(date +%m%d)
experiment_name=chem
output_dir="{/path/to/outputs}/${model}/${date}/${experiment_name}_bs${batch_size}_lr${lr}_ep${epochs}_${data_basis}"
save_path=$output_dir/checkpoints/
nproc_per_node=4
#liger needs transformers 4.52

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data \
    data.val_files=$train_data \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=$batch_size \
    data.max_length=$max_length \
    optim.lr=$lr \
    optim.lr_scheduler=$lr_scheduler \
    +optim.optimizer=$optimizer \
    optim.clip_grad=$clip_grad \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$model \
    model.use_liger=False \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$experiment_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=2360 \
    trainer.test_freq=-1 \
    trainer.total_epochs=$epochs \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
