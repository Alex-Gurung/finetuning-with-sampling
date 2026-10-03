"""Full-parameter SFT on a prompt/response parquet with the paper's chemistry settings.

lr 5e-5, 2 epochs, global batch 16, max length 2048, cosine schedule, gradient clipping at 1.0, and
bf16 weights as in the paper's verl run (model_dtype=bf16). The rest follows verl's SFT defaults:
AdamW betas 0.9/0.95, weight decay 0.01, 10% warmup. Plain data parallel, one model copy per GPU;
the loss covers the response tokens only.

    torchrun --nproc_per_node 8 train_sft.py --data groot_qwen/sft.parquet \
        --model Qwen/Qwen2.5-7B-Instruct --out checkpoints/groot_qwen
"""

import argparse
import os

import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

GLOBAL_BATCH = 16


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = pd.read_parquet(args.data)
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": response}],
            }
            for prompt, response in zip(rows["prompt"], rows["response"], strict=True)
        ]
    )
    config = SFTConfig(
        output_dir=args.out,
        model_init_kwargs={"dtype": torch.bfloat16},
        num_train_epochs=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        adam_beta2=0.95,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GLOBAL_BATCH // int(os.environ.get("WORLD_SIZE", "1")),
        max_length=2048,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset)
    trainer.train()
    # Olmo-3-7B-Instruct-SFT ships temperature and top_p without do_sample, which saving rejects.
    trainer.model.generation_config.do_sample = True
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
