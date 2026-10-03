"""Full-parameter SFT on a prompt/response parquet with the paper's chemistry settings.

lr 5e-5, 2 epochs, global batch 16, max length 2048, cosine schedule, gradient clipping at 1.0,
and verl's SFT defaults for the rest (AdamW betas 0.9/0.95, weight decay 0.01, 10% warmup).
The loss covers the response tokens only.

    torchrun --nproc_per_node 8 train_sft.py --data groot_qwen/sft.parquet \
        --model Qwen/Qwen2.5-7B-Instruct --out checkpoints/groot_qwen
"""

import argparse
import os

import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

GLOBAL_BATCH = 16
PER_DEVICE_BATCH = 2


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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        adam_beta2=0.95,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GLOBAL_BATCH // (PER_DEVICE_BATCH * world_size),
        max_length=2048,
        bf16=True,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset)
    trainer.train()
    # Olmo-3-7B-Instruct-SFT ships temperature and top_p without do_sample, which saving rejects.
    trainer.model.generation_config.do_sample = True
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(args.out)
    if trainer.accelerator.is_main_process:
        trainer.processing_class.save_pretrained(args.out)


if __name__ == "__main__":
    main()
