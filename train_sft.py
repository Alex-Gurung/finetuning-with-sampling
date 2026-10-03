"""Full-parameter SFT on a prompt/response parquet with the paper's chemistry settings.

The defaults are the paper's chemistry run: lr 5e-5, 2 epochs, global batch 16. The paper tunes
each SFT run over epochs {1, 2}, lr {5e-5, 1e-5, 5e-6} and batch {16, 32, 64}. Fixed: max length
2048, cosine schedule, gradient clipping at 1.0, and bf16 weights as in the paper's verl run
(model_dtype=bf16). The rest follows verl's SFT defaults: AdamW betas 0.9/0.95, weight decay 0.01,
10% warmup. Plain data parallel, one model copy per GPU; the loss covers the response tokens only.
Examples are formatted as verl formats them: the chat-templated prompt, then the response followed by
the tokenizer's EOS token (TRL appends it), so a fine-tuned base model stops where the eval expects.

    torchrun --nproc_per_node 8 train_sft.py --data groot_qwen/sft.parquet \
        --model Qwen/Qwen2.5-7B-Instruct --out checkpoints/groot_qwen
"""

import argparse
import os

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16, help="global batch size")
    parser.add_argument("--seed", type=int, default=42, help="data order (42 is TRL's default)")
    parser.add_argument("--per_device", type=int, default=1,
                        help="sequences per GPU per forward pass; gradient accumulation makes up the global batch")
    parser.add_argument("--no_checkpointing", action="store_true",
                        help="keep activations instead of recomputing them in the backward pass (faster, more memory)")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="stop after this many optimizer steps without saving (timing runs; the schedule follows it)")
    args = parser.parse_args()

    rows = pd.read_parquet(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        for prompt in rows["prompt"]
    ]
    dataset = Dataset.from_dict({"prompt": prompts, "completion": list(rows["response"])})
    config = SFTConfig(
        output_dir=args.out,
        model_init_kwargs={"dtype": torch.bfloat16},
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        adam_beta2=0.95,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.per_device,
        gradient_accumulation_steps=args.batch // (args.per_device * int(os.environ.get("WORLD_SIZE", "1"))),
        max_length=2048,
        gradient_checkpointing=not args.no_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=args.max_steps,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    if args.max_steps > 0:
        return
    # Olmo-3-7B-Instruct-SFT ships temperature and top_p without do_sample, which saving rejects.
    trainer.model.generation_config.do_sample = True
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
