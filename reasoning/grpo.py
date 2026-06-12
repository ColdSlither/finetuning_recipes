import unsloth

import argparse
from pathlib import Path

try:
    from reasoning.env import (
        SEED,
        format_row,
        trl_reward_functions,
    )
    from reasoning.eval_logging import build_eval_logging_trainer
except ModuleNotFoundError:
    from env import (
        SEED,
        format_row,
        trl_reward_functions,
    )
    from eval_logging import build_eval_logging_trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO reasoning training with Unsloth."
    )
    parser.add_argument(
        "--model_path",
        "-m",
        default="paperbd/neuraltxt-135M-reasoning-base",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="models/reasoning_grpo",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        default="paperbd/paper_instructions_300K-v1",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", "-g", type=int, default=4)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--num_train_rows", "-n", type=int, default=1000)
    parser.add_argument("--num_eval_rows", type=int, default=32)
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory for per-evaluation-step JSONL logs.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fast_inference", action="store_true")
    return parser.parse_args()


def main():
    import sys

    args = parse_args()
    if sys.platform != "darwin":
        import transformers.utils.generic

        transformers.utils.generic._is_mlx_available = False

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    log_dir = Path(args.log_dir or f"{args.output_dir}/log")
    trainer_class = build_eval_logging_trainer(GRPOTrainer, log_dir)

    print(f"Loading model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_r,
        gpu_memory_utilization=0.8,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=args.lora_r >= 64,
    )

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train").shuffle(seed=SEED)
    num_eval_rows = min(args.num_eval_rows, len(dataset))
    num_train_rows = min(args.num_train_rows, len(dataset) - num_eval_rows)
    if num_train_rows == 0 or num_eval_rows == 0:
        raise ValueError("Training and evaluation datasets must both be non-empty.")

    eval_dataset = dataset.select(range(num_eval_rows))
    train_dataset = dataset.select(
        range(num_eval_rows, num_eval_rows + num_train_rows)
    )
    train_dataset = train_dataset.map(
        format_row,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        format_row,
        remove_columns=eval_dataset.column_names,
    )

    training_args = GRPOConfig(
        temperature=0.8,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_seq_length // 2,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to="none",
        output_dir=args.output_dir,
        bf16=True,
        seed=SEED,
    )

    print(
        f"Model: {args.model_path}\n"
        f"Train rows: {len(train_dataset)}\n"
        f"Eval rows: {len(eval_dataset)}\n"
        f"Max steps: {args.max_steps}\n"
        f"Generations per prompt: {args.num_generations}\n"
        f"LoRA rank: {args.lora_r}\n"
    )

    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        reward_funcs=trl_reward_functions(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    model.save_pretrained(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"Saved to {args.output_dir}/final")
    print(f"Evaluation logs saved to {log_dir}")


if __name__ == "__main__":
    main()
