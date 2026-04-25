"""GRPO training entrypoint for BioOperatorEnv.

Usage (typically run on Colab or HF compute, not local CPU):

    python training/grpo_train.py \
        --model_id Qwen/Qwen2.5-3B-Instruct \
        --stage 1 \
        --max_steps 200 \
        --output_dir checkpoints/qwen3-bioperator-lora

Stages (per design spec §7):
    0 = format school (format_only reward, normal-baseline scenario)
    1 = easy DO recovery (full reward, do-recovery-easy)
    2 = productive recovery (full reward, easy + medium mixed)
    3 = multi-fault (full reward, all 3 disturbance scenarios)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# Note: imports of unsloth / trl / transformers happen inside main() so this
# module can be imported on a CPU-only machine without errors.

from training.reward_fn import format_only_reward_fn, reward_fn
from training.rollout import build_dataset


_STAGE_CONFIG = {
    0: {"task_ids": ["normal-baseline"], "reward": format_only_reward_fn,
        "n_samples": 128, "name": "format_school"},
    1: {"task_ids": ["do-recovery-easy"], "reward": reward_fn,
        "n_samples": 256, "name": "easy_do_recovery"},
    2: {"task_ids": ["do-recovery-easy", "do-recovery-medium"], "reward": reward_fn,
        "n_samples": 384, "name": "productive_recovery"},
    3: {"task_ids": ["do-recovery-easy", "do-recovery-medium", "aeration-limit-hard"],
        "reward": reward_fn, "n_samples": 512, "name": "multi_fault"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--stage", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="checkpoints/qwen3-bioperator-lora")
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    cfg = _STAGE_CONFIG[args.stage]
    print(f"[grpo_train] Stage {args.stage} ({cfg['name']}): tasks={cfg['task_ids']}")

    # 1) Build dataset
    print(f"[grpo_train] Building dataset ({cfg['n_samples']} prompts)...")
    rows = build_dataset(num_samples=cfg["n_samples"],
                          task_ids=cfg["task_ids"],
                          seed=args.seed)
    print(f"[grpo_train] Dataset built: {len(rows)} rows")

    # 2) Convert to HF dataset
    from datasets import Dataset
    ds = Dataset.from_list(rows)

    # 3) Load model + tokenizer (Unsloth fast path)
    print(f"[grpo_train] Loading model {args.model_id} (4bit={args.load_in_4bit})...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # 4) Configure GRPOTrainer
    print("[grpo_train] Configuring GRPOTrainer...")
    from trl import GRPOConfig, GRPOTrainer
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        logging_steps=5,
        save_steps=max(args.max_steps // 4, 50),
        seed=args.seed,
        report_to=args.report_to,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[cfg["reward"]],
        args=grpo_cfg,
        train_dataset=ds,
    )

    # 5) Train
    print("[grpo_train] Starting training...")
    trainer.train()

    # 6) Save adapter (LoRA-only; do NOT merge into 4-bit base — see hack §16)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[grpo_train] Saved adapter to {args.output_dir}")

    # 7) Quick post-training inspection
    print("[grpo_train] Sample post-training generations:")
    for row in rows[:3]:
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.5,
                              do_sample=True, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        print(f"  prompt-tail: ...{prompt[-80:]}")
        print(f"  completion : {text.strip()[:160]}")
        print()


if __name__ == "__main__":
    main()
