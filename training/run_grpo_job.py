# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.3",
#     "transformers>=4.46",
#     "peft>=0.11",
#     "trl>=0.10",
#     "bitsandbytes>=0.43",
#     "accelerate>=0.30",
#     "datasets>=2.20",
#     "wandb>=0.17",
#     "huggingface_hub>=0.24",
#     "unsloth",
#     "unsloth_zoo",
#     "scipy>=1.11",
#     "numpy>=1.26",
#     "pandas>=2.0",
#     "pydantic>=2.5",
#     "matplotlib>=3.8",
#     "tqdm>=4.66",
# ]
# ///
"""Self-contained GRPO training job for HF Jobs.

Submit via:

    hf auth login
    hf jobs uv run \\
        --flavor a100-large \\
        -s WANDB_API_KEY=<your_wandb_key> \\
        -s HF_TOKEN=<your_hf_write_token> \\
        https://github.com/Json604/openenv-bioreactor/raw/main/training/run_grpo_job.py \\
        --max_steps=200 \\
        --num_generations=8 \\
        --max_completion_length=256 \\
        --push_to_hub=Json604/qwen3b-bioperator-lora

`--flavor` follows HF Spaces hardware naming. Pass HF_TOKEN explicitly as
`-s HF_TOKEN=hf_xxx` (the bare `-s HF_TOKEN` form silently fails to inject
the token into the container; lesson learned the hard way).

The script clones the repo into a temp dir, builds the dataset, loads the
model + LoRA via Unsloth, runs GRPO, saves the adapter, pushes it to a
Hugging Face model repo, and writes a reward-curve PNG.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/Json604/openenv-bioreactor.git"
REPO_DIR = Path("/tmp/openenv-bioreactor")


def _ensure_repo() -> None:
    """Clone the repo so the bioperator_env / training packages import."""
    if REPO_DIR.exists():
        return
    print(f"[run_grpo_job] cloning {REPO_URL} -> {REPO_DIR}", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless GRPO training for BioOperatorEnv")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--task_id", default="do-recovery-easy")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="/tmp/qwen3-bioperator-lora")
    parser.add_argument("--push_to_hub", default=None,
                        help="If set, push the LoRA adapter to this HF model repo.")
    parser.add_argument("--wandb_project", default="bioperator-env")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    # 1) Get the repo on PYTHONPATH
    _ensure_repo()
    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    # 2) wandb (optional, opt-in via env var)
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"grpo-{args.max_steps}steps-G{args.num_generations}",
            config=vars(args),
            tags=["hackathon", "hf-jobs", args.task_id, args.model_id.split("/")[-1]],
        )
        report_to = "wandb"
    else:
        print("[run_grpo_job] WANDB_API_KEY not set; logging locally only.")
        report_to = "none"

    # 3) Build the GRPO dataset
    from datasets import Dataset  # type: ignore
    from training.rollout import build_dataset  # type: ignore
    print(f"[run_grpo_job] building {args.num_samples} prompts from task '{args.task_id}'", flush=True)
    rows = build_dataset(num_samples=args.num_samples, task_ids=[args.task_id], seed=args.seed)
    ds = Dataset.from_list(rows)
    print(f"[run_grpo_job] dataset rows: {len(ds)}", flush=True)

    # 4) Load model + attach LoRA via Unsloth
    print(f"[run_grpo_job] loading {args.model_id} in 4-bit + LoRA(r={args.lora_rank})", flush=True)
    from unsloth import FastLanguageModel  # type: ignore
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
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

    # 5) Configure + run GRPO
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
    from training.reward_fn import reward_fn  # type: ignore
    print(f"[run_grpo_job] starting GRPO: max_steps={args.max_steps} G={args.num_generations}", flush=True)
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=5,
        save_steps=max(args.max_steps // 4, 25),
        report_to=report_to,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()

    # 6) Save adapter (LoRA only, do NOT merge -- per HACK_INST §16)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[run_grpo_job] adapter saved to {args.output_dir}", flush=True)

    # 7) Optional push to HF Hub so notebook 04 can load via TrainedQwenAgent
    if args.push_to_hub:
        if not os.environ.get("HF_TOKEN"):
            print("[run_grpo_job] HF_TOKEN not set; skipping push_to_hub")
        else:
            print(f"[run_grpo_job] pushing adapter to https://huggingface.co/{args.push_to_hub}", flush=True)
            try:
                model.push_to_hub(args.push_to_hub, token=os.environ["HF_TOKEN"], private=False)
                tokenizer.push_to_hub(args.push_to_hub, token=os.environ["HF_TOKEN"], private=False)
                print("[run_grpo_job] push complete")
            except Exception as e:
                print(f"[run_grpo_job] push_to_hub failed: {e}")

    # 8) Plot reward curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        log_history = trainer.state.log_history
        rewards = [(h["step"], h.get("reward")) for h in log_history if "reward" in h]
        if rewards:
            xs, ys = zip(*rewards)
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("GRPO step")
            plt.ylabel("Mean reward")
            plt.title("BioOperatorEnv GRPO training reward")
            plt.grid(alpha=0.3)
            out = REPO_DIR / "results" / "reward_curve.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[run_grpo_job] reward curve saved to {out}")
    except Exception as e:
        print(f"[run_grpo_job] reward-curve plot skipped: {e}")

    print("[run_grpo_job] DONE", flush=True)


if __name__ == "__main__":
    main()
