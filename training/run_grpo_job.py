# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.3,<2.7",
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
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# ///
"""Self-contained GRPO training job for HF Jobs.

Submit via:

    hf auth login
    hf jobs uv run \\
        --flavor a100-large \\
        -s WANDB_API_KEY=<your_wandb_key> \\
        -s HF_TOKEN \\
        https://github.com/Json604/openenv-bioreactor/raw/main/training/run_grpo_job.py \\
        --max_steps=200 \\
        --num_generations=8 \\
        --max_completion_length=256 \\
        --push_to_hub=Json604/qwen3b-bioperator-lora

`--flavor` follows HF Spaces hardware naming. Run `hf jobs hardware` for
the full list. `-s HF_TOKEN` with no value reuses the token from
`hf auth login`. Script args go directly after the URL (no `--` needed;
hf jobs uv run treats the script URL as positional and everything after
as SCRIPT_ARGS).

The script clones the repo into a temp dir, builds the dataset, loads the
model + LoRA, runs GRPO, saves the adapter, and (optionally) pushes the
adapter to a Hugging Face model repo so the demo notebook can load it.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/Json604/openenv-bioreactor.git"
REPO_DIR = Path("/tmp/openenv-bioreactor")
ARTIFACTS_DIR = Path("/tmp/bioperator-artifacts")


def _ensure_repo() -> None:
    """Clone the repo so the bioperator_env / training packages import."""
    if REPO_DIR.exists():
        return
    print(f"[run_grpo_job] cloning {REPO_URL} -> {REPO_DIR}", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)


# ---------- end-of-training eval helpers ----------

_JSON_RE = re.compile(r"\{[^{}]*\}", flags=re.DOTALL)


def _generate_action_text(model, tokenizer, prompt: str,
                           max_new_tokens: int = 96,
                           temperature: float = 0.5) -> str:
    """Single greedy-ish generation. Used for eval, not training."""
    import torch  # type: ignore
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)


def _parse_first_json(text: str):
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _rollout_episode(model, tokenizer, task_id: str, seed: int,
                     eval_temperature: float = 0.5) -> dict:
    """Run one full episode using `model` as the policy. Returns metrics + DO trajectory."""
    from bioperator_env.env import BioOperatorEnv  # type: ignore
    from bioperator_env.prompt import build_prompt  # type: ignore
    env = BioOperatorEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    do_min_safe = obs.setpoints_or_limits.get("DO_min_safe_pct", 20.0)
    do_traj = [obs.measurements["dissolved_oxygen_pct"]]
    rewards = []
    n_above = 0
    n_format_valid = 0
    n_steps = 0
    done = False
    while not done:
        prompt = build_prompt(obs)
        text = _generate_action_text(model, tokenizer, prompt,
                                      temperature=eval_temperature)
        parsed = _parse_first_json(text) or {
            "feed_delta_L_h": 0, "aeration_delta_vvm": 0.0,
            "agitation_delta_rpm": 0, "reason": "fallback",
        }
        obs, r, done, info = env.step(parsed)
        rewards.append(r)
        do_traj.append(obs.measurements["dissolved_oxygen_pct"])
        n_steps += 1
        if obs.measurements["dissolved_oxygen_pct"] >= do_min_safe:
            n_above += 1
        if info.get("format_valid", False):
            n_format_valid += 1
    return {
        "task_id": task_id,
        "seed": seed,
        "do_traj": do_traj,
        "total_reward": float(sum(rewards)),
        "do_above_floor_pct": (n_above / max(n_steps, 1)) * 100.0,
        "format_valid_pct": (n_format_valid / max(n_steps, 1)) * 100.0,
        "safety_violations": int(env.safety_violations),
        "n_steps": n_steps,
    }


def _run_inline_eval(model, tokenizer, task_id: str,
                     seeds: list[int],
                     output_dir: Path,
                     eval_temperature: float = 0.5) -> dict:
    """After training: roll out trained (adapter active) vs untrained (adapter
    disabled) on the same seeds. Saves before_after_demo.png and
    eval_metrics.json. Returns the metrics dict."""
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    trained_runs, untrained_runs = [], []
    for seed in seeds:
        print(f"[eval] trained seed={seed} ...", flush=True)
        trained_runs.append(_rollout_episode(model, tokenizer, task_id, seed,
                                              eval_temperature))
        print(f"[eval] untrained (adapter disabled) seed={seed} ...", flush=True)
        try:
            ctx = model.disable_adapter()
        except AttributeError:
            # Fallback: PeftModel must expose disable_adapter; if not, skip.
            print("[eval] disable_adapter() unavailable; skipping untrained eval")
            return {"trained_runs": trained_runs, "untrained_runs": []}
        with ctx:
            untrained_runs.append(_rollout_episode(model, tokenizer, task_id,
                                                    seed, eval_temperature))

    def _avg(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else None

    metrics = {
        "task_id": task_id,
        "seeds": list(seeds),
        "trained": {
            "do_above_floor_pct_mean": _avg(trained_runs, "do_above_floor_pct"),
            "total_reward_mean":      _avg(trained_runs, "total_reward"),
            "format_valid_pct_mean":  _avg(trained_runs, "format_valid_pct"),
            "safety_violations_mean": _avg(trained_runs, "safety_violations"),
            "per_seed": [{k: v for k, v in r.items() if k != "do_traj"}
                         for r in trained_runs],
        },
        "untrained": {
            "do_above_floor_pct_mean": _avg(untrained_runs, "do_above_floor_pct"),
            "total_reward_mean":      _avg(untrained_runs, "total_reward"),
            "format_valid_pct_mean":  _avg(untrained_runs, "format_valid_pct"),
            "safety_violations_mean": _avg(untrained_runs, "safety_violations"),
            "per_seed": [{k: v for k, v in r.items() if k != "do_traj"}
                         for r in untrained_runs],
        },
    }
    # Improvement deltas (only meaningful if both arms ran)
    if untrained_runs:
        u = metrics["untrained"]["do_above_floor_pct_mean"]
        t = metrics["trained"]["do_above_floor_pct_mean"]
        if u is not None and t is not None:
            metrics["delta"] = {
                "do_above_floor_pct_abs": t - u,
                "do_above_floor_pct_rel": ((t - u) / u * 100.0) if u > 0 else None,
            }

    # Plot DO trajectories: one line per seed per arm
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for r in untrained_runs:
        axes[0].plot(r["do_traj"], color="#e34a33", alpha=0.7,
                      label=f"untrained seed={r['seed']}")
    for r in trained_runs:
        axes[0].plot(r["do_traj"], color="#31a354", alpha=0.85,
                      label=f"trained seed={r['seed']}")
    axes[0].axhline(20, color="grey", linestyle="--", label="DO_min_safe=20%")
    axes[0].set_ylabel("Dissolved O2 %")
    axes[0].set_title(f"Trained vs untrained Qwen-3B on {task_id} (n={len(seeds)})")
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    # Total-reward bar chart underneath
    arms, vals_mean, vals_std = [], [], []
    if untrained_runs:
        arms.append("untrained")
        vals_mean.append(_avg(untrained_runs, "total_reward"))
        vals_std.append(float(np.std([r["total_reward"] for r in untrained_runs])))
    arms.append("trained")
    vals_mean.append(_avg(trained_runs, "total_reward"))
    vals_std.append(float(np.std([r["total_reward"] for r in trained_runs])))
    axes[1].bar(arms, vals_mean, yerr=vals_std, capsize=5,
                 color=["#e34a33", "#31a354"][: len(arms)])
    axes[1].set_ylabel(f"Mean total reward (n={len(seeds)})")
    axes[1].set_title("Episode reward")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = output_dir / "before_after_demo.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Save metrics JSON
    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] saved {plot_path} and {metrics_path}", flush=True)
    print(f"[eval] summary: trained={metrics['trained']['do_above_floor_pct_mean']:.1f}%  "
          f"untrained={metrics['untrained']['do_above_floor_pct_mean']:.1f}%  "
          f"(do_above_floor)", flush=True)
    return metrics


def _upload_artifacts_to_hub(repo_id: str, files: list[Path], token: str) -> None:
    """Upload each file to the same model repo as the adapter."""
    from huggingface_hub import HfApi  # type: ignore
    api = HfApi(token=token)
    for f in files:
        if not f.exists():
            print(f"[upload] skip missing {f}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"add {f.name} from training run",
            )
            print(f"[upload] -> {repo_id}/{f.name}")
        except Exception as e:
            print(f"[upload] failed for {f.name}: {e}")


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
    parser.add_argument("--eval_seeds", default="1,4,5,7,42",
                        help="Comma-separated seeds for end-of-training eval. "
                             "Defaults are env seeds where the random/passive "
                             "baselines actually push DO toward the safety "
                             "floor on do-recovery-medium (probed 2026-04-26). "
                             "Empty string disables eval.")
    parser.add_argument("--eval_task", default="do-recovery-medium",
                        help="Scenario for end-of-training eval.")
    parser.add_argument("--eval_temperature", type=float, default=0.5)
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

    # 4) Force-initialize CUDA so Unsloth's accelerator detection works.
    #    On HF Jobs uv-run containers, importing unsloth can race ahead of the
    #    nvidia driver init (CUDA error 802, "system not yet initialized"),
    #    which makes Unsloth conclude there is no GPU. We retry torch.cuda
    #    init with backoff and explicit driver pokes to break the race.
    import time, ctypes  # type: ignore
    import torch  # type: ignore
    print(f"[run_grpo_job] torch={torch.__version__}  cuda_built={torch.version.cuda}",
          flush=True)

    def _try_cuda_init() -> bool:
        """Return True iff torch.cuda becomes usable. Uses ctypes to poke the
        nvidia driver if the first torch attempt hits 'system not yet
        initialized'."""
        for delay in (0, 2, 5, 10):
            if delay:
                time.sleep(delay)
            # Reset torch's cached "no cuda" answer if needed.
            try:
                torch.cuda._initialized = False  # type: ignore[attr-defined]
                torch.cuda._lazy_init()           # type: ignore[attr-defined]
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.zeros(1, device="cuda").mul_(1)
                    torch.cuda.synchronize()
                    return True
                except Exception as e:
                    print(f"[run_grpo_job] CUDA tensor probe failed: {e}", flush=True)
            else:
                # Try to wake the driver via libcuda.cuInit(0)
                try:
                    libcuda = ctypes.CDLL("libcuda.so.1")
                    rc = libcuda.cuInit(0)
                    print(f"[run_grpo_job] libcuda.cuInit(0) returned {rc}", flush=True)
                except OSError as e:
                    print(f"[run_grpo_job] libcuda.so.1 not loadable: {e}", flush=True)
        return False

    if not _try_cuda_init():
        raise RuntimeError(
            "torch.cuda is unavailable after retries on a GPU flavor. The "
            "container likely lacks an attached nvidia device. Try `hf jobs "
            "run` with a CUDA-baked image (e.g. pytorch/pytorch:cuda12.4) "
            "instead of `uv run`."
        )
    print(f"[run_grpo_job] CUDA ready: device_count={torch.cuda.device_count()}  "
          f"name={torch.cuda.get_device_name(0)}", flush=True)

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

    # 8) Plot training reward curve
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    reward_curve_path = ARTIFACTS_DIR / "reward_curve.png"
    log_history_path = ARTIFACTS_DIR / "training_log_history.json"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        log = trainer.state.log_history
        log_history_path.write_text(json.dumps(log, indent=2, default=str))
        rewards = [(h["step"], h.get("reward")) for h in log if "reward" in h]
        if rewards:
            xs, ys = zip(*rewards)
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("GRPO step")
            plt.ylabel("Mean reward")
            plt.title("BioOperatorEnv GRPO training reward")
            plt.grid(alpha=0.3)
            plt.savefig(reward_curve_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[run_grpo_job] reward curve saved to {reward_curve_path}")
    except Exception as e:
        print(f"[run_grpo_job] reward-curve plot skipped: {e}")

    # 9) End-of-training eval: trained vs adapter-disabled (untrained) on same seeds
    eval_metrics = None
    if args.eval_seeds.strip():
        try:
            seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
            print(f"[run_grpo_job] running eval: task={args.eval_task} seeds={seeds}",
                  flush=True)
            eval_metrics = _run_inline_eval(
                model=model,
                tokenizer=tokenizer,
                task_id=args.eval_task,
                seeds=seeds,
                output_dir=ARTIFACTS_DIR,
                eval_temperature=args.eval_temperature,
            )
        except Exception as e:
            print(f"[run_grpo_job] eval skipped: {type(e).__name__}: {e}", flush=True)
    else:
        print("[run_grpo_job] --eval_seeds empty; skipping inline eval")

    # 10) Push artifacts (reward curve, before/after plot, eval metrics) to the
    #     same Hub repo as the adapter.
    if args.push_to_hub and os.environ.get("HF_TOKEN"):
        artifact_files = [
            reward_curve_path,
            log_history_path,
            ARTIFACTS_DIR / "before_after_demo.png",
            ARTIFACTS_DIR / "eval_metrics.json",
        ]
        try:
            _upload_artifacts_to_hub(args.push_to_hub, artifact_files,
                                      os.environ["HF_TOKEN"])
        except Exception as e:
            print(f"[run_grpo_job] artifact upload failed: {e}", flush=True)

    print("[run_grpo_job] DONE", flush=True)


if __name__ == "__main__":
    main()
