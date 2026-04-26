"""Pull H200 training artifacts (reward curve, before/after plot, eval
metrics, training log) from the HF Hub adapter repo into results/.

Usage:
    python scripts/fetch_training_artifacts.py [--repo Json604/qwen3b-bioperator-lora]

After a successful `training/run_grpo_job.py` run with --push_to_hub, the
job uploads:
  - reward_curve.png
  - before_after_demo.png
  - eval_metrics.json
  - training_log_history.json
  - (and the adapter files: adapter_model.safetensors, etc.)

This script downloads just the four artifact files into results/ and
prints the headline numbers from eval_metrics.json so we can update
the README/Blog.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

ARTIFACTS = (
    "reward_curve.png",
    "before_after_demo.png",
    "eval_metrics.json",
    "training_log_history.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Json604/qwen3b-bioperator-lora")
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in ARTIFACTS:
        try:
            path = hf_hub_download(repo_id=args.repo, filename=fname,
                                   local_dir=str(out_dir))
            print(f"  fetched {fname} -> {path}")
        except Exception as e:
            print(f"  [skip] {fname}: {type(e).__name__}: {e}")

    metrics_path = out_dir / "eval_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        print()
        print("=== Eval summary ===")
        print(f"task: {m.get('task_id')}  seeds: {m.get('seeds')}")
        for arm in ("trained", "untrained"):
            d = m.get(arm) or {}
            do_pct = d.get("do_above_floor_pct_mean")
            reward = d.get("total_reward_mean")
            fmt = d.get("format_valid_pct_mean")
            sv = d.get("safety_violations_mean")
            if do_pct is None:
                continue
            print(f"  {arm:9s}: DO_above_floor={do_pct:.1f}%  "
                  f"reward={reward:.2f}  format_valid={fmt:.1f}%  "
                  f"safety_violations={sv:.2f}")
        delta = m.get("delta")
        if delta:
            abs_d = delta.get("do_above_floor_pct_abs")
            rel_d = delta.get("do_above_floor_pct_rel")
            print(f"  delta:    +{abs_d:.1f}pp DO_above_floor  "
                  f"({rel_d:+.1f}% relative)")


if __name__ == "__main__":
    main()
