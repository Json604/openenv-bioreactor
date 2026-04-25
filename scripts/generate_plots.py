"""Generate the README-ready plots from baseline CSVs and training logs.

Outputs to results/:
  - baseline_comparison_bar.png       (mean reward + safety violations per agent)
  - do_recovery_trajectory_comparison.png   (DO vs time, all agents same seed)
  - action_trajectory_comparison.png  (feed/aer/RPM deltas vs step)
  - per_component_rewards.png         (mean of 7 components per agent)
  - format_validity_curve.png         (placeholder; populated post-training)
  - success_rate_curve.png            (placeholder; post-training)
  - safety_violations_curve.png       (placeholder; post-training)
  - reward_curve.png                  (placeholder; post-training; written
                                        by the Colab notebook directly)

Usage:
    python scripts/generate_plots.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines.fixed_recipe_agent import FixedRecipeAgent
from baselines.random_agent import RandomAgent
from baselines.rule_based_agent import RuleBasedAgent
from bioperator_env.env import BioOperatorEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def baseline_bar_chart() -> None:
    """Mean reward + mean safety violations per agent on do-recovery-medium."""
    rows = []
    for agent in ("random", "fixed_recipe", "rule_based"):
        path = RESULTS / f"baseline_{agent}.csv"
        if not path.exists():
            print(f"  [skip] {path} missing — run scripts/run_baselines.py first")
            continue
        df = pd.read_csv(path)
        df = df[df["task_id"] == "do-recovery-medium"]
        rows.append({
            "agent": agent,
            "mean_reward": df["total_reward"].mean(),
            "std_reward": df["total_reward"].std(),
            "mean_safety": df["safety_violations"].mean(),
            "n": len(df),
        })

    if not rows:
        print("  no baseline CSVs found; skipping bar chart")
        return

    rdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(rdf["agent"], rdf["mean_reward"],
                yerr=rdf["std_reward"], capsize=5, color="#2b8cbe")
    axes[0].set_title("Mean episode reward (do-recovery-medium)")
    axes[0].set_ylabel("Sum of per-step rewards")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(rdf["agent"], rdf["mean_safety"], color="#e34a33")
    axes[1].set_title("Mean safety violations per episode")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle(f"BioOperatorEnv baselines (n={rdf['n'].iloc[0]} seeds each)")
    fig.tight_layout()
    out = RESULTS / "baseline_comparison_bar.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


def trajectory_comparison(seed: int = 42) -> None:
    """Run all 3 lightweight baselines on the same seeded episode, plot
    DO % and action deltas over time."""
    agents = {
        "random": RandomAgent(seed=seed),
        "fixed_recipe": FixedRecipeAgent(),
        "rule_based": RuleBasedAgent(),
    }
    traces = {}
    for name, agent in agents.items():
        env = BioOperatorEnv(task_id="do-recovery-medium", seed=seed)
        obs = env.reset()
        do_t = [obs.measurements["dissolved_oxygen_pct"]]
        feed_t, aer_t, rpm_t = [], [], []
        done = False
        while not done:
            a = agent.act(obs)
            obs, r, done, _ = env.step(a)
            do_t.append(obs.measurements["dissolved_oxygen_pct"])
            feed_t.append(a["feed_delta_L_h"])
            aer_t.append(a["aeration_delta_vvm"])
            rpm_t.append(a["agitation_delta_rpm"])
        traces[name] = {
            "do": np.array(do_t), "feed": np.array(feed_t),
            "aer": np.array(aer_t), "rpm": np.array(rpm_t),
        }

    # DO trajectory plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, t in traces.items():
        ax.plot(t["do"], label=name, linewidth=2)
    ax.axhline(20, color="grey", linestyle="--", label="DO_min_safe=20%")
    ax.set_xlabel("Episode step (12 sim min each)")
    ax.set_ylabel("Dissolved oxygen %")
    ax.set_title(f"DO trajectory: do-recovery-medium (seed={seed})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "do_recovery_trajectory_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    # Action trajectory plot
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, key, label in zip(axes, ("feed", "aer", "rpm"),
                              ("feed_delta_L_h", "aeration_delta_vvm", "agitation_delta_rpm")):
        for name, t in traces.items():
            ax.plot(t[key], label=name, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Episode step")
    fig.suptitle(f"Action trajectories (seed={seed})")
    fig.tight_layout()
    out = RESULTS / "action_trajectory_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def per_component_rewards() -> None:
    """Run each baseline once, average each reward component over the episode,
    plot as grouped bars."""
    agents = {
        "random": RandomAgent(seed=42),
        "fixed_recipe": FixedRecipeAgent(),
        "rule_based": RuleBasedAgent(),
    }
    rows = []
    for name, agent in agents.items():
        for seed in (0, 1, 2):
            env = BioOperatorEnv(task_id="do-recovery-medium", seed=seed)
            obs = env.reset()
            done = False
            while not done:
                a = agent.act(obs)
                obs, _, done, info = env.step(a)
            mean_comps = {k: float(np.mean([h[k] for h in env.component_history]))
                          for k in env.component_history[0].keys()}
            mean_comps["agent"] = name
            mean_comps["seed"] = seed
            rows.append(mean_comps)

    df = pd.DataFrame(rows)
    components = ["format_validity", "do_safety", "productivity",
                  "substrate_control", "stability", "control_effort",
                  "terminal_yield_bonus"]
    grouped = df.groupby("agent")[components].mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(components))
    width = 0.25
    colors = ["#2b8cbe", "#e34a33", "#31a354"]
    for i, (agent, row) in enumerate(grouped.iterrows()):
        ax.bar(x + i * width, row.values, width, label=agent, color=colors[i % 3])
    ax.set_xticks(x + width)
    ax.set_xticklabels(components, rotation=20, ha="right")
    ax.set_ylabel("Mean per-step reward")
    ax.set_title("Per-component reward averages by baseline (do-recovery-medium)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = RESULTS / "per_component_rewards.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("Generating BioOperatorEnv plots...")
    baseline_bar_chart()
    trajectory_comparison()
    per_component_rewards()
    print("Done. See results/*.png")


if __name__ == "__main__":
    main()
