---
title: BioOperatorEnv
emoji: "🧪"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BioOperatorEnv

**A flight simulator for autonomous bioreactor operators.**

OpenEnv Hackathon India 2026 · Theme: *World Modeling / Professional Tasks*

---

## Why this exists

Real bioreactors are alive. They brew medicines like penicillin in tanks the size of a small house. They are slow, expensive, dangerous to mistakes, and quality-sensitive. **You cannot let an AI agent learn by ruining real batches.**

So we built the practice rink.

BioOperatorEnv takes a **published industrial penicillin simulator** (IndPenSim by Goldrick et al., *Journal of Biotechnology* 2015 — the gold-standard reference for fed-batch fermentation modeling) and turns it into an OpenEnv-compatible environment where an LLM can read a SCADA-style plant console, output a structured JSON control action, and learn — through GRPO + LoRA — to run the reactor better than untrained baselines.

> Previous papers built better autopilots for one bioprocess controller at a time. We're building the simulator where future autonomous bioreactor operators learn to fly.

---

## What the agent sees

A SCADA console snapshot. No equations. No hidden state. Just what a real plant operator sees on their screen:

```json
{
  "time_h": 42.0,
  "batch_phase": "production",
  "measurements": {
    "temperature_C": 25.0, "pH": 6.5, "dissolved_oxygen_pct": 22.0,
    "substrate_g_L": 0.15, "volume_L": 80000.0,
    "OUR": 0.5, "CER": 0.4
  },
  "setpoints_or_limits": {
    "DO_min_safe_pct": 20.0, "substrate_max_g_L": 0.30
  },
  "current_controls": {
    "feed_rate_L_h": 80.0, "aeration_rate_vvm": 0.85,
    "agitation_rpm": 100.0
  },
  "recent_trends": {"DO": "falling_fast", "pH": "stable",
                    "temperature": "stable", "substrate": "rising"},
  "alarm": "DO_NEAR_LOW_LIMIT",
  "previous_action": {"feed_delta_L_h": 5, "aeration_delta_vvm": 0,
                      "agitation_delta_rpm": 0}
}
```

## What the agent does

Picks one of 27 discrete operator actions every 12 simulated minutes, as JSON:

```json
{
  "feed_delta_L_h": -5,
  "aeration_delta_vvm": 0.10,
  "agitation_delta_rpm": 0,
  "reason": "DO falling after a feed bump. Reduce feed and add aeration first."
}
```

The pH and temperature loops stay on autopilot — like a real plant. The agent only nudges the operator-level controls.

## How it learns

```
LLM observation -> JSON action -> simulator advances 12 sim min
                                 -> 7 reward components computed
                                 -> GRPO updates LoRA adapter weights
```

Reward is **deliberately split into 7 independent components** so the model can't game one signal:

| Component | What it watches | Weight |
|---|---|---|
| `format_validity` | Was the action valid JSON in range? | 0.05 |
| `do_safety` | Dissolved oxygen above the safe floor? | 0.30 |
| `productivity` | Penicillin growing? | 0.20 |
| `substrate_control` | Substrate in the healthy band? | 0.15 |
| `stability` | Temperature and pH near setpoints? | 0.10 |
| `control_effort` | Avoiding wild swings? | 0.10 |
| `terminal_yield_bonus` | Total penicillin at episode end | 0.10 |

## The plant engine

We did **not** invent a new bioreactor model. We **ported the validated MATLAB IndPenSim simulator to Python** and verified the port reproduces the published trajectories.

![Python port vs MATLAB calibration](docs/calibration/python_vs_matlab.png)

*Calibration overlay: Python port of IndPenSim (orange dashed) vs MATLAB/Octave reference (blue solid) on the same batch. Tight variables T, pH, V, DO are within published spec bands. Full per-variable error bands and a known-deviation explanation in [docs/calibration/calibration_report.md](docs/calibration/calibration_report.md).*

## Baseline results

We compared three lightweight baselines (no LLM yet) on `do-recovery-medium`, 5 seeds each:

![Baseline bar chart](results/baseline_comparison_bar.png)

The DO trajectories tell the story:

![DO trajectory comparison](results/do_recovery_trajectory_comparison.png)

A **random** policy (chaotic JSON) regularly slams DO down toward the 20% safety floor. The fixed-recipe (do-nothing) baseline cruises. The naive **rule-based operator** over-corrects — a sign that "intervene aggressively at every alarm" is *worse* than doing nothing. The trained LLM's job is to learn when to intervene and when to leave the plant alone.

## Tasks

| Task ID | Difficulty | Scenario |
|---|---|---|
| `do-recovery-easy` | easy | Mild substrate fault → small DO drop, recoverable with feed cut + aeration bump |
| `do-recovery-medium` | medium | Larger substrate fault, must balance DO safety with productivity |
| `aeration-limit-hard` | hard | Aeration fault during high-density fermentation |
| `normal-baseline` | dev | No-fault sanity check |

## Try it

### Local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"task_id": "do-recovery-medium", "seed": 42}'
curl -s -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0}}'
curl -s http://localhost:8000/state | head -c 300
```

### Docker

```bash
docker build -t bioperator-env .
docker run --rm -p 8000:7860 bioperator-env
```

### Train your own LLM operator (Colab)

[`notebooks/03_train_grpo.ipynb`](notebooks/03_train_grpo.ipynb) is a self-contained Colab notebook that:
1. Pulls this repo + installs Unsloth + TRL
2. Builds a 256-prompt GRPO dataset from the env
3. Loads Qwen 2.5 3B Instruct in 4-bit with LoRA(rank=16)
4. Trains 200 GRPO steps and plots the reward curve
5. Saves the LoRA adapter

### See the trained agent in action

[`notebooks/04_demo.ipynb`](notebooks/04_demo.ipynb) runs the untrained Qwen and the trained Qwen on the same seeded scenario, side-by-side. The output is `results/before_after_demo.png`.

## Repo structure

```
bioperator_env/         the importable Python package
  plant/                Python port of IndPenSim (33 ODEs, params, controllers)
  env.py                BioOperatorEnv (reset/step/state)
  rewards.py            7-component reward composer
  scenarios.py          4 task definitions
  prompt.py, trends.py, alarms.py
server/app.py           FastAPI server (/, /health, /tasks, /reset, /step, /state)
baselines/              6 baseline agents (random, fixed, rule, untrained-Qwen,
                        Claude zero-shot, trained-Qwen)
training/               TRL GRPOTrainer + Unsloth glue
notebooks/              4 Jupyter notebooks (smoke/baselines/train/demo)
scripts/                run_baselines.py, generate_plots.py, calibrate_against_matlab.py
docs/                   spec, plan, calibration report + plot
results/                baseline CSVs + 9 plots
IndPenSim/              MATLAB source (used only to generate output_5/ reference)
tests/                  ~115 tests, all green
```

## Engineering quality

- **115/115 tests passing** (`pytest tests/`).
- **Calibration tested** against MATLAB reference on every commit.
- **Anti-cheat by construction**: agent never receives the 33-vector ODE state; pH/T PIDs are inaccessible; all action values clipped to literal sets before integration; reward components logged independently so single-signal exploits show up immediately.

## Limitations

- This is one specific bioprocess (penicillin fed-batch at 100,000 L). Each new process needs its own plant adapter.
- The MATLAB `mu_x` saturation rule (`indpensim.m §194-200`) is not yet ported, which causes our late-batch yields to overshoot MATLAB by ~25%. Documented in the calibration report.
- Real deployment to a physical reactor would require validation, regulatory review, and human supervision — far beyond this hackathon's scope.

## Links

- **Spec:** [`docs/superpowers/specs/2026-04-26-bioperatorenv-design.md`](docs/superpowers/specs/2026-04-26-bioperatorenv-design.md)
- **Implementation plan:** [`docs/superpowers/plans/2026-04-26-bioperatorenv.md`](docs/superpowers/plans/2026-04-26-bioperatorenv.md)
- **Calibration report:** [`docs/calibration/calibration_report.md`](docs/calibration/calibration_report.md)
- **Original simulator paper:** Goldrick et al., *J. Biotech* 2015. [DOI: 10.1016/j.jbiotec.2014.10.029](https://doi.org/10.1016/j.jbiotec.2014.10.029)
- **HF Space:** *(populate with the HF Space URL after first push)*
- **Training W&B run:** *(populate after the first GRPO run on Colab)*
- **2-min video:** *(populate after recording)*

---

*BioOperatorEnv: previous papers built better autopilots. We built the simulator where future autonomous bioreactor operators learn to fly.*
