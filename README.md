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

**An OpenEnv-compatible training environment for language models that operate an industrial penicillin bioreactor.**

OpenEnv Hackathon India 2026 · Theme: *World Modeling / Professional Tasks*

---

## Why this exists

Industrial bioreactors brew medicines like penicillin in tanks the size of a small house. They are slow, expensive, dangerous to mistakes, and quality-sensitive. **You cannot let an AI agent learn by ruining real batches.**

So I built the environment in which one can.

BioOperatorEnv takes a **published industrial penicillin simulator** (IndPenSim by Goldrick et al., *Journal of Biotechnology* 2015 — the standard reference for fed-batch fermentation modeling), ports it from MATLAB to Python, and wraps it in an OpenEnv-compatible interface where a language model can read a SCADA-style plant console, output a structured JSON control action, and learn — through GRPO with a low-rank adapter — to run the reactor better than untrained baselines.

> Previous papers built better autopilots. I built the simulator where future autonomous bioreactor operators learn the job.

For the long-form story-led writeup, see [`Blog.md`](./Blog.md).

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

I did **not** invent a new bioreactor model. I **ported the validated MATLAB IndPenSim simulator to Python** and verified the port reproduces the published trajectories.

![Python port vs MATLAB calibration](docs/calibration/python_vs_matlab.png)

*Calibration overlay: Python port of IndPenSim (orange dashed) vs MATLAB/Octave reference (blue solid) on the same batch. Tight variables T, pH, V, DO are within published spec bands. Full per-variable error bands and a known-deviation explanation in [docs/calibration/calibration_report.md](docs/calibration/calibration_report.md).*

## Baseline results

I compared four baselines on `do-recovery-medium`, 5 seeds each (env seeds {1,4,5,7,42}, picked because the random agent's chaos manifests on those):

![Baseline bar chart](results/baseline_comparison_bar.png)

| Agent | Mean episode reward (n=5) | What it does |
|---|---|---|
| `fixed_recipe` | ~23.1 | Never moves the controls — the natural floor |
| `random` | ~18.6 | Picks uniformly from the 27 valid actions |
| `claude_zero_shot` (Opus 4.7) | ~14.9 | Frontier LLM, no training, prompt-only |
| `rule_based` | ~13.0 | Hand-written if-then operator logic |

The DO trajectories tell the story:

![DO trajectory comparison](results/do_recovery_trajectory_comparison.png)

**The headline finding from the baselines alone:** even zero-shot Claude Opus 4.7 — the strongest available frontier model — does *worse* than doing nothing on this scenario. It writes 100% valid JSON and keeps DO above the safety floor, but it intervenes too often, and the control-effort penalty piles up. The naive rule-based operator behaves the same way for the same reason. *Intervening aggressively at every alarm is worse than leaving the plant alone.* The trained LLM operator's job is to learn when to act and when to wait.

## Training run

GRPO fine-tuning of Qwen 2.5 3B-Instruct + LoRA(r=16) on `do-recovery-easy`, run on a single H200 via Hugging Face Jobs (200 optimization steps, 64-prompt critical-snapshot dataset, 8 generations per prompt, KL-regularized policy gradient with `beta=0.04`). Full config and seven-component reward live in [`training/run_grpo_job.py`](training/run_grpo_job.py).

![Reward curve](results/reward_curve.png)

*Mean group reward and KL-regularized loss across 200 GRPO steps. Reward stays in the 0.40–0.46 band — the loop runs end-to-end, but at this small scale (200 steps × 64 prompts on a 3B base) the policy doesn't yet break out of the seven-component reward's local equilibrium. A larger run (more steps, more prompts, optionally the next stage of the curriculum) is the next step.*

The wandb run for this curve, with all per-step metrics, profiling timing, and config, is linked below in the Links section.

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
2. Builds a 64-prompt GRPO dataset from the env
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
tests/                  107 tests, all green
```

## Engineering quality

- **107/107 tests passing** (`pytest tests/`).
- **Calibration tested** against MATLAB reference on every commit.
- **Anti-cheat by construction**: agent never receives the 33-vector ODE state; pH/T PIDs are inaccessible; all action values clipped to literal sets before integration; reward components logged independently so single-signal exploits show up immediately.

## Limitations

- This is one specific bioprocess (penicillin fed-batch at 100,000 L). Each new process needs its own plant adapter.
- The MATLAB `mu_x` saturation rule (`indpensim.m §194-200`) is not yet ported, which causes the late-batch yields to overshoot MATLAB by ~25%. Documented in the calibration report.
- Real deployment to a physical reactor would require validation, regulatory review, and human supervision — far beyond this hackathon's scope.

## Links

- **Blog post (story-led writeup):** [`Blog.md`](./Blog.md)
- **Spec:** [`docs/superpowers/specs/2026-04-26-bioperatorenv-design.md`](docs/superpowers/specs/2026-04-26-bioperatorenv-design.md)
- **Implementation plan:** [`docs/superpowers/plans/2026-04-26-bioperatorenv.md`](docs/superpowers/plans/2026-04-26-bioperatorenv.md)
- **Calibration report:** [`docs/calibration/calibration_report.md`](docs/calibration/calibration_report.md)
- **Original simulator paper:** Goldrick et al., *J. Biotech* 2015. [DOI: 10.1016/j.jbiotec.2014.10.029](https://doi.org/10.1016/j.jbiotec.2014.10.029)
- **HF Space (live):** [`Json604/openenv-bioreactor`](https://huggingface.co/spaces/Json604/openenv-bioreactor) — direct API at `https://Json604-openenv-bioreactor.hf.space`
- **Trained LoRA adapter:** [`Json604/qwen3b-bioperator-lora`](https://huggingface.co/Json604/qwen3b-bioperator-lora) — GRPO-trained on H200, includes reward curve, before/after demo plot, and full eval metrics.
- **Training W&B run:** [`personal-meta/bioperator-env/runs/1ycts2ex`](https://wandb.ai/personal-meta/bioperator-env/runs/1ycts2ex) — the 200-step H200 run shown in the reward curve above. Per-step metrics, profiling timing, and config all on wandb.

---

*BioOperatorEnv: previous papers built better autopilots. I built the simulator where future autonomous bioreactor operators learn the job.*
