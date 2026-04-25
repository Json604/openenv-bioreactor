# BioOperatorEnv — Design Spec

**Date:** 2026-04-26
**Status:** Draft for review
**Authors:** Kartikey + Claude
**Hackathon:** OpenEnv Hackathon India 2026 (Round 2)
**Theme:** #3.1 — World Modeling / Professional Tasks

---

## 1. Pitch

**BioOperatorEnv is a flight simulator for autonomous bioreactor operators.**

Real bioprocess plants are too expensive, slow, and dangerous to let an AI agent learn on. We turn an industrial-scale (100,000 L) penicillin fermentation simulator — IndPenSim by Goldrick et al. (Journal of Biotechnology 2015), originally MATLAB — into an OpenEnv-compatible Python environment. An LLM agent reads a SCADA-style plant console and emits structured JSON control actions. We train a small open-weight model (Qwen 3) with TRL+GRPO+Unsloth+LoRA against a multi-component verifiable reward, and demonstrate measurable improvement over six baselines including untrained open and closed SOTA models.

**Why this is a frontier problem the hackathon wants:** the agent must maintain consistent internal state across a long, partially observable trajectory; infer hidden biological dynamics from delayed lab measurements; choose structured actions under safety constraints; and recover from disturbances. That is the literal definition of Theme 3.1 ("dynamic systems where the model is expected to do real hard work instead of exploiting shortcuts").

---

## 2. Rubric Mapping

| Criterion | Weight | How we score |
|---|---|---|
| **Environment Innovation** | 40% | Faithful Python port of a peer-reviewed industrial simulator; SCADA-style plant-console abstraction; multi-fault scenario generator; structured-JSON action space; partial observability with delayed offline measurements. No team is shipping this. |
| **Storytelling** | 30% | Plain-English README ("flight simulator" metaphor, no jargon), 2-min video, before/after action examples, six-baseline bar chart, MATLAB↔Python calibration overlay, demo trajectory plots with annotated turning points. |
| **Showing improvement** | 20% | Reward curves (overall + per-component), success-rate curve, safety-violations curve, format-validity curve, baseline-vs-trained DO trajectory, action-trajectory comparison. All saved as `.png` and embedded in README. |
| **Reward & training pipeline** | 10% | 7 independent reward components with anti-hacking design; locked simulator state; JSON schema validation; 4-stage curriculum; Unsloth + TRL `GRPOTrainer`; Colab notebook. |

---

## 3. The Plant Engine (Python port of IndPenSim)

### 3.1 Source of truth
- MATLAB code in `IndPenSim/` (Goldrick 2015, J. Biotech).
- Reference Octave-generated trajectories in `IndPenSim/output_5/`:
  - 3 normal batches (Faults=0)
  - 1 substrate-fault batch (Faults=3)
  - 1 aeration-fault batch (Faults=1)
  - 1150 timesteps per batch at h=0.2 h sampling, 230 h total.
  - Columns we have: `Time, Fg, RPM, Fs, Fa, Fb, Fc, Fh, Fw, pressure, Fremoved, S, DO2, P, V, Wt, pH, T, Q, CO2outgas, Fpaa, Foil, OUR, O2, CER` + offline (`X_offline, P_offline, PAA_offline, NH3_offline, Viscosity_offline`).

### 3.2 What we port
Re-implement `indpensim_ode.m` (33 ODEs) in Python:
- 10 process states: S, DO2, O2_offgas, P, V, Wt, pH(H+), T, Q, viscosity
- Structured biomass: A0, A1, A3, A4 (4 morphological regions)
- Vacuole population balance: n0..n9, nm, phi0 (12 vars)
- Gas/dissolved CO2: CO2outgas, CO2_d
- Nutrients: PAA, NH3
- Diagnostic mu_p, mu_x

### 3.3 Implementation notes
- Solver: `scipy.integrate.solve_ivp(method='BDF', rtol=1e-5, atol=1e-7)` — BDF is the SciPy equivalent of MATLAB's `ode15s`.
- 105-parameter vector matches `Parameter_list.m` exactly.
- Inhibition model = full (Inhib=2): pH, T, DO2, CO2, PAA, N.
- Same 8 disturbance channels (low-pass-filtered noise on mu_p, mu_x, c_s, c_oil, abc, PAA_c, T_cin, O2_in).
- pH stored internally as H+ concentration; converted at observation boundary.
- Numerical floor at 0.001 to prevent integration explosions (matches MATLAB).
- Step time matches MATLAB: h = 0.2 h (12 min sim time per env step).

### 3.4 Calibration acceptance criterion
The Python port is "faithful" if, run with the same seed/IC/recipe as a MATLAB batch, these curves stay within these bands across the full 230 h:

| Variable | Acceptable absolute error | Acceptable relative error |
|---|---|---|
| Biomass (X) | ±2 g/L | ±10% |
| Penicillin (P) | ±0.5 g/L | ±15% |
| Dissolved O2 | ±2% saturation | ±15% |
| Substrate (S) | ±0.5 g/L | ±20% |
| Volume (V) | ±2000 L | ±5% |
| pH | ±0.1 | n/a |
| T | ±0.5 K | n/a |

Total final-yield (kg) within ±15% of MATLAB. README ships an overlay plot (`docs/calibration/python_vs_matlab.png`) with all batches superimposed.

If we miss the bands, we tune parameters in priority order: kla_constant (alpha_kla) → m_O2_X → Y_O2_X → mu_h → growth-rate Arrhenius constants.

---

## 4. OpenEnv Interface

### 4.1 API surface
Standard OpenEnv (Gym-style):

```
GET  /              → metadata + version
GET  /health        → liveness
GET  /tasks         → list of available scenarios
POST /reset         → {task_id, seed} → Observation
POST /step          → {action} → (Observation, reward, done, info)
GET  /state         → full debug state (server-side only; not for agent context)
```

### 4.2 Action model

```python
class BioOperatorAction(BaseModel):
    feed_delta_L_h: Literal[-5, 0, 5]
    aeration_delta_vvm: Literal[-0.10, 0.0, 0.10]
    agitation_delta_rpm: Literal[-5, 0, 5]
    reason: Optional[str] = None  # max 200 chars, NOT used for reward at MVP
```

Discrete 27-arm space (3 × 3 × 3). At each step, the env clips to absolute-rate safety limits before applying:

| Variable | Min | Max | Unit |
|---|---|---|---|
| Fs (substrate feed) | 0 | 200 | L/h |
| Fg (aeration) | 20 | 120 | L/h (mapped from vvm × V) |
| RPM | 80 | 200 | — |

PID controllers for pH (acid/base) and temperature (cooling/heating) **remain active and out of agent's reach**. This is correct operator behavior AND the main anti-cheat measure.

### 4.3 Observation model

```python
class BioOperatorObservation(BaseModel):
    time_h: float
    batch_phase: Literal["inoculation", "growth", "production", "stationary"]
    measurements: dict   # online sensors (12-min cadence)
    setpoints_or_limits: dict
    current_controls: dict
    recent_trends: dict  # qualitative labels per variable
    alarm: Optional[str]
    previous_action: Optional[dict]
    offline_lab: Optional[dict]  # only populated every 12h with 4h delay
    instruction: str  # static prompt scaffold
```

#### Measurements (online, every step)
- `temperature_C`, `pH`, `dissolved_oxygen_pct`, `substrate_g_L`, `volume_L`, `OUR`, `CER`, `CO2_outgas_pct`, `O2_outgas_pct`

#### Setpoints/limits (static per task)
- `temperature_target_C`, `pH_target`, `DO_min_safe_pct`, `substrate_max_g_L`, `substrate_min_g_L`

#### Current controls (visible state of the manipulated variables)
- `feed_rate_L_h`, `aeration_rate_vvm`, `agitation_rpm`, `cooling_valve_pct`, `pressure_bar`

#### Recent trends (last 5 steps, qualitative)
- For each of {DO, pH, temperature, substrate}: one of `"rising_fast", "rising", "stable", "falling", "falling_fast"`. Computed from linear regression over the trailing window.

#### Alarms
- Single string when active: `"DO_NEAR_LOW_LIMIT"`, `"S_OVERSHOOT"`, `"TEMP_DRIFT"`, `"PH_DRIFT"`, or `null`.

#### Offline lab (delayed; this is where partial observability lives)
- Every 12 h with 4 h delay: `biomass_g_L_t-4h`, `penicillin_g_L_t-4h`, `PAA_mg_L_t-4h`. NaN otherwise.

#### Instruction (static)
> "You are an operator running a 100,000 L penicillin fermenter. Read the console state, choose the next safe control action for the next 10 minutes. Reply with valid JSON matching the schema."

### 4.4 State (server-side only)
Full ODE state vector + episode metadata. Exposed via `/state` for debugging only — **NOT** included in any prompt the LLM sees. Agent cannot read or modify it.

### 4.5 Episode definition
- Episode start: t = 40 h (production phase entry, just before the first scripted disturbance for the chosen scenario).
- Step interval (control interval): dt = 0.2 h = 12 min sim time. **Matches IndPenSim's native sample rate exactly** — one agent action ↔ one MATLAB simulation sample. Avoids any sub-stepping or aliasing between the agent loop and the calibrated dynamics.
- Max episode length: 50 steps = 10 h sim time.
- Termination conditions:
  - timeout: 50 steps
  - safety violation: DO < 5% sustained for >3 consecutive steps → terminate, large penalty
  - successful recovery: DO ≥ DO_min_safe for last 10 steps AND substrate in band → terminate, success bonus
  - integrator failure: any state goes NaN → terminate, log

---

## 5. Reward Design

### 5.1 The 7 starting components

All clamped to documented ranges. Computed independently. Logged independently.

| # | Name | Range | Formula sketch |
|---|---|---|---|
| 1 | `format_validity` | {0, 1} | 1 if action JSON parses & all values in literal sets; 0 otherwise (penalty −1 in stages 1+) |
| 2 | `do_safety` | [-1, 1] | piecewise: ≥25% → +1; 20–25% → +0.3; 15–20% → −0.5; <15% → −1; <5% → −1 + termination |
| 3 | `productivity` | [0, 1] | normalized ΔP per step, clipped at 95th-percentile of MATLAB reference |
| 4 | `substrate_control` | [-1, 1] | +1 if 0.05 ≤ S ≤ 0.20 g/L; linear ramp to 0 at edges; −1 outside [0, 0.5] |
| 5 | `stability` | [0, 1] | Gaussian on tracking errors of T (sp 25°C), pH (sp 6.5) |
| 6 | `control_effort` | [-1, 0] | penalty: −(0.05·\|feed_delta\| + 0.5·\|aer_delta\| + 0.05·\|rpm_delta\|), normalized |
| 7 | `terminal_yield_bonus` | [0, 1] | sparse, only at episode end: normalized total penicillin produced this episode |

### 5.2 Total reward
```
r_step = 0.05 · format
       + 0.30 · do_safety
       + 0.20 · productivity
       + 0.15 · substrate_control
       + 0.10 · stability
       + 0.10 · control_effort
        (+ 0.10 · terminal_yield_bonus, only on last step)
```

Final per-step reward is clamped to `[-1, +1]`. Total episode reward = sum.

### 5.3 Reserve components (add only on observed failure mode)

| Add when… | Component | Formula |
|---|---|---|
| Sample inspection shows `+5,−5,+5,−5` action chains | `oscillation_penalty` | −0.1 per direction-flip in last 3 steps |
| Trained model produces valid actions but garbage `reason` strings | `reason_coherence` | small LLM-as-judge weight, max 0.05 |
| Agent learns "crash DO once, recover, net positive" | `safety_cooldown` | extra −0.5 per step DO < safe in trailing 10 steps |

Adding any of these requires a new commit + a 1-paragraph README note ("after observing X, we added reward Y").

### 5.4 Anti-hacking checklist (per hack_instructions §8)

- ✅ Multiple independent components (7+)
- ✅ JSON schema validation; invalid actions get format penalty AND default to action `{0,0,0}` (do nothing)
- ✅ Action values clipped to literal sets before applying
- ✅ All manipulated variables clipped to absolute safety limits before integration
- ✅ Hidden ODE state never shown to agent; episode-internal only
- ✅ pH/temperature PID controllers locked behind agent's reach
- ✅ Timeouts (max 50 steps) prevent infinite loops
- ✅ Per-component reward logging — visible reward hacking shows up as one column rising while others collapse
- ✅ During training: log 5 sample completions per 100 steps; manual inspection gate at end of each curriculum stage
- ✅ Reward function code lives in `graders.py` and is **never imported by the agent prompt path** — agent cannot inspect it

---

## 6. Scenarios

| Task ID | Difficulty | What happens at t=40h |
|---|---|---|
| `do-recovery-easy` | easy | small substrate disturbance (Faults=3 mild), DO drops 8 percentage points |
| `do-recovery-medium` | medium | larger substrate fault, DO drops 15 points, recovery requires aeration + feed cut |
| `aeration-limit-hard` | hard | aeration fault (Faults=1) + concurrent disturbance, agent must use feed cut + RPM up |
| `normal-baseline` | dev | no faults, just steady-state operation; for testing reward sanity |

All scenarios re-use the calibrated Python plant; only initial conditions and disturbance schedules vary.

---

## 7. Curriculum

| Stage | Scenarios | Reward weights | Promotion criterion | Steps |
|---|---|---|---|---|
| **0 — Format school** | `normal-baseline` only | format_validity only | ≥90% format-valid over last 50 rollouts | ~200 GRPO updates |
| **1 — Easy DO recovery** | `do-recovery-easy` | full reward, but reduced DO penalty | mean episode reward > baseline-rule-based | ~500 |
| **2 — Productive recovery** | `do-recovery-easy` + `do-recovery-medium` | full reward, full weights | mean reward > untrained-Qwen by ≥0.15 | ~1000 |
| **3 — Multi-fault** | all 3 disturbance scenarios mixed | full reward + safety_cooldown if needed | beats fixed-recipe baseline on success rate | ~1000 |

If compute runs out we ship after Stage 2. If it ALL works fast, Stage 4 (full operator: add Fpaa_delta, dT_sp, dpH_sp).

---

## 8. Training Pipeline

### 8.1 Stack
- **Base model:** Qwen 3 4B-Instruct (default). Promote to Qwen 3 7B if HF GPU credits comfortably allow rollouts in budget. Fallback to Qwen 2.5 1.5B-Instruct if GPU is tight.
- **LoRA via Unsloth:** rank=16, alpha=32, target=`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`. Loaded in 4-bit (QLoRA).
- **Trainer:** `trl.GRPOTrainer`.
- **Logger:** Weights & Biases (W&B). All metrics logged. Run URL goes in README.

### 8.2 Hyperparams (starting point)
```
num_generations (group size G) = 8
learning_rate = 5e-6
max_prompt_length = 2048
max_completion_length = 256
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
beta (KL coef) = 0.04
temperature = 0.7
top_p = 0.95
max_steps = 2000  (across all curriculum stages)
warmup_ratio = 0.05
lr_scheduler = "cosine"
```

### 8.3 Reward function signature for TRL
```python
def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = parse_action(completion)
        # env.step is performed inside the trainer's rollout loop, not here
        # We compute per-component rewards from (action, env_state_at_prompt)
        components = compute_reward_components(action, kwargs["env_state"])
        rewards.append(weighted_sum(components))
    return rewards
```
A custom rollout collator drives the env and feeds `env_state` to the reward function via `kwargs`.

### 8.4 Save path
- Save LoRA adapter only (do **not** merge into 4-bit base). `model.save_pretrained("./checkpoints/qwen3-4b-bioperator-lora")`.
- For inference / demo: load base in 16-bit, attach adapter. Document this clearly in the Colab.
- (Hack instructions §16 explicit warning about merging 4-bit + LoRA naively.)

### 8.5 Compute budget plan
- Single GPU (HF onsite credits, target L40S/A100 40GB).
- Rollout rate target: ≥10 env steps/sec (Python ODE step ≤30 ms; group-of-8 generations dominate).
- Total wall-clock estimate: 4–6 h for full curriculum on 4B; 8–12 h for 7B.
- Decision gate: if Stage 0 doesn't reach format-valid ≥90% in 1 h, we drop model size.

---

## 9. Baselines (six total)

| Baseline | Implementation | Where it lives |
|---|---|---|
| `random_agent` | uniform over the 27 actions | `baselines/random_agent.py` |
| `fixed_recipe` | re-runs the SBC recipe (Fs/Fg/RPM directly from `fctrl_indpensim.m`) — i.e. "do nothing, let the plant do its thing" | `baselines/fixed_recipe_agent.py` |
| `rule_based` | if-then operator rules (DO < threshold → cut feed + bump aeration; S > max → cut feed; etc.) | `baselines/rule_based_agent.py` |
| `untrained_open_llm` | Qwen 3 4B-Instruct, zero-shot, same prompt | `baselines/untrained_qwen_agent.py` |
| `closed_sota_zero_shot` | Claude 4.6 Sonnet via API, zero-shot, same prompt | `baselines/claude_zero_shot_agent.py` |
| `trained_open_llm` | Our trained Qwen 3 4B + LoRA | `baselines/trained_qwen_agent.py` |

All run against the **same set of seeded scenarios** (e.g., 30 episodes per task). Numbers + plots ship in `results/`.

---

## 10. Required Artifacts

### 10.1 Plots (per hack_instructions: readable, labelled, embedded in README)

| File | What it shows | Caption |
|---|---|---|
| `docs/calibration/python_vs_matlab.png` | Overlay of Python port vs MATLAB output on key state variables across 5 batches | "Python port of IndPenSim faithfully reproduces the published industrial simulator. Curves shown for biomass, DO, penicillin, pH, temperature; shaded band = MATLAB reference range." |
| `results/reward_curve.png` | Total reward vs training step | "Total episode reward rises from baseline to plateau over 2000 GRPO steps." |
| `results/per_component_rewards.png` | All 7 reward components, separately, vs training step | "Each reward component improves; productivity and DO-safety drive the headline gain." |
| `results/success_rate_curve.png` | % of episodes reaching successful recovery vs training step | "Success rate rises from ~10% (untrained) to ~60% (trained) on `do-recovery-medium`." |
| `results/safety_violations_curve.png` | DO-floor violations per episode vs training step | "Safety violations drop by ~80% during Stage 2." |
| `results/format_validity_curve.png` | Fraction of completions that parse as valid JSON | "Stage 0 of the curriculum drives format validity to >95% before any process reward is added." |
| `results/baseline_comparison_bar.png` | Mean episode reward, success rate, safety violations across all 6 baselines on `do-recovery-medium` | "Trained Qwen 3 4B beats both untrained Qwen 3 4B and Claude 4.6 zero-shot on safety + productivity." |
| `results/do_recovery_trajectory_comparison.png` | DO vs time, same seed, all 6 agents overlaid | "Untrained agent crashes DO; trained agent recovers cleanly." |
| `results/action_trajectory_comparison.png` | Feed/Aer/RPM deltas vs time, untrained vs trained | "Trained agent reduces feed first, then bumps aeration; untrained agent fixates on feed-up." |
| `results/before_after_action_examples.md` | ~3 sample prompt+completion pairs, untrained vs trained, with `reason` string | n/a (text) |

### 10.2 Notebooks
- `notebooks/01_env_smoke_test.ipynb` — reset/step/reward sanity loop with random agent
- `notebooks/02_baselines.ipynb` — runs all 6 baselines on all scenarios, dumps result CSVs and plots
- `notebooks/03_train_grpo.ipynb` — Colab-ready training notebook (Unsloth + TRL GRPO). Judges should be able to open + run.
- `notebooks/04_demo.ipynb` — loads trained adapter, runs trained-vs-untrained on a fixed seed, generates the comparison plots

### 10.3 Video / mini-blog
- `<2 min YouTube video` linked from README.
- Script structure: problem (20s) → environment + console (40s) → training curve (20s) → before/after demo (40s).
- Backup: HF mini-blog with same content.

### 10.4 README
Plain language. ~600 words target. Sections:
1. **The problem (no jargon).** "Bioreactors brew medicines. They're alive, expensive, and unforgiving. We can't let AI agents practice on real ones."
2. **The simulator.** "We took a published industrial simulator (originally MATLAB) and ported it to Python. Here's the proof it matches" → calibration plot.
3. **The console.** Screenshot of an example observation with one-line annotations of each field.
4. **The agent's job.** "Pick one of 27 control actions every 10 simulated minutes. Output JSON. Don't kill the cells."
5. **The reward.** Table of 7 components in plain English. "Each one watches a different thing so the model can't game one."
6. **Training.** "We start from Qwen 3 (a small open AI). We use GRPO (a recent RL algorithm). LoRA (so it fits on one GPU). Unsloth (so it's fast)."
7. **Results.** Embed the 7 key plots with captions.
8. **Baselines.** Bar chart with one-line takeaway.
9. **Try it.** HF Space link, Colab link, repo link.
10. **Limitations.** "This is one specific bioprocess. Each new process needs its own plant adapter. Real deployment requires far more validation."
11. **Links.** Video, blog, slides, W&B run.

---

## 11. Repo Structure

```
meta_env/
├─ README.md                              # the plain-language story
├─ openenv.yaml                           # OpenEnv manifest (latest spec)
├─ Dockerfile                             # HF Space deployment
├─ requirements.txt
├─ pyproject.toml
│
├─ bioperator_env/                        # the package (importable)
│   ├─ __init__.py
│   ├─ models.py                          # pydantic Action/Observation/State
│   ├─ env.py                             # BioOperatorEnv class (reset/step/state)
│   ├─ scenarios.py                       # task definitions + disturbance schedules
│   ├─ rewards.py                         # 7 reward components + composer
│   ├─ trends.py                          # qualitative trend labels
│   ├─ alarms.py                          # alarm rules
│   ├─ prompt.py                          # prompt template
│   └─ plant/
│       ├─ __init__.py
│       ├─ ode.py                         # the 33 ODEs (port of indpensim_ode.m)
│       ├─ params.py                      # 105-param vector (port of Parameter_list.m)
│       ├─ controllers.py                 # pH and T PIDs (port of fctrl + PIDSimple3)
│       ├─ disturbances.py                # 8 disturbance channels
│       └─ recipe.py                      # SBC recipes (the fixed baseline)
│
├─ server/
│   ├─ app.py                             # FastAPI server
│   └─ openenv_compat.py                  # OpenEnv Environment base class wiring
│
├─ baselines/
│   ├─ random_agent.py
│   ├─ fixed_recipe_agent.py
│   ├─ rule_based_agent.py
│   ├─ untrained_qwen_agent.py
│   ├─ claude_zero_shot_agent.py
│   └─ trained_qwen_agent.py
│
├─ training/
│   ├─ grpo_train.py                      # main training entrypoint
│   ├─ rollout.py                         # env-driven rollout collator
│   ├─ reward_fn.py                       # TRL-compatible reward callable
│   └─ inspect_generations.py             # mid-training sample dump
│
├─ notebooks/
│   ├─ 01_env_smoke_test.ipynb
│   ├─ 02_baselines.ipynb
│   ├─ 03_train_grpo.ipynb
│   └─ 04_demo.ipynb
│
├─ docs/
│   ├─ calibration/
│   │   ├─ python_vs_matlab.png
│   │   └─ calibration_report.md
│   └─ superpowers/specs/
│       └─ 2026-04-26-bioperatorenv-design.md   ← this file
│
├─ results/
│   ├─ reward_curve.png
│   ├─ per_component_rewards.png
│   ├─ success_rate_curve.png
│   ├─ safety_violations_curve.png
│   ├─ format_validity_curve.png
│   ├─ baseline_comparison_bar.png
│   ├─ do_recovery_trajectory_comparison.png
│   ├─ action_trajectory_comparison.png
│   └─ before_after_action_examples.md
│
├─ scripts/
│   ├─ run_baselines.py
│   ├─ generate_plots.py
│   ├─ calibrate_against_matlab.py        # uses IndPenSim/output_5/
│   └─ smoke_test.sh
│
├─ tests/
│   ├─ test_models.py
│   ├─ test_plant_calibration.py          # asserts Python port stays inside MATLAB band
│   ├─ test_rewards.py                    # each component exercised on synthetic states
│   ├─ test_env_loop.py                   # reset/step roundtrip
│   └─ test_anti_hacking.py               # invalid JSON, out-of-range vals, hidden state leaks
│
├─ checkpoints/
│   └─ qwen3-4b-bioperator-lora/          # LoRA adapter only (post-training)
│
└─ IndPenSim/                             # MATLAB source + Octave reference
    └─ output_5/                          # already populated
```

The Round 1 artifacts (`bioreactor_env.py`, `tasks.py`, `graders.py`, `models.py`, `server/app.py`, `inference.py`) are **deleted** — Round 2 is a clean rebuild. The HF Space gets a force-push.

---

## 12. Deployment

- HF Space, Docker SDK, `app_port: 7860`. Push from local repo.
- `Dockerfile`: Python 3.11 base, install requirements, expose 7860, run uvicorn.
- HF Space README front-matter (already known to be required: `sdk: docker`, `app_port: 7860`).
- Colab notebook `03_train_grpo.ipynb` is the "minimum training script" the rubric demands. Self-contained: clones repo, installs Unsloth + TRL, runs 200 demo steps to prove the loop, then loads our pre-trained adapter for the demo.

---

## 13. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Python ODE port doesn't match MATLAB | medium | Calibration test; tune parameter list; if still off, use SciPy LSODA (closer to MATLAB ode15s) |
| GRPO training stalls (zero reward) | medium | Stage 0 format-school curriculum; SFT warm-up fallback (synthesize ~50 valid actions, run 1 epoch SFT before GRPO) |
| Rollout speed too slow on the GPU we get | medium | drop to Qwen 1.5B; reduce group size from 8 to 4; skip Stage 3 |
| HF Space build fails | low | early deploy gate (Phase 4 of the plan); test container locally with `docker run` before push |
| Closed-SOTA baseline (Claude/GPT) too good and trivializes story | low | report it honestly; emphasize "we trained a 4B to beat untrained 4B" as the core result; closed-SOTA framed as "expensive ceiling" not "comparison" |
| Reward hacking detected late | medium | mid-training inspection at end of every curriculum stage; if found, add reserve component + retrain from last checkpoint |
| MATLAB output_5 has missing/garbled columns | low (already verified — 35 columns, 5751 rows, sane yields) | use the columns we have; biomass online not present, will be inferred or pulled from offline lag |

---

## 14. Acceptance Criteria

The submission is "ship-ready" when ALL of:

- [ ] `python -m pytest tests/` green
- [ ] Calibration overlay plot exists and is honest (we report deviations if any)
- [ ] HF Space `/reset` returns 200 and `/step` returns valid Observation
- [ ] `openenv validate --verbose` passes
- [ ] All 6 baselines have results CSVs + bar chart
- [ ] At least one trained adapter checkpoint exists, with reward curves
- [ ] Trained agent measurably beats untrained Qwen on at least success_rate AND safety_violations on `do-recovery-medium`
- [ ] All 9 plot files exist in `results/` and are embedded in README with captions
- [ ] Colab notebook `03_train_grpo.ipynb` runs end-to-end on a fresh runtime
- [ ] README is ≤700 words, plain language, non-bio-readable
- [ ] 2-min video posted, linked in README
- [ ] W&B run linked in README

---

## 15. Open questions / deferred decisions

- Final base model size (4B vs 7B): decided onsite based on observed GPU.
- Whether to add a Stage 4 (full operator with Fpaa/setpoint actions): decided after Stage 2 results.
- Whether to add `reason_coherence` reward: decided after first sample inspection at end of Stage 1.
- Exact prompt scaffold wording: tuned after Stage 0 format-rate measurement.

---

**End of design spec.**
