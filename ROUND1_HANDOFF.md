# Round 1 Handoff

This file is the full context dump for the `openenv-bioreactor` submission as it exists after the final Round 1 resubmission.

Use this later if:
- you need to remember what was built
- you get selected for the next round
- you want to explain the project to someone else
- you want to keep improving without re-reading the whole codebase from scratch

## 1. Final Submission Status

At the end of this round, the project satisfied the required pass/fail checks:

- Hugging Face Space deployed and responded
- `POST /reset` returned HTTP `200`
- Docker image built successfully
- `openenv validate` passed
- repo contained 3 tasks with graders
- rewards and scores were normalized to `[0, 1]`
- root `inference.py` existed and used the `OpenAI` client

The submission had already cleared Phase 1 and Phase 2 on an earlier version, and then was upgraded and re-pushed with stricter inference formatting and a stronger environment design.

## 2. What The Project Is

This is an OpenEnv-compatible RL environment for bioreactor control.

The benchmark simulates a simplified fermentation process where an agent must:
- maintain dissolved oxygen
- maintain mixing quality
- manage nutrient depletion
- grow biomass
- prevent overflow byproduct formation
- avoid foam risk
- avoid shear damage from excessive stirring
- finish the batch in a good terminal operating state

The core idea is that this should feel more like a real process-control and bioprocess optimization problem than a toy setpoint tracker.

## 3. File Overview

Main files:

- `tasks.py`
  - task definitions
  - task-specific operating regimes
  - initial conditions
  - target setpoints
  - terminal objectives
  - disturbances

- `bioreactor_env.py`
  - main environment implementation
  - reset/step/state logic
  - dynamics update
  - action handling
  - task application

- `graders.py`
  - per-step reward logic
  - final trajectory scoring
  - phase summaries

- `models.py`
  - typed Pydantic models
  - action/observation/state schemas

- `server/app.py`
  - FastAPI app
  - `/`
  - `/health`
  - `/tasks`
  - `/reset`
  - `/step`
  - `/state`

- `openenv.yaml`
  - OpenEnv metadata
  - task list
  - observation/action/reward references

- `inference.py`
  - root baseline script
  - reads required env vars
  - uses OpenAI client
  - logs `[START]`, `[STEP]`, `[END]`

- `Dockerfile`
  - deployment image for HF Space

- `README.md`
  - public description of benchmark and usage

- `SUBMISSION_INSTRUCTIONS.md`
  - exact operational checklist for pushing and validating

- `ROUND1_HANDOFF.md`
  - this file

## 4. Evolution Of The Environment

### Version 1

The first version was a minimal oxygen/mixing/nutrient controller:
- observation: oxygen, mixing, nutrient
- actions: increase/decrease stirrer, increase/decrease oxygen, do nothing
- simple reward around target oxygen/mixing/nutrient
- 3 difficulty tiers

This version was enough to pass structural and deep validation.

### Why It Was Not Strong Enough To Leave Alone

It was compliant, but still too close to a simple setpoint controller.

Weaknesses:
- not enough process tradeoff depth
- no explicit production objective
- no biomass target
- no terminal production outcome
- feed behavior was not controllable by the agent

### Version 2

The environment was upgraded to include:
- biomass concentration
- byproduct load
- feed-rate-driven process phases
- foam risk
- shear damage
- phase summaries for startup/growth/stress

This made the environment more like real fermentation control.

### Version 3

The final upgrade added:
- explicit feed control in the action space
- task-specific terminal objectives
- stricter inference formatting to match the provided sample more closely

This is the final Round 1 version.

## 5. Final Action Space

The final action space is:

- `0` = increase stirrer speed
- `1` = decrease stirrer speed
- `2` = increase oxygen input
- `3` = decrease oxygen input
- `4` = do nothing
- `5` = increase feed rate
- `6` = decrease feed rate

Reason for this upgrade:
- feed is a real control lever in fermentation
- controlling feed makes the benchmark much more interesting
- it creates a meaningful tradeoff between growth and metabolic overflow

## 6. Final Observation Space

The final observation includes:

- `oxygen_level`
- `mixing_uniformity`
- `nutrient_concentration`
- `biomass_concentration`
- `byproduct_load`
- `feed_rate`

Also exposed:
- task metadata
- target values
- terminal objective values
- reward
- score
- done

## 7. Hidden / Server-Side State

The environment also tracks:

- `stirrer_speed`
- `oxygen_input`
- `feed_rate`
- `foam_risk`
- `shear_damage`
- `cumulative_reward`
- `phase_scores`

These are kept in the environment state and exposed through `/state`.

## 8. Final Task Design

There are 3 tasks:

### `startup-stabilization-easy`

Intent:
- recover after inoculation
- grow clean biomass
- keep the reactor in a stable productive band

This is the easiest scenario and is meant to be recoverable with straightforward control.

### `fed-batch-optimization-medium`

Intent:
- survive and exploit feed pulses
- keep growth high
- avoid overflow metabolite accumulation

This is more difficult because growth and risk rise together.

### `oxygen-limited-recovery-hard`

Intent:
- handle high-density fermentation near oxygen limitation
- recover from disturbances
- finish with strong biomass and acceptable quality

This is the hardest scenario because viscosity, oxygen demand, and terminal objectives all matter.

## 9. Reward Logic

Per-step reward in `graders.py` includes:

- oxygen tracking
- mixing tracking
- nutrient balance
- biomass production score
- byproduct purity score
- safety score
- growth bonus
- penalties for actuator intensity and foam

Everything is clamped into `[0, 1]`.

The reward is shaped to give meaningful partial progress, not just terminal success/failure.

## 10. Final Score Logic

Final trajectory score includes:

- average reward
- final biomass achievement
- terminal objective score
- fraction of time in a safe operating region
- efficiency term
- survival / episode completion
- collapse penalty

This was important because the judging criteria emphasized:
- meaningful reward shaping
- non-trivial graders
- scores in `[0, 1]`

## 11. Terminal Objectives

One of the last major upgrades was adding terminal objectives.

Each task now defines:
- `terminal_biomass_target`
- `terminal_byproduct_limit`
- `terminal_nutrient_range`
- `terminal_oxygen_floor`

Why this matters:
- it stops the benchmark from being only about local stepwise control
- it forces trajectory planning toward a good end-of-batch state
- it makes the environment feel more like a production process benchmark

## 12. Phase Summaries

Another later upgrade was phase-aware grading.

Trajectory points are divided into:
- `startup`
- `growth`
- `stress`

The environment computes phase summaries containing:
- average reward
- average biomass
- safety summary

These are exposed in:
- `env.phase_scores`
- `/step` response `info`
- `/state`

Why this was added:
- more interpretable grading
- more useful debugging
- better benchmark story for later rounds

## 13. Baseline Inference Script

The baseline script is `inference.py`.

It was later tightened to match the sample spec more closely.

### Required environment variables

Present in the script:

- `LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")`
- `API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")`
- `MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")`
- `HF_TOKEN = os.getenv("HF_TOKEN")`

Notes:
- defaults are only set for `API_BASE_URL` and `MODEL_NAME`
- `HF_TOKEN` has no default
- script uses `OpenAI`

### Logging format

The script prints:

- `[START] task=<...> env=<...> model=<...>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,...>`

Later changes to `inference.py` were specifically made because the sample script formatting looked stricter than our earlier version.

### Memory upgrade

The baseline prompt was improved to include:
- previous action
- previous reward
- previous state snippets

This makes the baseline more trajectory-aware without adding local model compute.

## 14. Why The Environment Still Fits Hardware Limits

The evaluator machine budget was:
- `2 vCPU`
- `8 GB RAM`

This repo should fit comfortably because:
- no model is hosted locally
- all LLM calls go to external APIs
- no RL training loop
- no CFD
- no heavy solvers
- short episodes
- tiny in-memory state
- FastAPI + Python only

The runtime cost is dominated by:
- web server overhead
- lightweight math
- external API latency during inference

## 15. Validation History

What was explicitly validated during development:

- syntax via `python3 -m compileall`
- local environment smoke tests
- `openenv validate --verbose`
- Docker build
- HF Space `/reset` returns `200`
- provided `validate-submission.sh` passed all `3/3` checks

Important outcome:
- the repo was in a fully passing validator state at the end

## 16. Hugging Face Space Notes

The Space was configured as a Docker Space.

Important fix made during setup:
- Hugging Face needed front matter at the top of `README.md`

Added metadata:
- `sdk: docker`
- `app_port: 7860`
- title / emoji / colors

Without that, HF showed a README config error.

## 17. Submission / Judging Context

The first submitted version already got an email confirming:
- Phase 1 passed
- Phase 2 passed
- officially in judging

This means the project was already structurally sound before the later improvements.

The later changes were focused on making the benchmark more competitive, not just more compliant.

## 18. Most Important Strategic Improvements We Made

If someone asks what made the project stronger, the answer is:

1. moved from setpoint control to process optimization
2. added biomass and byproduct
3. added feed as an explicit control lever
4. added safety mechanisms like foam and shear
5. added terminal objectives
6. added phase-aware grading
7. tightened inference formatting for evaluator compatibility

## 19. What Was Probably The Highest-Value Improvement

The single most important upgrade was:

**making feed rate an explicit action and tying final score to end-of-batch objectives**

Why:
- it introduced real delayed tradeoffs
- it made the environment less toy-like
- it made task success depend on trajectory quality, not just staying near a target each step

## 20. If Selected For The Next Round

If this project moves forward, the best next-round directions are:

### A. Better baselines

- rule-based controller baseline
- stronger LLM prompting baseline
- maybe compare several prompting strategies

### B. Better realism

- explicit product quality metric
- pH / temperature coupling
- oxygen transfer coefficient proxy
- dynamic batch phase switching

### C. Better evaluation

- benchmark cards with expected failure modes
- ablations of what each action dimension contributes
- stronger documentation of why this matters for agent evaluation

### D. Better presentation

- diagrams of state/action/reward flow
- examples of good vs bad trajectories
- benchmark motivation framed for RL + agentic control researchers

## 21. Known Non-Issues

Things that looked suspicious but were okay:

- HF warning about missing repo card metadata
  - fixed by adding YAML front matter to `README.md`

- old startup timestamp in HF logs
  - not the best signal for latest deployment
  - commit SHA in build logs was the better proof

- `RequestsDependencyWarning` during local validation
  - validator still passed
  - not a blocker

## 22. Things To Be Careful About Later

- do not casually break `inference.py` formatting
- do not remove root `inference.py`
- do not move `openenv.yaml`
- do not add heavyweight dependencies without checking runtime budget
- do not change task IDs unless the metadata, docs, and inference flow are updated together

## 23. Final Practical Summary

This repo ended Round 1 as:

- validator-passing
- deployable
- HF-live
- stronger than the first accepted version
- still lightweight
- more realistic than a simple setpoint benchmark

If you read only one sentence from this file later, read this:

**The final benchmark is a lightweight but richer bioprocess-control environment where the agent explicitly manages oxygen, mixing, and feed to maximize biomass while minimizing byproduct, foam, and shear, with task-specific terminal objectives and phase-aware scoring.**
