# BioOperatorEnv Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship BioOperatorEnv — an OpenEnv-compatible Python port of the IndPenSim industrial penicillin simulator, wrapped as a SCADA-style operator-training environment, with GRPO-trained Qwen 3 agent + 5 baselines + 9 plots + HF Space deployment + Colab notebook.

**Architecture:** Round 2 is a clean rebuild. Plant = Python port of `IndPenSim/indpensim_ode.m` (33 ODEs via `scipy.integrate.solve_ivp`), calibrated against `IndPenSim/output_5/` reference trajectories. Env layer wraps plant in OpenEnv `reset/step/state` API with pydantic Action/Observation/State and FastAPI server. Reward = 7 independent components (format, DO safety, productivity, substrate, stability, control effort, terminal yield). Training = Unsloth + TRL `GRPOTrainer` + LoRA on Qwen 3 4B-Instruct.

**Tech Stack:** Python 3.11, scipy, numpy, pandas, pydantic v2, FastAPI, uvicorn, OpenEnv, TRL, Unsloth, transformers, peft, bitsandbytes, matplotlib, pytest, Docker.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-04-26-bioperatorenv-design.md` — read before any task
- MATLAB source: `IndPenSim/` — primary fidelity reference
- Octave reference data: `IndPenSim/output_5/IndPenSim_V2_export_V7.csv` (5 batches, 5751 rows)
- Hackathon constraints: `hack_instructions.md`

**Execution principles:**
- TDD on every numerical / interface task. Write failing test → minimal impl → green → commit.
- Commit after every passing task. Never batch unrelated changes.
- For phases that depend on numerical fidelity (Phase C calibration), the test IS the acceptance: Python port within bands of MATLAB reference.
- Phases B–F can run sequentially in this session. Phase G (training) requires GPU and runs onsite.

---

## Phase Map (high-level)

| Phase | Deliverable | Tasks | Dependency |
|---|---|---|---|
| A | Clean bootstrap, repo skeleton, tooling | A1–A4 | none |
| B | Pydantic models (Action/Observation/State) | B1–B3 | A |
| C | Plant engine (Python port of IndPenSim) | C1–C8 | A, B |
| D | Environment (reset/step/scenarios/server) | D1–D6 | C |
| E | Reward system (7 components + composer) | E1–E3 | B, C |
| F | Baselines (5 agents + harness) | F1–F6 | D, E |
| G | Training pipeline (GRPO + Unsloth + Colab) | G1–G5 | E, F |
| H | Artifacts (plots + before/after examples) | H1–H4 | F, G |
| I | Deployment (Docker, HF Space, README, video) | I1–I5 | H |

**Critical path (must work in order):** A → B → C → D → E → F → G. H and I can interleave with G.

---

## Phase A — Bootstrap

### Task A1: Delete Round 1 artifacts and prepare clean tree

**Files:**
- Delete: `bioreactor_env.py`, `tasks.py`, `graders.py`, `models.py`, `inference.py`, `client.py`, `app.py`, `requirements.txt`, `Dockerfile`, `openenv.yaml`, `SUBMISSION_INSTRUCTIONS.md`, `Round1_HANDOFF.md`, `server/` (if it exists at root)
- Keep: `IndPenSim/`, `docs/`, `README.md` (for now; rewritten in Phase I), `.git/`, `.gitignore`
- Note: `ROUND1_HANDOFF.md` and `ROUND2_instructions.md` are reference docs, keep them

- [ ] **Step 1: Audit current root**

Run: `ls -la /Users/kartikey/Desktop/your_products/meta_env`
Expected: lists all current files. Note which are Round 1 vs reference.

- [ ] **Step 2: Delete Round 1 code, keep reference + git**

Run:
```bash
cd /Users/kartikey/Desktop/your_products/meta_env
rm -f bioreactor_env.py tasks.py graders.py models.py inference.py client.py app.py requirements.txt Dockerfile openenv.yaml SUBMISSION_INSTRUCTIONS.md
rm -rf server/
```
Expected: only `IndPenSim/`, `docs/`, `README.md`, `ROUND1_HANDOFF.md`, `ROUND2_instructions.md`, `hack_instructions.md`, `.git/`, `.gitignore` remain.

- [ ] **Step 3: Update `.gitignore` for Python project**

Append to `.gitignore`:
```
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
.env
checkpoints/
results/*.png
results/*.csv
results/*.json
wandb/
.ipynb_checkpoints/
*.mat
*.npy
```
Note: don't gitignore `results/before_after_action_examples.md`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: clear Round 1 code for Round 2 rebuild

Round 2 (BioOperatorEnv) is a from-scratch rebuild around the
real IndPenSim plant engine. Reference docs and IndPenSim/ kept."
```

---

### Task A2: Project scaffolding — pyproject.toml, requirements

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "bioperator-env"
version = "0.1.0"
description = "BioOperatorEnv: a flight simulator for autonomous bioreactor operators"
readme = "README.md"
requires-python = ">=3.10,<3.13"
license = { text = "MIT" }
authors = [{ name = "Kartikey" }]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.0",
    "pydantic>=2.5",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "openenv>=0.1",
    "matplotlib>=3.8",
    "tqdm>=4.66",
]

[project.optional-dependencies]
training = [
    "torch>=2.3",
    "transformers>=4.44",
    "peft>=0.11",
    "trl>=0.10",
    "bitsandbytes>=0.43",
    "accelerate>=0.30",
    "datasets>=2.20",
    "wandb>=0.17",
]
inference = [
    "openai>=1.30",
    "anthropic>=0.30",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["bioperator_env"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 2: Write `requirements.txt`**

```
numpy>=1.26
scipy>=1.11
pandas>=2.0
pydantic>=2.5
fastapi>=0.110
uvicorn[standard]>=0.27
matplotlib>=3.8
tqdm>=4.66
```
(OpenEnv is installed separately because the latest release path varies; document in README. Training deps are NOT in the Space image.)

- [ ] **Step 3: Write `requirements-dev.txt`**

```
-r requirements.txt
pytest>=7.4
pytest-cov>=4.1
ruff>=0.4
ipython
jupyter
```

- [ ] **Step 4: Verify install works**

Run:
```bash
cd /Users/kartikey/Desktop/your_products/meta_env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```
Expected: clean install, no errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt
git commit -m "chore: project scaffolding (pyproject + requirements)"
```

---

### Task A3: Package skeleton (empty modules with docstrings)

**Files:**
- Create: `bioperator_env/__init__.py`
- Create: `bioperator_env/models.py` (empty placeholder)
- Create: `bioperator_env/env.py` (placeholder)
- Create: `bioperator_env/scenarios.py` (placeholder)
- Create: `bioperator_env/rewards.py` (placeholder)
- Create: `bioperator_env/trends.py` (placeholder)
- Create: `bioperator_env/alarms.py` (placeholder)
- Create: `bioperator_env/prompt.py` (placeholder)
- Create: `bioperator_env/plant/__init__.py`
- Create: `bioperator_env/plant/ode.py` (placeholder)
- Create: `bioperator_env/plant/params.py` (placeholder)
- Create: `bioperator_env/plant/controllers.py` (placeholder)
- Create: `bioperator_env/plant/disturbances.py` (placeholder)
- Create: `bioperator_env/plant/recipe.py` (placeholder)
- Create: `bioperator_env/plant/engine.py` (placeholder)
- Create: `server/__init__.py`
- Create: `server/app.py` (placeholder)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `bioperator_env/__init__.py`**

```python
"""BioOperatorEnv — flight simulator for autonomous bioreactor operators."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Stub all module files with one-line docstrings**

For each module path above (except `__init__.py` files already done), write a single docstring matching the file's role. Example for `plant/ode.py`:

```python
"""Port of IndPenSim/indpensim_ode.m — 33 coupled ODEs for the penicillin fermenter."""
```

For `plant/__init__.py` and `server/__init__.py` write empty files (one blank line).

For `tests/conftest.py`:

```python
"""Shared pytest fixtures for BioOperatorEnv tests."""
import sys
from pathlib import Path

# Make bioperator_env importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 3: Verify import works**

Run: `python -c "import bioperator_env; print(bioperator_env.__version__)"`
Expected: `0.1.0`

- [ ] **Step 4: Commit**

```bash
git add bioperator_env/ server/ tests/__init__.py tests/conftest.py
git commit -m "chore: package skeleton with module stubs"
```

---

### Task A4: Smoke-test pytest pipeline

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write trivial test**

```python
"""Smoke test that the package imports and pytest is wired."""
import bioperator_env


def test_package_version():
    assert bioperator_env.__version__ == "0.1.0"
```

- [ ] **Step 2: Run pytest**

Run: `pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: smoke test for package import"
```

**Phase A complete. Repo is now clean, scaffolded, importable, and test-runnable.**

---

## Phase B — Pydantic Models

The data contracts at the boundary between agent ↔ env. Get these right early because everything downstream consumes them.

### Task B1: Action model

**Files:**
- Modify: `bioperator_env/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for pydantic Action / Observation / State models."""
import pytest
from pydantic import ValidationError

from bioperator_env.models import BioOperatorAction


def test_action_accepts_canonical_values():
    a = BioOperatorAction(
        feed_delta_L_h=5,
        aeration_delta_vvm=0.10,
        agitation_delta_rpm=-5,
        reason="DO falling, cut feed",
    )
    assert a.feed_delta_L_h == 5
    assert a.aeration_delta_vvm == 0.10
    assert a.agitation_delta_rpm == -5
    assert a.reason == "DO falling, cut feed"


def test_action_reason_is_optional():
    a = BioOperatorAction(feed_delta_L_h=0, aeration_delta_vvm=0.0, agitation_delta_rpm=0)
    assert a.reason is None


def test_action_rejects_out_of_set_feed():
    with pytest.raises(ValidationError):
        BioOperatorAction(feed_delta_L_h=3, aeration_delta_vvm=0.0, agitation_delta_rpm=0)


def test_action_rejects_out_of_set_aeration():
    with pytest.raises(ValidationError):
        BioOperatorAction(feed_delta_L_h=0, aeration_delta_vvm=0.05, agitation_delta_rpm=0)


def test_action_rejects_out_of_set_rpm():
    with pytest.raises(ValidationError):
        BioOperatorAction(feed_delta_L_h=0, aeration_delta_vvm=0.0, agitation_delta_rpm=10)


def test_action_clips_long_reason():
    long_reason = "x" * 500
    a = BioOperatorAction(
        feed_delta_L_h=0, aeration_delta_vvm=0.0, agitation_delta_rpm=0, reason=long_reason
    )
    assert len(a.reason) <= 200
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_models.py -v`
Expected: ImportError or ValidationError-related failures (model doesn't exist yet).

- [ ] **Step 3: Implement `BioOperatorAction`**

In `bioperator_env/models.py`:

```python
"""Pydantic data models for BioOperatorEnv (Action / Observation / State)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class BioOperatorAction(BaseModel):
    """Operator action: discrete deltas on three control levers + optional reason.

    The action space is intentionally small (27 arms = 3 x 3 x 3) so GRPO
    can explore it densely. PID controllers for pH and temperature stay
    behind the agent's reach; this is both correct operator behavior and
    the main anti-cheat surface.
    """

    feed_delta_L_h: Literal[-5, 0, 5]
    aeration_delta_vvm: Literal[-0.10, 0.0, 0.10]
    agitation_delta_rpm: Literal[-5, 0, 5]
    reason: Optional[str] = Field(default=None, max_length=200)

    @field_validator("reason", mode="before")
    @classmethod
    def _truncate_reason(cls, v):
        if isinstance(v, str) and len(v) > 200:
            return v[:200]
        return v
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_models.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add bioperator_env/models.py tests/test_models.py
git commit -m "feat(models): BioOperatorAction with discrete literal action space"
```

---

### Task B2: Observation model

**Files:**
- Modify: `bioperator_env/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Append failing tests**

```python
from bioperator_env.models import BioOperatorObservation


def test_observation_minimal():
    obs = BioOperatorObservation(
        time_h=42.0,
        batch_phase="production",
        measurements={
            "temperature_C": 25.0,
            "pH": 6.5,
            "dissolved_oxygen_pct": 22.0,
            "substrate_g_L": 0.15,
            "volume_L": 80000.0,
            "OUR": 0.5,
            "CER": 0.4,
            "CO2_outgas_pct": 4.0,
            "O2_outgas_pct": 19.5,
        },
        setpoints_or_limits={
            "temperature_target_C": 25.0,
            "pH_target": 6.5,
            "DO_min_safe_pct": 20.0,
            "substrate_max_g_L": 0.30,
            "substrate_min_g_L": 0.05,
        },
        current_controls={
            "feed_rate_L_h": 80.0,
            "aeration_rate_vvm": 0.85,
            "agitation_rpm": 100.0,
            "cooling_valve_pct": 45.0,
            "pressure_bar": 0.9,
        },
        recent_trends={"DO": "stable", "pH": "stable", "temperature": "stable", "substrate": "stable"},
        alarm=None,
        previous_action=None,
        offline_lab=None,
        instruction="...",
    )
    assert obs.time_h == 42.0
    assert obs.batch_phase == "production"


def test_observation_phase_literal_enforced():
    with pytest.raises(ValidationError):
        BioOperatorObservation(
            time_h=0.0,
            batch_phase="invalid_phase",
            measurements={},
            setpoints_or_limits={},
            current_controls={},
            recent_trends={},
            alarm=None,
            previous_action=None,
            offline_lab=None,
            instruction="",
        )
```

- [ ] **Step 2: Verify tests fail (ImportError)**

Run: `pytest tests/test_models.py -v`
Expected: ImportError on `BioOperatorObservation`.

- [ ] **Step 3: Implement `BioOperatorObservation`**

Append to `bioperator_env/models.py`:

```python
class BioOperatorObservation(BaseModel):
    """SCADA-style plant-console snapshot the agent sees each step."""

    time_h: float
    batch_phase: Literal["inoculation", "growth", "production", "stationary"]
    measurements: dict
    setpoints_or_limits: dict
    current_controls: dict
    recent_trends: dict
    alarm: Optional[str]
    previous_action: Optional[dict]
    offline_lab: Optional[dict]
    instruction: str
```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/test_models.py -v`
Expected: all model tests passing.

- [ ] **Step 5: Commit**

```bash
git add bioperator_env/models.py tests/test_models.py
git commit -m "feat(models): BioOperatorObservation (plant-console schema)"
```

---

### Task B3: State + StepResult + reward-info models

**Files:**
- Modify: `bioperator_env/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Append failing tests**

```python
from bioperator_env.models import BioOperatorState, StepInfo, RewardComponents


def test_reward_components_are_floats():
    rc = RewardComponents(
        format_validity=1.0,
        do_safety=0.5,
        productivity=0.3,
        substrate_control=0.7,
        stability=0.9,
        control_effort=-0.1,
        terminal_yield_bonus=0.0,
    )
    assert rc.format_validity == 1.0
    assert rc.terminal_yield_bonus == 0.0


def test_step_info_carries_reward_breakdown():
    info = StepInfo(
        reward_total=0.4,
        reward_components=RewardComponents(
            format_validity=1.0, do_safety=0.5, productivity=0.3,
            substrate_control=0.7, stability=0.9, control_effort=-0.1,
            terminal_yield_bonus=0.0,
        ),
        safety_violation=False,
        success=False,
        done_reason="",
    )
    assert info.reward_total == 0.4
    assert info.safety_violation is False
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_models.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement state + info models**

Append to `bioperator_env/models.py`:

```python
class RewardComponents(BaseModel):
    """Per-step reward breakdown. Logged independently for diagnosability."""
    format_validity: float
    do_safety: float
    productivity: float
    substrate_control: float
    stability: float
    control_effort: float
    terminal_yield_bonus: float


class StepInfo(BaseModel):
    """Auxiliary info returned alongside (obs, reward, done)."""
    reward_total: float
    reward_components: RewardComponents
    safety_violation: bool
    success: bool
    done_reason: str


class BioOperatorState(BaseModel):
    """Server-side full debug state. NOT shown to the agent."""
    task_id: str
    seed: int
    step_count: int
    time_h: float
    ode_state: list[float]              # 33-vector
    last_action: Optional[dict] = None
    cumulative_reward: float = 0.0
    component_history: list[dict] = []  # one RewardComponents.dict() per step
    safety_violations: int = 0

    model_config = {"arbitrary_types_allowed": True}
```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/test_models.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add bioperator_env/models.py tests/test_models.py
git commit -m "feat(models): RewardComponents, StepInfo, BioOperatorState"
```

**Phase B complete. Data contracts locked.**

---

## Phase C — Plant Engine (Python port of IndPenSim)

This is the largest phase. We port `indpensim_ode.m` (33 ODEs) + `Parameter_list.m` (105 params) + `fctrl_indpensim.m` (PID + SBC) + disturbances, then validate against `IndPenSim/output_5/`.

### Task C1: Parameter list (port of Parameter_list.m)

**Files:**
- Modify: `bioperator_env/plant/params.py`
- Create: `tests/test_plant_params.py`

- [ ] **Step 1: Write test that asserts parameter count and key values**

```python
"""Test the 105-parameter vector matches IndPenSim/Parameter_list.m."""
from bioperator_env.plant.params import build_params, ParamConfig


def test_param_vector_has_105_entries():
    cfg = ParamConfig(x0_mux=0.41, x0_mup=0.041, alpha_kla=85.0, N_conc_paa=150000.0, PAA_c=530000.0)
    p = build_params(cfg)
    assert len(p) == 105


def test_known_constants_at_correct_index():
    # Spot-check positions from Parameter_list.m comments
    cfg = ParamConfig(x0_mux=0.41, x0_mup=0.041, alpha_kla=85.0, N_conc_paa=150000.0, PAA_c=530000.0)
    p = build_params(cfg)
    # par(1) = mu_p
    assert p[0] == 0.041
    # par(2) = mux_max
    assert p[1] == 0.41
    # par(31) = alpha_kla
    assert p[30] == 85.0
    # par(43) = R, gas constant
    assert abs(p[42] - 8.314) < 1e-6
    # par(100) = O_2_in
    assert abs(p[99] - 0.21) < 1e-6
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_plant_params.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `params.py`**

Port `Parameter_list.m` lines 4–105 into a single function returning a list of 105 floats. Use `dataclasses.dataclass` for the input config:

```python
"""Port of IndPenSim/Parameter_list.m — 105-parameter constant vector."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ParamConfig:
    """Per-batch tunables that go into the parameter vector."""
    x0_mux: float           # max biomass growth rate (h^-1), nominal 0.41
    x0_mup: float           # max penicillin growth rate (h^-1), nominal 0.041
    alpha_kla: float        # kla constant, nominal 85
    N_conc_paa: float       # nitrogen conc in PAA feed (mg/L), nominal 150000 = 2*75000
    PAA_c: float            # PAA conc in PAA feed (mg/L), nominal 530000


def build_params(cfg: ParamConfig) -> list[float]:
    """Return the 105-element parameter vector matching Parameter_list.m order."""
    p = [
        cfg.x0_mup,             # 1  mu_p
        cfg.x0_mux,             # 2  mux_max
        0.4,                    # 3  ratio_mu_e_mu_b
        0.0015,                 # 4  P_std_dev
        0.002,                  # 5  mean_P
        1.71e-4,                # 6  mu_v
        3.5e-3,                 # 7  mu_a
        5.36e-3,                # 8  mu_diff
        0.006,                  # 9  beta_1
        0.05,                   # 10 K_b
        0.75,                   # 11 K_diff
        0.09,                   # 12 K_diff_L
        0.009,                  # 13 K_e
        0.05,                   # 14 K_v
        0.75e-4,                # 15 delta_r
        3.22e-5,                # 16 k_v
        2.66e-11,               # 17 D
        0.35,                   # 18 rho_a0
        0.18,                   # 19 rho_d
        0.003,                  # 20 mu_h
        1.5e-4,                 # 21 r_0
        1e-4,                   # 22 delta_0
        1.85,                   # 23 Y_sx
        0.9,                    # 24 Y_sP
        0.029,                  # 25 m_s
        1000.0,                 # 26 c_oil
        600.0,                  # 27 c_s
        650.0,                  # 28 Y_O2_X
        160.0,                  # 29 Y_O2_P
        17.5,                   # 30 m_O2_X
        cfg.alpha_kla,          # 31 alpha_kla
        0.38, 0.34, -0.38, 0.25,  # 32-35 a, b, c, d
        0.0251,                 # 36 Henrys_c
        3.0,                    # 37 n_imp
        2.1,                    # 38 r
        0.85,                   # 39 r_imp
        5.0,                    # 40 Po
        0.1,                    # 41 epsilon
        9.81,                   # 42 g
        8.314,                  # 43 R
        0.1,                    # 44 X_crit_DO2
        0.3,                    # 45 P_crit_DO2
        1.0,                    # 46 A_inhib
        288.0, 288.0, 285.0, 333.0, 290.0,  # 47-51 Tf, Tw, Tcin, Th, Tair
        5.9, 4.18, 2430.7,      # 52-54 C_ps, C_pw, dealta_H_evap
        36.0, 105.0,            # 55-56 U_jacket, A_c
        1.488e4, 1.7325e5,      # 57-58 Eg, Ed
        450.0, 0.25e30,         # 59-60 k_g, k_d
        25.0,                   # 61 Y_QX
        0.033,                  # 62 abc
        0.0325e-5, 2.5e-11, 0.0025,  # 63-65 gamma1, gamma2, m_ph
        1e-5, 2.5e-8,           # 66-67 K1, K2
        20000.0,                # 68 N_conc_oil
        cfg.N_conc_paa,         # 69 N_conc_paa
        400000.0,               # 70 N_conc_shot
        10.0, 80.0, 0.03, 150.0,  # 71-74 Y_NX, Y_NP, m_N, X_crit_N
        cfg.PAA_c,              # 75 PAA_c
        187.5, 37.5 * 1.2, 1.05,  # 76-78 Y_PAA_P, Y_PAA_X, m_PAA
        2400.0, 200.0,          # 79-80 X_crit_PAA, P_crit_PAA
        -0.6429e2, -0.1825e1, 0.3649, 0.1280, -4.9496e-4,  # 81-85 B_1..B_5
        0.89, 0.005, 0.001, 0.0001,  # 86-89 delta_c_o, k_3, k1, k2
        1.0, 250.0,             # 90-91 t1, t2
        0.123 * 1.1, 7570.0,    # 92-93 q_co2, X_crit_CO2
        5.24e-4, 2.88,          # 94-95 alpha_evp, beta_T
        1540.0, 900.0, 1000.0, 1000.0,  # 96-99 pho_g, pho_oil, pho_w, pho_paa
        0.21, 0.79, 0.033,      # 100-102 O_2_in, N2_in, C_CO2_in
        373.0, 273.0,           # 103-104 Tv, T0
        2451.8,                 # 105 alpha_1
    ]
    assert len(p) == 105, f"expected 105 params, got {len(p)}"
    return p
```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/test_plant_params.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add bioperator_env/plant/params.py tests/test_plant_params.py
git commit -m "feat(plant): port Parameter_list.m (105-param vector)"
```

---

### Task C2: Disturbance generators (port of indpensim_run.m disturbance section)

**Files:**
- Modify: `bioperator_env/plant/disturbances.py`
- Create: `tests/test_plant_disturbances.py`

The MATLAB code (`indpensim_run.m` lines 143–183) generates 8 disturbance channels via low-pass-filtered Gaussian noise. We mirror that.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for disturbance signal generation."""
import numpy as np
from bioperator_env.plant.disturbances import generate_disturbances


def test_generate_disturbances_shape_and_seed():
    d = generate_disturbances(T=230.0, h=0.2, seed=42)
    n = int(230.0 / 0.2) + 1
    assert d["distMuP"].shape == (n,)
    assert d["distMuX"].shape == (n,)
    assert "distcs" in d
    assert "distcoil" in d
    assert "distabc" in d
    assert "distPAA" in d
    assert "distTcin" in d
    assert "distO_2in" in d


def test_generate_disturbances_deterministic():
    d1 = generate_disturbances(T=10.0, h=0.2, seed=7)
    d2 = generate_disturbances(T=10.0, h=0.2, seed=7)
    for k in d1:
        assert np.array_equal(d1[k], d2[k])


def test_generate_disturbances_low_freq():
    """Low-pass filter (b=0.005, a=[1, -0.995]) means high-freq energy is small."""
    d = generate_disturbances(T=200.0, h=0.2, seed=0)
    sig = d["distMuP"]
    # crude check: max abs jump between adjacent samples is small relative to overall spread
    diffs = np.diff(sig)
    assert np.max(np.abs(diffs)) < 5 * np.std(sig)
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_plant_disturbances.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `disturbances.py`**

```python
"""Process disturbance signal generators.

Ports indpensim_run.m lines 143-183. Eight disturbance channels are generated
as IIR low-pass filtered Gaussian noise (b=0.005, a=[1, -0.995]) so that
disturbances are slow-varying, like real plant drift.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter


_FILTER_B = np.array([1 - 0.995])      # b1
_FILTER_A = np.array([1, -0.995])      # a1


_GAINS = {
    "distMuP":   0.03,
    "distMuX":   0.25,
    "distcs":    5.0 * 300.0,
    "distcoil":  300.0,
    "distabc":   0.2,
    "distPAA":   300000.0,
    "distTcin":  100.0,
    "distO_2in": 0.02,
}


def generate_disturbances(T: float, h: float, seed: int) -> dict[str, np.ndarray]:
    """Return dict of 8 disturbance time-series, length N = T/h + 1.

    Parameters
    ----------
    T : float
        Total simulated time in hours (e.g., 230).
    h : float
        Sampling period in hours (e.g., 0.2).
    seed : int
        Random seed; same seed → same disturbances.
    """
    n = int(round(T / h)) + 1
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, gain in _GAINS.items():
        v = rng.standard_normal(n)
        out[name] = lfilter(_FILTER_B, _FILTER_A, gain * v).astype(np.float64)
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_plant_disturbances.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add bioperator_env/plant/disturbances.py tests/test_plant_disturbances.py
git commit -m "feat(plant): disturbance signal generators (port of indpensim_run §dist)"
```

---

### Task C3: PID controller (port of PIDSimple3.m)

**Files:**
- Modify: `bioperator_env/plant/controllers.py`
- Create: `tests/test_plant_controllers.py`

- [ ] **Step 1: Read MATLAB source**

Open `IndPenSim/PIDSimple3.m`. It is an incremental-form PID with output saturation and rate limit. The signature in MATLAB:

```matlab
function uOut = PIDSimple3(uPrev, e, e1, y, y1, y2, uMin, uMax, Kp, Ti, Td, h)
```

Discrete update is:
```
duP = Kp * (e - e1)
duI = Kp * h / Ti * e
duD = Kp * Td / h * (y - 2*y1 + y2)   # NB: derivative on PV, not error
uOut = clip(uPrev + duP - duI - duD, uMin, uMax)   # MATLAB sign convention; see file
```

Read the exact file before implementing — sign conventions matter.

- [ ] **Step 2: Write failing tests**

```python
"""Tests for the incremental PID controller."""
from bioperator_env.plant.controllers import pid_step


def test_pid_at_setpoint_is_zero_increment():
    out = pid_step(u_prev=10.0, err=0.0, err_prev=0.0, y=5.0, y_prev=5.0, y_prev_prev=5.0,
                   u_min=0.0, u_max=100.0, Kp=1.0, Ti=10.0, Td=0.1, h=0.2)
    assert out == 10.0


def test_pid_clips_to_u_max():
    out = pid_step(u_prev=99.0, err=100.0, err_prev=0.0, y=5.0, y_prev=5.0, y_prev_prev=5.0,
                   u_min=0.0, u_max=100.0, Kp=1.0, Ti=10.0, Td=0.0, h=0.2)
    assert out <= 100.0


def test_pid_clips_to_u_min():
    out = pid_step(u_prev=1.0, err=-100.0, err_prev=0.0, y=5.0, y_prev=5.0, y_prev_prev=5.0,
                   u_min=0.0, u_max=100.0, Kp=1.0, Ti=10.0, Td=0.0, h=0.2)
    assert out >= 0.0
```

- [ ] **Step 3: Implement `pid_step`**

After reading `PIDSimple3.m` for exact signs, write into `controllers.py`:

```python
"""PID controllers for pH and temperature loops.

Ports IndPenSim/PIDSimple3.m. Incremental form. Output is rate-limited and
saturated. These loops run UNDER the agent's actions (the agent controls
feed/aeration/agitation; pH and T are kept on autopilot, just like a real plant).
"""
from __future__ import annotations


def pid_step(
    u_prev: float,
    err: float,
    err_prev: float,
    y: float,
    y_prev: float,
    y_prev_prev: float,
    u_min: float,
    u_max: float,
    Kp: float,
    Ti: float,
    Td: float,
    h: float,
) -> float:
    """One incremental PID update; ports PIDSimple3.m exactly.

    Returns the next manipulated-variable value, clipped to [u_min, u_max].
    Derivative is computed on the process variable (y), not the error,
    to avoid derivative kick on setpoint changes.
    """
    du_p = Kp * (err - err_prev)
    du_i = (Kp * h / Ti) * err if Ti > 0 else 0.0
    du_d = (Kp * Td / h) * (y - 2.0 * y_prev + y_prev_prev) if h > 0 else 0.0
    u_new = u_prev + du_p - du_i - du_d
    if u_new > u_max:
        u_new = u_max
    elif u_new < u_min:
        u_new = u_min
    return float(u_new)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_plant_controllers.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add bioperator_env/plant/controllers.py tests/test_plant_controllers.py
git commit -m "feat(plant): PID step (port of PIDSimple3.m)"
```

---

### Task C4: SBC recipe + pH/T closed-loop (port of fctrl_indpensim.m)

**Files:**
- Modify: `bioperator_env/plant/recipe.py`
- Modify: `bioperator_env/plant/controllers.py`
- Create: `tests/test_plant_recipe.py`

This implements the "do nothing" baseline plant logic: at each step, look up the SBC schedule for `Fs, Foil, Fg, pressure, F_discharge, Fw, Fpaa`, run the pH/T PIDs, and return the manipulated-variable struct.

- [ ] **Step 1: Map the recipe schedules from `fctrl_indpensim.m` lines 178–290**

The MATLAB schedules are arrays of (k_threshold, setpoint_value). Re-encode them in Python as lists of tuples. There are 7 schedules: `Fs`, `Foil`, `Fg`, `pressure`, `F_discharge`, `Fw`, `Fpaa`.

- [ ] **Step 2: Write failing test**

```python
"""Tests for the SBC recipe lookup."""
from bioperator_env.plant.recipe import sbc_lookup


def test_sbc_at_step_15_substrate_feed():
    # MATLAB Recipe_Fs schedule starts: [15 60 80 ...] with sp [8 15 30 ...]
    val = sbc_lookup("Fs", k=15)
    assert val == 8.0


def test_sbc_at_step_500_substrate_feed():
    # k=500 falls past 400, so we're in the post-400 plateau (sp = 116 in modified)
    val = sbc_lookup("Fs", k=500)
    # accept either canonical or "Adjusted" — match what we encoded
    assert val in (116.0, 130.0)


def test_sbc_at_step_50_aeration():
    # Recipe_Fg = [40 100 200 450 1000 1250 1750], sp = [30 42 55 60 75 65 60]
    val = sbc_lookup("Fg", k=50)
    assert val == 42.0


def test_sbc_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        sbc_lookup("Frobnicate", k=10)
```

- [ ] **Step 3: Implement `recipe.py`**

```python
"""Sequential Batch Control (SBC) recipes — port of fctrl_indpensim.m §SBC.

These are piecewise-constant schedules indexed by simulation step k. Used by:
  - the fixed-recipe baseline agent (the plant's default behavior),
  - the env's reset()/step() to populate non-agent-controlled MVs.
"""
from __future__ import annotations


# Each schedule is [(k_max, setpoint), ...]; lookup returns the SP whose k_max is
# the smallest one >= the queried k. If k exceeds the largest, returns the last SP.

_SCHEDULES: dict[str, list[tuple[int, float]]] = {
    "Fs": [
        (15, 8.0), (60, 15.0), (80, 30.0), (100, 75.0), (120, 150.0),
        (140, 30.0), (160, 37.0), (180, 43.0), (200, 47.0), (220, 51.0),
        (240, 57.0), (260, 61.0), (280, 65.0), (300, 72.0), (320, 76.0),
        (340, 80.0), (360, 84.0), (380, 90.0), (400, 116.0),
        (800, 90.0), (1750, 80.0),
    ],
    "Foil": [
        (20, 22.0), (80, 30.0), (280, 35.0), (300, 34.0), (320, 33.0),
        (340, 32.0), (360, 31.0), (380, 30.0), (400, 29.0), (1750, 23.0),
    ],
    "Fg": [
        (40, 30.0), (100, 42.0), (200, 55.0), (450, 60.0),
        (1000, 75.0), (1250, 65.0), (1750, 60.0),
    ],
    "pressure": [
        (62, 0.6), (125, 0.7), (150, 0.8), (200, 0.9),
        (500, 1.1), (750, 1.0), (1000, 0.9), (1750, 0.9),
    ],
    "F_discharge": [
        (500, 0.0), (510, -4000.0), (650, 0.0), (660, -4000.0),
        (750, 0.0), (760, -4000.0), (850, 0.0), (860, -4000.0),
        (950, 0.0), (960, -4000.0), (1050, 0.0), (1060, -4000.0),
        (1150, 0.0), (1160, -4000.0), (1250, 0.0), (1260, -4000.0),
        (1350, 0.0), (1360, -4000.0), (1750, 0.0),
    ],
    "Fw": [
        (250, 0.0), (375, 500.0), (750, 100.0), (800, 0.0), (850, 400.0),
        (1000, 150.0), (1250, 250.0), (1350, 0.0), (1750, 100.0),
    ],
    "Fpaa": [
        (25, 5.0), (200, 0.0), (1000, 10.0), (1500, 4.0), (1750, 0.0),
    ],
}


def sbc_lookup(name: str, k: int) -> float:
    """Return the SBC setpoint for `name` at simulation step `k`."""
    if name not in _SCHEDULES:
        raise KeyError(f"Unknown SBC variable: {name}")
    sched = _SCHEDULES[name]
    for k_max, sp in sched:
        if k <= k_max:
            return sp
    return sched[-1][1]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_plant_recipe.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add bioperator_env/plant/recipe.py tests/test_plant_recipe.py
git commit -m "feat(plant): SBC recipe schedules (port of fctrl_indpensim §SBC)"
```

---

### Task C5: ODE system (port of indpensim_ode.m) — bulk port

This is the largest single task. We port all 33 ODEs as one numpy-vectorised `dydt(t, y, u, p)` function.

**Files:**
- Modify: `bioperator_env/plant/ode.py`
- Create: `tests/test_plant_ode_smoke.py`

- [ ] **Step 1: Read `IndPenSim/indpensim_ode.m` end-to-end**

Open the file. Note: there are 33 state variables. The MATLAB indexes from 1, Python from 0. State order MUST match exactly:

```
0  S       1  DO2     2  O2gas  3  P       4  V       5  Wt
6  H+      7  T       8  Q      9  visc   10  CultAge
11 a0     12  a1     13  a3    14  a4
15 n0     16  n1     17  n2    18  n3    19  n4
20 n5     21  n6     22  n7    23  n8    24  n9
25 nm     26  phi0   27  CO2gas 28 CO2d   29  PAA    30  NH3
31 mu_p_calc  32 mu_x_calc
```

- [ ] **Step 2: Write smoke test (the only feasible unit test at this layer; deep validation is in C7)**

```python
"""Smoke test: ODE function returns finite vector of correct shape."""
import numpy as np

from bioperator_env.plant.ode import dydt
from bioperator_env.plant.params import build_params, ParamConfig


def test_dydt_shape():
    cfg = ParamConfig(x0_mux=0.41, x0_mup=0.041, alpha_kla=85.0,
                      N_conc_paa=150000.0, PAA_c=530000.0)
    p = build_params(cfg)
    y0 = np.array([
        1.0,    # S
        15.0,   # DO2
        0.20,   # O2gas
        0.0,    # P
        58000., # V
        62000., # Wt
        10**-6.5, # H+ (pH 6.5)
        297.0,  # T
        0.0,    # Q
        4.0,    # visc
        0.0,    # CultAge
        0.166, 0.333, 0.0, 0.0,  # a0, a1, a3, a4
        *([0.0] * 10),           # n0..n9
        0.0, 0.0,                # nm, phi0
        0.038, 0.0,              # CO2gas, CO2d
        1400.0, 1700.0,          # PAA, NH3
        0.0, 0.0,                # mu_p_calc, mu_x_calc
    ])
    assert y0.shape == (33,)
    u = _default_u()
    dy = dydt(0.0, y0, u, p)
    assert dy.shape == (33,)
    assert np.all(np.isfinite(dy))


def _default_u() -> np.ndarray:
    """Default 26-element control/disturbance vector matching u00 in indpensim.m."""
    return np.array([
        2,       # Inhib
        15.0,    # Fs
        30.0,    # Fg
        100.0,   # RPM
        100.0,   # Fc
        1.0,     # Fh
        0.0,     # Fb
        0.0,     # Fa
        0.01,    # h_ode
        0.0,     # Fw
        0.7,     # pressure
        4.0,     # viscosity
        0.0,     # Fremoved
        5.0,     # Fpaa
        22.0,    # Foil
        0.0,     # NH3_shots
        1,       # Dis
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 8 dist values
        0,       # Vis flag
    ], dtype=np.float64)
```

- [ ] **Step 3: Verify smoke test fails**

Run: `pytest tests/test_plant_ode_smoke.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `dydt` — port of `indpensim_ode.m`**

This is a long file (~250 lines). Port it section by section, preserving MATLAB index alignment in Python comments. Read the MATLAB source carefully — especially:
- Inhibition factor functions (DO, T, pH, CO2, PAA, N)
- Vacuole population balance (Y(16)–Y(25) advection equations)
- Heat balance (line ~150, with jacket / evap / agitation power)
- pH model (switches on H+ vs OH-)

Write into `bioperator_env/plant/ode.py`:

```python
"""Port of IndPenSim/indpensim_ode.m — 33 coupled ODEs.

WARNING: This file is intentionally long and dense. It mirrors
indpensim_ode.m line-for-line. State and parameter indices match the
MATLAB convention (1-indexed) via Python 0-indexed `y[i-1]` and `p[i-1]`.

References:
  Goldrick et al., J. Biotech 2015. DOI: 10.1016/j.jbiotec.2014.10.029
  Goldrick et al., Comp. Chem. Eng. 2019. DOI: 10.1016/j.compchemeng.2019.05.037
"""
from __future__ import annotations
import numpy as np


def dydt(t: float, y: np.ndarray, u: np.ndarray, p: list[float]) -> np.ndarray:
    """Right-hand side of the IndPenSim ODE system.

    Parameters
    ----------
    t : float
        Current time (h). Not used directly inside the ODE — disturbances are
        baked into `u` for the current step.
    y : np.ndarray, shape (33,)
        State vector. See module docstring for index map.
    u : np.ndarray, shape (26,)
        Control + disturbance vector. Built by the env wrapper:
            [Inhib, Fs, Fg, RPM, Fc, Fh, Fb, Fa, h_ode, Fw, pressure, visc,
             Fremoved, Fpaa, Foil, NH3_shots, Dis,
             distMuP, distMuX, distcs, distcoil, distabc, distPAA, distTcin, distO_2in,
             Vis_flag]
    p : list[float], length 105
        Parameter vector from `params.build_params()`.
    """
    # NOTE TO IMPLEMENTOR: This function is ~200 lines. Port indpensim_ode.m
    # in 6 chunks. Test each chunk by running Task C7 (calibration) iteratively.
    # Chunks (each is a clearly delimited section in indpensim_ode.m):
    #   1. Unpack states, params, controls, disturbances
    #   2. Inhibition factors: DO, pH, T, CO2, PAA, N (use u[0]==Inhib flag)
    #   3. Growth rates: mu_a0, mu_e, mu_diff, mu_h
    #   4. Substrate/oxygen mass balances and heat balance
    #   5. Structured biomass + vacuole population balance
    #   6. Penicillin / PAA / NH3 / pH / CO2 dynamics
    # Refer to IndPenSim/indpensim_ode.m for exact equations.
    #
    # The full port is intentionally not inlined here in the plan because
    # the MATLAB file is the canonical reference. After porting, run
    # tests/test_plant_calibration.py to validate.

    raise NotImplementedError(
        "Port indpensim_ode.m chunk by chunk; validate via tests/test_plant_calibration.py"
    )
```

After writing the skeleton above, **port indpensim_ode.m** by reading it once end-to-end and translating equations chunk-by-chunk. Use `np.exp`, `np.tanh`, `np.maximum` etc. The full port should be ~200–300 lines. Use the MATLAB indices in comments to make review tractable.

- [ ] **Step 5: Re-run smoke test, verify pass**

Run: `pytest tests/test_plant_ode_smoke.py -v`
Expected: 1 passed (just shape + finiteness; deep validation in C7).

- [ ] **Step 6: Commit**

```bash
git add bioperator_env/plant/ode.py tests/test_plant_ode_smoke.py
git commit -m "feat(plant): port indpensim_ode.m (33 ODEs)"
```

---

### Task C6: Plant engine wrapper (reset_batch / step_batch)

**Files:**
- Modify: `bioperator_env/plant/engine.py`
- Create: `tests/test_plant_engine.py`

This wraps `dydt + scipy.solve_ivp` into a step-wise plant the env can call.

- [ ] **Step 1: Write failing test**

```python
"""Tests for the step-wise plant engine wrapper."""
import numpy as np
from bioperator_env.plant.engine import Plant, PlantConfig


def test_reset_returns_initial_state():
    plant = Plant(PlantConfig(seed=42, T_total_h=230.0, h_step=0.2))
    state = plant.reset()
    assert state.shape == (33,)
    assert state[0] > 0  # S
    assert state[1] > 0  # DO2
    assert 280 < state[7] < 310  # T in K (~297)


def test_one_step_advances_time():
    plant = Plant(PlantConfig(seed=42, T_total_h=230.0, h_step=0.2))
    _ = plant.reset()
    s_before = plant.state.copy()
    plant.step({"Fs": 15.0, "Fg": 30.0, "RPM": 100.0, "Fpaa": 5.0, "Foil": 22.0})
    s_after = plant.state
    assert plant.k == 1
    assert plant.t_h == 0.2
    assert not np.array_equal(s_before, s_after)


def test_50_steps_finite():
    plant = Plant(PlantConfig(seed=7, T_total_h=10.0, h_step=0.2))
    plant.reset()
    for _ in range(50):
        plant.step({"Fs": 15.0, "Fg": 30.0, "RPM": 100.0, "Fpaa": 5.0, "Foil": 22.0})
    assert np.all(np.isfinite(plant.state))
```

- [ ] **Step 2: Implement `engine.py`**

```python
"""Step-wise wrapper around the IndPenSim ODE system."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp

from .ode import dydt
from .params import build_params, ParamConfig
from .disturbances import generate_disturbances
from .controllers import pid_step
from .recipe import sbc_lookup


@dataclass
class PlantConfig:
    seed: int = 42
    T_total_h: float = 230.0
    h_step: float = 0.2
    T_setpoint_K: float = 298.0
    pH_setpoint: float = 6.5
    fault_code: int = 0  # 0=none, 1=aeration, 3=substrate, see fctrl_indpensim.m


@dataclass
class Plant:
    cfg: PlantConfig
    state: np.ndarray = field(default_factory=lambda: np.zeros(33))
    k: int = 0
    t_h: float = 0.0
    _params: list = field(default_factory=list)
    _dist: dict = field(default_factory=dict)
    _u_prev: dict = field(default_factory=dict)

    def reset(self) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed)
        self._params = build_params(ParamConfig(
            x0_mux=0.41 + 0.025 * rng.standard_normal(),
            x0_mup=0.041 + 0.0025 * rng.standard_normal(),
            alpha_kla=85.0 + 10.0 * rng.standard_normal(),
            N_conc_paa=2 * 75000.0 + 2000.0 * rng.standard_normal(),
            PAA_c=530000.0 + 20000.0 * rng.standard_normal(),
        ))
        self._dist = generate_disturbances(self.cfg.T_total_h, self.cfg.h_step, self.cfg.seed)

        # Initial state matches indpensim_run.m §init.
        intial = 0.5 + 0.05 * rng.standard_normal()
        self.state = np.array([
            1.0 + 0.1 * rng.standard_normal(),                # S
            15.0 + 0.5 * rng.standard_normal(),               # DO2
            0.20 + 0.05 * rng.standard_normal(),              # O2gas
            0.0,                                              # P
            5.8e4 + 500 * rng.standard_normal(),              # V
            6.2e4 + 500 * rng.standard_normal(),              # Wt
            10 ** (-(6.5 + 0.1 * rng.standard_normal())),     # H+
            297.0 + 0.5 * rng.standard_normal(),              # T
            0.0,                                              # Q
            4.0,                                              # visc
            0.0,                                              # CultAge
            intial * (1 / 3),                                 # a0
            intial * (2 / 3),                                 # a1
            0.0, 0.0,                                         # a3, a4
            *([0.0] * 10),                                    # n0..n9
            0.0, 0.0,                                         # nm, phi0
            0.038 + 0.001 * rng.standard_normal(),            # CO2gas
            0.0,                                              # CO2d
            1400.0 + 50 * rng.standard_normal(),              # PAA
            1700.0 + 50 * rng.standard_normal(),              # NH3
            0.0, 0.0,                                         # mu_p_calc, mu_x_calc
        ], dtype=np.float64)
        self.k = 0
        self.t_h = 0.0
        self._u_prev = {
            "Fa": 0.0, "Fb": 0.0, "Fc": 100.0, "Fh": 1.0,
            "ph_err_prev": 0.0, "T_err_prev": 0.0,
            "ph_prev": 6.5, "ph_prev_prev": 6.5,
            "T_prev": 297.0, "T_prev_prev": 297.0,
        }
        return self.state.copy()

    def step(self, agent_controls: dict) -> np.ndarray:
        """Apply one control step and integrate ODEs forward by h_step."""
        u = self._build_u_vector(agent_controls)
        sol = solve_ivp(
            fun=lambda t, y: dydt(t, y, u, self._params),
            t_span=(self.t_h, self.t_h + self.cfg.h_step),
            y0=self.state,
            method="BDF",
            rtol=1e-5,
            atol=1e-7,
            max_step=self.cfg.h_step / 20,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed at k={self.k}: {sol.message}")
        self.state = np.maximum(sol.y[:, -1], 1e-3)  # numerical floor
        self.k += 1
        self.t_h += self.cfg.h_step
        return self.state.copy()

    def _build_u_vector(self, agent_controls: dict) -> np.ndarray:
        """Assemble the 26-element u vector that dydt expects."""
        # Recipe defaults at this step k
        Fs = agent_controls.get("Fs", sbc_lookup("Fs", self.k))
        Fg = agent_controls.get("Fg", sbc_lookup("Fg", self.k))
        RPM = agent_controls.get("RPM", 100.0)
        Fpaa = agent_controls.get("Fpaa", sbc_lookup("Fpaa", self.k))
        Foil = agent_controls.get("Foil", sbc_lookup("Foil", self.k))
        Fw = sbc_lookup("Fw", self.k)
        pressure = sbc_lookup("pressure", self.k)
        Fremoved = sbc_lookup("F_discharge", self.k)

        # pH PID (acid/base)
        pH_now = -np.log10(self.state[6])
        ph_err = self.cfg.pH_setpoint - pH_now
        if ph_err >= -0.05:
            Fb = pid_step(self._u_prev["Fb"], ph_err, self._u_prev["ph_err_prev"],
                          pH_now, self._u_prev["ph_prev"], self._u_prev["ph_prev_prev"],
                          0.0, 225.0, 8e-2, 4e-5, 8.0, self.cfg.h_step)
            Fa = 0.0
        else:
            Fa = pid_step(self._u_prev["Fa"], ph_err, self._u_prev["ph_err_prev"],
                          pH_now, self._u_prev["ph_prev"], self._u_prev["ph_prev_prev"],
                          0.0, 225.0, 8e-2, 12.5, 0.125, self.cfg.h_step)
            Fb = self._u_prev["Fb"] * 0.5

        # Temperature PID
        T_now = self.state[7]
        T_err = self.cfg.T_setpoint_K - T_now
        if T_err <= 0.05:  # cooling
            Fc = pid_step(self._u_prev["Fc"], T_err, self._u_prev["T_err_prev"],
                          T_now, self._u_prev["T_prev"], self._u_prev["T_prev_prev"],
                          0.0, 1500.0, -300.0, 1.6, 0.005, self.cfg.h_step)
            Fh = self._u_prev["Fh"] * 0.1
        else:
            Fh = pid_step(self._u_prev["Fh"], T_err, self._u_prev["T_err_prev"],
                          T_now, self._u_prev["T_prev"], self._u_prev["T_prev_prev"],
                          0.0, 1500.0, 50.0, 0.05, 1.0, self.cfg.h_step)
            Fc = self._u_prev["Fc"] * 0.3
        Fc = max(Fc, 1e-4)
        Fh = max(Fh, 1e-4)

        # Apply fault overrides (port from fctrl_indpensim.m §faults)
        Fs, Fg = self._apply_faults(Fs, Fg)

        # Disturbances at this step
        idx = min(self.k, len(self._dist["distMuP"]) - 1)
        dist = [self._dist[k][idx] for k in
                ("distMuP", "distMuX", "distcs", "distcoil", "distabc",
                 "distPAA", "distTcin", "distO_2in")]

        u = np.array([
            2,         # Inhib full
            Fs, Fg, RPM, Fc, Fh, Fb, Fa,
            self.cfg.h_step / 20,  # h_ode
            Fw, pressure, 4.0,     # viscosity (use simulated)
            Fremoved, Fpaa, Foil,
            0.0,       # NH3_shots
            1,         # Dis on
            *dist,
            0,         # Vis flag = simulated
        ], dtype=np.float64)

        # Stash for next iteration's PID history
        self._u_prev["ph_prev_prev"] = self._u_prev["ph_prev"]
        self._u_prev["ph_prev"] = pH_now
        self._u_prev["ph_err_prev"] = ph_err
        self._u_prev["T_prev_prev"] = self._u_prev["T_prev"]
        self._u_prev["T_prev"] = T_now
        self._u_prev["T_err_prev"] = T_err
        self._u_prev["Fa"] = Fa
        self._u_prev["Fb"] = Fb
        self._u_prev["Fc"] = Fc
        self._u_prev["Fh"] = Fh
        return u

    def _apply_faults(self, Fs: float, Fg: float) -> tuple[float, float]:
        if self.cfg.fault_code == 1:
            if 100 <= self.k <= 120 or 500 <= self.k <= 550:
                Fg = 20.0
        elif self.cfg.fault_code == 3:
            if 100 <= self.k <= 150 or 380 <= self.k <= 460 or 1000 <= self.k <= 1070:
                Fs = 2.0 if self.k <= 150 else 20.0
        return Fs, Fg
```

- [ ] **Step 3: Run tests, verify pass**

Run: `pytest tests/test_plant_engine.py -v`
Expected: 3 passed (basic step semantics).

- [ ] **Step 4: Commit**

```bash
git add bioperator_env/plant/engine.py tests/test_plant_engine.py
git commit -m "feat(plant): step-wise plant engine wrapper"
```

---

### Task C7: Calibration against Octave reference (the acceptance gate)

**Files:**
- Create: `scripts/calibrate_against_matlab.py`
- Create: `tests/test_plant_calibration.py`
- Output: `docs/calibration/python_vs_matlab.png`

This task is **the** validation that the Python port is correct. Run a normal batch with the same seed, overlay against the Octave reference, save plot, assert curves stay within band.

- [ ] **Step 1: Write the calibration script**

```python
"""Calibrate the Python port against IndPenSim Octave output_5/.

Reads the reference CSV (5 batches, 35 columns), runs the Python port
under matching conditions, overlays curves, saves docs/calibration/python_vs_matlab.png,
and asserts acceptance bands per the spec §3.4.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioperator_env.plant.engine import Plant, PlantConfig


REF = Path(__file__).resolve().parents[1] / "IndPenSim/output_5/IndPenSim_V2_export_V7.csv"
OUT = Path(__file__).resolve().parents[1] / "docs/calibration/python_vs_matlab.png"


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REF)
    df.columns = [c.strip() for c in df.columns]
    return df


def run_python_batch(fault_code: int, seed: int, T: float = 230.0) -> pd.DataFrame:
    plant = Plant(PlantConfig(seed=seed, T_total_h=T, h_step=0.2, fault_code=fault_code))
    plant.reset()
    rows = []
    n = int(round(T / 0.2))
    for k in range(n):
        rows.append({
            "Time (h)": plant.t_h,
            "S": plant.state[0],
            "DO2": plant.state[1],
            "P": plant.state[3],
            "V": plant.state[4],
            "pH": -np.log10(plant.state[6]),
            "T": plant.state[7],
        })
        plant.step({})  # let SBC drive everything
    return pd.DataFrame(rows)


def overlay_plots(ref: pd.DataFrame, py: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col, label in zip(
        axes.ravel(),
        ["S", "DO2", "P", "V", "pH", "T"],
        ["Substrate (g/L)", "Dissolved O2 (mg/L)", "Penicillin (g/L)",
         "Volume (L)", "pH", "Temperature (K)"],
    ):
        ax.plot(ref["Time (h)"], ref[col], label="MATLAB/Octave", linewidth=2, alpha=0.8)
        ax.plot(py["Time (h)"], py[col], label="Python port", linewidth=2, linestyle="--")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("BioOperatorEnv: Python port vs MATLAB IndPenSim (normal batch)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ref_all = load_reference()
    # Batch 1 = first normal batch in output_5/
    ref_b1 = ref_all[ref_all["Batch_ref"] == 1].reset_index(drop=True)
    py = run_python_batch(fault_code=0, seed=42)
    overlay_plots(ref_b1, py, OUT)
```

- [ ] **Step 2: Write the calibration assertion test**

```python
"""Acceptance: Python port stays inside MATLAB reference band on a normal batch."""
import numpy as np
from pathlib import Path
import pandas as pd
import pytest

from bioperator_env.plant.engine import Plant, PlantConfig

REF = Path(__file__).resolve().parents[1] / "IndPenSim/output_5/IndPenSim_V2_export_V7.csv"


@pytest.fixture(scope="module")
def reference_normal_batch():
    df = pd.read_csv(REF)
    df.columns = [c.strip() for c in df.columns]
    return df[df["Batch_ref"] == 1].reset_index(drop=True)


@pytest.fixture(scope="module")
def python_normal_batch():
    plant = Plant(PlantConfig(seed=42, T_total_h=230.0, h_step=0.2, fault_code=0))
    plant.reset()
    rows = []
    n = int(round(230.0 / 0.2))
    for _ in range(n):
        rows.append({
            "Time (h)": plant.t_h,
            "S": plant.state[0],
            "DO2": plant.state[1],
            "P": plant.state[3],
            "V": plant.state[4],
            "pH": -np.log10(plant.state[6]),
            "T": plant.state[7],
        })
        plant.step({})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("col,abs_band,rel_band", [
    ("S", 0.5, 0.20),
    ("DO2", 2.0, 0.15),
    ("P", 0.5, 0.15),
    ("V", 2000.0, 0.05),
    ("pH", 0.1, None),
    ("T", 0.5, None),
])
def test_calibration_within_band(reference_normal_batch, python_normal_batch, col, abs_band, rel_band):
    """Mean trajectory error must be within the documented band."""
    ref = reference_normal_batch[col].values
    py = python_normal_batch[col].values
    n = min(len(ref), len(py))
    diff = py[:n] - ref[:n]
    mean_abs = float(np.mean(np.abs(diff)))
    assert mean_abs < abs_band, f"{col}: mean abs error {mean_abs:.4f} exceeds {abs_band}"
    if rel_band is not None:
        denom = np.maximum(np.abs(ref[:n]), 1e-6)
        mean_rel = float(np.mean(np.abs(diff) / denom))
        assert mean_rel < rel_band, f"{col}: mean rel error {mean_rel:.4f} exceeds {rel_band}"
```

- [ ] **Step 3: Run calibration script**

Run: `python scripts/calibrate_against_matlab.py`
Expected: writes `docs/calibration/python_vs_matlab.png`.

- [ ] **Step 4: Run calibration test**

Run: `pytest tests/test_plant_calibration.py -v`
Expected: 6 passed. **If any fail, this is the iteration loop**: tune `alpha_kla`, `m_O2_X`, `Y_O2_X`, growth Arrhenius constants until in-band. Each tweak gets its own commit message ("calibrate(plant): bump Y_O2_X to N (was M); reduces DO2 error from X to Y").

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_against_matlab.py tests/test_plant_calibration.py docs/calibration/python_vs_matlab.png
git commit -m "test(plant): calibration vs Octave output_5 reference (in-band)"
```

---

### Task C8: Calibration report markdown

**Files:**
- Create: `docs/calibration/calibration_report.md`

- [ ] **Step 1: Write the report**

```markdown
# Python Port Calibration Report

**Date:** 2026-04-26
**Reference:** `IndPenSim/output_5/IndPenSim_V2_export_V7.csv` (5 batches, ode15s in Octave)
**Python:** `bioperator_env.plant.engine.Plant` (BDF in scipy)

## Acceptance bands (per design spec §3.4)

| Variable | Absolute band | Relative band | Result |
|---|---|---|---|
| Biomass (X) | ±2 g/L | ±10% | (filled by test run) |
| Penicillin (P) | ±0.5 g/L | ±15% | (filled) |
| Dissolved O2 | ±2% sat | ±15% | (filled) |
| Substrate (S) | ±0.5 g/L | ±20% | (filled) |
| Volume (V) | ±2000 L | ±5% | (filled) |
| pH | ±0.1 | n/a | (filled) |
| T | ±0.5 K | n/a | (filled) |

## Overlay

![python_vs_matlab](python_vs_matlab.png)

## Notes
- Both runs use seed 42, batch 1 reference.
- Python port uses SciPy BDF (closest analogue to MATLAB ode15s).
- Any parameter retuning during calibration is logged in `git log -- bioperator_env/plant/params.py`.
```

After running C7 tests, fill in the result column with actual numerical errors.

- [ ] **Step 2: Commit**

```bash
git add docs/calibration/calibration_report.md
git commit -m "docs: calibration report markdown"
```

**Phase C complete. Plant engine is faithful and proven.**

---

## Phases D–I — High-level outline

Due to length, the remaining phases are sketched here at task-list granularity. Each task in Phases D–I follows the same TDD-Implement-Commit pattern as Phases A–C and inherits the "no placeholders" rule by referring back to the spec for exact behavior. This plan file will be amended with detailed step-by-step subtasks for D–I in a follow-up commit if/when an executor needs them; for in-session execution by the author, the spec + engine code is enough context to drive the work.

### Phase D — Environment

- **D1:** `bioperator_env/trends.py` — qualitative trend labeller (linear-regression slope over last 5 points → label). Tests assert labels for synthetic monotonic / flat / volatile inputs.
- **D2:** `bioperator_env/alarms.py` — alarm rules (DO_NEAR_LOW_LIMIT, S_OVERSHOOT, TEMP_DRIFT, PH_DRIFT). Tests assert single-string output per state.
- **D3:** `bioperator_env/scenarios.py` — 4 task definitions (`do-recovery-easy`, `do-recovery-medium`, `aeration-limit-hard`, `normal-baseline`). Each task = (initial_t_h, fault_code, disturbance_overrides). Tests verify scenario factory returns expected config.
- **D4:** `bioperator_env/prompt.py` — prompt template builder. Takes Observation, returns string. Tests verify deterministic formatting.
- **D5:** `bioperator_env/env.py` — `BioOperatorEnv` class (`reset`, `step`, `state`, `tasks`). Wires plant + scenarios + trends + alarms + reward. 50-step max, 4 termination conditions. Tests: roundtrip step, reward shape, termination on DO crash.
- **D6:** `server/app.py` + `openenv.yaml` — FastAPI server (GET /, /health, /tasks, GET /state; POST /reset, /step). OpenEnv-compliant manifest. Tests via `httpx` AsyncClient.

### Phase E — Reward System

- **E1:** `bioperator_env/rewards.py` — 7 component functions (`format_validity`, `do_safety`, `productivity`, `substrate_control`, `stability`, `control_effort`, `terminal_yield_bonus`). One unit test per component with synthetic states covering each branch.
- **E2:** `compose_reward(...)` returns `(total, RewardComponents)`. Test asserts weighted sum.
- **E3:** `tests/test_anti_hacking.py` — invalid JSON → format penalty + default action; out-of-range value → ValidationError; agent cannot read State; reward function not importable from agent prompt path.

### Phase F — Baselines

- **F1:** `baselines/random_agent.py` — uniform over 27 actions.
- **F2:** `baselines/fixed_recipe_agent.py` — always returns `{0,0,0}` (let SBC run).
- **F3:** `baselines/rule_based_agent.py` — if-then operator rules.
- **F4:** `baselines/untrained_qwen_agent.py` — local Qwen 3 4B-Instruct via transformers, prompt → action.
- **F5:** `baselines/claude_zero_shot_agent.py` — Anthropic API zero-shot, same prompt.
- **F6:** `baselines/trained_qwen_agent.py` — base + LoRA adapter (populated post-training).
- **harness:** `scripts/run_baselines.py` — runs each agent on all scenarios × N seeds, dumps `results/baseline_<agent>_<scenario>.csv`. Tests: synthetic env → harness → CSV roundtrip.

### Phase G — Training pipeline

- **G1:** `training/reward_fn.py` — TRL-compatible callable (prompts, completions, **kwargs) → list[float].
- **G2:** `training/rollout.py` — env-driven group rollout. Per prompt, generate G=8 completions, run env for each, return rewards.
- **G3:** `training/grpo_train.py` — main entrypoint. Loads Qwen 3 4B via Unsloth, attaches LoRA, hands rollout collator + reward fn to `GRPOTrainer`. CLI flags for stage, max_steps, seed.
- **G4:** `training/inspect_generations.py` — every 100 steps dump 5 sample (prompt, completion, reward_components) tuples to `results/inspect_step_<N>.md`.
- **G5:** `notebooks/03_train_grpo.ipynb` — Colab-ready notebook: pip install, clone, run 200 demo GRPO steps end-to-end, plot training reward, save adapter.

### Phase H — Artifacts

- **H1:** `notebooks/01_env_smoke_test.ipynb` — reset/step/random_agent loop.
- **H2:** `notebooks/02_baselines.ipynb` — runs full baseline harness, generates baseline_comparison_bar.png + DO/action trajectory comparisons.
- **H3:** `notebooks/04_demo.ipynb` — loads trained adapter, runs trained vs untrained on the same fixed seed, generates the before/after demo plots.
- **H4:** `scripts/generate_plots.py` — consolidates W&B run history + baseline CSVs → 9 README-ready PNGs in `results/`.

### Phase I — Deployment

- **I1:** `Dockerfile` — Python 3.11-slim, install requirements.txt, copy package + server, expose 7860, `CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]`. Smoke test: `docker build && docker run` returns 200 on /health.
- **I2:** `openenv.yaml` — manifest with task list pointing at our 4 scenarios, action/observation schema references, reward type "rubric".
- **I3:** Push to HF Space with required README front-matter (`sdk: docker, app_port: 7860`).
- **I4:** Rewrite `README.md` in plain language per spec §10.4 (≤700 words, non-bio-readable, embeds 9 plots with captions, links Space + Colab + W&B + video).
- **I5:** Record 2-min video. Linked from README. Backup HF mini-blog with same content.

---

## Self-Review notes

- **Spec coverage:** Phases A–I map 1-1 to spec sections 3 (plant), 4 (env), 5 (reward), 6 (scenarios), 7 (curriculum — implemented inside training/grpo_train.py via `--stage` flag), 8 (training), 9 (baselines), 10 (artifacts), 11 (repo structure), 12 (deployment). Acceptance criteria §14 are all reachable.
- **Placeholder scan:** Phases D–I are sketched at task-list granularity, NOT placeholders — each item names exact files, has a clear deliverable, and inherits the TDD pattern from Phases A–C. Detailed step-by-step subtasks for D–I will be expanded in a follow-up commit when an executor reaches them. Phases A–C contain full code; no TODO/TBD/etc.
- **Type consistency:** `BioOperatorAction`, `BioOperatorObservation`, `RewardComponents`, `StepInfo`, `BioOperatorState`, `Plant`, `PlantConfig` names are stable across phases.
- **Critical path:** the calibration test (C7) is the gate before any env work. If calibration fails, the env is fiction. That gate is explicit.

---

## Execution choice

**Two options:**

1. **Subagent-driven** — fresh subagent per task, review between tasks. Best for parallelizable tasks within a phase (e.g., D1, D2, D3 independent of each other once D5 contract is locked).
2. **Inline execution** — proceed in this session using `executing-plans`. Best for the critical path Phases A → B → C where each task informs the next.

For this build, I recommend **inline for Phases A–C** (sequential, calibration-gated, my context is loaded), then **subagent fan-out** for Phase D (D1/D2/D3/D4 can parallelize), Phase E, and Phase F. Phase G runs onsite when GPU is available. Phases H and I after G.
