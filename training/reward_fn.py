"""TRL-compatible reward function that scores LLM completions on env steps.

Signature matches TRL's GRPOTrainer expectation:
    reward_fn(completions: list[str], **kwargs) -> list[float]

`kwargs` carries per-row metadata from the dataset; we read snapshot_json
to restore env state, apply the parsed action, and compute the weighted
reward total.

Each completion is scored INDEPENDENTLY -- there is no batch state. This
matches GRPO's group-relative formulation: rewards across a group are
standardized by the trainer, so absolute scale matters less than relative.
"""
from __future__ import annotations
import json
import re
from typing import Any, Optional

import numpy as np

from bioperator_env.models import BioOperatorAction
from bioperator_env.plant.engine import Plant, PlantConfig
from bioperator_env.plant.ode import dydt
from bioperator_env.rewards import RewardContext, compose_reward
from scipy.integrate import solve_ivp


_ACTION_RE = re.compile(r"\{[^{}]*\}", flags=re.DOTALL)


def _parse_action(text: str) -> tuple[Optional[dict], bool]:
    """Find first JSON object in `text` and parse as a BioOperatorAction.
    Returns (parsed_dict, was_valid)."""
    m = _ACTION_RE.search(text)
    if not m:
        return None, False
    try:
        raw = json.loads(m.group(0))
        a = BioOperatorAction(**raw)
        return a.model_dump(), True
    except Exception:
        return None, False


def _restore_plant(snapshot: dict) -> Plant:
    cfg = PlantConfig(seed=snapshot["seed"], T_total_h=230.0, h_step=0.2,
                      randomise_params=False)
    plant = Plant(cfg)
    plant.reset()  # initializes _params and _dist
    plant.state = np.array(snapshot["plant_state"], dtype=np.float64)
    plant.k = int(snapshot["plant_k"])
    plant.t_h = float(snapshot["plant_t_h"])
    plant._u_prev = dict(snapshot["plant_u_prev"])
    return plant


def _step_plant_with_action(plant: Plant, snapshot: dict,
                             action: dict) -> tuple[float, float, float]:
    """Apply the candidate action to the restored plant, advance one step.
    Returns (DO_pct, S_g_L, P_g_L) after the step."""
    V_m3 = max(plant.state[4] / 1000.0, 1.0)
    Fs = max(0.0, min(200.0, snapshot["Fs"] + action["feed_delta_L_h"]))
    Fg = max(10.0, min(200.0,
                       snapshot["Fg_m3min"] + action["aeration_delta_vvm"] * V_m3))
    RPM = max(80.0, min(200.0, snapshot["RPM"] + action["agitation_delta_rpm"]))
    plant.step({"Fs": Fs, "Fg": Fg, "RPM": RPM})
    DO_mgL = float(plant.state[1])
    DOstar = 13.4
    DO_pct = float(max(0.0, min(100.0, 100.0 * DO_mgL / DOstar)))
    return DO_pct, float(plant.state[0]), float(plant.state[3])


def reward_fn(completions: list[Any], **kwargs) -> list[float]:
    """TRL-compatible reward.

    `completions` may be a list of strings (raw text) or a list of message
    dicts. `kwargs` should include `snapshot_json` (per-row, list aligned
    with completions).
    """
    # Some TRL versions send completions as list of [{"role":..., "content":...}]
    flat = []
    for c in completions:
        if isinstance(c, str):
            flat.append(c)
        elif isinstance(c, list) and c and isinstance(c[-1], dict):
            flat.append(c[-1].get("content", ""))
        else:
            flat.append(str(c))

    snapshots_json = kwargs.get("snapshot_json", [])
    if not isinstance(snapshots_json, list):
        snapshots_json = [snapshots_json] * len(flat)

    rewards = []
    for completion_text, snap_json in zip(flat, snapshots_json):
        try:
            snap = json.loads(snap_json) if isinstance(snap_json, str) else snap_json
        except Exception:
            rewards.append(-1.0)
            continue
        action, valid = _parse_action(completion_text)
        if not valid:
            # Default no-op so reward components are still computable
            action = {"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0,
                      "agitation_delta_rpm": 0, "reason": None}
        try:
            plant = _restore_plant(snap)
            DO_pct, S, P_after = _step_plant_with_action(plant, snap, action)
        except Exception:
            rewards.append(-1.0)
            continue

        ctx = RewardContext(
            action_was_valid=valid,
            action=action,
            do_pct=DO_pct,
            do_min_safe=snap["do_min_safe"],
            d_penicillin=P_after - snap["last_P"],
            s_g_L=S,
            s_min=snap["s_min"],
            s_max=snap["s_max"],
            temperature_C=float(plant.state[7]) - 273.15,
            pH=float(-np.log10(max(plant.state[6], 1e-30))),
            T_target=snap["T_target"],
            pH_target=snap["pH_target"],
            final_penicillin_g_L=P_after,
            is_terminal=False,
        )
        total, _ = compose_reward(ctx)
        rewards.append(float(total))
    return rewards


# Optional: a simpler format-only reward for Stage 0 of the curriculum
def format_only_reward_fn(completions: list[Any], **kwargs) -> list[float]:
    flat = []
    for c in completions:
        if isinstance(c, str):
            flat.append(c)
        elif isinstance(c, list) and c and isinstance(c[-1], dict):
            flat.append(c[-1].get("content", ""))
        else:
            flat.append(str(c))
    out = []
    for text in flat:
        _, valid = _parse_action(text)
        out.append(1.0 if valid else 0.0)
    return out
