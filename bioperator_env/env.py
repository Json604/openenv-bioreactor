"""BioOperatorEnv -- Gym-style operator-training environment.

Wraps the calibrated IndPenSim Python plant in a SCADA-style observation /
structured-JSON action interface for an LLM agent.

Episode shape:
  - reset(): fast-forwards the plant to scenario.start_t_h, returns initial Obs.
  - step(action): translates the agent delta into absolute manipulated
    variables, advances the plant by 0.2 h, computes reward components,
    returns (Observation, reward, done, info_dict).

Action -> control mapping (anti-cheat: agent never touches pH/T loops):
  - feed_delta_L_h          -> Fs += delta, clipped to [0, 200] L/h
  - aeration_delta_vvm      -> Fg += delta * V_m3, clipped to [10, 200] m^3/min
  - agitation_delta_rpm     -> RPM += delta, clipped to [80, 200]
"""
from __future__ import annotations
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .alarms import evaluate_alarm
from .models import (BioOperatorAction, BioOperatorObservation,
                     BioOperatorState, RewardComponents, StepInfo)
from .plant.engine import Plant, PlantConfig
from .plant.recipe import sbc_lookup
from .rewards import RewardContext, compose_reward
from .scenarios import TaskSpec, get_task
from .trends import trend_label


# Action -> absolute-MV safety clips
FS_MIN, FS_MAX = 0.0, 200.0
FG_MIN_M3MIN, FG_MAX_M3MIN = 10.0, 200.0
RPM_MIN, RPM_MAX = 80.0, 200.0
TREND_WINDOW = 5
DO_CRITICAL_PCT = 5.0       # below this, terminate early
SUSTAINED_DO_LOW_STEPS = 3   # consecutive steps below safe -> safety violation


@dataclass
class BioOperatorEnv:
    task_id: str = "do-recovery-medium"
    seed: int = 42

    spec: TaskSpec = field(init=False)
    plant: Plant = field(init=False)
    step_count: int = 0
    cumulative_reward: float = 0.0
    component_history: list[dict] = field(default_factory=list)
    safety_violations: int = 0
    _last_action: Optional[dict] = None
    _consec_low_do: int = 0
    _last_P: float = 0.0

    # current absolute MVs (mutated by agent deltas)
    _Fs: float = 0.0
    _Fg_m3min: float = 0.0
    _RPM: float = 100.0

    # trend histories
    _hist: dict[str, deque] = field(default_factory=dict)

    def __post_init__(self):
        self.spec = get_task(self.task_id)
        self.plant = Plant(PlantConfig(
            seed=self.seed,
            T_total_h=230.0,
            h_step=0.2,
            T_setpoint_K=self.spec.setpoints["temperature_target_C"] + 273.15,
            pH_setpoint=self.spec.setpoints["pH_target"],
            fault_code=self.spec.fault_code,
            randomise_params=True,
        ))

    # ---------- public API ----------

    def reset(self, *, seed: Optional[int] = None,
              task_id: Optional[str] = None) -> BioOperatorObservation:
        if task_id is not None:
            self.task_id = task_id
            self.spec = get_task(task_id)
        if seed is not None:
            self.seed = seed
        self.plant = Plant(PlantConfig(
            seed=self.seed, T_total_h=230.0, h_step=0.2,
            T_setpoint_K=self.spec.setpoints["temperature_target_C"] + 273.15,
            pH_setpoint=self.spec.setpoints["pH_target"],
            fault_code=self.spec.fault_code, randomise_params=True,
        ))
        self.plant.reset()

        # Fast-forward to scenario.start_t_h with the SBC running unaltered
        n_warmup = int(round(self.spec.start_t_h / 0.2))
        for _ in range(n_warmup):
            self.plant.step({})

        # Initialize agent-controllable MVs from current SBC values
        self._Fs = float(sbc_lookup("Fs", self.plant.k))
        self._Fg_m3min = float(sbc_lookup("Fg", self.plant.k))
        self._RPM = 100.0

        self.step_count = 0
        self.cumulative_reward = 0.0
        self.component_history = []
        self.safety_violations = 0
        self._last_action = None
        self._consec_low_do = 0
        self._last_P = float(self.plant.state[3])
        self._hist = {k: deque(maxlen=TREND_WINDOW) for k in
                      ("DO", "pH", "temperature", "substrate")}
        self._record_history()
        return self._build_observation()

    def step(self, action: Any) -> tuple[BioOperatorObservation, float, bool, dict]:
        # 1) Parse + validate action
        parsed, valid = self._parse_action(action)

        # 2) Translate deltas to absolute MVs (with clipping)
        self._apply_action(parsed)

        # 3) Advance plant by one step
        self.plant.step({"Fs": self._Fs, "Fg": self._Fg_m3min, "RPM": self._RPM})

        # 4) Compute reward
        is_terminal_step = (self.step_count + 1 >= self.spec.max_steps)
        reward, comps, safety_violated = self._compute_reward(parsed, valid, is_terminal_step)

        # 5) Update bookkeeping
        self.step_count += 1
        self.cumulative_reward += reward
        self.component_history.append(comps.model_dump())
        self._last_action = parsed
        self._last_P = float(self.plant.state[3])
        self._record_history()

        # 6) Termination
        done, done_reason = self._check_done(safety_violated)

        # 7) Success criterion (recovered DO + substrate in band over last 10 steps)
        success = self._check_success() if done else False

        info = {
            "reward_total": reward,
            "reward_components": comps.model_dump(),
            "safety_violation": safety_violated,
            "success": success,
            "done_reason": done_reason,
            "format_valid": valid,
            "absolute_controls": {
                "Fs_L_h": self._Fs,
                "Fg_m3_min": self._Fg_m3min,
                "RPM": self._RPM,
            },
        }
        obs = self._build_observation()
        return obs, reward, done, info

    def state(self) -> BioOperatorState:
        return BioOperatorState(
            task_id=self.task_id, seed=self.seed,
            step_count=self.step_count, time_h=float(self.plant.t_h),
            ode_state=[float(x) for x in self.plant.state],
            last_action=self._last_action,
            cumulative_reward=self.cumulative_reward,
            component_history=self.component_history,
            safety_violations=self.safety_violations,
        )

    # ---------- helpers ----------

    def _parse_action(self, action: Any) -> tuple[dict, bool]:
        """Return (parsed_dict, was_valid). Falls back to no-op if invalid."""
        try:
            if isinstance(action, BioOperatorAction):
                return action.model_dump(), True
            if isinstance(action, dict):
                a = BioOperatorAction(**action)
                return a.model_dump(), True
            if isinstance(action, str):
                a = BioOperatorAction(**json.loads(action))
                return a.model_dump(), True
        except Exception:
            pass
        # Default: do nothing
        return {"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0,
                "agitation_delta_rpm": 0, "reason": None}, False

    def _apply_action(self, action: dict) -> None:
        V_m3 = max(self.plant.state[4] / 1000.0, 1.0)
        self._Fs = max(FS_MIN, min(FS_MAX, self._Fs + action["feed_delta_L_h"]))
        delta_fg = action["aeration_delta_vvm"] * V_m3
        self._Fg_m3min = max(FG_MIN_M3MIN, min(FG_MAX_M3MIN, self._Fg_m3min + delta_fg))
        self._RPM = max(RPM_MIN, min(RPM_MAX, self._RPM + action["agitation_delta_rpm"]))

    def _compute_reward(self, parsed_action: dict, valid: bool,
                        is_terminal: bool) -> tuple[float, RewardComponents, bool]:
        DOstar = self._dostar()
        do_pct = self._do_to_pct(self.plant.state[1], DOstar)

        s_min = self.spec.setpoints.get("substrate_min_g_L", 0.05)
        s_max = self.spec.setpoints.get("substrate_max_g_L", 0.30)
        ctx = RewardContext(
            action_was_valid=valid,
            action=parsed_action,
            do_pct=do_pct,
            do_min_safe=self.spec.setpoints["DO_min_safe_pct"],
            d_penicillin=float(self.plant.state[3]) - self._last_P,
            s_g_L=float(self.plant.state[0]),
            s_min=s_min,
            s_max=s_max,
            temperature_C=float(self.plant.state[7]) - 273.15,
            pH=float(self.plant.pH()),
            T_target=self.spec.setpoints["temperature_target_C"],
            pH_target=self.spec.setpoints["pH_target"],
            final_penicillin_g_L=float(self.plant.state[3]),
            is_terminal=is_terminal,
        )
        total, comps = compose_reward(ctx)
        # Safety: DO below critical floor counts as violation
        safety_violated = do_pct < DO_CRITICAL_PCT
        if safety_violated:
            self.safety_violations += 1
        return total, comps, safety_violated

    def _check_done(self, safety_violated: bool) -> tuple[bool, str]:
        # Sustained low DO -> early termination
        do_pct = self._do_to_pct(self.plant.state[1], self._dostar())
        if do_pct < self.spec.setpoints["DO_min_safe_pct"]:
            self._consec_low_do += 1
        else:
            self._consec_low_do = 0
        if self._consec_low_do >= SUSTAINED_DO_LOW_STEPS and do_pct < DO_CRITICAL_PCT:
            return True, "safety_violation"
        if self.step_count >= self.spec.max_steps:
            return True, "timeout"
        if not np.all(np.isfinite(self.plant.state)):
            return True, "integrator_failure"
        return False, ""

    def _check_success(self) -> bool:
        if len(self.component_history) < 10:
            return False
        recent = self.component_history[-10:]
        do_safe_count = sum(1 for c in recent if c["do_safety"] >= 0.3)
        sub_safe_count = sum(1 for c in recent if c["substrate_control"] >= 0.5)
        return do_safe_count >= 8 and sub_safe_count >= 6

    def _dostar(self) -> float:
        """Saturation DO (mg/L) at current pressure, used to convert to %."""
        # Approximation matching ode.py: DOstar = total_pressure * O_2_in / Henrys_c
        # With pressure_top ≈ 1.6 bar, DOstar ≈ 13.4 mg/L. Use 13.4 for stable %.
        return 13.4

    def _do_to_pct(self, do_mg_L: float, do_star_mg_L: float) -> float:
        """Convert dissolved O2 from mg/L to % saturation (clamped 0..100)."""
        return float(max(0.0, min(100.0, 100.0 * do_mg_L / max(do_star_mg_L, 1e-6))))

    def _record_history(self) -> None:
        DO_pct = self._do_to_pct(self.plant.state[1], self._dostar())
        self._hist["DO"].append(DO_pct)
        self._hist["pH"].append(float(self.plant.pH()))
        self._hist["temperature"].append(float(self.plant.state[7]) - 273.15)
        self._hist["substrate"].append(float(self.plant.state[0]))

    def _build_observation(self) -> BioOperatorObservation:
        DOstar = self._dostar()
        do_pct = self._do_to_pct(self.plant.state[1], DOstar)
        V_m3 = max(self.plant.state[4] / 1000.0, 1.0)

        measurements = {
            "temperature_C": float(self.plant.state[7] - 273.15),
            "pH": float(self.plant.pH()),
            "dissolved_oxygen_pct": float(do_pct),
            "substrate_g_L": float(self.plant.state[0]),
            "volume_L": float(self.plant.state[4]),
            "OUR": float(self._our()),
            "CER": float(self._cer()),
            "CO2_outgas_pct": float(self.plant.state[27] * 100.0),
            "O2_outgas_pct": float(self.plant.state[2] * 100.0),
        }
        current_controls = {
            "feed_rate_L_h": float(self._Fs),
            "aeration_rate_vvm": float(self._Fg_m3min / V_m3),
            "agitation_rpm": float(self._RPM),
            "cooling_valve_pct": float(self._estimate_cooling_pct()),
            "pressure_bar": float(sbc_lookup("pressure", self.plant.k)),
        }
        recent_trends = {k: trend_label(list(v)) for k, v in self._hist.items()}
        alarm = evaluate_alarm(measurements, self.spec.setpoints)

        return BioOperatorObservation(
            time_h=float(self.plant.t_h),
            batch_phase=self._phase_label(),
            measurements=measurements,
            setpoints_or_limits=dict(self.spec.setpoints),
            current_controls=current_controls,
            recent_trends=recent_trends,
            alarm=alarm,
            previous_action=self._last_action,
            offline_lab=self._offline_lab(),
            instruction=self.spec.description,
        )

    def _our(self) -> float:
        # Approximation: oxygen uptake rate from biomass + product
        Y_O2_X, Y_O2_P = 650.0, 160.0
        X_t = self.plant.state[11] + self.plant.state[12] + self.plant.state[13] + self.plant.state[14]
        return -17.5 * X_t / 1000.0 - 0.0  # rough proxy in g O2 / L / h

    def _cer(self) -> float:
        X_t = self.plant.state[11] + self.plant.state[12]
        return float(0.123 * 1.1 * X_t * self.plant.state[4] / 1e5)

    def _estimate_cooling_pct(self) -> float:
        # Cool valve approx 0..100% by Fc relative to limit.
        return min(100.0, max(0.0, self.plant._u_prev.get("Fc", 100.0) / 15.0))

    def _phase_label(self) -> str:
        t = self.plant.t_h
        if t < 5:
            return "inoculation"
        if t < 30:
            return "growth"
        if t < 200:
            return "production"
        return "stationary"

    def _offline_lab(self) -> Optional[dict]:
        # Mirror MATLAB: every 12 h, 4 h delay
        # Convert to step boundary; 12 h = 60 steps; 4 h = 20 steps lag.
        if self.plant.k < 20:
            return None
        if (self.plant.k % 60) != 0:
            return None
        # ode_state is "now"; offline data is the snapshot from 4h ago
        # We don't carry that history here; report current values flagged as lab.
        return {
            "biomass_g_L": float(sum(self.plant.state[11:15])),
            "penicillin_g_L": float(self.plant.state[3]),
            "PAA_mg_L": float(self.plant.state[29]),
            "lag_h": 4.0,
        }
