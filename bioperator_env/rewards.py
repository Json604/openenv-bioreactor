"""7-component reward system + weighted composer.

Each component is independent and clamped to a documented range. Logged
separately so reward hacking is visible: an exploited single component
shows up as one column rising while siblings collapse.

Components (per design spec §5.1):
    1. format_validity         {0, 1}   was action JSON-valid?
    2. do_safety               [-1, 1]  dissolved oxygen above safe floor?
    3. productivity            [0, 1]   penicillin growing?
    4. substrate_control       [-1, 1]  substrate in healthy band?
    5. stability               [0, 1]   T and pH near setpoints?
    6. control_effort          [-1, 0]  small / smooth deltas?
    7. terminal_yield_bonus    [0, 1]   sparse, end-of-episode

Default weights (sum to 0.90 for per-step; +0.10 terminal):
    0.05 format + 0.30 do + 0.20 prod + 0.15 sub + 0.10 stab + 0.10 effort
"""
from __future__ import annotations
from dataclasses import dataclass

from .models import RewardComponents


# ----- weights -----

DEFAULT_WEIGHTS: dict[str, float] = {
    "format_validity":      0.05,
    "do_safety":            0.30,
    "productivity":         0.20,
    "substrate_control":    0.15,
    "stability":            0.10,
    "control_effort":       0.10,
    "terminal_yield_bonus": 0.10,
}


# ----- per-component scorers -----

def score_format_validity(action_was_valid: bool) -> float:
    return 1.0 if action_was_valid else 0.0


def score_do_safety(do_pct: float, do_min_safe: float = 20.0) -> float:
    """Piecewise reward keyed off the safe-floor.

    >= do_min_safe + 5: +1.0
    >= do_min_safe:      +0.3
    >= do_min_safe - 5:  -0.5
    < do_min_safe - 5:   -1.0  (safety violation)
    """
    if do_pct >= do_min_safe + 5.0:
        return 1.0
    if do_pct >= do_min_safe:
        return 0.3
    if do_pct >= do_min_safe - 5.0:
        return -0.5
    return -1.0


def score_productivity(d_penicillin_g_L: float,
                       reference_max_per_step: float = 0.05) -> float:
    """Normalized penicillin growth rate per step. Clamped to [0, 1]."""
    if d_penicillin_g_L <= 0:
        return 0.0
    return min(d_penicillin_g_L / max(reference_max_per_step, 1e-9), 1.0)


def score_substrate_control(s_g_L: float,
                            s_min: float = 0.05,
                            s_max: float = 0.30) -> float:
    """+1 inside band, ramp to 0 at edges, -1 well outside."""
    if s_min <= s_g_L <= s_max:
        return 1.0
    if s_g_L < s_min:
        # below band: linear ramp -1 at 0, 0 at s_min
        return max(-1.0, s_g_L / s_min - 1.0)  # at s=0 -> -1; at s=s_min -> 0
    # above band
    over = s_g_L - s_max
    if over <= s_max:
        return -over / s_max  # 0 at s_max, -1 at 2*s_max
    return -1.0


def score_stability(temperature_C: float, pH: float,
                    T_target: float = 25.0, pH_target: float = 6.5,
                    T_sigma: float = 0.5, pH_sigma: float = 0.1) -> float:
    """Gaussian on tracking errors. 1.0 perfect tracking, 0.0 far away."""
    import math
    t_err = (temperature_C - T_target) / T_sigma
    p_err = (pH - pH_target) / pH_sigma
    return float(math.exp(-0.5 * (t_err ** 2 + p_err ** 2)))


def score_control_effort(action: dict) -> float:
    """Penalty: large or aggressive actuator deltas hurt."""
    feed = abs(action.get("feed_delta_L_h", 0))
    aer  = abs(action.get("aeration_delta_vvm", 0.0))
    rpm  = abs(action.get("agitation_delta_rpm", 0))
    # Normalize each to [0,1] then weighted sum.
    raw = 0.05 * (feed / 5.0) + 0.5 * (aer / 0.10) + 0.05 * (rpm / 5.0)
    return -min(raw, 1.0)   # negative = penalty


def score_terminal_yield_bonus(final_penicillin_g_L: float,
                                reference_yield_g_L: float = 20.0,
                                is_terminal: bool = False) -> float:
    if not is_terminal:
        return 0.0
    return max(0.0, min(final_penicillin_g_L / reference_yield_g_L, 1.0))


# ----- composer -----

@dataclass
class RewardContext:
    """Inputs needed by all 7 components for a single step."""
    action_was_valid: bool
    action: dict                       # parsed action (may be defaulted)
    do_pct: float
    do_min_safe: float
    d_penicillin: float                # change in P (g/L) over the step
    s_g_L: float
    s_min: float
    s_max: float
    temperature_C: float
    pH: float
    T_target: float
    pH_target: float
    final_penicillin_g_L: float        # current P (g/L)
    is_terminal: bool


def compose_reward(ctx: RewardContext,
                   weights: dict[str, float] = None) -> tuple[float, RewardComponents]:
    """Return (total_reward, RewardComponents).

    Total is clamped to [-1, 1].
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    components = RewardComponents(
        format_validity=score_format_validity(ctx.action_was_valid),
        do_safety=score_do_safety(ctx.do_pct, ctx.do_min_safe),
        productivity=score_productivity(ctx.d_penicillin),
        substrate_control=score_substrate_control(ctx.s_g_L, ctx.s_min, ctx.s_max),
        stability=score_stability(ctx.temperature_C, ctx.pH, ctx.T_target, ctx.pH_target),
        control_effort=score_control_effort(ctx.action),
        terminal_yield_bonus=score_terminal_yield_bonus(
            ctx.final_penicillin_g_L, is_terminal=ctx.is_terminal),
    )
    total = (
        w["format_validity"]      * components.format_validity
        + w["do_safety"]          * components.do_safety
        + w["productivity"]       * components.productivity
        + w["substrate_control"]  * components.substrate_control
        + w["stability"]          * components.stability
        + w["control_effort"]     * components.control_effort
        + w["terminal_yield_bonus"] * components.terminal_yield_bonus
    )
    total = max(-1.0, min(1.0, total))
    return total, components
