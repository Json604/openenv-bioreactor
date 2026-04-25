"""PID controllers for pH and temperature loops.

Ports IndPenSim/PIDSimple3.m exactly. Incremental form:

    u_k = u_{k-1} + Kp * (P + I + D)

with
    P = e_k - e_{k-1}
    I = e_k * h / Ti        (only when Ti > 1e-7)
    D = -Td/h * (y_k - 2 y_{k-1} + y_{k-2})   (only when Td > 0.001)

Output saturated to [u_min, u_max]. Derivative computed on the PV (not the
error) to avoid derivative kick on setpoint changes.

These loops run UNDER the agent's actions (the agent controls feed/aeration/
agitation; pH and T are kept on autopilot, just like a real plant operator
trusts the loop tuner not to fight every PID).
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
    """One incremental PID update; ports PIDSimple3.m exactly."""
    P = err - err_prev
    I = (err * h / Ti) if Ti > 1e-7 else 0.0
    D = (-Td / h * (y - 2.0 * y_prev + y_prev_prev)) if Td > 0.001 else 0.0
    u_new = u_prev + Kp * (P + I + D)
    if u_new > u_max:
        u_new = u_max
    elif u_new < u_min:
        u_new = u_min
    return float(u_new)
