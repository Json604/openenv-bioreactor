"""Smoke test: ODE function returns finite vector of correct shape."""
import numpy as np

from bioperator_env.plant.ode import dydt
from bioperator_env.plant.params import build_params, ParamConfig


def _default_y0() -> np.ndarray:
    """Initial state matching indpensim_run.m §init nominal values."""
    return np.array([
        1.0,        # 0  S
        15.0,       # 1  DO2
        0.20,       # 2  O2gas
        0.0,        # 3  P
        58000.0,    # 4  V
        62000.0,    # 5  Wt
        10 ** -6.5, # 6  H+ (pH 6.5)
        297.0,      # 7  T
        0.0,        # 8  Q
        4.0,        # 9  visc
        0.0,        # 10 integ
        0.166,      # 11 a0
        0.333,      # 12 a1
        0.0,        # 13 a3
        0.0,        # 14 a4
        *([0.0] * 10),   # 15..24 n0..n9
        0.0,        # 25 nm
        0.0,        # 26 phi0
        0.038,      # 27 CO2gas
        0.0,        # 28 CO2d
        1400.0,     # 29 PAA
        1700.0,     # 30 NH3
        0.0,        # 31 mu_p_calc
        0.0,        # 32 mu_x_calc
    ], dtype=np.float64)


def _default_u() -> np.ndarray:
    return np.array([
        2,       # Inhib (full model)
        15.0,    # Fs
        30.0,    # Fg (m^3/min)
        100.0,   # RPM
        100.0,   # Fc
        1.0,     # Fh
        0.0,     # Fb
        0.0,     # Fa
        0.01,    # h_ode
        0.0,     # Fw
        0.7,     # pressure
        4.0,     # viscosity_in
        0.0,     # F_discharge
        5.0,     # Fpaa
        22.0,    # Foil
        0.0,     # NH3_shots
        1,       # Dis on
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 8 disturbances at zero
        0,       # Vis flag = simulated
    ], dtype=np.float64)


def test_dydt_shape():
    p = build_params(ParamConfig())
    dy = dydt(0.0, _default_y0(), _default_u(), p)
    assert dy.shape == (33,)
    assert np.all(np.isfinite(dy))


def test_dydt_substrate_responds_to_feed():
    """Increasing Fs should make dS/dt larger (substrate accumulates)."""
    p = build_params(ParamConfig())
    u = _default_u()
    dy_low = dydt(0.0, _default_y0(), u, p)

    u_high = u.copy()
    u_high[1] = 100.0  # Fs higher
    dy_high = dydt(0.0, _default_y0(), u_high, p)

    assert dy_high[0] > dy_low[0]  # substrate accumulates faster


def test_dydt_volume_responds_monotonically_to_feed():
    """Higher Fs => higher dV/dt (whether or not absolute sign flips)."""
    p = build_params(ParamConfig())
    u_low = _default_u()
    dy_low = dydt(0.0, _default_y0(), u_low, p)

    u_high = u_low.copy()
    u_high[1] = 200.0  # Fs much higher
    dy_high = dydt(0.0, _default_y0(), u_high, p)

    assert dy_high[4] > dy_low[4]
