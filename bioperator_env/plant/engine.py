"""Step-wise wrapper around the IndPenSim ODE system.

This is the conversion of the MATLAB full-batch loop in `indpensim.m` to a
reset-then-step interface that an RL environment can drive one action at a
time. It owns:
  - the parameter vector (built from ParamConfig per batch)
  - the disturbance time-series (8 channels, indexed by step k)
  - the closed-loop pH and T PIDs (acid/base, cooling/heating)
  - the SBC recipes (used as defaults when the agent doesn't override)
  - fault overrides (Faults=1..3) per fctrl_indpensim.m
"""
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
    """Per-batch configuration for the Plant."""
    seed: int = 42
    T_total_h: float = 230.0
    h_step: float = 0.2
    T_setpoint_K: float = 298.0
    pH_setpoint: float = 6.5
    fault_code: int = 0   # 0=none, 1=aeration, 2=pressure, 3=substrate, 4=base, 5=coolant
    randomise_params: bool = True


@dataclass
class Plant:
    """Step-wise IndPenSim plant.

    Usage:
        plant = Plant(PlantConfig(seed=42))
        plant.reset()
        for k in range(50):
            plant.step({})            # all defaults from SBC
        # or override agent-controllable variables:
            plant.step({"Fs": 80, "Fg": 60, "RPM": 110})
    """
    cfg: PlantConfig
    state: np.ndarray = field(default_factory=lambda: np.zeros(33))
    k: int = 0
    t_h: float = 0.0
    _params: list = field(default_factory=list)
    _dist: dict = field(default_factory=dict)
    _u_prev: dict = field(default_factory=dict)

    # ----- public API -----

    def reset(self) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.randomise_params:
            cfg = ParamConfig(
                x0_mux=0.41 + 0.025 * rng.standard_normal(),
                x0_mup=0.041 + 0.0025 * rng.standard_normal(),
                alpha_kla=85.0 + 10.0 * rng.standard_normal(),
                N_conc_paa=2.0 * 75000.0 + 2000.0 * rng.standard_normal(),
                PAA_c=530000.0 + 20000.0 * rng.standard_normal(),
            )
        else:
            cfg = ParamConfig()
        self._params = build_params(cfg)
        self._dist = generate_disturbances(self.cfg.T_total_h, self.cfg.h_step, self.cfg.seed)

        # Initial state per indpensim_run.m §init.
        intial = 0.5 + 0.05 * rng.standard_normal()
        S0 = 1.0 + 0.1 * rng.standard_normal()
        DO0 = 15.0 + 0.5 * rng.standard_normal()
        O2gas0 = 0.20 + 0.05 * rng.standard_normal()
        V0 = 5.8e4 + 500.0 * rng.standard_normal()
        Wt0 = 6.2e4 + 500.0 * rng.standard_normal()
        pH0 = 6.5 + 0.1 * rng.standard_normal()
        T0 = 297.0 + 0.5 * rng.standard_normal()
        CO2gas0 = 0.038 + 0.001 * rng.standard_normal()
        PAA0 = 1400.0 + 50.0 * rng.standard_normal()
        NH3_0 = 1700.0 + 50.0 * rng.standard_normal()

        self.state = np.array([
            S0, DO0, O2gas0, 0.0, V0, Wt0,
            10 ** (-pH0), T0, 0.0, 4.0, 0.0,
            intial * (1.0 / 3.0), intial * (2.0 / 3.0), 0.0, 0.0,
            *([0.0] * 10), 0.0, 0.0,
            CO2gas0, 0.0,
            PAA0, NH3_0,
            0.0, 0.0,
        ], dtype=np.float64)
        self.k = 0
        self.t_h = 0.0
        self._u_prev = {
            "Fa": 0.0, "Fb": 0.0, "Fc": 100.0, "Fh": 1.0,
            "ph_err_prev": 0.0, "T_err_prev": 0.0,
            "ph_prev": pH0, "ph_prev_prev": pH0,
            "T_prev": T0, "T_prev_prev": T0,
        }
        return self.state.copy()

    def step(self, agent_controls: dict) -> np.ndarray:
        """Apply one control step and integrate ODEs forward by h_step."""
        u = self._build_u_vector(agent_controls)
        try:
            sol = solve_ivp(
                fun=lambda t, y: dydt(t, y, u, self._params),
                t_span=(self.t_h, self.t_h + self.cfg.h_step),
                y0=self.state,
                method="BDF",
                rtol=1e-5,
                atol=1e-7,
                max_step=self.cfg.h_step / 20.0,
            )
            if not sol.success:
                # Fallback to LSODA for robustness if BDF fails on some steps
                sol = solve_ivp(
                    fun=lambda t, y: dydt(t, y, u, self._params),
                    t_span=(self.t_h, self.t_h + self.cfg.h_step),
                    y0=self.state,
                    method="LSODA",
                    rtol=1e-4,
                    atol=1e-6,
                )
        except Exception:  # numerical blowup
            sol = None

        if sol is None or not sol.success:
            # Hold state; flag will be visible to env via NaN check
            new_state = self.state.copy()
        else:
            new_state = sol.y[:, -1]

        # Numerical floor matching MATLAB indpensim.m: ONLY non-positive values
        # get bumped to 1e-3; positive small values (e.g. H+ ~ 1e-7) are kept.
        new_state = np.where(new_state <= 0.0, 1e-3, new_state)
        self.state = new_state
        # DO2 has explicit MATLAB clamp: if <2, set to 1
        if self.state[1] < 2.0:
            self.state[1] = 1.0

        self.k += 1
        self.t_h += self.cfg.h_step
        return self.state.copy()

    # ----- helpers -----

    def pH(self) -> float:
        return float(-np.log10(max(self.state[6], 1e-30)))

    def _build_u_vector(self, agent_controls: dict) -> np.ndarray:
        # Recipe defaults at this step
        Fs = float(agent_controls.get("Fs", sbc_lookup("Fs", self.k)))
        Fg = float(agent_controls.get("Fg", sbc_lookup("Fg", self.k)))
        RPM = float(agent_controls.get("RPM", 100.0))
        Fpaa = float(agent_controls.get("Fpaa", sbc_lookup("Fpaa", self.k)))
        Foil = float(agent_controls.get("Foil", sbc_lookup("Foil", self.k)))
        Fw = float(sbc_lookup("Fw", self.k))
        pressure = float(sbc_lookup("pressure", self.k))
        F_discharge = float(sbc_lookup("F_discharge", self.k))

        # pH PID (acid/base) -- mirrors fctrl_indpensim.m §pH
        pH_now = self.pH()
        pH_err = self.cfg.pH_setpoint - pH_now
        if pH_err >= -0.05:
            Fb = pid_step(self._u_prev["Fb"], pH_err, self._u_prev["ph_err_prev"],
                          pH_now, self._u_prev["ph_prev"], self._u_prev["ph_prev_prev"],
                          0.0, 225.0, 8e-2, 4e-5, 8.0, self.cfg.h_step)
            Fa = 0.0
        else:
            Fa = pid_step(self._u_prev["Fa"], pH_err, self._u_prev["ph_err_prev"],
                          pH_now, self._u_prev["ph_prev"], self._u_prev["ph_prev_prev"],
                          0.0, 225.0, 8e-2, 12.5, 0.125, self.cfg.h_step)
            Fb = self._u_prev["Fb"] * 0.5

        # Temperature PID -- mirrors fctrl_indpensim.m §T
        T_now = float(self.state[7])
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

        # Apply faults if active
        Fs, Fg, pressure, Fb, Fc = self._apply_faults(Fs, Fg, pressure, Fb, Fc)

        # Disturbances at this k
        idx = min(self.k, len(self._dist["distMuP"]) - 1)
        d = [self._dist[name][idx] for name in
             ("distMuP", "distMuX", "distcs", "distcoil", "distabc",
              "distPAA", "distTcin", "distO_2in")]

        u = np.array([
            2,                          # Inhib full
            Fs, Fg, RPM, Fc, Fh, Fb, Fa,
            self.cfg.h_step / 20.0,     # h_ode
            Fw, pressure, 4.0,          # viscosity_in (unused when vis_flag=0)
            F_discharge, Fpaa, Foil,
            0.0,                        # NH3_shots
            1,                          # Dis on
            *d,
            0,                          # vis_flag = use simulated viscosity y[9]
        ], dtype=np.float64)

        # Update PID histories for next iteration
        self._u_prev["ph_prev_prev"] = self._u_prev["ph_prev"]
        self._u_prev["ph_prev"] = pH_now
        self._u_prev["ph_err_prev"] = pH_err
        self._u_prev["T_prev_prev"] = self._u_prev["T_prev"]
        self._u_prev["T_prev"] = T_now
        self._u_prev["T_err_prev"] = T_err
        self._u_prev["Fa"] = Fa
        self._u_prev["Fb"] = Fb
        self._u_prev["Fc"] = Fc
        self._u_prev["Fh"] = Fh
        return u

    def _apply_faults(self, Fs, Fg, pressure, Fb, Fc):
        f = self.cfg.fault_code
        k = self.k
        if f in (1, 6) and (100 <= k <= 120 or 500 <= k <= 550):
            Fg = 20.0
        if f in (2, 6) and (500 <= k <= 520 or 1000 <= k <= 1200):
            pressure = 2.0
        if f in (3, 6):
            if 100 <= k <= 150:
                Fs = 2.0
            elif 380 <= k <= 460 or 1000 <= k <= 1070:
                Fs = 20.0
        if f in (4, 6):
            if 400 <= k <= 420:
                Fb = 5.0
            elif 700 <= k <= 800:
                Fb = 10.0
        if f in (5, 6):
            if 350 <= k <= 450:
                Fc = 2.0
            elif 1200 <= k <= 1350:
                Fc = 10.0
        return Fs, Fg, pressure, Fb, Fc
