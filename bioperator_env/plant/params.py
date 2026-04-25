"""Port of IndPenSim/Parameter_list.m — 105-parameter constant vector.

Reference: Goldrick et al., J. Biotechnology 2015. DOI: 10.1016/j.jbiotec.2014.10.029
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ParamConfig:
    """Per-batch tunables that go into the parameter vector.

    Defaults match the nominal values in indpensim_run.m.
    """
    x0_mux: float = 0.41           # max biomass growth rate (h^-1)
    x0_mup: float = 0.041          # max penicillin growth rate (h^-1)
    alpha_kla: float = 85.0        # kla constant
    N_conc_paa: float = 150000.0   # nitrogen conc in PAA feed (mg/L) = 2*75000
    PAA_c: float = 530000.0        # PAA conc in PAA feed (mg/L)


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
        0.38, 0.34, -0.38, 0.25,            # 32-35 a, b, c, d
        0.0251,                             # 36 Henrys_c
        3.0,                                # 37 n_imp
        2.1,                                # 38 r
        0.85,                               # 39 r_imp
        5.0,                                # 40 Po
        0.1,                                # 41 epsilon
        9.81,                               # 42 g
        8.314,                              # 43 R
        0.1,                                # 44 X_crit_DO2
        0.3,                                # 45 P_crit_DO2
        1.0,                                # 46 A_inhib
        288.0, 288.0, 285.0, 333.0, 290.0,  # 47-51 Tf, Tw, Tcin, Th, Tair
        5.9, 4.18, 2430.7,                  # 52-54 C_ps, C_pw, dealta_H_evap
        36.0, 105.0,                        # 55-56 U_jacket, A_c
        1.488e4, 1.7325e5,                  # 57-58 Eg, Ed
        450.0, 0.25e30,                     # 59-60 k_g, k_d
        25.0,                               # 61 Y_QX
        0.033,                              # 62 abc
        0.0325e-5, 2.5e-11, 0.0025,         # 63-65 gamma1, gamma2, m_ph
        1e-5, 2.5e-8,                       # 66-67 K1, K2
        20000.0,                            # 68 N_conc_oil
        cfg.N_conc_paa,                     # 69 N_conc_paa
        400000.0,                           # 70 N_conc_shot
        10.0, 80.0, 0.03, 150.0,            # 71-74 Y_NX, Y_NP, m_N, X_crit_N
        cfg.PAA_c,                          # 75 PAA_c
        187.5, 37.5 * 1.2, 1.05,            # 76-78 Y_PAA_P, Y_PAA_X, m_PAA
        2400.0, 200.0,                      # 79-80 X_crit_PAA, P_crit_PAA
        -0.6429e2, -0.1825e1, 0.3649,
        0.1280, -4.9496e-4,                 # 81-85 B_1..B_5
        0.89, 0.005, 0.001, 0.0001,         # 86-89 delta_c_o, k_3, k1, k2
        1.0, 250.0,                         # 90-91 t1, t2
        0.123 * 1.1, 7570.0,                # 92-93 q_co2, X_crit_CO2
        5.24e-4, 2.88,                      # 94-95 alpha_evp, beta_T
        1540.0, 900.0, 1000.0, 1000.0,      # 96-99 pho_g, pho_oil, pho_w, pho_paa
        0.21, 0.79, 0.033,                  # 100-102 O_2_in, N2_in, C_CO2_in
        373.0, 273.0,                       # 103-104 Tv, T0
        2451.8,                             # 105 alpha_1
    ]
    p = [float(v) for v in p]
    assert len(p) == 105, f"expected 105 params, got {len(p)}"
    return p
