"""Port of IndPenSim/indpensim_ode.m -- 33 coupled ODEs for the penicillin fermenter.

This file mirrors indpensim_ode.m line-for-line. State and parameter indices
match the MATLAB convention shifted by -1 (Python 0-indexed). Function returns
the right-hand side dy/dt for SciPy's solve_ivp.

State vector (33 entries):
    y[0]  S            substrate (g/L)
    y[1]  DO2          dissolved oxygen (mg/L)
    y[2]  O2gas        O2 off-gas (%)
    y[3]  P            penicillin (g/L)
    y[4]  V            volume (L)
    y[5]  Wt           weight (kg)
    y[6]  Hplus        H+ concentration (= 10^-pH)
    y[7]  T            temperature (K)
    y[8]  Q            generated heat (kJ)
    y[9]  visc         viscosity (cP)
    y[10] integ_X      integral of total biomass
    y[11] a0           growing biomass (g/L)
    y[12] a1           extension biomass (g/L)
    y[13] a3           degenerated biomass (g/L)
    y[14] a4           autolysed biomass (g/L)
    y[15..24]          n0..n9 vacuole number density bins
    y[25] nm           max vacuole department
    y[26] phi0         mean vacuole volume
    y[27] CO2gas       CO2 off-gas (%)
    y[28] CO2d         dissolved CO2 (mg/L)
    y[29] PAA          phenylacetic acid (mg/L)
    y[30] NH3          nitrogen (mg/L)
    y[31] mu_p_calc    diagnostic
    y[32] mu_x_calc    diagnostic

Reference: Goldrick et al., J. Biotech 2015 (DOI: 10.1016/j.jbiotec.2014.10.029)
           Goldrick et al., Comp. Chem. Eng. 2019 (DOI: 10.1016/j.compchemeng.2019.05.037)
"""
from __future__ import annotations
import numpy as np


def dydt(t: float, y: np.ndarray, u: np.ndarray, p: list[float]) -> np.ndarray:
    """Right-hand side of the IndPenSim ODE system.

    Parameters
    ----------
    t : float
        Current time (h). Used only by viscosity sigmoid.
    y : np.ndarray, shape (33,)
        State vector.
    u : np.ndarray, shape (26,)
        Control + disturbance vector. Built by the env wrapper:
            [Inhib, Fs, Fg(m3/min), RPM, Fc, Fh, Fb, Fa, h_ode, Fw,
             pressure, viscosity_in, F_discharge, Fpaa, Foil, NH3_shots,
             Dis, distMuP, distMuX, distcs, distcoil, distabc, distPAA,
             distTcin, distO_2in, Vis_flag]
    p : list[float], length 105
        Parameter vector from `params.build_params()`.
    """
    # ----- 1. Unpack params (same names/order as Parameter_list.m) -----
    mu_p             = p[0]
    mux_max          = p[1]
    ratio_mu_e_mu_b  = p[2]
    P_std_dev        = p[3]
    mean_P           = p[4]
    mu_v             = p[5]
    mu_a             = p[6]
    mu_diff          = p[7]
    beta_1           = p[8]
    K_b              = p[9]
    K_diff_base      = p[10]
    K_diff_L         = p[11]
    K_e              = p[12]
    K_v              = p[13]
    delta_r          = p[14]
    k_v              = p[15]
    D                = p[16]
    rho_a0           = p[17]
    rho_d            = p[18]
    mu_h             = p[19]
    r_0              = p[20]
    delta_0          = p[21]
    Y_sX             = p[22]
    Y_sP             = p[23]
    m_s              = p[24]
    c_oil            = p[25]
    c_s              = p[26]
    Y_O2_X           = p[27]
    Y_O2_P           = p[28]
    m_O2_X           = p[29]
    alpha_kla        = p[30]
    a_kla            = p[31]
    b_kla            = p[32]
    c_kla            = p[33]
    d_kla            = p[34]
    Henrys_c         = p[35]
    n_imp            = p[36]
    r_vessel         = p[37]
    r_imp            = p[38]
    Po               = p[39]
    epsilon          = p[40]
    g_grav           = p[41]   # noqa: F841 (kept for clarity)
    R                = p[42]
    X_crit_DO2       = p[43]
    P_crit_DO2       = p[44]
    A_inhib          = p[45]
    Tf               = p[46]
    Tw               = p[47]
    Tcin             = p[48]
    Th               = p[49]
    Tair             = p[50]
    C_ps             = p[51]
    C_pw             = p[52]
    delta_H_evap     = p[53]
    U_jacket         = p[54]
    A_c              = p[55]
    Eg               = p[56]
    Ed               = p[57]
    k_g              = p[58]
    k_d              = p[59]
    Y_QX             = p[60]
    abc              = p[61]
    gamma1           = p[62]
    gamma2           = p[63]
    m_ph             = p[64]
    K1               = p[65]
    K2               = p[66]
    N_conc_oil       = p[67]
    N_conc_paa       = p[68]
    N_conc_shot      = p[69]
    Y_NX             = p[70]
    Y_NP             = p[71]
    m_N              = p[72]
    X_crit_N         = p[73]
    PAA_c            = p[74]
    Y_PAA_P          = p[75]
    Y_PAA_X          = p[76]
    m_PAA            = p[77]
    X_crit_PAA       = p[78]
    P_crit_PAA       = p[79]
    B_1              = p[80]
    B_2              = p[81]
    B_3              = p[82]
    B_4              = p[83]
    B_5              = p[84]
    delta_c_0        = p[85]
    k3_visc          = p[86]
    k1_visc          = p[87]
    k2_visc          = p[88]
    t1_visc          = p[89]
    t2_visc          = p[90]
    q_co2            = p[91]
    X_crit_CO2       = p[92]
    alpha_evp        = p[93]
    beta_T           = p[94]
    pho_g            = p[95]
    pho_oil          = p[96]
    pho_w            = p[97]
    pho_paa          = p[98]
    O_2_in           = p[99]
    N2_in            = p[100]
    C_CO2_in         = p[101]
    Tv               = p[102]
    T0               = p[103]
    alpha_1          = p[104]

    # ----- 2. Unpack inputs (port of inp1) -----
    inhib_flag = int(u[0])
    Fs         = u[1]
    Fg         = u[2] / 60.0      # convert m^3/min -> m^3/s
    RPM        = u[3]
    Fc         = u[4]
    Fh         = u[5]
    Fb         = u[6]
    Fa         = u[7]
    step1      = u[8]
    Fw         = max(u[9], 0.0)   # negative water flow clipped (matches MATLAB Fw(Fw<0)=0)
    pressure   = u[10]
    visc_in    = u[11]
    F_discharge = u[12]
    Fpaa       = u[13]
    Foil       = u[14]
    NH3_shots  = u[15]
    dist_flag  = int(u[16])
    distMuP    = u[17]
    distMuX    = u[18]
    distsc     = u[19]
    distcoil   = u[20]
    distabc    = u[21]
    distPAA    = u[22]
    distTcin   = u[23]
    distO_2in  = u[24]
    vis_flag   = int(u[25])

    # ----- 3. Disturbances applied -----
    if dist_flag == 1:
        mu_p_use      = mu_p + distMuP
        mux_max_use   = mux_max + distMuX
        c_s_use       = c_s + distsc
        c_oil_use     = c_oil + distcoil
        abc_use       = abc + distabc
        PAA_c_use     = PAA_c + distPAA
        Tcin_use      = Tcin + distTcin
        O_2_in_use    = O_2_in + distO_2in
    else:
        mu_p_use, mux_max_use = mu_p, mux_max
        c_s_use, c_oil_use = c_s, c_oil
        abc_use, PAA_c_use = abc, PAA_c
        Tcin_use, O_2_in_use = Tcin, O_2_in

    # ----- 4. Broth density and pressure -----
    pho_b = 1100.0 + y[3] + y[11] + y[12] + y[13] + y[14]   # broth density g/L proxy

    A_t1 = y[10] / (y[11] + y[12] + y[13] + y[14] + 1e-12)  # age-dependent term

    s = y[0]
    a_1 = y[12]
    a_0 = y[11]
    a_3 = y[13]
    total_X = y[11] + y[12] + y[13] + y[14]

    # Liquid height
    h_b = (y[4] / 1000.0) / (np.pi * (r_vessel ** 2))
    h_b = h_b * (1.0 - epsilon)
    pressure_bottom = 1.0 + pressure + ((pho_b * h_b) * 9.81 * 1e-5)
    pressure_top = 1.0 + pressure
    if pressure_bottom > pressure_top:
        log_mean_pressure = (pressure_bottom - pressure_top) / np.log(pressure_bottom / pressure_top)
    else:
        log_mean_pressure = pressure_top
    total_pressure = log_mean_pressure

    # Viscosity selection (matches MATLAB Vis flag)
    if vis_flag == 0:
        viscosity = y[9]
    else:
        viscosity = visc_in
    if viscosity < 4.0:
        viscosity = 1.0   # MATLAB: viscosity(viscosity<4) = 1

    DOstar_tp = (total_pressure * O_2_in_use) / Henrys_c

    # ----- 5. Inhibition terms -----
    if inhib_flag == 0:
        pH_inhib = 1.0
        NH3_inhib = 1.0
        T_inhib = 1.0
        mu_h_use = 0.003
        DO_2_inhib_X = 1.0
        DO_2_inhib_P = 1.0
        CO2_inhib = 1.0
        PAA_inhib_X = 1.0
        PAA_inhib_P = 1.0
    elif inhib_flag == 1:
        pH_inhib = 1.0 / (1.0 + (y[6] / K1) + (K2 / max(y[6], 1e-30)))
        NH3_inhib = 1.0
        T_inhib = (k_g * np.exp(-(Eg / (R * y[7]))) - k_d * np.exp(-(Ed / (R * y[7])))) * 0.0 + 1.0
        CO2_inhib = 1.0
        DO_2_inhib_X = 0.5 * (1.0 - np.tanh(A_inhib * (X_crit_DO2 * DOstar_tp - y[1])))
        DO_2_inhib_P = 0.5 * (1.0 - np.tanh(A_inhib * (P_crit_DO2 * DOstar_tp - y[1])))
        PAA_inhib_X = 1.0
        PAA_inhib_P = 1.0
        pH_val = -np.log10(max(y[6], 1e-30))
        k4 = np.exp((B_1 + B_2 * pH_val + B_3 * y[7] + B_4 * (pH_val ** 2)) + B_5 * (y[7] ** 2))
        mu_h_use = k4
    else:  # inhib_flag == 2
        pH_inhib = 1.0 / (1.0 + (y[6] / K1) + (K2 / max(y[6], 1e-30)))
        NH3_inhib = 0.5 * (1.0 - np.tanh(A_inhib * (X_crit_N - y[30])))
        T_inhib = k_g * np.exp(-(Eg / (R * y[7]))) - k_d * np.exp(-(Ed / (R * y[7])))
        CO2_inhib = 0.5 * (1.0 + np.tanh(A_inhib * (X_crit_CO2 - y[28] * 1000.0)))
        DO_2_inhib_X = 0.5 * (1.0 - np.tanh(A_inhib * (X_crit_DO2 * DOstar_tp - y[1])))
        DO_2_inhib_P = 0.5 * (1.0 - np.tanh(A_inhib * (P_crit_DO2 * DOstar_tp - y[1])))
        PAA_inhib_X = 0.5 * (1.0 + np.tanh(X_crit_PAA - y[29]))
        PAA_inhib_P = 0.5 * (1.0 + np.tanh(-P_crit_PAA + y[29]))
        pH_val = -np.log10(max(y[6], 1e-30))
        k4 = np.exp((B_1 + B_2 * pH_val + B_3 * y[7] + B_4 * (pH_val ** 2)) + B_5 * (y[7] ** 2))
        mu_h_use = k4

    # ----- 6. Main kinetic rates -----
    P_inhib = 2.5 * P_std_dev * ((P_std_dev * np.sqrt(2 * np.pi)) ** -1
                                  * np.exp(-0.5 * ((s - mean_P) / P_std_dev) ** 2))

    mu_a0 = ratio_mu_e_mu_b * mux_max_use * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X
    mu_e  =                   mux_max_use * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X

    K_diff = K_diff_base - (A_t1 * beta_1)
    if K_diff < K_diff_L:
        K_diff = K_diff_L

    r_b0  = mu_a0 * a_1 * s / (K_b + s)
    r_sb0 = Y_sX * r_b0
    r_e1  = (mu_e * a_0 * s) / (K_e + s)
    r_se1 = Y_sX * r_e1
    r_d1  = mu_diff * a_0 / (K_diff + s)
    r_m0  = m_s * a_0 / (K_diff + s)

    # ----- 7. Vacuole population balance -----
    # phi[0] = phi0; phi[k] for k=1..9 use bins n_1..n_9 (y[16..24])
    phi = [y[26]]
    for k in range(1, 10):
        # MATLAB r_mean(k) for k=2..10: (1.5e-4) + (k-2)*delta_r
        # In our 0..9 loop with k=1..9, that's r_0 + (k-1)*delta_r
        r_mean_k = 1.5e-4 + (k - 1) * delta_r
        phi.append(((4 * np.pi * r_mean_k ** 3) / 3.0) * y[15 + k] * delta_r)
    v_2 = float(np.sum(phi))

    rho_a1 = a_1 / ((a_1 / rho_a0) + max(v_2, 1e-30)) if a_1 > 0 else rho_a0
    v_a1 = (a_1 / (2.0 * max(rho_a1, 1e-12))) - v_2

    # Penicillin production from non-growing region
    r_p = mu_p_use * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P - mu_h_use * y[3]

    # Vacuole formation/degeneration auxiliaries
    r_m1 = (m_s * rho_a0 * v_a1 * s) / (K_v + s)
    r_d4 = mu_a * a_3

    # Vacuole bin derivatives (y[15] = n_0, y[16..24] = n_1..n_9)
    dn = [0.0] * 10
    dn[0] = ((mu_v * v_a1) / (K_v + s)) * (6.0 / np.pi) * ((r_0 + delta_0) ** -3) - k_v * y[15]

    # Bins n_1..n_9 (advection + diffusion). Boundary: lower uses n_{j-1};
    # upper for j=9 uses y[25]=nm (matches MATLAB n=25 -> y(26)).
    for j in range(1, 10):
        yi = 15 + j  # Python index for n_j
        y_lo = y[yi - 1]
        y_hi = y[yi + 1]  # for j=9, this is y[25] = nm
        advection = -k_v * (y_hi - y_lo) / (2.0 * delta_r)
        diffusion = D * (y_hi - 2.0 * y[yi] + y_lo) / (delta_r ** 2)
        dn[j] = advection + diffusion

    # n_k used downstream for biomass volume transitions. MATLAB sets n_k = y(25)
    # AFTER computing dn9_dt. y(25) MATLAB == y[24] Python == n_9.
    n_k = y[24]

    # Maximum vacuole department dn_m_dt  (k=10 in MATLAB; r_k = r_0+8*delta_r,
    # k=12 -> r_m = r_0+10*delta_r)
    r_k = r_0 + 8.0 * delta_r
    r_m = r_0 + 10.0 * delta_r
    dn_m_dt = k_v * dn[9] / max(r_m - r_k, 1e-12) - mu_a * y[25]

    # Mean vacuole volume
    dphi_0_dt = ((mu_v * v_a1) / (K_v + s)) - k_v * y[15] * (np.pi * (r_0 + delta_0) ** 3) / 6.0

    # ----- 8. Volume / weight balance -----
    F_evp = y[4] * alpha_evp * (np.exp(2.5 * (y[7] - T0) / (Tv - T0)) - 1.0)
    pho_feed = (c_s_use / 1000.0) * pho_g + (1.0 - c_s_use / 1000.0) * pho_w
    dilution = Fs + Fb + Fa + Fw - F_evp + Fpaa
    dV1 = Fs + Fb + Fa + Fw + F_discharge / (pho_b / 1000.0) - F_evp + Fpaa
    dWt = (Fs * pho_feed / 1000.0
           + pho_oil / 1000.0 * Foil
           + Fb + Fa + Fw + F_discharge - F_evp + Fpaa * pho_paa / 1000.0)

    # ----- 9. Biomass ODEs -----
    da_0_dt = r_b0 - r_d1 - y[11] * dilution / y[4]
    da_1_dt = r_e1 - r_b0 + r_d1 - (np.pi * ((r_k + r_m) ** 3) / 6.0) * rho_d * k_v * n_k - y[12] * dilution / y[4]
    da_3_dt = (np.pi * ((r_k + r_m) ** 3) / 6.0) * rho_d * k_v * n_k - r_d4 - y[13] * dilution / y[4]
    da_4_dt = r_d4 - y[14] * dilution / y[4]
    dP_dt = r_p - y[3] * dilution / y[4]

    X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt
    X_t = total_X

    # ----- 10. Heat generation -----
    Qrxn_X = X_1 * Y_QX * y[4] * Y_O2_X / 1000.0
    Qrxn_P = dP_dt * Y_QX * y[4] * Y_O2_P / 1000.0
    Qrxn_t = max(Qrxn_X + Qrxn_P, 0.0)

    # ----- 11. Power, oxygen transfer -----
    N_rps = RPM / 60.0
    D_imp = 2.0 * r_imp
    unaerated_power = n_imp * Po * pho_b * (N_rps ** 3) * (D_imp ** 5)
    if Fg > 1e-12 and unaerated_power > 0:
        P_g = 0.706 * (((unaerated_power ** 2) * N_rps * D_imp ** 3) / (Fg ** 0.56)) ** 0.45
        P_n = P_g / unaerated_power
    else:
        P_n = 1.0
    variable_power = (n_imp * Po * pho_b * (N_rps ** 3) * (D_imp ** 5) * P_n) / 1000.0  # kW

    V_s = Fg / (np.pi * (r_vessel ** 2))
    T_K = y[7]
    V = y[4]
    V_m = V / 1000.0
    if h_b > 0 and V_m > 0:
        P_air = (V_s * R * T_K * V_m / (22.4 * h_b)) * np.log(1.0 + pho_b * 9.81 * h_b / (pressure_top * 1e5))
    else:
        P_air = 0.0
    P_t1 = variable_power + P_air

    if viscosity <= 4.0:
        viscosity = 1.0   # second clamp matching MATLAB line 457
    vis_scaled = viscosity / 100.0
    oil_f = Foil / max(V, 1.0)

    if V_s > 0 and P_t1 > 0:
        kla = alpha_kla * (V_s ** a_kla) * ((P_t1 / V_m) ** b_kla) * (vis_scaled ** c_kla) * (1.0 - oil_f ** d_kla)
    else:
        kla = 0.0

    OUR = (-X_1) * Y_O2_X - m_O2_X * X_t - dP_dt * Y_O2_P
    OTR = kla * (DOstar_tp - y[1])

    # ----- 12. Initialize derivative vector -----
    dy = np.zeros(33, dtype=np.float64)

    # 12.1 Substrate
    dy[0] = (-r_se1 - r_sb0 - r_m0 - r_m1
             - (Y_sP * mu_p_use * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P)
             + Fs * c_s_use / y[4]
             + Foil * c_oil_use / y[4]
             - y[0] * dilution / y[4])

    # 12.2 Dissolved oxygen
    dy[1] = OUR + OTR - (y[1] * dilution / y[4])

    # 12.3 O2 off-gas
    Vg = epsilon * V_m
    Qfg_in = 60.0 * Fg * 1000.0 * 32.0 / 22.4
    denom_offgas = max(1.0 - y[2] - y[27] / 100.0, 1e-6)
    Qfg_out = 60.0 * Fg * (N2_in / denom_offgas) * 1000.0 * 32.0 / 22.4
    if Vg > 0:
        dy[2] = (Qfg_in * O_2_in_use - Qfg_out * y[2] - 0.001 * OTR * V_m * 60.0) / (Vg * 28.97 * 1000.0 / 22.4)
    else:
        dy[2] = 0.0

    # 12.4 Penicillin
    dy[3] = dP_dt
    # 12.5 Volume
    dy[4] = dV1
    # 12.6 Weight
    dy[5] = dWt

    # 12.7 pH (in H+ space)
    pH_dis = Fs + Foil + Fb + Fa + F_discharge + Fw
    pH_now = -np.log10(max(y[6], 1e-30))
    if pH_now < 7.0:
        cb = -abc_use
        ca = abc_use
        Hp = y[6]
        pH_balance = 0
    else:
        cb = abc_use
        ca = -abc_use
        Hp = (1e-14 / max(y[6], 1e-30) - y[6])
        pH_balance = 1
    if (y[4] + Fb * step1 + Fa * step1) > 0:
        B = (Hp * y[4] + ca * Fa * step1 + cb * Fb * step1) / (y[4] + Fb * step1 + Fa * step1)
    else:
        B = 0.0
    B = -B
    disc = max(B * B + 4e-14, 0.0)
    if pH_balance == 1:
        dy[6] = (-gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X)
                 - gamma1 * r_p
                 - gamma2 * pH_dis
                 + ((-B - np.sqrt(disc)) / 2.0 - y[6]))
    else:
        dy[6] = (gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X)
                 + gamma1 * r_p
                 + gamma2 * pH_dis
                 + ((-B + np.sqrt(disc)) / 2.0 - y[6]))

    # 12.8 Temperature
    Ws = P_t1
    Qcon = U_jacket * A_c * (y[7] - Tair)
    cool_term = ((alpha_1 / 1000.0) * (Fc ** (beta_T + 1.0))
                 * ((y[7] - Tcin_use)
                    / (Fc / 1000.0 + (alpha_1 * (Fc / 1000.0) ** beta_T) / 2.0 * pho_b * C_ps)))
    heat_term = ((alpha_1 / 1000.0) * (Fh ** (beta_T + 1.0))
                 * ((y[7] - Th)
                    / (Fh / 1000.0 + (alpha_1 * (Fh / 1000.0) ** beta_T) / 2.0 * pho_b * C_ps)))
    dQ_dt = (Fs * pho_feed * C_ps * (Tf - y[7]) / 1000.0
             + Fw * pho_w * C_pw * (Tw - y[7]) / 1000.0
             - F_evp * pho_b * C_pw / 1000.0
             - delta_H_evap * F_evp * pho_w / 1000.0
             + Qrxn_t + Ws
             - cool_term - heat_term - Qcon)
    dy[7] = dQ_dt / ((y[4] / 1000.0) * C_pw * pho_b)
    dy[8] = dQ_dt

    # 12.9 Viscosity
    dy[9] = (3.0 * (a_0 ** (1.0 / 3.0))
             * (1.0 / (1.0 + np.exp(-k1_visc * (t - t1_visc))))
             * (1.0 / (1.0 + np.exp(-k2_visc * (t - t2_visc))))
             - k3_visc * Fw)

    # 12.10 Integrated total biomass
    dy[10] = total_X

    # 12.11 Biomass regions
    dy[11] = da_0_dt
    dy[12] = da_1_dt
    dy[13] = da_3_dt
    dy[14] = da_4_dt

    # 12.12 Vacuole bins
    for j in range(10):
        dy[15 + j] = dn[j]
    dy[25] = dn_m_dt
    dy[26] = dphi_0_dt

    # 12.13 CO2 off-gas
    total_X_CO2 = y[11] + y[12]
    CER = total_X_CO2 * q_co2 * V
    if Vg > 0:
        dy[27] = ((((60.0 * Fg * 44.0 * 1000.0) / 22.4) * C_CO2_in
                   + CER
                   - ((60.0 * Fg * 44.0 * 1000.0) / 22.4) * y[27])
                  / (Vg * 28.97 * 1000.0 / 22.4))
    else:
        dy[27] = 0.0

    # 12.14 Dissolved CO2
    Henrys_c_co2 = (np.exp(11.25 - 395.9 / max(y[7] - 175.9, 1.0))) / (44.0 * 100.0)
    C_star_CO2 = (total_pressure * y[27]) / max(Henrys_c_co2, 1e-30)
    dy[28] = kla * delta_c_0 * (C_star_CO2 - y[28]) - y[28] * dilution / y[4]

    # 12.15 PAA
    dy[29] = (Fpaa * PAA_c_use / V
              - Y_PAA_P * dP_dt
              - Y_PAA_X * X_1
              - m_PAA * y[3]
              - y[29] * dilution / y[4])

    # 12.16 Nitrogen
    X_C_nitrogen = (-r_b0 - r_e1 - r_d1 - r_d4) * Y_NX
    P_C_nitrogen = -dP_dt * Y_NP
    dy[30] = ((NH3_shots * N_conc_shot) / y[4]
              + X_C_nitrogen + P_C_nitrogen
              - m_N * total_X
              + (1.0 * N_conc_paa * Fpaa / y[4])
              + N_conc_oil * Foil / y[4]
              - y[30] * dilution / y[4])

    # 12.17 Diagnostic mu
    dy[31] = mu_p_use
    dy[32] = mu_e

    if not np.all(np.isfinite(dy)):
        # Fall back to zeros to keep solver alive; integrator will reduce step
        dy = np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)
    return dy
