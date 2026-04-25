"""Test the 105-parameter vector matches IndPenSim/Parameter_list.m."""
from bioperator_env.plant.params import build_params, ParamConfig


def _default_cfg() -> ParamConfig:
    return ParamConfig(x0_mux=0.41, x0_mup=0.041, alpha_kla=85.0,
                       N_conc_paa=150000.0, PAA_c=530000.0)


def test_param_vector_has_105_entries():
    p = build_params(_default_cfg())
    assert len(p) == 105


def test_known_constants_at_correct_index():
    """Spot-check positions against Parameter_list.m comments (1-indexed in MATLAB)."""
    p = build_params(_default_cfg())
    assert p[0] == 0.041    # par(1) mu_p
    assert p[1] == 0.41     # par(2) mux_max
    assert p[30] == 85.0    # par(31) alpha_kla
    assert abs(p[42] - 8.314) < 1e-6  # par(43) R
    assert abs(p[99] - 0.21) < 1e-6   # par(100) O_2_in
    assert p[68] == 150000.0          # par(69) N_conc_paa
    assert p[74] == 530000.0          # par(75) PAA_c


def test_param_values_are_floats():
    p = build_params(_default_cfg())
    assert all(isinstance(v, float) for v in p)
