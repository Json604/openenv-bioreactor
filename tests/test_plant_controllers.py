"""Tests for the incremental PID controller (port of PIDSimple3.m)."""
from bioperator_env.plant.controllers import pid_step


def test_pid_at_setpoint_holds():
    """If error is zero and PV is steady, control is unchanged."""
    out = pid_step(u_prev=10.0, err=0.0, err_prev=0.0,
                   y=5.0, y_prev=5.0, y_prev_prev=5.0,
                   u_min=0.0, u_max=100.0, Kp=1.0, Ti=10.0, Td=0.1, h=0.2)
    assert out == 10.0


def test_pid_clips_to_u_max():
    out = pid_step(u_prev=99.0, err=100.0, err_prev=0.0,
                   y=5.0, y_prev=5.0, y_prev_prev=5.0,
                   u_min=0.0, u_max=100.0, Kp=1.0, Ti=10.0, Td=0.0, h=0.2)
    assert out == 100.0


def test_pid_clips_to_u_min():
    out = pid_step(u_prev=1.0, err=-100.0, err_prev=0.0,
                   y=5.0, y_prev=5.0, y_prev_prev=5.0,
                   u_min=0.0, u_max=100.0, Kp=1.0, Ti=10.0, Td=0.0, h=0.2)
    assert out == 0.0


def test_pid_proportional_response():
    """Pure P-action: positive err -> output rises."""
    out = pid_step(u_prev=10.0, err=5.0, err_prev=0.0,
                   y=0.0, y_prev=0.0, y_prev_prev=0.0,
                   u_min=0.0, u_max=100.0, Kp=2.0, Ti=0.0, Td=0.0, h=0.2)
    # P = (5 - 0) = 5; u = 10 + 2*5 = 20
    assert out == 20.0


def test_pid_integral_only_for_nonzero_Ti():
    """When Ti>1e-7 the integral kicks in."""
    out_no_i = pid_step(u_prev=10.0, err=2.0, err_prev=2.0,
                        y=0.0, y_prev=0.0, y_prev_prev=0.0,
                        u_min=0.0, u_max=100.0, Kp=1.0, Ti=0.0, Td=0.0, h=0.2)
    out_with_i = pid_step(u_prev=10.0, err=2.0, err_prev=2.0,
                          y=0.0, y_prev=0.0, y_prev_prev=0.0,
                          u_min=0.0, u_max=100.0, Kp=1.0, Ti=1.0, Td=0.0, h=0.2)
    assert out_no_i == 10.0  # P = 0 because err == err_prev; I disabled
    assert out_with_i > out_no_i  # I = err*h/Ti = 0.4 > 0
