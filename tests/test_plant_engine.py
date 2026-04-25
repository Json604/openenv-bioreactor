"""Tests for the step-wise plant engine wrapper."""
import numpy as np

from bioperator_env.plant.engine import Plant, PlantConfig


def test_reset_returns_initial_state():
    plant = Plant(PlantConfig(seed=42, T_total_h=230.0, h_step=0.2, randomise_params=False))
    state = plant.reset()
    assert state.shape == (33,)
    assert state[0] > 0    # S
    assert state[1] > 0    # DO2
    assert 280 < state[7] < 310  # T in K (~297)
    assert plant.k == 0
    assert plant.t_h == 0.0


def test_one_step_advances_time():
    plant = Plant(PlantConfig(seed=42, T_total_h=230.0, h_step=0.2, randomise_params=False))
    plant.reset()
    s_before = plant.state.copy()
    plant.step({})
    assert plant.k == 1
    assert abs(plant.t_h - 0.2) < 1e-9
    assert not np.array_equal(s_before, plant.state)
    assert np.all(np.isfinite(plant.state))


def test_50_steps_finite_no_crash():
    plant = Plant(PlantConfig(seed=7, T_total_h=20.0, h_step=0.2, randomise_params=True))
    plant.reset()
    for _ in range(50):
        plant.step({})
    assert np.all(np.isfinite(plant.state))
    assert plant.state[0] > 0    # S
    assert plant.state[4] > 0    # V


def test_agent_can_override_feed():
    plant_low = Plant(PlantConfig(seed=42, T_total_h=10.0, h_step=0.2, randomise_params=False))
    plant_low.reset()
    plant_high = Plant(PlantConfig(seed=42, T_total_h=10.0, h_step=0.2, randomise_params=False))
    plant_high.reset()
    for _ in range(10):
        plant_low.step({"Fs": 0.0})
        plant_high.step({"Fs": 200.0})
    # Low feed -> less substrate accumulated than high feed
    assert plant_high.state[4] >= plant_low.state[4]


def test_pH_helper():
    plant = Plant(PlantConfig(seed=42, T_total_h=10.0, h_step=0.2, randomise_params=False))
    plant.reset()
    pH = plant.pH()
    assert 6.0 < pH < 7.0
