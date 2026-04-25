"""Tests for the SBC recipe lookup (port of fctrl_indpensim.m §SBC)."""
import pytest

from bioperator_env.plant.recipe import sbc_lookup, available_schedules


def test_sbc_at_step_15_substrate_feed():
    # Recipe_Fs starts at k=15 with sp=8.0
    assert sbc_lookup("Fs", k=15) == 8.0


def test_sbc_step_50_substrate_feed_in_range_2():
    # Between k=15 and k=60: sp=15
    assert sbc_lookup("Fs", k=50) == 15.0


def test_sbc_step_50_aeration():
    # Recipe_Fg = [40, 100, 200, 450, 1000, 1250, 1750]
    # sp        = [30,  42,  55,  60,   75,   65,   60]
    # k=50 falls in (40, 100] -> 42
    assert sbc_lookup("Fg", k=50) == 42.0


def test_sbc_beyond_last_step_uses_terminal():
    val = sbc_lookup("Fs", k=99999)
    assert val == 80.0  # last entry of modified Fs schedule


def test_sbc_unknown_raises():
    with pytest.raises(KeyError):
        sbc_lookup("Frobnicate", k=10)


def test_available_schedules():
    keys = available_schedules()
    assert "Fs" in keys
    assert "Fg" in keys
    assert "Fpaa" in keys
    assert len(keys) == 7
