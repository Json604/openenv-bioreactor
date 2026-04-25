"""Smoke test that the package imports and pytest is wired."""
import bioperator_env


def test_package_version():
    assert bioperator_env.__version__ == "0.1.0"
