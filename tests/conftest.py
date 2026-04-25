"""Shared pytest fixtures for BioOperatorEnv tests."""
import sys
from pathlib import Path

# Make bioperator_env importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
