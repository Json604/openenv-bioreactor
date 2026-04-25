"""Fixed-recipe baseline: never moves the controls.

Equivalent to "let the IndPenSim SBC recipe run uninterrupted." Useful as
the natural floor: any agent worth its keep should beat this on disturbance
scenarios and tie/beat it on normal scenarios.
"""
from __future__ import annotations

from bioperator_env.models import BioOperatorObservation


class FixedRecipeAgent:
    name = "fixed_recipe"

    def act(self, obs: BioOperatorObservation) -> dict:  # noqa: ARG002
        return {
            "feed_delta_L_h": 0,
            "aeration_delta_vvm": 0.0,
            "agitation_delta_rpm": 0,
            "reason": "follow recipe",
        }
