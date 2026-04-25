"""Random baseline: picks uniformly among the 27 valid actions."""
from __future__ import annotations
import random
from typing import Optional

from bioperator_env.models import BioOperatorObservation


_FEED = (-5, 0, 5)
_AER = (-0.10, 0.0, 0.10)
_RPM = (-5, 0, 5)


class RandomAgent:
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, obs: BioOperatorObservation) -> dict:  # noqa: ARG002
        return {
            "feed_delta_L_h": self.rng.choice(_FEED),
            "aeration_delta_vvm": self.rng.choice(_AER),
            "agitation_delta_rpm": self.rng.choice(_RPM),
            "reason": "random",
        }
