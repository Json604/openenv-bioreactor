"""Rule-based operator: simple if-then logic encoding what a human operator
would do at the console. Strong baseline; serves as the "good ol' PID" the
LLM has to beat on the rubric.
"""
from __future__ import annotations

from bioperator_env.models import BioOperatorObservation


class RuleBasedAgent:
    name = "rule_based"

    def act(self, obs: BioOperatorObservation) -> dict:
        m = obs.measurements
        sp = obs.setpoints_or_limits

        DO = m.get("dissolved_oxygen_pct", 25.0)
        DO_min = sp.get("DO_min_safe_pct", 20.0)
        S = m.get("substrate_g_L", 0.15)
        S_max = sp.get("substrate_max_g_L", 0.30)
        S_min = sp.get("substrate_min_g_L", 0.05)
        trend_DO = obs.recent_trends.get("DO", "stable")

        # Default no-op
        feed = 0
        aer = 0.0
        rpm = 0
        reason_parts = []

        # Priority: DO crisis dominates everything else.
        do_crisis = DO < DO_min or trend_DO == "falling_fast"

        if do_crisis:
            aer = 0.10
            feed = -5
            reason_parts.append("DO low or falling fast: cut feed, bump aeration")
        else:
            if DO < DO_min + 3:
                aer = 0.10
                reason_parts.append("DO marginal: bump aeration")
            elif DO > DO_min + 15:
                aer = -0.10
                reason_parts.append("DO comfortable: trim aeration")

            # Substrate management only when DO is safe
            if S > S_max:
                feed = -5
                reason_parts.append("substrate over band: cut feed")
            elif S < S_min:
                feed = 5
                reason_parts.append("substrate starved: bump feed")

        if not reason_parts:
            reason_parts.append("steady operation")

        return {
            "feed_delta_L_h": feed,
            "aeration_delta_vvm": aer,
            "agitation_delta_rpm": rpm,
            "reason": "; ".join(reason_parts),
        }
