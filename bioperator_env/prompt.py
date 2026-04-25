"""Prompt template builder: BioOperatorObservation -> str.

The prompt is intentionally compact and structured: the LLM should output
a JSON action, not prose. Format mirrors what a SCADA console would show
a plant operator.
"""
from __future__ import annotations
import json
from typing import Optional

from .models import BioOperatorObservation


SYSTEM_PROMPT = (
    "You are an operator running a 100,000 L industrial penicillin "
    "fermenter. Read the plant-console state and choose the next safe "
    "control action for the next 12 simulated minutes. "
    "Respond with valid JSON ONLY, matching this schema:\n"
    '  {"feed_delta_L_h": -5|0|5, '
    '"aeration_delta_vvm": -0.10|0.0|0.10, '
    '"agitation_delta_rpm": -5|0|5, '
    '"reason": "<short string, optional>"}\n'
    "Goals: keep dissolved oxygen above the safe floor, keep substrate in "
    "its healthy band, grow biomass and penicillin, avoid wild control "
    "swings. Reply with the JSON object only."
)


def format_observation(obs: BioOperatorObservation) -> str:
    """Compact JSON-formatted plant console for the LLM prompt."""
    payload = {
        "time_h": round(obs.time_h, 2),
        "batch_phase": obs.batch_phase,
        "measurements": _round_dict(obs.measurements),
        "setpoints_or_limits": _round_dict(obs.setpoints_or_limits),
        "current_controls": _round_dict(obs.current_controls),
        "recent_trends": obs.recent_trends,
        "alarm": obs.alarm,
        "previous_action": obs.previous_action,
        "offline_lab": obs.offline_lab,
    }
    return json.dumps(payload, separators=(",", ":"))


def build_prompt(obs: BioOperatorObservation,
                 system: Optional[str] = None) -> str:
    """Return the full prompt: system instruction + serialized observation."""
    head = system if system is not None else SYSTEM_PROMPT
    return f"<system>\n{head}\n</system>\n<observation>\n{format_observation(obs)}\n</observation>\n<action>"


def _round_dict(d: dict, ndigits: int = 3) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, float):
            out[k] = round(v, ndigits)
        else:
            out[k] = v
    return out
