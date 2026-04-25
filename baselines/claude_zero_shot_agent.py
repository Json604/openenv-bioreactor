"""Closed-SOTA baseline: Anthropic Claude (latest available) zero-shot.

This baseline gives us a "smart-but-untrained" comparison point. Our
GRPO-trained 4B model should beat THIS on safety + productivity if the
training works. Even if it doesn't, the comparison tells a clear story.

Loads the Anthropic SDK lazily so the rest of the system stays runnable
without an API key.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

from bioperator_env.models import BioOperatorObservation
from bioperator_env.prompt import SYSTEM_PROMPT, build_prompt


class ClaudeZeroShotAgent:
    name = "claude_zero_shot"

    def __init__(self, model: str = "claude-sonnet-4-6",
                 max_tokens: int = 256, temperature: float = 0.5):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import anthropic  # local import
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def act(self, obs: BioOperatorObservation) -> dict:
        self._ensure_client()
        prompt = build_prompt(obs, system=SYSTEM_PROMPT)
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
        except Exception as e:
            text = f"<error: {e}>"
        parsed = _extract_first_json(text)
        return parsed or {
            "feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0,
            "reason": f"claude: parse-failed: {text[:80]}",
        }


def _extract_first_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
