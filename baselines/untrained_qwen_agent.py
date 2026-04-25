"""Untrained Qwen 3 (or compatible HF instruct model) zero-shot baseline.

Loads the base instruct model via transformers, prompts with
`bioperator_env.prompt.build_prompt(...)`, parses the JSON action.

This baseline does NOT load by default at import time -- only when
.act() is called -- so the rest of the system stays usable on a CPU box.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

from bioperator_env.models import BioOperatorObservation
from bioperator_env.prompt import SYSTEM_PROMPT, build_prompt


class UntrainedQwenAgent:
    name = "untrained_qwen"

    def __init__(self,
                 model_id: str = "Qwen/Qwen2.5-3B-Instruct",
                 device: Optional[str] = None,
                 max_new_tokens: int = 128,
                 temperature: float = 0.7):
        self.model_id = model_id
        self.device = device or os.environ.get("BIOPERATOR_DEVICE", "cuda")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map=self.device,
        )
        self._model.eval()

    def act(self, obs: BioOperatorObservation) -> dict:
        self._ensure_loaded()
        import torch  # local import
        prompt = build_prompt(obs, system=SYSTEM_PROMPT)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = _extract_first_json(text)
        return parsed or {
            "feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0,
            "reason": f"untrained_qwen: failed to parse: {text[:80]}",
        }


def _extract_first_json(text: str) -> Optional[dict]:
    """Find first {...} block in text and try to parse it as a dict."""
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
