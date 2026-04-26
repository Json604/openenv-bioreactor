"""Trained Qwen 3 + LoRA adapter baseline.

This subclasses UntrainedQwenAgent but additionally attaches a LoRA adapter
saved by `training/grpo_train.py`. Used in `notebooks/04_demo.ipynb` for the
before/after comparison.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from .untrained_qwen_agent import UntrainedQwenAgent


class TrainedQwenAgent(UntrainedQwenAgent):
    name = "trained_qwen"

    def __init__(self,
                 adapter_path: Optional[str] = None,
                 model_id: str = "Qwen/Qwen2.5-3B-Instruct",
                 allow_no_adapter: bool = False,
                 **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.adapter_path = adapter_path or os.environ.get(
            "BIOPERATOR_LORA", "checkpoints/qwen3-bioperator-lora"
        )
        self.allow_no_adapter = allow_no_adapter

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from peft import PeftModel  # local import
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        base = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map=self.device,
        )
        adapter = self.adapter_path
        # Hub repo id like "Json604/qwen3b-bioperator-lora" -> let PEFT fetch it.
        is_hub_id = isinstance(adapter, str) and "/" in adapter and not Path(adapter).exists()
        if is_hub_id:
            self._model = PeftModel.from_pretrained(base, adapter)
        elif Path(adapter).exists():
            self._model = PeftModel.from_pretrained(base, str(adapter))
        else:
            if not self.allow_no_adapter:
                raise FileNotFoundError(
                    f"TrainedQwenAgent: adapter not found at '{adapter}'. "
                    f"Set BIOPERATOR_LORA env var to a local path or HF Hub repo id "
                    f"(e.g. 'Json604/qwen3b-bioperator-lora'), or pass "
                    f"allow_no_adapter=True to silently fall back to the base model."
                )
            self._model = base
        self._model.eval()
