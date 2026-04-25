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
                 **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.adapter_path = adapter_path or os.environ.get(
            "BIOPERATOR_LORA", "checkpoints/qwen3-bioperator-lora"
        )

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from peft import PeftModel  # local import
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        base = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map=self.device,
        )
        adapter_dir = Path(self.adapter_path)
        if adapter_dir.exists():
            self._model = PeftModel.from_pretrained(base, str(adapter_dir))
        else:
            # Fall back to base model if adapter not present (allows the demo
            # notebook to run pre-training and post-training side-by-side).
            self._model = base
        self._model.eval()
