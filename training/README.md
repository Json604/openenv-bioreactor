# Training pipeline (TRL GRPO + Unsloth)

Two ways to train: **HF Jobs** (recommended for real runs) or **Colab** (good for quick experiments).

---

## HF Jobs — recommended

`training/run_grpo_job.py` is a self-contained PEP-723 script. HF Jobs uses `uv` to install its declared dependencies and run it on the GPU you pick.

### One-time setup

```bash
pip install -U huggingface_hub
hf auth login   # paste your HF token (any with read+write scope)
```

Get a [Weights & Biases API key](https://wandb.ai/authorize) for live training curves.
Get a [HF write token](https://huggingface.co/settings/tokens) if you want the trained adapter pushed to the Hub.

### Submit a full training run

```bash
hf jobs uv run \
    --gpu A100-80GB \
    --secrets WANDB_API_KEY=<your_wandb_key> \
    --secrets HF_TOKEN=<your_hf_write_token> \
    https://github.com/Json604/openenv-bioreactor/raw/main/training/run_grpo_job.py \
    -- \
    --max_steps=200 \
    --num_generations=8 \
    --max_completion_length=256 \
    --temperature=1.0 \
    --task_id=do-recovery-easy \
    --push_to_hub=Json604/qwen3b-bioperator-lora
```

The job clones the repo, builds the dataset, trains GRPO, saves the LoRA adapter, and pushes it to `https://huggingface.co/Json604/qwen3b-bioperator-lora`.

### GPU choice

| GPU | $/hr | Full 200-step training | Recommended for |
|---|---|---|---|
| `T4-small` | ~$0.50 | ~3 h | smoke tests |
| `L4` | ~$1 | ~1.5 h | budget |
| `A10G-large` | ~$1.50 | ~1 h | balanced |
| **`A100-80GB`** | **~$4** | **~30 min** | **headline run, ~$2 cost** |
| `H100` | ~$8 | ~15 min | fastest iteration |

### Monitor

```bash
hf jobs ls                     # list active jobs
hf jobs logs <JOB_ID>          # tail logs
```

The wandb run URL prints near the start of training; click it for live curves.

### After training — load the adapter

The demo notebook `notebooks/04_demo.ipynb` looks for the adapter in `checkpoints/qwen3-bioperator-lora` by default. Set `BIOPERATOR_LORA` to point at your HF Hub model id:

```python
os.environ["BIOPERATOR_LORA"] = "Json604/qwen3b-bioperator-lora"
```

The `TrainedQwenAgent` will pull it via `peft.PeftModel.from_pretrained`.

---

## Colab — fastest to start, but slower and less reliable

`notebooks/03_train_grpo.ipynb` does the same training interactively. Recommended for:
- First-time exploration ("does the loop run?")
- Inspecting individual GRPO updates step by step

Disadvantages:
- Idle disconnect risk
- T4 only (no A100 on free tier)
- 2–3× slower than A100

Open in Colab:

```
https://colab.research.google.com/github/Json604/openenv-bioreactor/blob/main/notebooks/03_train_grpo.ipynb
```

---

## Hyperparameter notes

The defaults in `run_grpo_job.py` reflect what worked in our 15-step Colab smoke (rising reward, non-zero reward_std). Tuning levers:

| Lever | Default | Why you'd change it |
|---|---|---|
| `--temperature` | 1.0 | lower (0.7) if outputs look incoherent; higher (1.2) if reward_std stays ~0 |
| `--num_generations` | 8 | drop to 4 to halve generation cost; raises noisiness in advantage estimate |
| `--max_completion_length` | 256 | drop to 128 if `clipped_ratio < 0.3` (most completions terminate naturally) |
| `--max_steps` | 200 | drop to 100 for ablations; raise to 400 for stage 2 curriculum |
| `--task_id` | do-recovery-easy | `do-recovery-medium` for stage 2; `aeration-limit-hard` for stage 3 |
