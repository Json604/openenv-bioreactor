"""Baseline LLM policy runner for the bioreactor OpenEnv environment."""

from __future__ import annotations

import os
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from bioreactor_env import BioreactorEnv
from models import BioreactorObservation
from tasks import TASKS


ENV_NAME = "openenv-bioreactor"

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


def make_client() -> Any:
    if OpenAI is None:
        return None
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def format_prompt(observation: BioreactorObservation) -> str:
    return f"""You are controlling a real bioreactor process.

Task: {observation.task_id} ({observation.difficulty})
Goal: keep oxygen near {observation.target_oxygen:.2f}, mixing near {observation.target_mixing:.2f}, and nutrient near {observation.target_nutrient:.2f}.

Current state:
oxygen={observation.oxygen_level:.3f}
mixing={observation.mixing_uniformity:.3f}
nutrient={observation.nutrient_concentration:.3f}
step={observation.step}/{observation.max_steps}

Choose one action:
0: increase stirrer speed
1: decrease stirrer speed
2: increase oxygen input
3: decrease oxygen input
4: do nothing

Return only the action number."""


def extract_action(text: str) -> int | None:
    match = re.search(r"\b[0-4]\b", text.strip())
    if match is None:
        return None
    return int(match.group(0))


def choose_action(client: Any, observation: BioreactorObservation) -> tuple[int, str | None]:
    if client is None:
        if OpenAI is None:
            return 4, "ImportError:openai_package_not_installed"
        return 4, "MissingAPIKey:set_HF_TOKEN_or_OPENAI_API_KEY"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": format_prompt(observation)}],
            temperature=0,
            max_tokens=1,
        )
        content = response.choices[0].message.content or ""
        action = extract_action(content)
        if action is None:
            return 4, f"invalid_action:{content!r}"
        return action, None
    except Exception as exc:
        return 4, f"{type(exc).__name__}:{exc}"


def safe_error(error: str | None) -> str:
    if error is None:
        return "null"
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", error.strip())[:180]
    return safe or "unknown_error"


def run_task(task_id: str, client: Any) -> None:
    env = BioreactorEnv(task_id=task_id)
    rewards: list[float] = []
    errors: list[str] = []
    observation = env.reset(task_id=task_id)

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        while not observation.done:
            action, error = choose_action(client, observation)
            observation = env.step(action)
            rewards.append(observation.reward)
            if error is not None:
                errors.append(error)
            print(
                "[STEP] "
                f"step={len(rewards)} "
                f"action={action} "
                f"reward={observation.reward:.3f} "
                f"done={str(observation.done).lower()} "
                f"error={safe_error(error)}",
                flush=True,
            )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
    finally:
        env.close()
        rewards_text = ",".join(f"{reward:.3f}" for reward in rewards)
        success = not errors and observation.done and len(rewards) == env.task.max_steps
        print(
            "[END] "
            f"success={str(success).lower()} "
            f"task={task_id} "
            f"steps={len(rewards)} "
            f"score={env.score:.3f} "
            f"rewards={rewards_text}",
            flush=True,
        )


def main() -> None:
    client = make_client()
    for task_id in TASKS:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
