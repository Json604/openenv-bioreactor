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


def format_prompt(
    observation: BioreactorObservation,
    previous_action: int | None,
    previous_reward: float | None,
    previous_state: BioreactorObservation | None,
) -> str:
    previous_block = ""
    if previous_state is not None and previous_action is not None and previous_reward is not None:
        previous_block = f"""

Previous step:
action={previous_action}
reward={previous_reward:.3f}
previous_oxygen={previous_state.oxygen_level:.3f}
previous_mixing={previous_state.mixing_uniformity:.3f}
previous_biomass={previous_state.biomass_concentration:.3f}
previous_byproduct={previous_state.byproduct_load:.3f}"""

    return f"""You are controlling a real bioreactor process.

Task: {observation.task_id} ({observation.difficulty})
Goal: maximize biomass while keeping oxygen near {observation.target_oxygen:.2f}, mixing near {observation.target_mixing:.2f}, residual nutrient near {observation.target_nutrient:.2f}, and byproduct below {observation.max_safe_byproduct:.2f}.
End-of-batch objective: finish with biomass >= {observation.terminal_biomass_target:.2f}, byproduct <= {observation.terminal_byproduct_limit:.2f}, nutrient in [{observation.terminal_nutrient_low:.2f}, {observation.terminal_nutrient_high:.2f}], and oxygen above {observation.terminal_oxygen_floor:.2f}.

Current state:
oxygen={observation.oxygen_level:.3f}
mixing={observation.mixing_uniformity:.3f}
nutrient={observation.nutrient_concentration:.3f}
biomass={observation.biomass_concentration:.3f}
byproduct={observation.byproduct_load:.3f}
feed_rate={observation.feed_rate:.3f}
step={observation.step}/{observation.max_steps}
{previous_block}

Choose one action:
0: increase stirrer speed
1: decrease stirrer speed
2: increase oxygen input
3: decrease oxygen input
4: do nothing
5: increase feed rate
6: decrease feed rate

Return only the action number."""


def extract_action(text: str) -> int | None:
    match = re.search(r"\b[0-6]\b", text.strip())
    if match is None:
        return None
    return int(match.group(0))


def choose_action(
    client: Any,
    observation: BioreactorObservation,
    previous_action: int | None,
    previous_reward: float | None,
    previous_state: BioreactorObservation | None,
) -> tuple[int, str | None]:
    if client is None:
        if OpenAI is None:
            return 4, "ImportError:openai_package_not_installed"
        return 4, "MissingAPIKey:set_HF_TOKEN_or_OPENAI_API_KEY"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": format_prompt(
                        observation,
                        previous_action=previous_action,
                        previous_reward=previous_reward,
                        previous_state=previous_state,
                    ),
                }
            ],
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
    previous_action: int | None = None
    previous_reward: float | None = None
    previous_state: BioreactorObservation | None = None

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        while not observation.done:
            action, error = choose_action(
                client,
                observation,
                previous_action=previous_action,
                previous_reward=previous_reward,
                previous_state=previous_state,
            )
            previous_state = observation
            observation = env.step(action)
            rewards.append(observation.reward)
            previous_action = action
            previous_reward = observation.reward
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
