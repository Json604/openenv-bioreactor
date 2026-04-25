"""Baseline agents for BioOperatorEnv.

Each agent implements the same interface:

    class Agent:
        name: str
        def act(self, obs: BioOperatorObservation) -> dict:
            ...

The dict returned must validate as a BioOperatorAction. Agents that fail to
produce valid actions get reward=format_invalid via env.step's defaulting.
"""
