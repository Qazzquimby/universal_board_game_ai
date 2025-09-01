import random

from environments.base import BaseEnvironment, StateType, ActionType
from core.agent_interface import Agent


class RandomAgent(Agent):
    """A simple random agent for demonstration. Does not learn."""

    def __init__(self, env: BaseEnvironment):  # Use EnvInterface
        """
        Args:
            env: An instance of the BoardGameEnv (used for its copy method).
        """
        self.env = env  # Keep env reference mainly for copy()

    # Update action return type hint
    def act(self, env: BaseEnvironment) -> ActionType:
        """Choose a random valid action from the current environment state."""
        valid_actions = env.get_legal_actions()

        if valid_actions:
            return random.choice(valid_actions)
        # Return None if no action is possible
        return None
