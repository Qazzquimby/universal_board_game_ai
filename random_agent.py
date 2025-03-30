import random
from typing import Tuple

from temp_env import BoardGameEnv


class RandomAgent:
    """A simple random agent for demonstration."""

    def __init__(self, env: BoardGameEnv):
        self.env = env

    def act(self) -> Tuple[int, int]:
        """Choose a random valid action."""
        valid_actions = self.env.get_legal_actions()
        if valid_actions:
            return random.choice(valid_actions)
        return -1, -1  # Invalid action if no valid actions available
