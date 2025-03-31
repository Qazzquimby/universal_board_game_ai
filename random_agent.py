import random
from typing import Tuple

from temp_env import BoardGameEnv


from typing import Tuple, Dict, Any, List


class RandomAgent:
    """A simple random agent for demonstration."""

    def __init__(self, env: BoardGameEnv):
        self.env = env

    def act(self, state: Dict[str, Any]) -> Tuple[int, int]:
        """Choose a random valid action based on the provided state."""
        # Create a temporary environment copy and set its state
        temp_env = self.env.copy()
        temp_env.set_state(state)

        # Get legal actions from the temporary environment
        valid_actions = temp_env.get_legal_actions()

        if valid_actions:
            return random.choice(valid_actions)
        return -1, -1  # Invalid action if no valid actions available
