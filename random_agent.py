import random
from typing import Tuple, Dict, Any

# Use the generic EnvInterface
from core.env_interface import EnvInterface, StateType, ActionType
from core.agent_interface import Agent


class RandomAgent(Agent):
    """A simple random agent for demonstration. Does not learn."""

    def __init__(self, env: EnvInterface): # Use EnvInterface
        """
        Args:
            env: An instance of the BoardGameEnv (used for its copy method).
        """
        self.env = env # Keep env reference mainly for copy()

    # Update action return type hint
    def act(self, state: StateType) -> ActionType:
        """Choose a random valid action based on the provided state."""
        # Create a temporary environment copy and set its state
        temp_env = self.env.copy()
        temp_env.set_state(state)

        # Get legal actions from the temporary environment
        valid_actions = temp_env.get_legal_actions()

        if valid_actions:
            return random.choice(valid_actions)
        # Return None if no action is possible
        return None
