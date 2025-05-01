import random

from algorithms.mcts_old import MCTS_Old
from core.agent_interface import Agent
from environments.base import BaseEnvironment, StateType, ActionType


class MCTSAgent_Old(Agent):
    """Agent that uses MCTS to choose actions."""

    def __init__(
        self,
        env: BaseEnvironment,  # Use EnvInterface
        num_simulations: int = 100,
        exploration_constant: float = 1.41,
    ):
        """
        Args:
            env: The game environment instance (used for copy).
            num_simulations: Number of MCTS simulations per move.
            exploration_constant: Exploration constant (c) for UCB1.
        """
        self.env = env  # Keep env reference mainly for copy()
        # Ensure MCTSAgent uses the base MCTS class with UCB1
        self.mcts = MCTS_Old(
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            discount_factor=1.0,
        )

    def act(self, env: BaseEnvironment) -> ActionType:
        """
        Perform MCTS search and choose the best action.

        Returns:
            The chosen action (row, col).
        """
        # Ensure the MCTS root is reset or updated appropriately.
        # For stateless MCTS between moves, we run a fresh search each time.
        # If we wanted to reuse the tree, we'd need more logic here.
        self.reset()  # Reset the tree for a fresh search from the current state

        state = self.env.get_observation()

        root_node = self.mcts.search(self.env, state)
        assert root_node.children

        # if not root_node.children:
        #     # If root has no children after search (e.g., immediate terminal state or no legal moves)
        #     # Need a fallback strategy. Choosing randomly from legal moves is one option.
        #     print("Warning: MCTS root has no children after search. Falling back.")
        #     temp_env = self.env.copy()
        #     temp_env.set_state(state)
        #     legal_actions = temp_env.get_legal_actions()
        #     if legal_actions:
        #         return random.choice(legal_actions)
        #     else:
        #         # Return None if no action is possible, let caller handle it
        #         return None

        # Choose the action leading to the most visited child node
        child_visits = {
            action: node.visit_count for action, node in root_node.children.items()
        }
        best_action = max(child_visits, key=child_visits.get)

        return best_action

    def reset(self) -> None:
        """Reset the MCTS search tree."""
        # Use the reset_root method for consistency
        self.mcts.reset_root()
