from algorithms.mcts_old import MCTS_Old
from core.agent_interface import Agent
from environments.base import BaseEnvironment, ActionType


class MCTSAgent_Old(Agent):
    """Agent that uses MCTS to choose actions."""

    def __init__(
        self,
        env: BaseEnvironment,
        num_simulations: int = 100,
        exploration_constant: float = 1.41,
    ):
        """
        Args:
            env: The game environment instance (used for copy).
            num_simulations: Number of MCTS simulations per move.
            exploration_constant: Exploration constant (c) for UCB1.
        """
        self.env = env
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
        self.reset_turn()
        self.env = env
        root_node = self.mcts.search(self.env)
        assert root_node.children

        child_visits = {
            action: node.visit_count for action, node in root_node.children.items()
        }
        best_action = max(child_visits, key=child_visits.get)

        return best_action

    def reset_turn(self) -> None:
        """Reset the MCTS search tree."""
        self.mcts.reset_root()
