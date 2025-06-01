from typing import Optional

from algorithms.mcts import (
    MCTSOrchestrator,
    UCB1Selection,
    UniformExpansion,
    RandomRolloutEvaluation,
    StandardBackpropagation,
)
from core.agent_interface import Agent
from environments.base import BaseEnvironment, ActionType


class MCTSAgent(Agent):
    """Agent that uses MCTS (via MCTSOrchestrator) to choose actions."""

    def __init__(
        self,
        num_simulations: int = 100,
        exploration_constant: float = 1.41,
        rollout_max_depth: int = 100,
        discount_factor: float = 1.0,
    ):
        """
        Args:
            num_simulations: Number of MCTS simulations per move.
            exploration_constant: Exploration constant (c) for UCB1 selection.
            rollout_max_depth: Max depth for random rollouts during evaluation.
            discount_factor: Discount factor for rollout rewards (usually 1.0).
        """
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")
        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")

        selection_strategy = UCB1Selection(exploration_constant=exploration_constant)
        expansion_strategy = UniformExpansion()
        evaluation_strategy = RandomRolloutEvaluation(
            max_rollout_depth=rollout_max_depth, discount_factor=discount_factor
        )
        backpropagation_strategy = StandardBackpropagation()

        self.mcts_orchestrator = MCTSOrchestrator(
            selection_strategy=selection_strategy,
            expansion_strategy=expansion_strategy,
            evaluation_strategy=evaluation_strategy,
            backpropagation_strategy=backpropagation_strategy,
            num_simulations=num_simulations,
        )

        self._last_action: Optional[ActionType] = None

    def act(self, env: BaseEnvironment) -> Optional[ActionType]:
        self.mcts_orchestrator.set_root(state=env.state)
        self.mcts_orchestrator.search(env)

        chosen_action = self.mcts_orchestrator.get_policy().chosen_action
        assert chosen_action is not None
        self._last_action = chosen_action
        return chosen_action

    def reset(self) -> None:
        """Reset the MCTS search tree."""
        self.mcts_orchestrator.set_root()
        self._last_action = None
