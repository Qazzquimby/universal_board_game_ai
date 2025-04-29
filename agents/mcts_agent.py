import random
from typing import Optional

from loguru import logger

from algorithms.mcts import (
    MCTSOrchestrator,
    UCB1Selection,
    UniformExpansion,
    RandomRolloutEvaluation,
    StandardBackpropagation,
)
from core.agent_interface import Agent
from environments.base import BaseEnvironment, StateType, ActionType


class MCTSAgent(Agent):
    """Agent that uses MCTS (via MCTSOrchestrator) to choose actions."""

    def __init__(
        self,
        env: BaseEnvironment,
        num_simulations: int = 100,
        exploration_constant: float = 1.41,
        rollout_max_depth: int = 100,
        discount_factor: float = 1.0,
        temperature: float = 0.0,
        tree_reuse: bool = True,
    ):
        """
        Args:
            env: The game environment instance (used for copy).
            num_simulations: Number of MCTS simulations per move.
            exploration_constant: Exploration constant (c) for UCB1 selection.
            rollout_max_depth: Max depth for random rollouts during evaluation.
            discount_factor: Discount factor for rollout rewards (usually 1.0).
            temperature: Temperature for final action selection (0=deterministic).
            tree_reuse: Whether to reuse the MCTS tree between moves.
        """
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")
        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")

        self.env = env
        self.temperature = temperature
        self._tree_reuse = tree_reuse

        # Assemble the MCTS Orchestrator with "Pure MCTS" strategies
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
        self.mcts_orchestrator._tree_reuse_enabled = tree_reuse

        self._last_action: Optional[ActionType] = None

    def act(self, state: StateType, train: bool = False) -> Optional[ActionType]:
        """
        Perform MCTS search and choose the best action.

        Args:
            state: The current environment state observation.
            train: If True, uses temperature sampling for action selection. (Not used in pure MCTS agent currently)

        Returns:
            The chosen action, or None if no action is possible.
        """
        # If reusing tree, advance the root based on the opponent's last move (if known)
        # This agent doesn't track opponent moves, so we rely on the caller
        # having called `notify_opponent_action` or similar if tree reuse is desired.
        # For now, we assume the root is either fresh or correctly advanced externally.

        # If not reusing the tree, reset it before every search
        if not self._tree_reuse:
            self.reset()

        # The environment passed should match the state
        self.mcts_orchestrator.search(self.env, state)
        current_temp = self.temperature  # Could be overridden by `train` flag if needed
        chosen_action, _, action_visits = self.mcts_orchestrator.get_policy(
            temperature=current_temp
        )
        assert chosen_action

        if self._tree_reuse and chosen_action is not None:
            self.mcts_orchestrator.advance_root(chosen_action)

        self._last_action = chosen_action
        return chosen_action

    def reset(self) -> None:
        """Reset the MCTS search tree."""
        self.mcts_orchestrator.reset_root()
        self._last_action = None

    # Optional: Method to allow external advancement of the tree root
    # This is useful if the game loop manages turns and knows the opponent's move
    def notify_action_taken(self, action: ActionType) -> None:
        """
        Call this method after an action (either ours or opponent's) is taken
        in the environment to advance the MCTS root if tree reuse is enabled.
        """
        if self._tree_reuse:
            self.mcts_orchestrator.advance_root(action)
