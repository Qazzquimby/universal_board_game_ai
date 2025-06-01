from typing import Optional

import numpy as np
from cachetools import LRUCache
from loguru import logger

from algorithms.mcts import (
    UCB1Selection,
    UniformExpansion,
    RandomRolloutEvaluation,
    StandardBackpropagation,
    SelectionStrategy,
    ExpansionStrategy,
    EvaluationStrategy,
    BackpropagationStrategy,
    MCTSNode,
    DEBUG, PolicyResult,
)
from core.agent_interface import Agent
from environments.base import BaseEnvironment, ActionType, StateWithKey


class MCTSAgent(Agent):
    def __init__(
        self,
        num_simulations: int,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
    ):
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        self.selection_strategy = selection_strategy
        self.expansion_strategy = expansion_strategy
        self.evaluation_strategy = evaluation_strategy
        self.backpropagation_strategy = backpropagation_strategy

        self.num_simulations = num_simulations

        self.root: MCTSNode = None
        self._tree_reuse_enabled = True
        self._key_to_node = LRUCache(1024 * 8)

    def get_matching_node(self, key: int) -> Optional[MCTSNode]:
        return self._key_to_node.get(key, None)

    def set_root(self, state_with_key: StateWithKey):
        matching_node = self.get_matching_node(key=state_with_key.key)
        if matching_node:
            self.root = MCTSNode(state_with_key=state_with_key)
        else:
            self.root = MCTSNode()

    def act(self, env: BaseEnvironment) -> Optional[ActionType]:
        self.set_root(state=env.state)
        self.search(env)
        chosen_action = self.get_policy().chosen_action
        assert chosen_action is not None
        return chosen_action

    def search(self, env: BaseEnvironment) -> MCTSNode:
        """
        Run the MCTS search for a specified number of simulations.

        Args:
            env: The current environment instance.

        Returns:
            The root node of the search tree after simulations.
        """
        if DEBUG:
            current_env_state = env.get_observation()
            current_env_key = get_state_key(current_env_state)
            if self.root.state is None:
                self.root.state = current_env_state
            if self.root.state_key is None:
                self.root.state_key = current_env_key

        for sim_idx in range(self.num_simulations):
            sim_env = env.copy()

            # 1. Selection: Find a leaf node using the selection strategy.
            #    The strategy modifies sim_env to match the leaf node's state.
            selection_result = self.selection_strategy.select(self.root, sim_env)
            path = selection_result.path
            leaf_node = selection_result.leaf_node
            leaf_env = selection_result.leaf_env

            # 2. Expansion & Evaluation
            player_at_leaf = leaf_env.get_current_player()
            if leaf_env.is_game_over():
                winner = leaf_env.get_winning_player()
                if winner is None:  # draw
                    value = 0.0
                else:
                    value = 1.0 if winner == player_at_leaf else -1.0
            else:
                if not leaf_node.is_expanded():
                    self.expansion_strategy.expand(leaf_node, leaf_env)

                value = float(self.evaluation_strategy.evaluate(leaf_node, leaf_env))

            # 3. Backpropagation
            # The value should be from the perspective of the player whose turn it was at the leaf node.
            self.backpropagation_strategy.backpropagate(path, value, player_at_leaf)
        return self.root

    def get_policy(self) -> PolicyResult:
        """
        Calculates the final action policy based on root children visits.

        Returns:
            A PolicyResult dataclass instance.

        Raises:
            ValueError: If the root node has no children or no child has visits > 0
                        after simulations, indicating an issue (e.g., insufficient search
                        or problem during MCTS phases).
        """
        assert self.root.children

        action_visits = {
            action: child.visit_count for action, child in self.root.children.items()
        }

        valid_actions_visits = {a: v for a, v in action_visits.items() if v > 0}
        assert valid_actions_visits

        actions = list(valid_actions_visits.keys())
        visits = np.array(list(valid_actions_visits.values()), dtype=np.float64)

        probabilities = {}

        if self.temperature == 0:
            best_action_index = np.argmax(visits)
            chosen_action = actions[best_action_index]
            for i, act in enumerate(actions):
                probabilities[act] = 1.0 if i == best_action_index else 0.0
        else:
            assert self.temperature > 0

            log_visits = np.log(visits + 1e-10)
            log_probs = log_visits / self.temperature
            log_probs -= np.max(log_probs)
            exp_probs = np.exp(log_probs)
            probs_values = exp_probs / np.sum(exp_probs)

            probs_values /= np.sum(probs_values)
            if not np.isclose(np.sum(probs_values), 1.0):
                logger.warning(
                    f"Probabilities sum to {np.sum(probs_values)} after normalization. Re-normalizing."
                )
                probs_values /= np.sum(probs_values)

            for act, prob in zip(actions, probs_values):
                probabilities[act] = prob

            chosen_action_index = np.random.choice(len(actions), p=probs_values)
            chosen_action = actions[chosen_action_index]

        for zero_visit_action in action_visits:
            if zero_visit_action not in probabilities:
                probabilities[zero_visit_action] = 0.0

        assert chosen_action is not None
        return PolicyResult(
            chosen_action=chosen_action,
            action_probabilities=probabilities,
            action_visits=action_visits,
        )


def make_pure_mcts(num_simulations):
    return MCTSAgent(
        num_simulations=num_simulations,
        selection_strategy=UCB1Selection(exploration_constant=1.41),
        expansion_strategy=UniformExpansion(),
        evaluation_strategy=RandomRolloutEvaluation(
            max_rollout_depth=100, discount_factor=1
        ),
        backpropagation_strategy=StandardBackpropagation(),
    )
