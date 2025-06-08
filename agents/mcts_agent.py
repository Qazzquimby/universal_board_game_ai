from typing import Optional, Dict

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
    PolicyResult,
)
from core.agent_interface import Agent
from environments.base import BaseEnvironment, ActionType, StateWithKey


class MCTSNodeCache:
    def __init__(self):
        self.enabled = True
        self._key_to_node = LRUCache(1024 * 8)

    def get_matching_node(self, key: int) -> Optional[MCTSNode]:
        if self.enabled:
            return self._key_to_node.get(key, None)
        else:
            return None

    def cache_node(self, key: int, node: MCTSNode):
        if self.enabled:
            self._key_to_node[key] = node


EARLY_STOP_IF_CHANGE_IMPOSSIBLE_CHECK_FREQUENCY = 50


class MCTSAgent(Agent):
    def __init__(
        self,
        num_simulations: int,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
        temperature=0,
    ):
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        self.selection_strategy = selection_strategy
        self.expansion_strategy = expansion_strategy
        self.evaluation_strategy = evaluation_strategy
        self.backpropagation_strategy = backpropagation_strategy

        self.num_simulations = num_simulations
        self.temperature = temperature

        self.root: MCTSNode = None
        self.cache = MCTSNodeCache()

    def _ensure_state_is_root(self, state_with_key: StateWithKey):
        if self.root and self.root.state_with_key.key == state_with_key.key:
            return  # already root

        matching_node = self.cache.get_matching_node(key=state_with_key.key)
        if matching_node:
            self.root = matching_node
        else:
            self.root = MCTSNode(state_with_key=state_with_key)
            self.cache.cache_node(key=state_with_key.key, node=self.root)

    def act(self, env: BaseEnvironment) -> Optional[ActionType]:
        self._ensure_state_is_root(state_with_key=env.get_state_with_key())
        self.search(env=env)
        policy_result = self.get_policy()
        return policy_result.chosen_action

    def search(self, env: BaseEnvironment) -> MCTSNode:
        """
        Run the MCTS search for a specified number of simulations.

        Args:
            env: The current environment instance.

        Returns:
            The root node of the search tree after simulations.
        """
        self._ensure_state_is_root(state_with_key=env.get_state_with_key())

        for sim_idx in range(self.num_simulations):
            if (
                sim_idx
                and EARLY_STOP_IF_CHANGE_IMPOSSIBLE_CHECK_FREQUENCY
                and sim_idx % EARLY_STOP_IF_CHANGE_IMPOSSIBLE_CHECK_FREQUENCY == 0
            ):
                visit_counts = sorted(
                    [edge.num_visits for edge in self.root.edges.values()]
                )
                if len(visit_counts) < 2:
                    break
                max_visits = visit_counts[-1]
                second_most_visits = visit_counts[-2]
                sims_needed_to_change_mind = max_visits - second_most_visits
                remaining_sims = self.num_simulations - sim_idx
                if remaining_sims < sims_needed_to_change_mind:
                    break

            sim_env = env.copy()

            # 1. Selection: Find a leaf node using the selection strategy.
            #    The strategy modifies sim_env to match the leaf node's state.
            selection_result = self.selection_strategy.select(
                node=self.root, sim_env=sim_env, cache=self.cache
            )
            path = selection_result.path
            leaf_node = selection_result.leaf_node
            leaf_env = selection_result.leaf_env

            # 2. Expansion & Evaluation
            player_at_leaf = leaf_env.get_current_player()
            if not leaf_node.is_expanded:
                self.expansion_strategy.expand(leaf_node, leaf_env)
            value = float(self.evaluation_strategy.evaluate(leaf_node, leaf_env))
            player_to_value = {}
            for player in range(env.num_players):
                if player == player_at_leaf:
                    player_to_value[player] = value
                else:
                    player_to_value[player] = -value

            # 3. Backpropagation
            # The value should be from the perspective of the player whose turn it was at the leaf node.
            self.backpropagation_strategy.backpropagate(
                path=path, player_to_value=player_to_value
            )
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
        assert self.root

        action_visits: Dict[ActionType, int] = {
            action: edge.num_visits for action, edge in self.root.edges.items()
        }

        actions = list(action_visits.keys())
        visits = np.array(
            [action_visits[action] for action in actions], dtype=np.float64
        )

        if self.temperature == 0:
            best_action_index = np.argmax(visits)
            chosen_action = actions[best_action_index]
            probabilities = {
                act: (1.0 if act == chosen_action else 0.0) for act in actions
            }
        else:
            log_visits = np.log(visits + 1e-10)
            scaled_log_visits = log_visits / self.temperature
            scaled_log_visits -= np.max(scaled_log_visits)
            exp_scaled_log_visits = np.exp(scaled_log_visits)

            probs_values = exp_scaled_log_visits / np.sum(exp_scaled_log_visits)

            chosen_action_index = np.random.choice(len(actions), p=probs_values)
            chosen_action = actions[chosen_action_index]

            probabilities = {act: prob for act, prob in zip(actions, probs_values)}

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
