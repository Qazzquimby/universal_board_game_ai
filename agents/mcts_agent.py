from typing import Optional, Dict

import numpy as np

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
    MCTSNodeCache,
    EARLY_STOP_IF_CHANGE_IMPOSSIBLE_CHECK_FREQUENCY,
)
from core.agent_interface import Agent
from environments.base import BaseEnvironment, ActionType, StateWithKey


class BaseMCTSAgent(Agent):

    model_type: str = "mcts"

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
        self.node_cache = MCTSNodeCache()

    def set_root_to_state(self, state_with_key: StateWithKey):
        if self.root and self.root.state_with_key.key == state_with_key.key:
            return  # already root

        matching_node = self.node_cache.get_matching_node(key=state_with_key.key)
        if matching_node:
            self.root = matching_node
        else:
            self.root = MCTSNode(state_with_key=state_with_key)
            self.node_cache.cache_node(key=state_with_key.key, node=self.root)

    def _should_stop_early(self, sim_idx: int) -> bool:
        """Checks if the search can be stopped early."""
        return False  # Consider ucb tracking https://aistudio.google.com/prompts/1u8ZK7JpExSOXckEx9FfhrfSQTn5cFr8X

    def _expand_leaf(self, leaf_node: MCTSNode, leaf_env: BaseEnvironment, train: bool):
        """Default expansion logic. Can be overridden."""
        if not leaf_node.is_expanded:
            self.expansion_strategy.expand(leaf_node, leaf_env)

    def _run_simulation(
        self,
        env: BaseEnvironment,
        train: bool,
        remaining_sims: int,
        contender_actions: Optional[set],
    ):
        """Runs a single simulation from selection to backpropagation."""
        sim_env = env.copy()

        # 1. Selection
        selection_result = self.selection_strategy.select(
            node=self.root,
            sim_env=sim_env,
            cache=self.node_cache,
            remaining_sims=remaining_sims,
            contender_actions=contender_actions,
        )
        path = selection_result.path
        leaf_node = selection_result.leaf_node
        leaf_env = selection_result.leaf_env

        # 2. Expansion
        self._expand_leaf(leaf_node, leaf_env, train)

        # 3. Evaluation
        player_at_leaf = leaf_env.get_current_player()
        value = float(self.evaluation_strategy.evaluate(leaf_node, leaf_env))
        player_to_value = {}
        for player in range(env.num_players):
            if player == player_at_leaf:
                player_to_value[player] = value
            else:
                player_to_value[player] = -value

        # 4. Backpropagation
        self.backpropagation_strategy.backpropagate(
            path=path, player_to_value=player_to_value
        )

    def search(self, env: BaseEnvironment, train: bool = False) -> MCTSNode:
        """
        Run the MCTS search for a specified number of simulations.
        """
        self.set_root_to_state(state_with_key=env.get_state_with_key())

        contender_actions: Optional[set] = None

        for sim_idx in range(self.num_simulations):
            if self._should_stop_early(sim_idx):
                break

            remaining_sims = self.num_simulations - (sim_idx + 1)
            self._run_simulation(
                env,
                train,
                remaining_sims=remaining_sims,
                contender_actions=contender_actions,
            )

            contender_actions = self.get_new_contender_actions(
                contender_actions=contender_actions, remaining_sims=remaining_sims
            )
            if contender_actions and len(contender_actions) <= 1:
                break
        return self.root

    def get_policy_from_visits(self, temperature: float) -> PolicyResult:
        """
        Calculates the final action policy based on root children visits.
        This is a utility method, temperature must be passed explicitly.
        """
        assert self.root

        action_visits: Dict[ActionType, int] = {
            action: edge.num_visits for action, edge in self.root.edges.items()
        }

        actions = list(action_visits.keys())
        visits = np.array(
            [action_visits[action] for action in actions], dtype=np.float64
        )

        if not actions:
            return PolicyResult(
                chosen_action=None, action_probabilities={}, action_visits=action_visits
            )

        if temperature == 0:
            best_action_index = np.argmax(visits)
            chosen_action = actions[best_action_index]
            probabilities = {
                act: (1.0 if act == chosen_action else 0.0) for act in actions
            }
        else:
            log_visits = np.log(visits + 1e-10)
            scaled_log_visits = log_visits / temperature
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

    def act(self, env: BaseEnvironment, train: bool = False) -> Optional[ActionType]:
        raise NotImplementedError

    def get_new_contender_actions(self, contender_actions, remaining_sims):
        if contender_actions is None and self.root.is_expanded:
            contender_actions = set(self.root.edges.keys())
        if contender_actions and len(contender_actions) > 1:
            edges = self.root.edges
            max_visits = max(edge.num_visits for edge in edges.values())

            eliminated_actions = {
                action
                for action in contender_actions
                if max_visits - edges[action].num_visits > remaining_sims
            }
            if eliminated_actions:
                contender_actions -= eliminated_actions

        return contender_actions

    def reset_game(self) -> None:
        self.node_cache = MCTSNodeCache()


class MCTSAgent(BaseMCTSAgent):
    def __init__(
        self,
        num_simulations: int,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
        temperature=0,
    ):
        super().__init__(
            num_simulations=num_simulations,
            selection_strategy=selection_strategy,
            expansion_strategy=expansion_strategy,
            evaluation_strategy=evaluation_strategy,
            backpropagation_strategy=backpropagation_strategy,
        )
        self.temperature = temperature

    def act(self, env: BaseEnvironment, train: bool = False) -> Optional[ActionType]:
        self.search(env=env, train=train)
        policy_result = self.get_policy_from_visits(temperature=self.temperature)
        return policy_result.chosen_action

    def reset_game(self) -> None:
        self.node_cache = MCTSNodeCache()


def make_pure_mcts(num_simulations):
    return MCTSAgent(
        num_simulations=num_simulations,
        selection_strategy=UCB1Selection(exploration_constant=1.41),
        expansion_strategy=UniformExpansion(),
        evaluation_strategy=RandomRolloutEvaluation(
            max_rollout_depth=100, discount_factor=0.98
        ),
        backpropagation_strategy=StandardBackpropagation(),
    )
