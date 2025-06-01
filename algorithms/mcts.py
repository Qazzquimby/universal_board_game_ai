import math
import random
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
)
import abc
from dataclasses import dataclass, field

import torch
from loguru import logger
import torch.nn as nn
import numpy as np

from environments.base import ActionType, BaseEnvironment, StateWithKey

DEBUG = True


class MCTSNode:
    """Represents a node in the MCTS tree."""

    def __init__(
        self,
        state_with_key: StateWithKey,
        parent: Optional["MCTSNode"] = None,
        prior: float = 1.0,
    ):
        self.parent = parent
        self.prior: float = prior
        self.state_with_key = state_with_key

        self.children: Dict[ActionType, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0

    @property
    def value(self) -> float:
        """Calculates the mean value Q(s,a) of the node (action leading to this state)."""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        """Checks if the node has been expanded (i.e., has children)."""
        return bool(self.children)


@dataclass
class SelectionResult:
    """Holds the results of the MCTS selection phase."""

    path: List["MCTSNode"]
    leaf_node: "MCTSNode"
    leaf_env: BaseEnvironment


class SelectionStrategy(abc.ABC):
    @abc.abstractmethod
    def select(self, node: "MCTSNode", env: BaseEnvironment) -> SelectionResult:
        """
        Select a path from the given node down to a leaf node.

        Args:
            node: The starting node (usually the root).
            env: A copy of the environment corresponding to the starting node's state.
                 This environment instance will be modified during selection.

        Returns:
            A SelectionResult dataclass instance containing the path, leaf node,
            and the environment state corresponding to the leaf node.
        """
        pass


class ExpansionStrategy(abc.ABC):
    @abc.abstractmethod
    def expand(
        self,
        node: "MCTSNode",
        env: BaseEnvironment,
    ) -> None:
        """
        Expand a leaf node by adding children based on legal actions.

        Args:
            node: The leaf node to expand.
            env: The environment state corresponding to the leaf node.
                 Should not be modified by the expansion strategy itself.
        """
        pass


class EvaluationStrategy(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, node: "MCTSNode", env: BaseEnvironment) -> float:
        """
        Evaluate a leaf node to estimate its value.
        The value should be from the perspective of the player whose turn it is at the leaf node.

        Args:
            node: The leaf node to evaluate.
            env: The environment state corresponding to the leaf node.
                 Should not be modified by the evaluation strategy itself,
                 though internal copies might be made (e.g., for rollouts).

        Returns:
            The estimated value (float).
        """
        pass


class BackpropagationStrategy(abc.ABC):
    @abc.abstractmethod
    def backpropagate(
        self, path: List["MCTSNode"], value: float, player_at_leaf: int
    ) -> None:
        """
        Update statistics of nodes along the path based on the evaluation result.

        Args:
            path: The list of nodes from the root to the evaluated leaf (inclusive).
            value: The value obtained from the evaluation (e.g., rollout result or network value).
                   This value should be from the perspective of the player whose turn it was at the leaf node.
            player_at_leaf: The player whose turn it was at the leaf node. (Not strictly needed if value perspective is consistent)
        """
        pass


class DummyAlphaZeroNet(nn.Module):
    """
    A dummy network interface that mimics AlphaZeroNet but returns
    uniform policy priors and zero value. Used when no real network is provided.
    """

    def __init__(self, env: BaseEnvironment):
        super().__init__()
        self.env = env
        self._policy_size = self._calculate_policy_size(env)
        self._action_to_index = {}
        self._index_to_action = {}
        if hasattr(env, "map_action_to_policy_index") and hasattr(
            env, "map_policy_index_to_action"
        ):
            # Precompute mappings if possible
            # This might fail if legal actions depend heavily on state
            env.reset()
            legal_actions = env.get_legal_actions()
            for action in legal_actions:
                idx = env.map_action_to_policy_index(action)
                if idx is not None:
                    self._action_to_index[action] = idx
                    self._index_to_action[idx] = action
            env.reset()

    def _flatten_state(self, state_dict: dict) -> torch.Tensor:
        # Return a dummy tensor, as it won't be used for actual prediction
        return torch.zeros(1)

    def _calculate_input_size(self, env: BaseEnvironment) -> int:
        # Return a dummy size
        return 1

    def _calculate_policy_size(self, env: BaseEnvironment) -> int:
        return env.policy_vector_size

    def get_action_index(self, action: ActionType) -> Optional[int]:
        """Maps an action to its policy index."""
        action_key = tuple(action) if isinstance(action, list) else action
        idx = self._action_to_index.get(action_key)
        if idx is None and hasattr(self.env, "map_action_to_policy_index"):
            idx = self.env.map_action_to_policy_index(action_key)
            if idx is not None:
                self._action_to_index[action_key] = idx
                self._index_to_action[idx] = action_key
        return idx

    def get_action_from_index(self, index: int) -> Optional[ActionType]:
        """Maps a policy index back to an action."""
        action = self._index_to_action.get(index)
        if action is None and hasattr(self.env, "map_policy_index_to_action"):
            # Try on-the-fly mapping if not precomputed
            action = self.env.map_policy_index_to_action(index)
            if action is not None:  # Cache if found
                action_key = tuple(action) if isinstance(action, list) else action
                self._index_to_action[index] = action_key
                self._action_to_index[action_key] = index
        return action

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0] if x.dim() > 0 else 1
        policy_logits = torch.zeros(batch_size, self._policy_size)
        value = torch.zeros(batch_size, 1)
        return policy_logits, value

    def predict(self, state_dict: dict) -> Tuple[np.ndarray, float]:
        """Returns uniform policy probabilities and zero value."""
        policy_probs = np.ones(self._policy_size, dtype=np.float32) / self._policy_size
        value = 0.0
        return policy_probs, value

    def predict_batch(
        self, state_dicts: List[dict]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Returns uniform policy probabilities and zero values for a batch."""
        batch_size = len(state_dicts)
        policy_probs = np.ones(self._policy_size, dtype=np.float32) / self._policy_size
        policy_list = [policy_probs.copy() for _ in range(batch_size)]
        value_list = [0.0] * batch_size
        return policy_list, value_list


class UCB1Selection(SelectionStrategy):
    """Selects nodes using the UCB1 algorithm."""

    def __init__(self, exploration_constant: float):
        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")
        self.exploration_constant = exploration_constant

    def _score_child(self, child: MCTSNode, parent_visits: int) -> float:
        """Calculates the UCB1 score for a child node."""
        if child.visit_count == 0:
            return float("inf")
        assert parent_visits > 0

        # UCB Score = -Q(child) + C * P(child) * sqrt(log(N(parent)) / N(child))
        # We use -Q(child) because child.value is from the perspective of the player
        # at the child state. We need the value from the parent's perspective.
        exploitation_term = -child.value
        exploration_term = (
            self.exploration_constant
            * child.prior
            * math.sqrt(math.log(parent_visits) / child.visit_count)
        )
        return exploitation_term + exploration_term

    def select(self, node: MCTSNode, sim_env: BaseEnvironment) -> SelectionResult:
        """Select child node with highest UCB score until a leaf node is reached.
        Modifies sim_env"""
        path = [node]
        current_node = node

        while current_node.is_expanded() and not sim_env.is_game_over():
            if DEBUG:
                legal_actions_in_sim = sim_env.get_legal_actions()
                legal_action_keys_in_sim = {
                    tuple(a) if isinstance(a, list) else a for a in legal_actions_in_sim
                }
                child_action_keys = set(current_node.children.keys())

                assert legal_action_keys_in_sim == child_action_keys, (
                    f"State mismatch detected in select! Node's children actions do not match environment's legal actions.\n"
                    f"Node State Key: {current_node.state_key}\n"
                    f"Node Children Keys: {child_action_keys}\n"
                    f"Sim Env Legal Action Keys: {legal_action_keys_in_sim}\n"
                    f"Sim Env State: {sim_env.get_state_with_key()}"
                )
            assert current_node.children

            parent_visits = current_node.visit_count
            best_score = -float("inf")
            best_action = None
            best_child_node = None

            for action, child_node in current_node.children.items():
                score = self._score_child(child_node, parent_visits)
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child_node = child_node

            if DEBUG:
                assert (
                    best_action is not None
                ), "UCB1Selection failed to find a best action."
                assert (
                    best_child_node is not None
                ), "UCB1Selection failed to find a best child node."
                action_key = (
                    tuple(best_action) if isinstance(best_action, list) else best_action
                )
                assert (
                    action_key in current_node.children
                ), f"Selected action {action_key} not in node children {list(current_node.children.keys())}"

            done = sim_env.step(best_action).done

            if DEBUG:
                assert (
                    done == sim_env.is_game_over()
                ), f"Environment 'done' flag ({done}) inconsistent with is_game_over() ({sim_env.is_game_over()}) after action {best_action}"

            current_node = best_child_node
            path.append(current_node)
            if done:
                break
        return SelectionResult(path=path, leaf_node=current_node, leaf_env=sim_env)


class UniformExpansion(ExpansionStrategy):
    """Expands a node by creating children for all legal actions with uniform priors."""

    def expand(self, node: MCTSNode, env: BaseEnvironment) -> None:
        if node.is_expanded() or env.is_game_over():
            return

        current_state_with_key = env.get_state_with_key()
        assert node.state_with_key.key == current_state_with_key.key
        # node.state_with_key = current_state_with_key
        # todo hmm. Expanding shouldnt change the state
        # node.state = current_state_with_key
        # node.state_key = get_state_key(current_state_with_key)

        legal_actions = env.get_legal_actions()
        assert legal_actions
        num_legal_actions = len(legal_actions)
        uniform_prior = 1.0
        legal_action_keys = {
            tuple(a) if isinstance(a, list) else a for a in legal_actions
        }

        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action

            if DEBUG:
                if hasattr(env, "_is_valid_action"):
                    assert env._is_valid_action(
                        action
                    ), f"Action {action} from get_legal_actions() is considered invalid by _is_valid_action() in state {current_state_with_key}"
                assert (
                    action_key in legal_action_keys
                ), f"Action key {action_key} (from action {action}) not found in the initially generated legal action keys {legal_action_keys}. State: {current_state_with_key}"
                assert (
                    action_key not in node.children
                ), f"Attempting to expand action {action_key} which already exists as a child. State: {current_state_with_key}"

            child_env = env.copy()
            child_state_with_key = child_env.step(action).next_state_with_key
            child_node = MCTSNode(
                parent=node, prior=uniform_prior, state_with_key=child_state_with_key
            )
            node.children[action_key] = child_node

            if DEBUG:
                assert (
                    action_key in node.children
                ), f"Failed to add child node for action {action_key}"
                assert (
                    node.children[action_key] is child_node
                ), f"Incorrect child node added for action {action_key}"
        if DEBUG:
            assert (
                len(node.children) == num_legal_actions
            ), f"Number of children created ({len(node.children)}) does not match the number of legal actions ({num_legal_actions}). Legal actions: {legal_actions}, Children keys: {list(node.children.keys())}. State: {current_state_with_key}"


class RandomRolloutEvaluation(EvaluationStrategy):
    """Evaluates a node by performing a random rollout simulation."""

    def __init__(self, max_rollout_depth: int = 100, discount_factor: float = 1.0):
        self.max_rollout_depth = max_rollout_depth
        self.discount_factor = discount_factor  # Usually 1.0 for MCTS terminal rewards

    def evaluate(self, node: MCTSNode, env: BaseEnvironment) -> float:
        """Simulate game from the given environment state using random policy."""
        if env.is_game_over():
            player_at_leaf = env.get_current_player()
            winner = env.get_winning_player()
            if winner is None:
                return 0.0  # Draw
            # Value is from the perspective of the player whose turn it *would* be at the leaf
            return 1.0 if winner == player_at_leaf else -1.0

        player_at_rollout_start = env.get_current_player()
        sim_env = env.copy()
        steps = 0

        while not sim_env.is_game_over() and steps < self.max_rollout_depth:
            legal_actions = sim_env.get_legal_actions()
            action = random.choice(legal_actions)

            if DEBUG:
                action_key_rollout = (
                    tuple(action) if isinstance(action, list) else action
                )
                legal_action_keys_rollout = {
                    tuple(a) if isinstance(a, list) else a for a in legal_actions
                }
                assert (
                    action_key_rollout in legal_action_keys_rollout
                ), f"Rollout chose action {action_key_rollout} which is not in legal actions {legal_action_keys_rollout}"

            sim_env.step(action)
            steps += 1

        if steps >= self.max_rollout_depth:
            logger.warning(
                f"MCTS Rollout: Reached max depth ({self.max_rollout_depth}). Treating as draw."
            )
            return 0.0

        winner = sim_env.get_winning_player()
        if winner is None:
            value = 0.0
        elif winner == player_at_rollout_start:
            value = 1.0
        else:
            value = -1.0

        return value


class StandardBackpropagation(BackpropagationStrategy):
    """Updates node statistics by backpropagating the evaluation value."""

    def backpropagate(
        self,
        path: List[MCTSNode],
        value: float,
        player_at_leaf: int,
    ) -> None:
        """
        Backpropagate the evaluated value up the tree, updating node statistics.
        Assumes `value` is from the perspective of the player whose turn it was at the leaf node.
        """
        value_for_node = value

        for i, node_in_path in enumerate(reversed(path)):
            visits_before = node_in_path.visit_count
            value_before = node_in_path.total_value
            node_in_path.visit_count += 1
            node_in_path.total_value += value_for_node

            if DEBUG:
                assert (
                    node_in_path.visit_count == visits_before + 1
                ), f"Visit count did not increment correctly during backprop (Node {len(path)-1-i}). Before: {visits_before}, After: {node_in_path.visit_count}"
                assert math.isclose(
                    node_in_path.total_value, value_before + value_for_node
                ), f"Total value did not update correctly during backprop (Node {len(path)-1-i}). Before: {value_before}, Added: {value_for_node}, After: {node_in_path.total_value}"
            value_for_node *= -1.0


@dataclass
class PolicyResult:
    """Holds the results of the MCTS policy calculation."""

    chosen_action: ActionType
    action_probabilities: Dict[ActionType, float] = field(default_factory=dict)
    action_visits: Dict[ActionType, int] = field(default_factory=dict)
