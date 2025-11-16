import math
import random
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
    Iterator,
)
import abc
from dataclasses import dataclass, field

import torch
from cachetools import LRUCache
import torch.nn as nn
import numpy as np

from environments.base import (
    ActionType,
    BaseEnvironment,
    StateWithKey,
    StateType,
    DataFrame,
)

DEBUG = True

EARLY_STOP_IF_CHANGE_IMPOSSIBLE_CHECK_FREQUENCY = 50


def _get_current_player_from_state(state: StateType) -> int:
    """Gets the current player from a state dictionary, accommodating both old and new env styles."""
    # Old Pydantic-based state
    if "players" in state and isinstance(state.get("players"), dict):
        return state["players"]["current_index"]
    if "game" in state and isinstance(state.get("game"), DataFrame):
        return state["game"]["current_player"][0]
    raise ValueError(
        "Could not determine current player from the provided state representation."
    )


@dataclass
class PathStep:
    """A single step in the MCTS selection path."""

    node: "MCTSNode"
    # None iff first node
    action_taken_to_reach_this_node: Optional[ActionType]


class SearchPath:
    """The path taken during one MCTS selection phase."""

    def __init__(self, initial_node: "MCTSNode"):
        self._steps: List[PathStep] = []
        self._visited_keys: set[int] = set()
        self.add(node=initial_node, action_leading_to_node=None)

    def add(self, node: "MCTSNode", action_leading_to_node: Optional[ActionType]):
        self._visited_keys.add(node.state_with_key.key)
        self._steps.append(PathStep(node, action_leading_to_node))

    def has_visited_key(self, key: int) -> bool:
        return key in self._visited_keys

    def __iter__(self) -> Iterator[PathStep]:
        return reversed(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def last_node(self) -> "MCTSNode":
        if not self._steps:
            raise IndexError("SearchPath is empty, cannot get last node.")
        return self._steps[-1].node

    def get_step_details(
        self, steps_from_end: int
    ) -> Tuple["MCTSNode", Optional[ActionType], Optional["MCTSNode"]]:
        """
        Helper for backpropagation. Gets current node, action that led to it, and its parent.
        steps_from_end=0 is the leaf, index_from_end=1 is its parent, etc.
        Returns: (current_node, action_to_current, parent_node_of_current)
        Parent is None if current_node is root.
        Action is None if current_node is root.
        """
        actual_index = len(self._steps) - 1 - steps_from_end
        if actual_index < 0:
            raise IndexError("Index out of bounds for path steps.")

        current_step = self._steps[actual_index]
        current_node = current_step.node
        action_to_current = current_step.action_taken_to_reach_this_node

        parent_node = None
        if actual_index > 0:  # If not the root node
            parent_node = self._steps[actual_index - 1].node

        return current_node, action_to_current, parent_node


@dataclass
class Edge:
    """Represents an action from a state"""

    prior: float
    num_visits: int = 0
    total_value: float = 0.0  # from perspective of player taking the action
    child_node: Optional["MCTSNode"] = field(default=None, repr=False)

    @property
    def value(self) -> float:
        if self.num_visits == 0:
            return 0.0
        return self.total_value / self.num_visits


class MCTSNode:
    """Represents a node in the MCTS tree."""

    def __init__(
        self,
        state_with_key: StateWithKey,
    ):
        self.state_with_key = state_with_key

        self.edges: Dict[
            ActionType, Edge
        ] = {}  # should be index to edge, which should be list
        self.is_expanded = False

        # for value estimate, not actually needed
        self.num_visits = 0
        self.total_value = 0.0

    @property
    def value_estimate(self) -> float:
        # unused, I believe
        """Calculates the mean value Q(s,a) of the node (action leading to this state)."""
        if self.num_visits == 0:
            return 0.0
        return self.total_value / self.num_visits

    @property
    def current_player_index(self):
        if hasattr(self, "player_idx"):
            current_player_index = self.player_idx
        else:
            current_player_index = _get_current_player_from_state(
                self.state_with_key.state
            )
        return current_player_index


class MCTSNodeCache:
    def __init__(self):
        self.enabled = True
        self._key_to_node: LRUCache[int, MCTSNode] = LRUCache(1024 * 8)

    def get_matching_node(self, key: int) -> Optional[MCTSNode]:
        if self.enabled:
            return self._key_to_node.get(key, None)
        return None

    def cache_node(self, key: int, node: MCTSNode):
        if self.enabled:
            self._key_to_node[key] = node


@dataclass
class SelectionResult:
    """Holds the results of the MCTS selection phase."""

    path: SearchPath
    leaf_env: BaseEnvironment  # worried these may be large and waste memory. Not needed?

    @property
    def leaf_node(self):
        return self.path.last_node


class SelectionStrategy(abc.ABC):
    @abc.abstractmethod
    def select(
        self,
        node: "MCTSNode",
        sim_env: BaseEnvironment,
        cache: "MCTSNodeCache",
        remaining_sims: int,
        contender_actions: Optional[set],
    ) -> SelectionResult:
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
        self, path: SearchPath, player_to_value: Dict[int, float]
    ) -> None:
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

    def _score_edge(self, edge: Edge, parent_node_num_visits: int) -> float:
        """Calculates the UCB1 score for a child node."""
        if edge.num_visits == 0:
            return float("inf")

        # UCB Score = -Q(child) + C * P(child) * sqrt(log(N(parent)) / N(child))
        # We use -Q(child) because child.value is from the perspective of the player
        # at the child state. We need the value from the parent's perspective.
        exploitation_term = edge.value  # check if should be negative
        exploration_term = (
            self.exploration_constant
            * edge.prior
            * math.sqrt(math.log(parent_node_num_visits) / edge.num_visits)
        )
        return exploitation_term + exploration_term

    def select(
        self,
        node: MCTSNode,
        sim_env: BaseEnvironment,
        cache: "MCTSNodeCache",
        remaining_sims: int,
        contender_actions: Optional[set],
    ) -> SelectionResult:
        """Select child node with highest UCB score until a leaf node is reached.
        Modifies sim_env"""
        path = SearchPath(initial_node=node)
        current_node: MCTSNode = node
        # todo rename node to start node?

        while not sim_env.is_done:
            if not current_node.is_expanded:
                return SelectionResult(path=path, leaf_env=sim_env)

            best_action_index = self._select_action_index_from_edges(
                current_node=current_node,
                start_node=node,
                contender_actions=contender_actions,
            )
            legal_actions = sim_env.get_legal_actions()
            best_action = legal_actions[best_action_index]
            step_result = sim_env.step(best_action)

            if path.has_visited_key(step_result.next_state_with_key.key):
                # Cycle detected, terminate search path here.
                return SelectionResult(path=path, leaf_env=sim_env)

            next_node = cache.get_matching_node(key=step_result.next_state_with_key.key)
            if not next_node:
                next_node = MCTSNode(state_with_key=step_result.next_state_with_key)
                cache.cache_node(
                    key=step_result.next_state_with_key.key, node=next_node
                )
                path.add(node=next_node, action_leading_to_node=best_action_index)
                return SelectionResult(path=path, leaf_env=sim_env)

            current_node = next_node
            path.add(current_node, best_action_index)
        return SelectionResult(path=path, leaf_env=sim_env)

    def _select_action_index_from_edges(
        self,
        current_node: MCTSNode,
        start_node: MCTSNode,
        contender_actions: Optional[set],
    ) -> int:
        edges_to_consider = current_node.edges
        if current_node is start_node and contender_actions is not None:
            edges_to_consider = {
                action: edge
                for action, edge in current_node.edges.items()
                if action in contender_actions
            }

        best_score = -float("inf")
        best_action_index: Optional[ActionType] = None
        for action_index, edge in edges_to_consider.items():
            score = self._score_edge(
                edge=edge, parent_node_num_visits=current_node.num_visits
            )
            if score > best_score:
                best_score = score
                best_action_index = action_index

        assert best_action_index is not None
        return best_action_index


class UniformExpansion(ExpansionStrategy):
    """Expands a node by creating children for all legal actions with uniform priors."""

    def expand(self, node: MCTSNode, env_at_node: BaseEnvironment) -> None:
        if node.is_expanded or env_at_node.is_done:
            return

        legal_actions = env_at_node.get_legal_actions()
        assert legal_actions
        assert not node.edges
        for action_index, action in enumerate(legal_actions):
            node.edges[action_index] = Edge(prior=1.0)
        node.is_expanded = True


class RandomRolloutEvaluation(EvaluationStrategy):
    """Evaluates a node by performing a random rollout simulation."""

    def __init__(self, max_rollout_depth: int = 50, discount_factor: float = 1.0):
        self.max_rollout_depth = max_rollout_depth
        self.discount_factor = discount_factor  # Usually 1.0 for MCTS terminal rewards

    def evaluate(self, node: MCTSNode, env: BaseEnvironment) -> float:
        """Simulate game from the given environment state using random policy."""
        player_at_start = env.get_current_player()

        if env.is_done:
            winner = env.get_winning_player()
            if winner is None:
                return 0.0
            return 1.0 if winner == player_at_start else -1.0

        sim_env = env.copy()
        current_step = 0

        while not sim_env.is_done and current_step < self.max_rollout_depth:
            legal_actions = sim_env.get_legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            sim_env.step(action)
            current_step += 1

        if current_step >= self.max_rollout_depth:
            return 0.0

        winner = sim_env.get_winning_player()
        value = 0.0
        if winner is not None:
            value = 1.0 if winner == player_at_start else -1.0

        value *= self.discount_factor**current_step

        return value


class StandardBackpropagation(BackpropagationStrategy):
    """Updates node statistics by backpropagating the evaluation value."""

    def backpropagate(
        self, path: SearchPath, player_to_value: Dict[int, float]
    ) -> None:
        for i in range(len(path)):
            node, action_to_node, parent_of_node = path.get_step_details(
                steps_from_end=i
            )

            node.num_visits += 1
            node.total_value += player_to_value.get(node.current_player_index, 0.0)

            if parent_of_node and action_to_node is not None:
                # not start of path
                action_key = (
                    tuple(action_to_node)
                    if isinstance(action_to_node, list)
                    else action_to_node
                )

                edge_to_update = parent_of_node.edges[action_key]
                edge_to_update.num_visits += 1

                value = player_to_value.get(parent_of_node.current_player_index)
                edge_to_update.total_value += value


@dataclass
class PolicyResult:
    """Holds the results of the MCTS policy calculation."""

    chosen_action: ActionType
    action_probabilities: Dict[ActionType, float] = field(default_factory=dict)
    action_visits: Dict[ActionType, int] = field(default_factory=dict)
