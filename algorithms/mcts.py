import math
import random
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
)

import torch
from loguru import logger
import numpy as np
import abc
import torch.nn as nn

from core.config import AlphaZeroConfig
from environments.base import ActionType, StateType, BaseEnvironment
from typing import Protocol, Any

# Helper function moved outside class for potential reuse
def get_state_key(s: StateType) -> str:
    """Creates a hashable key from a state dictionary."""
    try:
        parts = []
        for k, v in sorted(s.items()):
            if isinstance(v, np.ndarray):
                # Use hash of bytes for numpy arrays
                parts.append(f"{k}:{hash(v.tobytes())}")
            elif isinstance(v, (list, tuple)):
                # Convert lists to tuples for hashing
                try:
                    parts.append(f"{k}:{hash(tuple(v))}")
                except TypeError:  # Handle unhashable elements within list/tuple if necessary
                    parts.append(f"{k}:{repr(v)}")  # Fallback to repr
            elif isinstance(v, dict):
                # Recursively handle nested dicts (or use repr as fallback)
                parts.append(f"{k}:{get_state_key(v)}")  # Simple recursive call
            else:
                # Use repr for other hashable types
                try:
                    hash(v)  # Check if hashable
                    parts.append(f"{k}:{repr(v)}")
                except TypeError:
                    logger.warning(
                        f"Unhashable type in state key generation: {type(v)} for key {k}. Using repr()."
                    )
                    parts.append(
                        f"{k}:{repr(v)}"
                    )  # Fallback for unhashable non-list/array types
        return "|".join(parts)
    except Exception as e:  # Catch broader exceptions during key generation
        logger.warning(
            f"State key generation failed unexpectedly: {e}. Falling back to simple str(). State: {s}"
        )
        # Fallback to string representation if complex hashing fails
        return str(s)


class SelectionStrategy(abc.ABC):
    @abc.abstractmethod
    def select(
        self, node: "MCTSNode", env: BaseEnvironment
    ) -> Tuple[List["MCTSNode"], "MCTSNode", BaseEnvironment]:
        """
        Select a path from the given node down to a leaf node.

        Args:
            node: The starting node (usually the root).
            env: A copy of the environment corresponding to the starting node's state.

        Returns:
            A tuple containing:
            - path: List of nodes from the root to the leaf (inclusive).
            - leaf_node: The selected leaf node.
            - leaf_env: The environment state corresponding to the leaf node.
        """
        pass


class ExpansionStrategy(abc.ABC):
    @abc.abstractmethod
    def expand(
        self,
        node: "MCTSNode",
        env: BaseEnvironment,
        network_output: Optional[Any] = None,
    ) -> None:
        """
        Expand a leaf node by adding children based on legal actions.
        May use network output (e.g., policy priors) if provided.

        Args:
            node: The leaf node to expand.
            env: The environment state corresponding to the leaf node.
            network_output: Optional output from the evaluation strategy (e.g., policy priors).
        """
        pass


class EvaluationStrategy(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, node: "MCTSNode", env: BaseEnvironment) -> Any:
        """
        Evaluate a leaf node to estimate its value.
        May also return other information needed for expansion (e.g., policy priors).

        Args:
            node: The leaf node to evaluate.
            env: The environment state corresponding to the leaf node.

        Returns:
            The evaluation result (e.g., float value, or tuple like (value, priors)).
            The exact type depends on the concrete strategy.
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
            player_at_leaf: The player whose turn it was at the leaf node.
        """
        pass


class NetworkInterface(Protocol):
    """Defines the interface for network predictions needed by MCTS strategies."""

    def predict(self, state_dict: StateType) -> Tuple[np.ndarray, float]:
        """Predict policy probabilities and value for a single state."""
        ...

    def predict_batch(
        self, state_dicts: List[StateType]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Predict policy probabilities and values for a batch of states."""
        ...


# --- End Interfaces ---


PredictResult = Tuple[
    np.ndarray, float
]  # Keep for now, might be superseded by EvaluationStrategy output type


class DummyAlphaZeroNet(nn.Module):
    """
    A dummy network interface that mimics AlphaZeroNet but returns
    uniform policy priors and zero value. Used when no real network is provided.
    """

    def __init__(self, env: BaseEnvironment):
        super().__init__()
        self.env = env
        self._policy_size = self._calculate_policy_size(env)
        # Store action mappings if available from env
        self._action_to_index = {}
        self._index_to_action = {}
        if hasattr(env, "map_action_to_policy_index") and hasattr(
            env, "map_policy_index_to_action"
        ):
            # Precompute mappings if possible
            # Note: This assumes a fixed set of actions, might not work for all envs
            try:
                # This might fail if legal actions depend heavily on state
                initial_state = env.reset()
                legal_actions = env.get_legal_actions()
                for action in legal_actions:
                    idx = env.map_action_to_policy_index(action)
                    if idx is not None:
                        self._action_to_index[action] = idx
                        self._index_to_action[idx] = action
                # Reset env again to be safe
                env.reset()
            except Exception as e:
                logger.warning(
                    f"Could not precompute action mappings for DummyNet: {e}"
                )
                # Mappings will be attempted on-the-fly

    def _flatten_state(self, state_dict: dict) -> torch.Tensor:
        # Return a dummy tensor, as it won't be used for actual prediction
        return torch.zeros(1)

    def _calculate_input_size(self, env: BaseEnvironment) -> int:
        # Return a dummy size
        return 1

    def _calculate_policy_size(self, env: BaseEnvironment) -> int:
        # Use the environment's policy vector size
        try:
            return env.policy_vector_size
        except AttributeError:
            logger.error("DummyNet requires env to have policy_vector_size attribute.")
            # Fallback, but likely indicates an issue
            return env.num_actions if hasattr(env, "num_actions") else 1

    def get_action_index(self, action: ActionType) -> Optional[int]:
        """Maps an action to its policy index."""
        action_key = tuple(action) if isinstance(action, list) else action
        idx = self._action_to_index.get(action_key)
        if idx is None and hasattr(self.env, "map_action_to_policy_index"):
            # Try on-the-fly mapping if not precomputed
            idx = self.env.map_action_to_policy_index(action_key)
            if idx is not None:  # Cache if found
                self._action_to_index[action_key] = idx
                self._index_to_action[idx] = action_key  # Assume inverse mapping exists
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
                self._action_to_index[
                    action_key
                ] = index  # Assume inverse mapping exists
        return action

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Not used directly, but required by nn.Module
        batch_size = x.shape[0] if x.dim() > 0 else 1
        # Return uniform policy logits (zeros -> uniform softmax) and zero value
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


# get_state_key moved above DummyAlphaZeroNet


class MCTSNode:
    def __init__(self, parent: Optional["MCTSNode"] = None, prior: float = 1.0):
        self.parent = parent
        self.prior = prior  # P(s,a) - Prior probability from the network
        self.visit_count = 0
        self.total_value = 0.0  # W(s,a) - Total action value accumulated
        self.children: Dict[ActionType, MCTSNode] = {}

    @property
    def value(self) -> float:  # Q(s,a) - Mean action value
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)

    def expand(self, action_priors: Dict[ActionType, float]):
        """
        Expand the node using prior probabilities from the network.
        Args:
            action_priors: A dictionary mapping legal actions to their prior probabilities.
        """
        children_added = 0
        for action, prior in action_priors.items():
            action_key = tuple(action) if isinstance(action, list) else action
            if action_key not in self.children:
                # logger.trace(f"  Node.expand: Creating child for action {action_key} with prior {prior:.4f}")
                self.children[action_key] = MCTSNode(parent=self, prior=prior)
                children_added += 1
            else:
                logger.warning(
                    f"  Node.expand: Child for action {action_key} already exists."
                    f" Was expand called multiple times?"
                )

        if action_priors:
            assert children_added > 0 or all(
                p == 0.0 for p in action_priors.values()
            ), f"Node expansion failed: non-zero priors provided but no children added. Priors: {action_priors}"
            assert self.children or all(
                p == 0.0 for p in action_priors.values()
            ), f"Node expansion failed: children dict empty despite non-zero priors. Priors: {action_priors}"


class MCTS:
    """Core MCTS algorithm implementation (UCB1 + Random Rollout)."""

    def __init__(
        self,
        exploration_constant: float = 1.41,  # Standard UCB1 exploration constant
        discount_factor: float = 1.0,  # Discount factor for rollout rewards
        num_simulations: int = 100,
    ):
        self.exploration_constant = exploration_constant
        self.discount_factor = (
            discount_factor  # Not used in standard UCB1 backprop here
        )
        self.num_simulations = num_simulations
        self.root = MCTSNode()

    def reset_root(self):
        """Resets the root node."""
        self.root = MCTSNode()

    def _select(
        self, node: MCTSNode, sim_env: BaseEnvironment
    ) -> Tuple[MCTSNode, BaseEnvironment]:
        """Select child node with highest UCB score until a leaf node is reached."""
        while node.is_expanded() and not sim_env.is_game_over():
            parent_visits = node.visit_count

            # Select the action corresponding to the child with the highest UCB score
            child_scores = {
                act: self._score_child(node=child, parent_visits=parent_visits)
                for act, child in node.children.items()
            }
            if not child_scores:
                logger.warning("MCTS _select: Node expanded but no children found.")
                break

            best_action = max(child_scores, key=child_scores.get)
            action, node = best_action, node.children[best_action]

            sim_env.step(action)
        return node, sim_env

    def _score_child(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate UCB1 score."""
        if node.visit_count == 0:
            return float("inf")
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / node.visit_count
        )
        q_value_for_parent = -node.value
        return q_value_for_parent + exploration_term

    def _expand(self, node: MCTSNode, env: BaseEnvironment):
        """Expand the leaf node by creating children for all legal actions."""
        if node.is_expanded() or env.is_game_over():
            return

        legal_actions = env.get_legal_actions()
        # Use uniform prior for standard MCTS expansion
        action_priors = {
            action: 1.0 / len(legal_actions) if legal_actions else 1.0
            for action in legal_actions
        }
        node.expand(action_priors)

    def _rollout(self, env: BaseEnvironment) -> float:
        """Simulate game from current state using random policy."""
        player_at_rollout_start = env.get_current_player()

        sim_env = env.copy()
        steps = 0
        max_steps = (
            env.width * env.height
            if hasattr(env, "width") and hasattr(env, "height")
            else 100
        )  # Safety break

        while not sim_env.is_game_over() and steps < max_steps:
            legal_actions = sim_env.get_legal_actions()
            if not legal_actions:
                logger.warning("MCTS _rollout: Game not over, but no legal actions.")
                break
            action = random.choice(legal_actions)
            sim_env.step(action)
            steps += 1

        if steps >= max_steps:
            logger.warning(
                f"MCTS _rollout: Reached max steps ({max_steps}). Treating as draw."
            )
            return 0.0

        winner = sim_env.get_winning_player()
        if winner is None:
            value = 0.0
        elif winner == player_at_rollout_start:
            value = 1.0
        else:
            value = -1.0
        # logger.debug(f"  Rollout: StartPlayer={player_at_rollout_start}, Winner={winner}, Value={value}")
        return value

    def _backpropagate(self, leaf_node: MCTSNode, value_from_leaf_perspective: float):
        """
        Backpropagate the evaluated value up the tree, updating node statistics.

        Args:
            leaf_node: The node where the simulation/evaluation ended.
            value_from_leaf_perspective: The outcome (+1, -1, 0) from the perspective
                                         of the player whose turn it was at the leaf_node.
        """
        current_node = leaf_node
        value_for_current_node = value_from_leaf_perspective

        while current_node is not None:
            current_node.visit_count += 1
            # logger.debug(f"  Backprop: Node={current_node}, ValueToAdd={value_for_current_node:.3f}, OldW={current_node.total_value:.3f}, OldN={current_node.visit_count - 1}, NewW={current_node.total_value + value_for_current_node:.3f}, NewN={current_node.visit_count}")
            current_node.total_value += value_for_current_node

            # Flip the value perspective for the parent node.
            value_for_current_node *= -1
            current_node = current_node.parent

    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode:
        """Run MCTS search from the given state using UCB1 and random rollouts."""
        self.reset_root()

        for sim_num in range(self.num_simulations):
            sim_env = env.copy()
            sim_env.set_state(state)

            # 1. Selection: Find a leaf node using UCB1.
            leaf_node, leaf_env_state = self._select(self.root, sim_env)

            # 2. Evaluation: Get the value of the leaf node.
            if not leaf_env_state.is_game_over():
                if not leaf_node.is_expanded():
                    self._expand(leaf_node, leaf_env_state)
                value = self._rollout(leaf_env_state)

            else:
                # Game ended during selection. Determine the outcome.
                # Value must be from the perspective of the player whose turn it was AT THE LEAF node.
                player_at_leaf = leaf_env_state.get_current_player()
                winner = leaf_env_state.get_winning_player()

                if winner is None:
                    value = 0.0
                elif winner == player_at_leaf:
                    value = 1.0
                else:
                    value = -1.0

                # logger.debug(f"  Terminal Found during Select: Winner={winner}, PlayerAtNode={player_at_leaf}, Value={value}")
            # 3. Backpropagation: Update nodes along the path from the leaf to the root.
            self._backpropagate(leaf_node, value)

        return self.root


class MCTSOrchestrator:
    """
    Orchestrates the MCTS process using pluggable strategies.
    Does not contain algorithm-specific logic (like UCB1, PUCT, rollouts).
    """

    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
        num_simulations: int,
        discount_factor: float = 1.0,  # May be used by some strategies
        # Add other core MCTS config as needed (e.g., tree reuse flag)
    ):
        self.selection_strategy = selection_strategy
        self.expansion_strategy = expansion_strategy
        self.evaluation_strategy = evaluation_strategy
        self.backpropagation_strategy = backpropagation_strategy
        self.num_simulations = num_simulations

        self.root: MCTSNode = MCTSNode()

    def reset_root(self):
        """Resets the root node."""
        self.root = MCTSNode()

    def advance_root(self, action: ActionType):
        """Advances the root node down the tree based on the chosen action."""
        action_key = tuple(action) if isinstance(action, list) else action
        if action_key in self.root.children:
            self.root = self.root.children[action_key]
            self.root.parent = None  # Detach from old parent
        else:
            # Action not found, likely means we need to start a fresh search
            logger.warning(
                f"Action {action_key} not found in root children during advance_root. Resetting root."
            )
            self.reset_root()

    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode:
        """
        Run the MCTS search for a specified number of simulations.

        Args:
            env: The *current* environment instance (used for copying).
            state: The current state dictionary corresponding to the env.

        Returns:
            The root node of the search tree after simulations.
        """
        # Ensure the root node corresponds to the initial state if reusing tree
        # (More complex logic needed here if root state doesn't match `state`)
        # For now, assume `advance_root` was called correctly or we start fresh.

        for _ in range(self.num_simulations):
            # Start each simulation with a fresh copy of the environment
            # set to the state corresponding to the *current* root node.
            # This assumes the root node *is* the state we want to search from.
            sim_env = env.copy()
            # TODO: If tree reuse is active, ensure sim_env matches root node's conceptual state.
            # For now, assume env passed to search IS the root state.
            sim_env.set_state(state)  # Set the copy to the state we are searching from

            # 1. Selection
            path, leaf_node, leaf_env = self.selection_strategy.select(
                self.root, sim_env
            )

            # 2. Evaluation & Expansion
            value = 0.0  # Default value
            player_at_leaf = (
                leaf_env.get_current_player()
            )  # Player whose turn it is at the leaf

            if leaf_env.is_game_over():
                # Game ended during selection or leaf is terminal
                winner = leaf_env.get_winning_player()
                if winner is None:
                    value = 0.0
                # Value is from the perspective of the player whose turn it *was* at the leaf
                elif winner == player_at_leaf:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # If node hasn't been expanded before (check needed?)
                # Evaluate the leaf node using the evaluation strategy
                # The evaluation strategy might return just value, or (value, priors)
                eval_output = self.evaluation_strategy.evaluate(leaf_node, leaf_env)

                # TODO: Standardize the output format of evaluate()
                # For now, assume it might be float or tuple (value, priors)
                priors = None
                if isinstance(eval_output, tuple):
                    value, priors = eval_output  # Assumes (value, priors) format
                elif isinstance(eval_output, (float, int)):
                    value = float(eval_output)
                else:
                    raise TypeError(
                        f"Unexpected evaluation output type: {type(eval_output)}"
                    )

                # Expand the node using the expansion strategy (priors are optional)
                # Only expand if the node isn't already terminal
                if not leaf_node.is_expanded():  # Avoid re-expanding
                    self.expansion_strategy.expand(leaf_node, leaf_env, priors)

            # 3. Backpropagation
            # Value must be from the perspective of the player at the leaf node
            self.backpropagation_strategy.backpropagate(path, value, player_at_leaf)

        return self.root

    def get_policy(
        self, temperature: float = 1.0
    ) -> Tuple[Optional[ActionType], Dict[ActionType, float], Dict[ActionType, int]]:
        """
        Calculates the final action policy based on root children visits.

        Args:
            temperature: Controls the exploration/exploitation trade-off in action selection.
                         temp=0 -> deterministic (choose best), temp=1 -> proportional to visits^1.

        Returns:
            Tuple: (chosen_action, action_probabilities, action_visits)
                   Returns (None, {}, {}) if root has no children.
        """
        if not self.root.children:
            logger.warning("MCTS get_policy: Root has no children.")
            return None, {}, {}

        action_visits = {
            action: child.visit_count for action, child in self.root.children.items()
        }
        actions = list(action_visits.keys())
        visits = np.array(
            list(action_visits.values()), dtype=np.float32
        )  # Use float for calculations

        if not actions:  # Should be redundant with the check above, but safe
            return None, {}, {}

        if temperature == 0:
            # Deterministic: choose the action with the highest visit count
            best_action_index = np.argmax(visits)
            chosen_action = actions[best_action_index]
            # Probabilities are 1 for the best action, 0 otherwise
            probabilities = {
                act: (1.0 if i == best_action_index else 0.0)
                for i, act in enumerate(actions)
            }

        else:
            # Temperature sampling
            # Add small epsilon to prevent division by zero if temperature is very high and visits are 0? No, visits should be > 0 if children exist.
            # Ensure visits are positive before exponentiation if temp is not integer
            visits_temp = visits ** (1.0 / temperature)
            total_visits_temp = np.sum(visits_temp)

            if total_visits_temp == 0:
                # Fallback if all temp-adjusted visits are zero (e.g., large temp, small visits)
                # Use uniform probability
                probs = np.ones_like(visits) / len(visits)
            else:
                probs = visits_temp / total_visits_temp

            # Ensure probabilities sum to 1 (handle potential floating point inaccuracies)
            probs /= np.sum(probs)

            # Choose action based on calculated probabilities
            chosen_action_index = np.random.choice(len(actions), p=probs)
            chosen_action = actions[chosen_action_index]
            probabilities = {act: prob for act, prob in zip(actions, probs)}

        return chosen_action, probabilities, action_visits


# Todo refactor into strategies
class AlphaZeroMCTS(MCTS):
    """MCTS algorithm adapted for AlphaZero (PUCT + Network Evaluation)."""

    def __init__(
        self,
        env: BaseEnvironment,
        config: AlphaZeroConfig,
        network_cache: Optional[Dict] = None,
    ):
        super().__init__(
            exploration_constant=config.cpuct,
            num_simulations=config.num_simulations,
        )
        self.config = config
        self.network_cache = network_cache if network_cache is not None else {}

        self.env = env  # Store base env for action mapping and config
        self.current_leaf_node = None
        self.current_leaf_env = None
        self.sim_count = 0
        self.training = True
        self.last_request_state: Optional[StateType] = None

    def _check_dynamic_termination(self) -> bool:
        """Checks if the MCTS search can terminate early based on visit counts."""
        if (
            not self.config.dynamic_simulations_enabled
            or self.sim_count < self.config.dynamic_simulations_min_visits
            or not self.root.children
        ):
            return False

        children_visits = [child.visit_count for child in self.root.children.values()]
        if len(children_visits) < 2:
            return False

        children_visits.sort(reverse=True)
        visit_delta = children_visits[0] - children_visits[1]

        return visit_delta >= self.config.dynamic_simulations_visit_delta

    def advance_root(self, action: ActionType):
        action_key = tuple(action) if isinstance(action, list) else action
        assert (
            action_key in self.root.children
        ), f"Action {action_key} not found in root children {list(self.root.children.keys())} during advance_root."

        self.root = self.root.children[action_key]
        self.root.parent = None  # Detach from the old parent

    def prepare_for_next_search(self, train: bool = False):
        """Resets simulation-specific state before starting the next search iteration."""
        self.sim_count = 0
        self.training = train
        self.current_leaf_node = None
        self.current_leaf_env = None

    def _expand(
        self,
        node: MCTSNode,
        env_state: BaseEnvironment,
        policy_priors_np: Optional[
            np.ndarray
        ] = None,  # not actually optional, just matching type
    ) -> None:
        assert policy_priors_np is not None

        if node.is_expanded() or env_state.is_game_over():
            return

        legal_actions = env_state.get_legal_actions()
        assert legal_actions, "Expanding from state with no legal actions"

        action_priors_dict = {}

        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action
            action_index = self.env.map_action_to_policy_index(action_key)

            if action_index is not None and 0 <= action_index < len(policy_priors_np):
                prior = policy_priors_np[action_index]
                action_priors_dict[action_key] = prior
            else:
                # Assign 0 prior if action is illegal or unmappable
                action_priors_dict[action_key] = 0.0
                logger.warning(
                    f"Action {action_key} (index {action_index}) invalid or out of policy bounds ({len(policy_priors_np)}) during expand."
                )

        node.expand(action_priors_dict)
        if action_priors_dict and any(p > 0 for p in action_priors_dict.values()):
            assert (
                node.children
            ), f"Node expansion failed: children dict is empty despite non-empty priors. Priors: {action_priors_dict}"

    def _score_child(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate the PUCT score for a node."""
        # Use self.exploration_constant as c_puct
        if node.visit_count == 0:
            q_value_for_parent = 0
            # Assign high score to encourage exploration of unvisited nodes
            # The U term dominates here. Use max(1, parent_visits) for sqrt robustness.
            u_score = (
                self.exploration_constant
                * node.prior
                * math.sqrt(max(1, parent_visits))
                # No division by (1 + N) for unvisited nodes in standard PUCT
            )
        else:
            # Q value is from the perspective of the player *at the parent node*.
            # Parent wants to maximize its value. If it's parent's turn at child, Q = V(child).
            # If it's opponent's turn at child, Q = -V(child).
            # node.value is always from the perspective of the player *at that node*.
            # So, parent Q = -node.value (assuming zero-sum).
            q_value_for_parent = -node.value

            # PUCT formula U term
            u_score = (
                self.exploration_constant
                * node.prior
                * math.sqrt(parent_visits)  # parent_visits > 0 guaranteed here
                / (1 + node.visit_count)
            )
        return q_value_for_parent + u_score

    def find_action_for_child(self, child_node: MCTSNode) -> Optional[ActionType]:
        """Find the action that leads to a given child node from the root."""
        if child_node.parent != self.root:
            return None  # Or maybe search recursively? For now, just check direct children.
        for action, node in self.root.children.items():
            if node == child_node:
                return action
        return None  # Child not found under root

    def get_terminal_value(self, leaf_env_state: BaseEnvironment) -> float:
        """Calculates the value of a terminal leaf node."""
        player_at_leaf = leaf_env_state.get_current_player()
        winner = leaf_env_state.get_winning_player()
        if winner is None:  # todo backwards?
            value = 0.0
        elif winner == player_at_leaf:
            value = 1.0
        else:
            value = -1.0
        return value

    def get_network_request(
        self, previous_response: Optional[Tuple[np.ndarray, float]] = None
    ) -> Optional[StateType]:
        if previous_response:
            policy_np, value = previous_response
            assert self.current_leaf_node is not None
            assert self.current_leaf_env is not None
            self._expand(self.current_leaf_node, self.current_leaf_env, policy_np)
            if (
                self.current_leaf_node == self.root
                and self.training
                and self.config.dirichlet_epsilon > 0
                and self.sim_count == 1  # Noise applied after first expansion of root
            ):
                self._apply_dirichlet_noise()
            self._backpropagate(self.current_leaf_node, value)
            self.current_leaf_node = None
            self.current_leaf_env = None
            self.last_request_state = None

        while self.sim_count < self.num_simulations:
            if self._check_dynamic_termination():
                break

            leaf_node, leaf_env, _ = self._select_leaf(self.root, self.env)

            if leaf_env.is_game_over():
                value = self.get_terminal_value(leaf_env)
                self._backpropagate(leaf_node, value)
                self.sim_count += 1
                continue

            leaf_state_obs = leaf_env.get_observation()
            state_key = get_state_key(leaf_state_obs)

            cached_result = self.network_cache.get(state_key)
            if cached_result:
                policy_np, value = cached_result
                self._expand(leaf_node, leaf_env, policy_np)
                if (
                    leaf_node == self.root
                    and self.training
                    and self.config.dirichlet_epsilon > 0
                    and self.sim_count == 0
                ):
                    self._apply_dirichlet_noise()
                self._backpropagate(leaf_node, value)
                self.sim_count += 1
                if self._check_dynamic_termination():
                    break
            else:
                # Cache Miss: Return state dict to manager
                self.current_leaf_node = leaf_node
                self.current_leaf_env = leaf_env
                self.last_request_state = leaf_state_obs
                return leaf_state_obs

        self.current_leaf_node = None
        self.current_leaf_env = None
        self.last_request_state = None
        return None

    def get_result(self) -> Tuple[ActionType, Dict[ActionType, int]]:
        """
        :return: chosen_action, action_visits
        """
        assert self.root.children, "MCTS Error: Root node has no children after search."

        action_visits = {
            action: child.visit_count for action, child in self.root.children.items()
        }
        visit_counts = np.array(list(action_visits.values()))
        actions = list(self.root.children.keys())

        total_visits = np.sum(visit_counts)
        assert total_visits > 0, "MCTS Error: Total visits for root children is zero."

        if self.training:
            # Temperature sampling (temp=1.0 means proportional to visits)
            temp = 1.0
            visit_counts_temp = visit_counts ** (1.0 / temp)
            action_probs = visit_counts_temp / np.sum(visit_counts_temp)
            if abs(np.sum(action_probs) - 1.0) > 1e-6:
                logger.warning(
                    f"Action probabilities sum to {np.sum(action_probs)}, renormalizing."
                )
                action_probs /= np.sum(action_probs)  # Renormalize
            chosen_action_index = np.random.choice(len(actions), p=action_probs)
            chosen_action = actions[chosen_action_index]
        else:
            chosen_action_index = np.argmax(visit_counts)
            chosen_action = actions[chosen_action_index]

        return chosen_action, action_visits

    def _select_leaf(
        self, root_node: MCTSNode, env: BaseEnvironment
    ) -> Tuple[MCTSNode, BaseEnvironment, List[MCTSNode]]:
        node = root_node
        sim_env = env.copy()
        search_path = [node]
        max_loops = (
            env.height * env.width + 5
            if hasattr(env, "height") and hasattr(env, "width")
            else 100
        )
        loop_count = 0

        while (
            node.is_expanded() and not sim_env.is_game_over() and loop_count < max_loops
        ):
            loop_count += 1
            parent_visits = node.visit_count
            assert node.children, f"Node {node} is expanded but has no children!"
            child_scores = {
                act: self._score_child(child, parent_visits)
                for act, child in node.children.items()
            }  # todo I see no tree reuse here?
            best_action = max(child_scores, key=child_scores.get)
            sim_env.step(best_action)
            node = node.children[best_action]
            search_path.append(node)

        return node, sim_env, search_path

    def _apply_dirichlet_noise(self) -> None:
        """Apply Dirichlet noise to the root node's children's priors."""
        if not self.root.children:
            return

        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(self.root.children)
        )
        for i, (_, child) in enumerate(self.root.children.items()):
            child.prior = (
                1 - self.config.dirichlet_epsilon
            ) * child.prior + self.config.dirichlet_epsilon * noise[i]
