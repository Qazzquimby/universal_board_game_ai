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
import abc
import torch.nn as nn
from dataclasses import dataclass, field

from environments.base import ActionType, StateType, BaseEnvironment

import numpy as np

DEBUG = True


def assert_states_are_equal(state1: StateType, state2: StateType):
    """Compares two state dictionaries, handling NumPy arrays correctly."""
    if state1.keys() != state2.keys():
        assert False
    for key in state1:
        val1 = state1[key]
        val2 = state2[key]
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                assert False
        elif val1 != val2:
            # Use standard equality for non-ndarray types
            assert False


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


class MCTSNode:
    """Represents a node in the MCTS tree."""

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
        state_key: Optional[str] = None,
    ):
        self.parent = parent
        self.prior: float = prior
        self.state_key: Optional[str] = state_key
        self.state: Optional[StateType] = None

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

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the node."""
        state_info = "State:Present" if self.state is not None else "State:None"
        return (
            f"MCTSNode(visits={self.visit_count}, value={self.value:.3f}, "
            f"children={len(self.children)}, {state_info}, state_key='{self.state_key}')"
        )


@dataclass
class SelectionResult:
    """Holds the results of the MCTS selection phase."""

    path: List["MCTSNode"]
    leaf_node: "MCTSNode"
    leaf_env: BaseEnvironment  # Environment state corresponding to the leaf node


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


class UCB1Selection(SelectionStrategy):
    """Selects nodes using the UCB1 algorithm."""

    def __init__(self, exploration_constant: float):
        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")
        self.exploration_constant = exploration_constant

    def _score_child(self, child: MCTSNode, parent_visits: int) -> float:
        """Calculates the UCB1 score for a child node."""
        if child.visit_count == 0:
            # Encourage exploration of unvisited nodes
            return float("inf")
        if (
            parent_visits == 0
        ):  # Should ideally not happen if root is visited before selection
            parent_visits = 1  # Avoid log(0) or division by zero

        # UCB Score = -Q(child) + C * P(child) * sqrt(log(N(parent)) / N(child))
        # We use -Q(child) because child.value is from the perspective of the player
        # at the child state. We need the value from the parent's perspective.
        exploitation_term = -child.value
        exploration_term = (
            self.exploration_constant
            * child.prior
            * math.sqrt(math.log(parent_visits) / child.visit_count)
        )
        # logger.trace(f"    Scoring Child: Visits={child.visit_count}, Value={child.value:.3f}, Prior={child.prior:.3f} -> Score={exploitation_term + exploration_term:.3f} (Exploit={exploitation_term:.3f}, Explore={exploration_term:.3f})")
        # logger.trace(f"    Scoring Child: Visits={child.visit_count}, Value={child.value:.3f}, Prior={child.prior:.3f} -> Score={exploitation_term + exploration_term:.3f} (Exploit={exploitation_term:.3f}, Explore={exploration_term:.3f})")
        return exploitation_term + exploration_term

    def select(self, node: MCTSNode, env: BaseEnvironment) -> SelectionResult:
        """Select child node with highest UCB score until a leaf node is reached."""
        path = [node]
        current_node = node
        # IMPORTANT: Work on the passed environment directly.
        # The caller (MCTSOrchestrator) is responsible for passing a copy
        # if the original state needs to be preserved outside the selection phase.
        sim_env = env

        while current_node.is_expanded() and not sim_env.is_game_over():
            # --- Critical Assertion ---
            # Check if the legal actions in the simulation environment match the children of the current node.
            # This is where the divergence likely occurs if there's a state mismatch.
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
                    f"Sim Env State: {sim_env.get_observation()}"
                )
            # --- End Critical Assertion ---

            assert current_node.children  # Should be guaranteed by is_expanded() check

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
                # This assertion is now implicitly covered by the critical assertion at the start of the loop

            # Apply the step to the simulation environment
            _, _, done = sim_env.step(best_action)

            if DEBUG:
                assert (
                    done == sim_env.is_game_over()
                ), f"Environment 'done' flag ({done}) inconsistent with is_game_over() ({sim_env.is_game_over()}) after action {best_action}"

            current_node = best_child_node
            path.append(current_node)
            if done:  # Stop if the action ended the game
                break

        # current_node is now the leaf node for this simulation
        # sim_env is the state corresponding to the leaf node
        return SelectionResult(path=path, leaf_node=current_node, leaf_env=sim_env)


class UniformExpansion(ExpansionStrategy):
    """Expands a node by creating children for all legal actions with uniform priors."""

    def expand(self, node: MCTSNode, env: BaseEnvironment) -> None:
        # env here represents the state of the node to be expanded.
        # We should not modify it directly during expansion logic.
        if node.is_expanded() or env.is_game_over():
            return  # Cannot expand already expanded or terminal nodes

        # Ensure the node being expanded has its state and key set
        # This might be redundant if set previously, but ensures consistency
        current_state = env.get_observation()
        node.state = current_state  # Store state in the node being expanded
        if node.state_key is None:  # Set key if not already set (e.g., for root)
            node.state_key = get_state_key(current_state)
        elif DEBUG:
            # If key exists, verify it matches the current env state
            assert node.state_key == get_state_key(
                current_state
            ), f"State key mismatch for node being expanded. Node key: {node.state_key}, Env key: {get_state_key(current_state)}"

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            # This can happen if the game ends exactly at this node after the parent's action
            # logger.debug(f"Expansion attempted on node with no legal actions (likely terminal). State: {current_state}")
            return

        # Assertion: Double-check env state hasn't become terminal unexpectedly
        assert (
            not env.is_game_over()
        ), f"Expansion called on a node corresponding to a terminal state. State: {current_state}"

        num_legal_actions = len(legal_actions)
        uniform_prior = 1.0 / num_legal_actions if num_legal_actions > 0 else 0.0

        # Convert legal actions to hashable keys for comparison and storage
        legal_action_keys = {
            tuple(a) if isinstance(a, list) else a for a in legal_actions
        }

        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action

            if DEBUG:
                # Use the env (representing the node's state) for validation checks
                if hasattr(env, "_is_valid_action"):
                    assert env._is_valid_action(
                        action
                    ), f"Action {action} from get_legal_actions() is considered invalid by _is_valid_action() in state {current_state}"
                # Assertion: Check key conversion and presence in the initial set
                assert (
                    action_key in legal_action_keys
                ), f"Action key {action_key} (from action {action}) not found in the initially generated legal action keys {legal_action_keys}. State: {current_state}"
                # Assertion: Ensure we are not overwriting an existing child
                assert (
                    action_key not in node.children
                ), f"Attempting to expand action {action_key} which already exists as a child. State: {current_state}"

            # Simulate action to get the resulting state for the child node
            child_env = env.copy()
            child_state, _, _ = child_env.step(action)
            child_state_key = get_state_key(child_state)

            # Create child node, storing the state and key it represents
            child_node = MCTSNode(
                parent=node, prior=uniform_prior, state_key=child_state_key
            )
            child_node.state = child_state  # Store resulting state in child
            node.children[action_key] = child_node

            if DEBUG:
                assert (
                    action_key in node.children
                ), f"Failed to add child node for action {action_key}"
                assert (
                    node.children[action_key] is child_node
                ), f"Incorrect child node added for action {action_key}"

        # Assertion: Ensure the number of children created matches the number of legal actions reported.
        if DEBUG:
            assert (
                len(node.children) == num_legal_actions
            ), f"Number of children created ({len(node.children)}) does not match the number of legal actions ({num_legal_actions}). Legal actions: {legal_actions}, Children keys: {list(node.children.keys())}. State: {current_state}"


class RandomRolloutEvaluation(EvaluationStrategy):
    """Evaluates a node by performing a random rollout simulation."""

    def __init__(self, max_rollout_depth: int = 100, discount_factor: float = 1.0):
        self.max_rollout_depth = max_rollout_depth
        self.discount_factor = discount_factor  # Usually 1.0 for MCTS terminal rewards

    def evaluate(self, node: MCTSNode, env: BaseEnvironment) -> float:
        """Simulate game from the given environment state using random policy."""
        # env represents the state of the node to be evaluated.
        # We should not modify it directly, so work with a copy for the rollout.

        if env.is_game_over():
            # If the node's state itself is terminal, determine the outcome directly
            player_at_leaf = env.get_current_player()  # Player whose turn it *would* be
            winner = env.get_winning_player()
            if winner is None:
                return 0.0  # Draw
            # Value is from the perspective of the player whose turn it *would* be at the leaf
            return 1.0 if winner == player_at_leaf else -1.0

        player_at_rollout_start = env.get_current_player()
        # Start the rollout simulation from a copy of the evaluation env
        sim_env = env.copy()
        steps = 0

        while not sim_env.is_game_over() and steps < self.max_rollout_depth:
            legal_actions = sim_env.get_legal_actions()
            if not legal_actions:
                logger.warning("MCTS Rollout: Game not over, but no legal actions.")
                # Treat as draw or assign penalty? Draw seems safer.
                return 0.0
            action = random.choice(legal_actions)

            if DEBUG:
                # Convert action to tuple if it's a list for consistent checking
                action_key_rollout = (
                    tuple(action) if isinstance(action, list) else action
                )
                legal_action_keys_rollout = {
                    tuple(a) if isinstance(a, list) else a for a in legal_actions
                }
                assert (
                    action_key_rollout in legal_action_keys_rollout
                ), f"Rollout chose action {action_key_rollout} which is not in legal actions {legal_action_keys_rollout}"

            _, _, done = sim_env.step(action)
            steps += 1
            # Apply discount factor here if needed for non-terminal rewards

        if steps >= self.max_rollout_depth:
            logger.warning(
                f"MCTS Rollout: Reached max depth ({self.max_rollout_depth}). Treating as draw."
            )
            return 0.0  # Treat hitting max depth as a draw

        # Game finished within rollout depth
        winner = sim_env.get_winning_player()
        if winner is None:
            value = 0.0  # Draw
        # Value must be from the perspective of the player whose turn it was AT THE START of the rollout
        elif winner == player_at_rollout_start:
            value = 1.0  # Win for the player at the start
        else:
            value = -1.0  # Loss for the player at the start

        # logger.trace(f"  Rollout Result: StartPlayer={player_at_rollout_start}, Winner={winner}, Value={value}")
        return value


class StandardBackpropagation(BackpropagationStrategy):
    """Updates node statistics by backpropagating the evaluation value."""

    def backpropagate(
        self,
        path: List[MCTSNode],
        value: float,
        player_at_leaf: int,  # player_at_leaf not needed here
    ) -> None:
        """
        Backpropagate the evaluated value up the tree, updating node statistics.
        Assumes `value` is from the perspective of the player whose turn it was at the leaf node.
        """
        current_node: Optional[MCTSNode] = path[-1]  # Start from the leaf
        # Value needs to be flipped for the parent if it's the other player's turn.
        # The value passed in `value` is from the perspective of the player AT THE LEAF.
        value_for_node = value

        for i, node_in_path in enumerate(reversed(path)):
            visits_before = node_in_path.visit_count
            value_before = node_in_path.total_value
            node_in_path.visit_count += 1

            # Add the value from the perspective of the player whose turn it was *at this node*.
            # value_for_node starts as the value from the leaf's perspective.
            # For the leaf's parent, the value is flipped (-1). For the grandparent, flipped again (+1), etc.
            # The value added should be value_for_node * (-1)^depth_difference
            # Alternatively, just flip value_for_node on each step up.
            node_in_path.total_value += value_for_node

            if DEBUG:
                assert (
                    node_in_path.visit_count == visits_before + 1
                ), f"Visit count did not increment correctly during backprop (Node {len(path)-1-i}). Before: {visits_before}, After: {node_in_path.visit_count}"
                # Check total_value update (allow for floating point issues)
                assert math.isclose(
                    node_in_path.total_value, value_before + value_for_node
                ), f"Total value did not update correctly during backprop (Node {len(path)-1-i}). Before: {value_before}, Added: {value_for_node}, After: {node_in_path.total_value}"

            # Flip the value perspective for the next node up (the parent).
            value_for_node *= -1.0


@dataclass
class PolicyResult:
    """Holds the results of the MCTS policy calculation."""

    chosen_action: ActionType
    action_probabilities: Dict[ActionType, float] = field(default_factory=dict)
    action_visits: Dict[ActionType, int] = field(default_factory=dict)


class MCTSOrchestrator:
    """
    Orchestrates the MCTS process using pluggable strategies.
    Manages the search tree and coordinates the selection, expansion,
    evaluation, and backpropagation phases.
    """

    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
        num_simulations: int,
        temperature: float = 0.0,
    ):
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        self.selection_strategy = selection_strategy
        self.expansion_strategy = expansion_strategy
        self.evaluation_strategy = evaluation_strategy
        self.backpropagation_strategy = backpropagation_strategy

        self.num_simulations = num_simulations
        self.temperature = temperature

        self.root: MCTSNode = MCTSNode()
        self._tree_reuse_enabled = True  # Default to reusing tree

    def set_root(self, state=None):
        """Resets the root node, discarding the existing tree."""
        if state is None:
            self.root = MCTSNode()
        else:
            # todo get node from tree with matching state key and set that to new root
            raise NotImplementedError
            # self.root.state = None

    def search(self, env: BaseEnvironment) -> MCTSNode:
        """
        Run the MCTS search for a specified number of simulations.

        Args:
            env: The current environment instance.

        Returns:
            The root node of the search tree after simulations.
        """
        # The 'env' passed here is the canonical environment at the root state.
        # Its state *must* match the state represented by self.root.
        if DEBUG:
            # Verify the passed env matches the state before starting search
            # Also, ensure the root node's state and key are correctly initialized/updated.
            current_env_state = env.get_observation()
            current_env_key = get_state_key(current_env_state)

            # Update root's state and key if they are None (first search or after reset)
            if self.root.state is None:
                self.root.state = current_env_state
            if self.root.state_key is None:
                self.root.state_key = current_env_key

            # Critical Assertion: Check if the root node's key matches the environment's key
            assert self.root.state_key == current_env_key, (
                f"Initial env state key mismatch in search. Root key: '{self.root.state_key}', Env key: '{current_env_key}'. "
                f"This likely means tree reuse advanced the root incorrectly or the provided env/state is wrong."
            )
            # Optional Debug Assertion: Check if root's stored state matches env state
            # assert_states_are_equal(self.root.state, current_env_state), "Root node's stored state differs from initial env state."

        for sim_idx in range(self.num_simulations):
            # Start each simulation with a fresh copy of the environment
            # set to the state corresponding to the *current* root node.
            sim_env = env.copy()  # IMPORTANT: Copy the env for the simulation run

            # logger.trace(f"\n--- Simulation {sim_idx+1}/{self.num_simulations} ---")
            # logger.trace(f"Starting search from root: {self.root}")

            # 1. Selection: Find a leaf node using the selection strategy.
            #    The strategy modifies sim_env to match the leaf node's state.
            selection_result = self.selection_strategy.select(self.root, sim_env)
            path = selection_result.path
            leaf_node = selection_result.leaf_node
            leaf_env = selection_result.leaf_env  # Env state is now at the leaf
            # logger.trace(f"Selection finished. Path length: {len(path)}, Leaf node: {leaf_node}")

            # 2. Expansion & Evaluation
            player_at_leaf = leaf_env.get_current_player()
            value: float = 0.0  # Initialize value

            if leaf_env.is_game_over():
                winner = leaf_env.get_winning_player()
                if winner is None:
                    value = 0.0  # Draw
                else:
                    # Value is from the perspective of the player whose turn it *would* be at the leaf
                    value = 1.0 if winner == player_at_leaf else -1.0
                # logger.trace(f"Leaf node is terminal. Winner: {winner}, Value: {value}")
            else:
                # Expand if necessary (pass the leaf_env which represents the state)
                if not leaf_node.is_expanded():
                    self.expansion_strategy.expand(leaf_node, leaf_env)
                    # logger.trace(f"Expanded leaf node. Children count: {len(leaf_node.children)}")

                # Evaluate the leaf node (pass the leaf_env)
                value = self.evaluation_strategy.evaluate(leaf_node, leaf_env)
                value = float(value)  # Ensure float
                # logger.trace(f"Evaluated leaf node. Value: {value:.4f}")

            # 3. Backpropagation
            # The value should be from the perspective of the player whose turn it was at the leaf node.
            # logger.trace(f"Backpropagating value {value:.4f} up path of length {len(path)}")
            self.backpropagation_strategy.backpropagate(path, value, player_at_leaf)
            # logger.trace(f"Backpropagation finished. Root: {self.root}")

        # logger.trace(f"\n--- Search Finished ({self.num_simulations} simulations) ---")
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

        # Get all children and their visit counts
        action_visits = {
            action: child.visit_count for action, child in self.root.children.items()
        }

        # Filter out actions with zero visits, as they shouldn't be chosen or have probability mass
        valid_actions_visits = {a: v for a, v in action_visits.items() if v > 0}
        assert valid_actions_visits

        actions = list(valid_actions_visits.keys())
        visits = np.array(
            list(valid_actions_visits.values()), dtype=np.float64
        )  # Use float64 for precision

        probabilities = {}

        if self.temperature == 0:
            # Deterministic: choose the action with the highest visit count
            best_action_index = np.argmax(visits)
            chosen_action = actions[best_action_index]
            # Probabilities are 1 for the best action, 0 otherwise (among visited actions)
            for i, act in enumerate(actions):
                probabilities[act] = 1.0 if i == best_action_index else 0.0
        else:
            # Temperature sampling
            if self.temperature <= 0:
                logger.warning(
                    f"Temperature ({self.temperature}) must be positive for sampling. Using temp=1.0 instead."
                )
                temperature = 1.0

            # Calculate visit counts raised to the power of (1 / temperature)
            # Use log-sum-exp trick for numerical stability if needed, but direct calc is often fine
            log_visits = np.log(visits + 1e-10)  # Add epsilon to avoid log(0)
            log_probs = log_visits / self.temperature
            # Subtract max for stability before exponentiating
            log_probs -= np.max(log_probs)
            exp_probs = np.exp(log_probs)
            probs_values = exp_probs / np.sum(exp_probs)

            # Ensure probabilities sum to 1 (handle potential floating point inaccuracies)
            probs_values /= np.sum(probs_values)
            if not np.isclose(np.sum(probs_values), 1.0):
                logger.warning(
                    f"Probabilities sum to {np.sum(probs_values)} after normalization. Re-normalizing."
                )
                probs_values /= np.sum(probs_values)  # Re-normalize if needed

            # Assign probabilities
            for act, prob in zip(actions, probs_values):
                probabilities[act] = prob

            # Choose action based on calculated probabilities
            try:
                chosen_action_index = np.random.choice(len(actions), p=probs_values)
                chosen_action = actions[chosen_action_index]
            except ValueError as e:
                logger.error(
                    f"Error choosing action with probabilities {probs_values} (sum={np.sum(probs_values)}): {e}. Falling back to max visits."
                )
                # Fallback to deterministic choice if sampling fails
                best_action_index = np.argmax(visits)
                chosen_action = actions[best_action_index]

        # Add zero probability for any actions that had zero visits initially
        for zero_visit_action in action_visits:
            if zero_visit_action not in probabilities:
                probabilities[zero_visit_action] = 0.0

        assert chosen_action is not None
        return PolicyResult(
            chosen_action=chosen_action,
            action_probabilities=probabilities,
            action_visits=action_visits,
        )
