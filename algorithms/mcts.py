import math
import random
import time
import contextlib
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
    Generator,
    Union,
    Literal,
)

import torch
from loguru import logger
import numpy as np
import torch.nn as nn

from core.config import MuZeroConfig, AlphaZeroConfig
from environments.base import ActionType, StateType, BaseEnvironment
from models.networks import AlphaZeroNet, MuZeroNet

PredictResult = Tuple[np.ndarray, float]


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


# Helper function to get state key (moved outside class)
def get_state_key(s: StateType) -> str:
    """Creates a hashable key from a state dictionary."""
    try:
        items = []
        # Sort items for consistency
        for k, v in sorted(s.items()):
            if isinstance(v, np.ndarray):
                items.append((k, v.tobytes()))
            elif isinstance(v, list):
                items.append((k, tuple(v)))  # Convert lists to tuples
            else:
                items.append((k, v))
        return str(tuple(items))
    except TypeError as e:
        logger.warning(
            f"State not easily hashable: {s}. Error: {e}. Using simple str()."
        )
        return str(s)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self._start_time = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000


class MCTSProfiler:
    """Collects timing data for MCTS searches."""

    def __init__(self):
        self.search_times_ms: List[float] = []
        self.network_times_ms: List[float] = []

    def record_search_time(self, duration_ms: float):
        self.search_times_ms.append(duration_ms)

    def record_network_time(self, duration_ms: float):
        self.network_times_ms.append(duration_ms)

    def get_average_search_time(self) -> float:
        return np.mean(self.search_times_ms) if self.search_times_ms else 0.0

    def get_average_network_time(self) -> float:
        # Note: Network time average is per network call, not per search
        return np.mean(self.network_times_ms) if self.network_times_ms else 0.0

    def get_total_search_time(self) -> float:
        return np.sum(self.search_times_ms)

    def get_total_network_time(self) -> float:
        return np.sum(self.network_times_ms)

    def get_num_searches(self) -> int:
        return len(self.search_times_ms)

    def get_num_network_calls(self) -> int:
        return len(self.network_times_ms)

    def reset(self):
        self.search_times_ms.clear()
        self.network_times_ms.clear()

    def report(self) -> str:
        num_searches = self.get_num_searches()
        num_net_calls = self.get_num_network_calls()
        if num_searches == 0:
            return "MCTS Profiler: No searches recorded."

        avg_search = self.get_average_search_time()
        avg_network = self.get_average_network_time()
        total_search = self.get_total_search_time()
        total_network = self.get_total_network_time()
        avg_net_calls_per_search = (
            num_net_calls / num_searches if num_searches > 0 else 0
        )
        avg_net_time_per_search = (
            total_network / num_searches if num_searches > 0 else 0
        )

        report_str = (
            f"MCTS Profiler Report ({num_searches} searches):\n"
            f"  Avg Search Time: {avg_search:.2f} ms\n"
            f"  Avg Network Call Time: {avg_network:.2f} ms\n"
            f"  Avg Network Calls per Search: {avg_net_calls_per_search:.2f}\n"
            f"  Avg Network Time per Search: {avg_net_time_per_search:.2f} ms\n"
            f"  Total Search Time: {total_search:.2f} ms\n"
            f"  Total Network Time: {total_network:.2f} ms"
        )
        return report_str


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
            # Ensure action is hashable if it's not already (e.g., list)
            action_key = tuple(action) if isinstance(action, list) else action
            if action_key not in self.children:
                # Add logging right before creating the child
                logger.trace(
                    f"  Node.expand: Creating child for action {action_key} with prior {prior:.4f}"
                )
                self.children[action_key] = MCTSNode(parent=self, prior=prior)
                children_added += 1
            else:
                # This case should ideally not happen if expand is called only once
                logger.warning(
                    f"  Node.expand: Child for action {action_key} already exists."
                )
        # Add assertion after the loop
        if action_priors and children_added == 0:
            logger.error(
                f"  Node.expand: action_priors provided but no children were added! Priors: {action_priors}"
            )
        assert (
            not action_priors or self.children
        ), "Node expansion failed: children dict is empty despite non-empty priors."


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
            discount_factor  # Note: Not used in standard UCB1 backprop here
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
            if not child_scores:  # Should not happen if node.is_expanded() is true
                logger.warning("MCTS _select: Node expanded but no children found.")
                break  # Cannot select further

            best_action = max(child_scores, key=child_scores.get)
            logger.debug(
                f"  Select: ParentVisits={parent_visits}, Scores={ {a: f'{s:.3f}' for a, s in child_scores.items()} }, Chosen={best_action}"
            )
            action, node = best_action, node.children[best_action]

            sim_env.step(action)
        return node, sim_env

    def _score_child(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate UCB1 score."""
        if node.visit_count == 0:
            # Must explore unvisited nodes
            return float("inf")

        # Ensure parent_visits is at least 1 to avoid math.log(0)
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / node.visit_count
        )

        # node.value is the average stored value from the perspective of the player AT 'node' (the child).
        # The parent selecting wants to maximize its own expected outcome.
        # Parent's Q for action leading to 'node' = - (node's value)
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
        logger.debug(
            f"  Rollout: StartPlayer={player_at_rollout_start}, Winner={winner}, Value={value}"
        )
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
            logger.debug(
               f"  Backprop: Node={current_node}, ValueToAdd={value_for_current_node:.3f}, OldW={current_node.total_value:.3f}, OldN={current_node.visit_count - 1}, NewW={current_node.total_value + value_for_current_node:.3f}, NewN={current_node.visit_count}"
            )
            current_node.total_value += value_for_current_node

            # Flip the value perspective for the parent node.
            value_for_current_node *= -1
            current_node = current_node.parent

    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode:
        """Run MCTS search from the given state using UCB1 and random rollouts."""
        self.reset_root()

        logger.debug(f"--- MCTS Search Start: State={state} ---")

        for sim_num in range(self.num_simulations):
            logger.debug(f" Simulation {sim_num+1}/{self.num_simulations}")
            # Start from the root node and a copy of the environment set to the initial state
            sim_env = env.copy()
            sim_env.set_state(state)  # Ensure simulation starts from the correct state

            # 1. Selection: Find a leaf node using UCB1.
            leaf_node, leaf_env_state = self._select(self.root, sim_env)

            # 2. Evaluation: Get the value of the leaf node.
            value = 0.0  # Initialize value
            if not leaf_env_state.is_game_over():
                # Expand the node if it hasn't been expanded yet
                if not leaf_node.is_expanded():
                    self._expand(leaf_node, leaf_env_state)
                    # After expansion, perform a rollout from this new leaf node
                    value = self._rollout(leaf_env_state)
                else:
                    # If already expanded (e.g., visited before), rollout from here
                    value = self._rollout(leaf_env_state)
            else:
                # Game ended during selection. Determine the outcome.
                # Value must be from the perspective of the player whose turn it was AT THE LEAF node.
                player_at_leaf = (
                    leaf_env_state.get_current_player()
                )  # Whose turn it *would* be
                winner = leaf_env_state.get_winning_player()

                if winner is None:
                    value = 0.0
                elif winner == player_at_leaf:
                    value = 1.0
                else:
                    value = -1.0

                logger.debug(
                    f"  Terminal Found during Select: Winner={winner}, PlayerAtNode={player_at_leaf}, Value={value}"
                )
            # 3. Backpropagation: Update nodes along the path from the leaf to the root.
            self._backpropagate(leaf_node, value)

        # Base MCTS search does not return timing information
        return self.root


class AlphaZeroMCTS(MCTS):
    """MCTS algorithm adapted for AlphaZero (PUCT + Network Evaluation)."""

    def __init__(
        self,
        env: BaseEnvironment,
        config: AlphaZeroConfig,
        network: Optional[AlphaZeroNet] = None,
    ):
        super().__init__(
            exploration_constant=config.cpuct,
            num_simulations=config.num_simulations,
        )
        self.config = config
        self.network = network if network is not None else DummyAlphaZeroNet(env)

    def _expand(
        self,
        node: MCTSNode,
        env_state: BaseEnvironment,
        policy_priors_np: Optional[
            np.ndarray
        ] = None,  # not actually optional, just matching type
    ) -> None:
        """
        Expand the leaf node. Uses network policy priors if provided, otherwise uniform.

        Args:
            node: The node to expand.
            env_state: The environment state at the node (used for legal actions).
            policy_priors_np: Optional numpy array of policy probabilities from the network.
        """
        assert policy_priors_np is not None

        if node.is_expanded() or env_state.is_game_over():
            return

        legal_actions = env_state.get_legal_actions()
        if not legal_actions:
            return  # Cannot expand if no legal actions

        action_priors_dict = {}

        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action
            action_index = self.network.get_action_index(action_key)

            # --- Add Debug Logging ---
            policy_len = len(policy_priors_np) if policy_priors_np is not None else -1
            logger.trace(
                f"  Expand Check: Action={action_key}, Index={action_index}, PolicyLen={policy_len}"
            )
            # --- End Debug Logging ---

            if action_index is not None and 0 <= action_index < policy_len:
                try:
                    prior = policy_priors_np[action_index]
                    action_priors_dict[action_key] = prior
                except IndexError as e:
                    logger.error(
                        f"  Expand Error: IndexError accessing policy_priors_np[{action_index}] (len={policy_len}). Action={action_key}. Error: {e}",
                        exc_info=True,
                    )
                    action_priors_dict[action_key] = 0.0  # Assign 0 prior on error
                except Exception as e:
                    logger.error(
                        f"  Expand Error: Unexpected error accessing policy_priors_np[{action_index}]. Action={action_key}. Error: {e}",
                        exc_info=True,
                    )
                    action_priors_dict[action_key] = 0.0  # Assign 0 prior on error
            else:
                # Assign 0 prior if action is illegal according to network mapping or out of bounds
                action_priors_dict[action_key] = 0.0
                logger.warning(
                    f"Action {action_key} (index {action_index}) not found in policy vector during expand."
                )

            # Log sorted priors for debugging
            sorted_priors = sorted(
                action_priors_dict.items(), key=lambda item: item[1], reverse=True
            )
            state_obs = env_state.get_observation()  # Get obs for logging
            logger.debug(
                f"  Expand (Network): State={state_obs.get('board', state_obs.get('piles', 'N/A'))}, Player={state_obs['current_player']}"
            )
            logger.debug(f"  Expand (Network): Legal Actions={legal_actions}")
            logger.debug(
                f"  Expand (Network={type(self.network).__name__}): Applied Priors (Top 5): { {a: f'{p:.3f}' for a, p in sorted_priors[:5]} }"
            )

        # Call the base node expansion method
        node.expand(action_priors_dict)

        # --- Add Assertion Immediately After Expansion ---
        # If priors were provided and non-zero, children should have been created.
        if action_priors_dict and any(p > 0 for p in action_priors_dict.values()):
            assert (
                node.children
            ), f"Node expansion failed: children dict is empty despite non-empty priors. Priors: {action_priors_dict}"
            logger.trace(
                f"  Expand Check OK: node.children populated. Keys: {list(node.children.keys())}"
            )
        elif not action_priors_dict:
            logger.trace(
                "  Expand Check OK: No priors provided, node.children expected to be empty."
            )
        # --- End Assertion ---

    def _backpropagate(self, leaf_node: MCTSNode, value_from_leaf_perspective: float):
        """
        Backpropagate the evaluated value up the tree, updating node statistics.
        (Inherited from base MCTS, logic is suitable for AlphaZero value backprop).

        Args:
            leaf_node: The node where the simulation/evaluation ended.
            value_from_leaf_perspective: The outcome (+1, -1, 0 or network prediction)
                                         from the perspective of the player whose turn
                                         it was at the leaf_node.
        """
        # Warning log removed - this is expected when root is uncached initially.
        super()._backpropagate(leaf_node, value_from_leaf_perspective)

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

    # Add helper method to AlphaZeroMCTS or MCTS
    def find_action_for_child(self, child_node: MCTSNode) -> Optional[ActionType]:
        """Find the action that leads to a given child node from the root."""
        if child_node.parent != self.root:
            return None  # Or maybe search recursively? For now, just check direct children.
        for action, node in self.root.children.items():
            if node == child_node:
                return action
        return None  # Child not found under root

    def _evaluate_terminal_leaf(
        self, leaf_node: MCTSNode, leaf_env_state: BaseEnvironment
    ) -> float:
        """Calculates the value of a terminal leaf node."""
        player_at_leaf = leaf_env_state.get_current_player()
        winner = leaf_env_state.get_winning_player()
        if winner is None:
            value = 0.0
        elif winner == player_at_leaf:
            value = 1.0
        else:
            value = -1.0
        logger.debug(
            f"  Sim Eval: Terminal state found. Winner={winner}, PlayerAtNode={player_at_leaf}, Value={value}"
        )
        return value

    def prepare_simulations(
        self, env: BaseEnvironment, state: StateType, train: bool
    ) -> Tuple[
        Dict[str, StateType],  # requests: state_key -> state_obs
        Dict[
            str, List[MCTSNode]
        ],  # pending_sims: state_key -> list of leaf_nodes waiting
        str,  # Return root state key
    ]:
        """
        Runs the selection phase for all simulations, collecting network requests
        and identifying completed simulations (terminal/cache hits).

        Args:
            env: The *real* environment instance (at the state to search from).
            state: The current state dictionary (matches env's state).

        Returns:
            Tuple containing:
            - requests: Dict mapping state_key -> state_obs for nodes needing network eval.
            - pending_sims: Dict mapping state_key -> list of (leaf_node, leaf_env_state, path)
                            tuples waiting for that network result.
            - completed_sims: List of (value, policy_or_None, path) tuples for simulations
                              that finished due to terminal state or cache hit.
        """
        self.reset_root()
        initial_state_obs = state
        log_prefix = f"AZ MCTS Prepare (Net={type(self.network).__name__}, Sims={self.num_simulations})"
        logger.debug(f"--- {log_prefix} Start: State={initial_state_obs} ---")

        # --- State for this search turn ---
        requests: Dict[str, StateType] = {}
        # Simplified pending_sims: maps state_key to list of leaf nodes waiting for that key's result
        pending_sims: Dict[str, List[MCTSNode]] = {}
        # Cache for network results *within this search* (cleared each call)
        self.evaluation_cache: Dict[str, PredictResult] = {}

        # --- Initial Root Evaluation & Expansion ---
        root_state_key = get_state_key(initial_state_obs)
        root_policy_np: Optional[np.ndarray] = None
        root_value: Optional[float] = None
        is_root_terminal = env.is_game_over()

        # Check if root state is terminal
        if is_root_terminal:
            logger.warning(
                f"{log_prefix} Root state is already terminal. Search is trivial."
            )
            # No simulations needed, return empty results (completed_sims removed)
            return requests, pending_sims, root_state_key

        # Check cache for root state (only if not terminal)
        if root_state_key in self.evaluation_cache:
            root_policy_np, root_value = self.evaluation_cache[root_state_key]
            logger.debug(f"{log_prefix} Root cache hit. Value={root_value:.3f}")
        else:
            logger.debug(f"{log_prefix} Root cache miss. Adding to requests.")
            requests[root_state_key] = initial_state_obs

        # If root was cached, expand it immediately IF policy is available
        if root_policy_np is not None:
            logger.debug(f"{log_prefix} Expanding root node immediately (cache hit).")
            root_env_state = env.copy()
            self._expand(self.root, root_env_state, root_policy_np)
            if train and self.root.is_expanded() and self.config.dirichlet_epsilon > 0:
                self._apply_dirichlet_noise()  # Apply noise right after root expansion
        else:
            logger.debug(f"{log_prefix} Root expansion deferred until network results.")

        if self.num_simulations <= 0:
            logger.warning(f"{log_prefix} num_simulations <= 0. Skipping search.")
            # Return root_state_key even if no sims run
            return requests, pending_sims, root_state_key

        # --- Run Selection Phase for all Simulations ---
        for sim_num in range(self.num_simulations):
            sim_log_prefix = f"  Sim {sim_num+1}/{self.num_simulations}:"
            logger.debug(f"{sim_log_prefix} Starting Selection...")

            # 1. Selection - Start from the root, pass the main env instance
            # Ensure root is used as starting point, even if expanded
            current_root_node = self.root
            leaf_node, leaf_env_state, search_path = self._select_leaf(
                current_root_node, env
            )

            # --- Handle Selection Result ---
            if leaf_node is None or leaf_env_state is None:
                logger.error(f"{sim_log_prefix} Selection failed. Skipping simulation.")
                # Optionally add a dummy completed_sim entry to track failure?
                # completed_sims.append((0.0, None, search_path)) # Example: Treat failure as draw
                continue

            # 2. Process Selected Leaf ---
            # Leaf is Terminal
            if leaf_env_state.is_game_over():
                value = self._evaluate_terminal_leaf(leaf_node, leaf_env_state)
                # Backpropagate immediately for terminal nodes
                self._backpropagate(leaf_node, value)
                # Don't add to completed_sims, it's already processed.
                continue  # Move to next simulation

            # Leaf needs evaluation (Cache Check / Request)
            leaf_state_obs = leaf_env_state.get_observation()
            state_key = get_state_key(leaf_state_obs)

            if state_key in self.evaluation_cache:
                # Cache Hit: Expand (if needed) and Backpropagate immediately
                policy_np, value = self.evaluation_cache[state_key]
                logger.debug(
                    f"{sim_log_prefix} Cache Hit for StateKey={state_key}, Value={value:.3f}"
                )
                if not leaf_node.is_expanded():
                    self._expand(leaf_node, leaf_env_state, policy_np)
                self._backpropagate(leaf_node, value)
                continue
            else:
                # Cache Miss: Record request and store pending simulation info
                logger.debug(
                    f"{sim_log_prefix} Cache Miss. Recording Request for StateKey={state_key}"
                )
                if state_key not in requests:
                    requests[state_key] = leaf_state_obs
                if state_key not in pending_sims:
                    pending_sims[state_key] = []
                # Store only the leaf node that needs the result
                pending_sims[state_key].append(leaf_node)

        num_pending = sum(len(v) for v in pending_sims.values())
        logger.debug(
            f"{log_prefix} Prepare complete. Requests: {len(requests)}, PendingSims: {num_pending}"
        )
        # Return root state key along with requests and simplified pending_sims
    # Removed prepare_simulations method

    def search(
        self, env: BaseEnvironment, state: StateType, train: bool = False
    ) -> Tuple[Optional[ActionType], Optional[np.ndarray]]:
        """
        Performs a synchronous MCTS search for evaluation or simple usage.
        Handles network calls immediately (no batching across games).

        Args:
            env: The environment instance at the current state.
            state: The current state dictionary.
            train: Whether to apply noise/temperature (typically False for evaluation).

        Returns:
            Tuple containing the chosen action and the calculated policy target.
        """
        self.reset_root()
        initial_state_obs = state
        log_prefix = f"AZ MCTS Sync Search (Net={type(self.network).__name__}, Sims={self.num_simulations})"
        logger.debug(f"--- {log_prefix} Start: State={initial_state_obs} ---")

        if env.is_game_over():
            logger.warning(f"{log_prefix} Root state is already terminal. No search.")
            return None, None

        # --- Run Simulations Synchronously ---
        for sim_num in range(self.num_simulations):
            sim_log_prefix = f"  Sync Sim {sim_num+1}/{self.num_simulations}:"
            logger.debug(f"{sim_log_prefix} Starting Selection...")

            # 1. Selection - Start from the root
            leaf_node, leaf_env_state, search_path = self._select_leaf(self.root, env)

            if leaf_node is None or leaf_env_state is None:
                logger.error(f"{sim_log_prefix} Selection failed. Skipping simulation.")
                continue

            # 2. Evaluate Leaf
            value = 0.0
            policy_np = None
            if leaf_env_state.is_game_over():
                value = self._evaluate_terminal_leaf(leaf_node, leaf_env_state)
            else:
                # Get network prediction immediately
                leaf_state_obs = leaf_env_state.get_observation()
                try:
                    policy_np, value = self.network.predict(leaf_state_obs)
                    # Expand the node using the prediction
                    if not leaf_node.is_expanded():
                        logger.debug(f"{sim_log_prefix} Expanding node {leaf_node}")
                        self._expand(leaf_node, leaf_env_state, policy_np)
                        # Apply noise right after root expansion on first sim if training
                        if train and sim_num == 0 and leaf_node == self.root and self.config.dirichlet_epsilon > 0:
                             self._apply_dirichlet_noise()

                except Exception as e:
                    logger.error(f"{sim_log_prefix} Network prediction/expansion failed: {e}")
                    # Assign default value (e.g., 0) and skip backprop? Or stop?
                    value = 0.0 # Assign neutral value on error

            # 3. Backpropagation
            self._backpropagate(leaf_node, value)

        # --- Select Action after all simulations ---
        if not self.root.children:
            logger.error(f"{log_prefix} Root node has no children after search.")
            return None, None

        visit_counts = np.array(
            [child.visit_count for child in self.root.children.values()]
        )
        actions = list(self.root.children.keys())

        if np.sum(visit_counts) == 0:
             logger.warning(f"{log_prefix} All child visit counts are zero. Choosing random.")
             chosen_action = random.choice(actions) if actions else None
        elif train: # Use temperature during training search if needed
             temperature = 1.0 # Simplified: Use temp=1 for exploration if train=True
             visit_counts_temp = visit_counts**(1.0 / temperature)
             action_probs = visit_counts_temp / np.sum(visit_counts_temp)
             chosen_action_index = np.random.choice(len(actions), p=action_probs)
             chosen_action = actions[chosen_action_index]
        else: # Greedy selection for evaluation
             chosen_action_index = np.argmax(visit_counts)
             chosen_action = actions[chosen_action_index]

        policy_target = self._calculate_policy_target(
            self.root, actions, visit_counts, env
        )

        logger.debug(f"{log_prefix} Chosen Action: {chosen_action}")
    # Removed process_results_and_select_action method

    # The entire incorrect _apply_dirichlet_noise definition below is removed.
    # The correct definition follows later in the file.
        log_prefix = "AZ MCTS Noise:"
        num_children = len(self.root.children)
        if num_children == 0:
            logger.warning(f"{log_prefix} Cannot apply noise, root has no children.")
            return

        noise = np.random.dirichlet([self.config.dirichlet_alpha] * num_children)
        noisy_priors = []
        child_items = list(self.root.children.items())  # Get fixed list

        for i, (action, child) in enumerate(child_items):
            original_prior = child.prior
            child.prior = (
                1 - self.config.dirichlet_epsilon
            ) * original_prior + self.config.dirichlet_epsilon * noise[i]
            noisy_priors.append((action, child.prior))

        sorted_noisy_priors = sorted(
            noisy_priors, key=lambda item: item[1], reverse=True
        )
        logger.debug(
            f"{log_prefix} Applied Dirichlet noise (alpha={self.config.dirichlet_alpha}, eps={self.config.dirichlet_epsilon})"
        )
        logger.debug(
            f"  Noisy Root Priors (Top 5): { {a: f'{p:.3f}' for a, p in sorted_noisy_priors[:5]} }"
        )

    # Modify signature to accept env directly
    def _select_leaf(
        self, root_node: MCTSNode, env: BaseEnvironment
    ) -> Tuple[Optional[MCTSNode], Optional[BaseEnvironment], List[MCTSNode]]:
        """
        Selects a leaf node starting from the root using PUCT scores.

        Args:
            root_node: The starting node for selection.
            env: The base environment instance (used for copying).
            initial_state: The state corresponding to the root node.

        Returns:
            A tuple containing:
            - The selected leaf node (or None if an error occurs).
            - The environment state at the leaf node (or None if error).
            - The list of nodes in the search path.
        """
        node = root_node
        # Start simulation from the env's current state (assumed to be initial_state)
        sim_env = env.copy()
        search_path = [node]
        sim_log_prefix = "  Sim Select:"  # Simplified prefix for helper

        selection_active = True
        loop_count = 0  # Add loop counter for debugging deep selections
        # Calculate safety break based on env dimensions passed in
        max_loops = (
            env.height * env.width + 5
            if hasattr(env, "height") and hasattr(env, "width")
            else 100
        )  # Fallback

        while (
            selection_active
            and node.is_expanded()
            and not sim_env.is_game_over()
            and loop_count < max_loops
        ):
            loop_count += 1
            parent_visits = node.visit_count
            logger.trace(
                f"{sim_log_prefix} Loop {loop_count}: Node={node}, ParentVisits={parent_visits}"
            )

            # --- Add Assertion ---
            # If node.is_expanded() is true (checked in while loop), node.children should exist and be non-empty.
            assert (
                node.children
            ), f"MCTS _select_leaf: Node {node} is expanded but has no children!"
            # --- End Assertion ---

            # Calculate scores for all children determined during expansion
            child_scores = {
                act: self._score_child(child, parent_visits)
                for act, child in node.children.items()
            }
            # --- Add detailed score logging ---
            log_scores = {str(a): f"{s:.3f}" for a, s in child_scores.items()}
            logger.trace(
                f"{sim_log_prefix} Loop {loop_count}: ChildScores={log_scores}"
            )
            # --- End detailed score logging ---

            best_action = max(child_scores, key=child_scores.get)
            logger.debug(
                f"{sim_log_prefix} Loop {loop_count}: ChosenAction={best_action} (Score={child_scores[best_action]:.3f})"
            )

            try:
                logger.debug(f"{sim_log_prefix} Stepping env with {best_action}")
                sim_env.step(best_action)
                logger.trace(
                    f"{sim_log_prefix} Loop {loop_count}: Moving to child node for action {best_action}"
                )
                selected_child_node = node.children[best_action]
                node = selected_child_node  # Update node for next loop iteration
                search_path.append(node)
                logger.trace(
                    f"{sim_log_prefix} Loop {loop_count}: New Node={node}, PathLen={len(search_path)}"
                )
            except (ValueError, KeyError) as e:
                logger.error(
                    f"{sim_log_prefix} Error stepping env or finding child for action {best_action}. Error: {e}"
                )
                return None, None, search_path

        logger.debug(
            f"{sim_log_prefix} Selection finished. Leaf node: {node}, PathLen: {len(search_path)}"
        )

        return node, sim_env, search_path

    def _calculate_policy_target(
        self, root_node, actions, visit_counts, env: BaseEnvironment
    ) -> Optional[np.ndarray]:
        """Calculates the policy target vector based on MCTS visit counts."""
        # Requires env access if network isn't available or doesn't have size info
        if self.network:
            policy_size = self.network._calculate_policy_size(env)
        else:
            try:
                policy_size = env.policy_vector_size
            except AttributeError:
                logger.error(
                    "Cannot determine policy size without network or env.policy_vector_size"
                )
                return None

        policy_target = np.zeros(policy_size, dtype=np.float32)
        total_visits = np.sum(visit_counts)

        if total_visits > 0:
            if self.network:
                for i, action in enumerate(actions):
                    action_key = tuple(action) if isinstance(action, list) else action
                    action_idx = self.network.get_action_index(action_key)
                    if action_idx is not None and 0 <= action_idx < policy_size:
                        policy_target[action_idx] = visit_counts[i] / total_visits
                    else:
                        logger.warning(
                            f"Action {action_key} could not be mapped to index during policy target calculation."
                        )
            else:  # Fallback logic (less ideal)
                if hasattr(env, "map_action_to_policy_index"):
                    for i, action in enumerate(actions):
                        action_key = (
                            tuple(action) if isinstance(action, list) else action
                        )
                        action_idx = env.map_action_to_policy_index(action_key)
                        if action_idx is not None and 0 <= action_idx < policy_size:
                            policy_target[action_idx] = visit_counts[i] / total_visits
                        else:
                            logger.warning(
                                f"Action {action_key} could not be mapped via env during policy target calculation."
                            )
                else:
                    logger.error(
                        "Cannot calculate policy target without network or env mapping."
                    )
                    return None  # Indicate failure

            # Normalize policy target
            current_sum = policy_target.sum()
            if current_sum > 1e-6:
                policy_target /= current_sum
            elif policy_target.size > 0:
                logger.warning(
                    f"Policy target sum is near zero ({current_sum}). Setting uniform distribution."
                )
                policy_target.fill(1.0 / policy_target.size)

            return policy_target
        else:
            logger.warning(
                "No visits recorded in MCTS root. Cannot calculate policy target."
            )
            return None


# Removed _calculate_policy_target from MCTS, agent should handle it.

# --- MuZero MCTS ---


class MuZeroMCTSNode:
    """Node specific to MuZero MCTS, storing hidden state."""

    def __init__(self, parent: Optional["MuZeroMCTSNode"] = None, prior: float = 0.0):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0  # W(s,a)
        self.reward = 0.0  # R(s,a) - Reward obtained reaching this node *from parent*
        self.hidden_state: Optional[
            torch.Tensor
        ] = None  # s - Hidden state represented by this node
        self.children: Dict[ActionType, "MuZeroMCTSNode"] = {}

    @property
    def value(self) -> float:  # Q(s,a) - Mean action value
        # MuZero Q value definition: Q(s, a) = R(s, a) + gamma * V(s')
        # The stored total_value W should reflect sum(G) where G = R + gamma*V
        # So, value = W / N should approximate Q.
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)

    def expand(
        self,
        policy_logits: torch.Tensor,
        network: MuZeroNet,
        legal_actions: Optional[List[ActionType]] = None,
    ):
        """
        Expand node using policy predictions from the network.
        Creates child nodes based on the predicted priors.
        If legal_actions is provided (only for root), filters expansion to legal actions.
        """
        policy_probs = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()

        if not hasattr(network, "get_action_from_index"):
            logger.error(
                "MuZeroNet needs get_action_from_index method for MCTS expansion."
            )
            return

        # Convert legal_actions list to set for efficient lookup if provided
        legal_action_set = (
            set(tuple(a) if isinstance(a, list) else a for a in legal_actions)
            if legal_actions is not None
            else None
        )

        for action_index, prior in enumerate(policy_probs):
            if prior > 1e-6:  # Small threshold to avoid tiny priors
                action = network.get_action_from_index(action_index)
                if action is not None:
                    action_key = tuple(action) if isinstance(action, list) else action

                    # If expanding root node, only add children for legal actions
                    if (
                        legal_action_set is not None
                        and action_key not in legal_action_set
                    ):
                        continue  # Skip illegal action at the root

                    if action_key not in self.children:
                        assert isinstance(
                            prior, (float, np.floating)
                        ), f"Prior type error: Expected float, got {type(prior)}"
                        self.children[action_key] = MuZeroMCTSNode(
                            parent=self, prior=prior
                        )
                else:
                    logger.warning(
                        f"Could not map action index {action_index} back to action."
                    )
                    pass


class MuZeroMCTS:
    """MCTS implementation adapted for MuZero's learned model."""

    def __init__(
        self,
        config: MuZeroConfig,
        network: MuZeroNet,
        profiler: Optional[MCTSProfiler] = None,
    ):
        self.config = config
        self.network = network
        self.profiler = profiler
        # Root node doesn't have a prior action, hidden state comes from representation(obs)
        self.root = MuZeroMCTSNode(prior=0.0)

    def reset_root(self):
        self.root = MuZeroMCTSNode(prior=0.0)

    def _puct_score(self, node: MuZeroMCTSNode, parent_visits: int) -> float:
        """
        Calculate the PUCT score for a node in MuZero MCTS.
        Using AlphaZero PUCT for now: Q(s,a) + U(s,a)
        Q(s,a) is from the parent's perspective (-node.value).
        """
        # Calculate U-value (exploration bonus)
        if node.visit_count == 0:
            q_value_for_parent = 0
            # Use max(1, parent_visits) for sqrt robustness
            u_value = (
                self.config.cpuct
                * node.prior
                * math.sqrt(max(1, parent_visits))
                # No division by (1 + node.visit_count)
            )
        else:
            # Q value from parent perspective
            q_value_for_parent = -node.value  # Assumes zero-sum game value definition
            u_value = (
                self.config.cpuct
                * node.prior
                * math.sqrt(parent_visits)  # parent_visits > 0 guaranteed
                / (1 + node.visit_count)
            )

        # MuZero paper uses: Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a)) * (c1 + log((N(s) + c2 + 1)/c2))
        # Using simpler AlphaZero PUCT for now:
        return q_value_for_parent + u_value

    def _select_child(
        self, node: MuZeroMCTSNode
    ) -> Tuple[Optional[ActionType], Optional[MuZeroMCTSNode]]:
        """
        Selects the child node with the highest PUCT score.
        Returns (None, None) if the node has no children.
        """
        parent_visits = node.visit_count
        if not node.children:
            # This should not happen if called after expansion check
            logger.error("MuZero _select_child called on node with no children.")
            # How to handle this? Maybe return the node itself? Or raise error?
            # For now, let's return None to signal failure
            return None, None

        best_item = max(
            node.children.items(),
            key=lambda item: self._puct_score(item[1], parent_visits),
        )
        return best_item  # Returns (action, child_node)

    def _backpropagate(self, node: MuZeroMCTSNode, value: float):
        """
        Backpropagate value using AlphaZero logic for now (flipping value).
        Assumes zero-sum game. MuZero's reward/discount needs careful handling.
        """
        current_node = node
        # Value is the predicted value V(s) from the leaf node perspective.
        value_for_current_node = value

        while current_node is not None:
            current_node.visit_count += 1
            logger.debug(
                f"  Backprop (MuZero): Node={current_node}, ValueToAdd={value_for_current_node:.3f}, OldW={current_node.total_value:.3f}, OldN={current_node.visit_count - 1}, NewW={current_node.total_value + value_for_current_node:.3f}, NewN={current_node.visit_count}"
            )
            # Standard MCTS backprop: Update W with value relative to player at node
            current_node.total_value += value_for_current_node

            # Flip perspective for parent in zero-sum game
            value_for_current_node = -value_for_current_node

            # Incorporate reward? MuZero: G = R + gamma*V. W should sum G.
            # If W sums G, then Q = W/N approximates G_avg.
            # Backprop needs to pass G up.
            # value_for_parent = current_node.reward + self.config.discount * value_for_current_node
            # Let's stick to simple AZ backprop for now.

            current_node = current_node.parent

    def search(
        self, observation_dict: dict, legal_actions: List[ActionType]
    ) -> MuZeroMCTSNode:
        """
        Run MuZero MCTS search starting from a root observation.

        Args:
            observation_dict: The initial observation from the *real* environment.
            legal_actions: The list of legal actions from the *real* environment state.
                           MuZero needs this for the root node expansion.
        """
        self.reset_root()

        search_timer = Timer() if self.profiler else contextlib.nullcontext()

        with search_timer:
            logger.debug(
                f"--- MuZero MCTS Search Start: Obs={observation_dict}, Legal={legal_actions} ---"
            )

            # Initial step: Get hidden state and initial prediction from observation
            value, reward, policy_logits, hidden_state = (
                None,
                None,
                None,
                None,
            )  # Initialize
            if self.profiler:
                with Timer() as net_timer:
                    (
                        value,  # V(s_0)
                        reward,  # r_0 (usually 0)
                        policy_logits,  # p(a|s_0)
                        hidden_state,  # h(o_1) -> s_0
                    ) = self.network.initial_inference(observation_dict)
                self.profiler.record_network_time(net_timer.elapsed_ms)
            else:
                (
                    value,
                    reward,
                    policy_logits,
                    hidden_state,
                ) = self.network.initial_inference(observation_dict)

            if hidden_state is None or policy_logits is None or value is None:
                logger.error(
                    "MuZero initial_inference failed to return expected values."
                )
                if self.profiler and isinstance(search_timer, Timer):
                    self.profiler.record_search_time(search_timer.elapsed_ms)
                return self.root

            self.root.hidden_state = hidden_state
            self.root.reward = reward.item() if reward is not None else 0.0

            if not legal_actions:
                logger.warning(
                    "No legal actions provided for MuZero MCTS root. Search cannot proceed."
                )
                if self.profiler and isinstance(search_timer, Timer):
                    # Record search time even if no legal actions
                    self.profiler.record_search_time(search_timer.elapsed_ms)
                return self.root

            # Expand root using policy and filtering by legal_actions
            self.root.expand(policy_logits, self.network, legal_actions)

            # TODO: Add Dirichlet noise to root priors here if training

            if not self.root.children:  # Check if root actually expanded
                logger.warning(
                    "MuZero MCTS root has no children after expansion (no legal actions had non-zero prior?)."
                )
                if self.profiler and isinstance(search_timer, Timer):
                    # Record search time even if no children
                    self.profiler.record_search_time(search_timer.elapsed_ms)
                return self.root

            # Run simulations
            for sim_num in range(self.config.num_simulations):
                logger.debug(
                    f" MuZero Simulation {sim_num+1}/{self.config.num_simulations}"
                )
                node = self.root
                search_path = [node]  # Keep track of nodes visited

                # 1. Selection: Traverse tree using PUCT until a leaf node is reached
                while node.is_expanded():
                    action, next_node = self._select_child(node)
                    if next_node is None:
                        logger.error("MuZero selection failed.")
                        break
                    logger.debug(
                        f"  MuZero Select: Action={action}, ChildNode={next_node}"
                    )
                    node = next_node
                    search_path.append(node)

                if node is None:
                    continue

                # 2. Expansion & Evaluation: Expand leaf node, get prediction from network
                parent = search_path[-2] if len(search_path) > 1 else self.root
                leaf_node = search_path[-1]

                action_taken = None
                for act, child in parent.children.items():
                    if child == leaf_node:
                        action_taken = act
                        break

                if action_taken is None and leaf_node is not self.root:
                    logger.error(
                        "Could not find action leading to leaf node during MuZero search."
                    )
                    continue

                leaf_value = 0.0
                if leaf_node is not self.root:
                    value_rec, reward_rec, policy_logits_rec, next_hidden_state_rec = (
                        None,
                        None,
                        None,
                        None,
                    )
                    if self.profiler:
                        with Timer() as net_timer:
                            (
                                value_rec,
                                reward_rec,
                                policy_logits_rec,
                                next_hidden_state_rec,
                            ) = self.network.recurrent_inference(
                                parent.hidden_state, action_taken
                            )
                        self.profiler.record_network_time(net_timer.elapsed_ms)
                    else:
                        (
                            value_rec,
                            reward_rec,
                            policy_logits_rec,
                            next_hidden_state_rec,
                        ) = self.network.recurrent_inference(
                            parent.hidden_state, action_taken
                        )

                    if (
                        next_hidden_state_rec is None
                        or policy_logits_rec is None
                        or value_rec is None
                    ):
                        logger.error(
                            "MuZero recurrent_inference failed to return expected values."
                        )
                        continue

                    leaf_node.hidden_state = next_hidden_state_rec
                    leaf_node.reward = (
                        reward_rec.item() if reward_rec is not None else 0.0
                    )
                    leaf_node.expand(policy_logits_rec, self.network)
                    leaf_value = value_rec.item()
                    logger.debug(
                        f"  MuZero Evaluate (Recurrent): Action={action_taken}, Reward={leaf_node.reward:.3f}, Value={leaf_value:.3f}"
                    )
                else:
                    leaf_value = value.item()
                    logger.debug(f"  MuZero Evaluate (Initial): Value={leaf_value:.3f}")

                # --- Backpropagation ---
                self._backpropagate(leaf_node, leaf_value)

        if self.profiler and isinstance(search_timer, Timer):
            self.profiler.record_search_time(search_timer.elapsed_ms)

        return self.root
