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

        # --- Add Check Immediately After Expansion ---
        if action_priors_dict and not node.children:
            logger.error(
                f"  Expand Check FAIL: node.expand finished but node.children is empty! Node: {node}"
            )
        elif node.children:
            logger.trace(
                f"  Expand Check OK: node.children populated. Keys: {list(node.children.keys())}"
            )
        # --- End Check ---

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

    def get_action_and_policy(
        self,
        env: BaseEnvironment,
        state: StateType,
        train: bool,  # Needed for temperature and noise
        current_step: int,  # Needed for temperature decay
    ) -> Tuple[Optional[ActionType], Optional[np.ndarray], List[Tuple[str, StateType]]]:
        """
        Performs the two-phase MCTS search:
        1. Runs simulations, collecting network prediction requests.
        2. (Implicitly expects caller to batch predict)
        3. Uses provided predictions to complete the search and select an action.

        Args:
            env: The *real* environment instance.
            state: The current state dictionary from the real environment.
            train: Boolean indicating if this is a training step (for noise/temp).
            current_step: The current step number in the episode (for temp decay).

        Returns:
            A tuple containing:
            - chosen_action: The selected action (or None if search fails).
            - policy_target: The calculated policy target (or None if search fails).
            - predict_requests: List of (state_key, state_obs) tuples needing prediction.
        """
        self.reset_root()
        initial_state_obs = state
        log_prefix = f"AZ MCTS Sync (Net={type(self.network).__name__}, Sims={self.num_simulations})"
        logger.debug(f"--- {log_prefix} Start: State={initial_state_obs} ---")

        # --- Phase 1: Run Simulations and Collect Network Requests ---
        pending_predict_requests: Dict[str, StateType] = {}
        # Cache for results *within this search* (cleared each call)
        evaluation_cache: Dict[str, PredictResult] = {}
        # Store the path taken for each simulation for backpropagation later
        simulation_paths: List[List[MCTSNode]] = []
        # Store the final value obtained for each simulation path
        simulation_values: List[Optional[float]] = []
        # Store leaf nodes, their env state, and path for evaluation after batch prediction
        nodes_awaiting_eval: Dict[
            str, List[Tuple[MCTSNode, BaseEnvironment, List[MCTSNode]]]
        ] = {}

        if self.num_simulations <= 0:
            logger.warning(f"{log_prefix} num_simulations <= 0. Skipping search.")
            return None, None, []

        for sim_num in range(self.num_simulations):
            sim_log_prefix = f"  Sim {sim_num+1}/{self.num_simulations}:"
            logger.debug(f"{sim_log_prefix} Starting Selection...")

            # 1. Selection
            leaf_node, leaf_env_state, search_path = self._select_leaf(
                self.root, env, initial_state_obs
            )

            if leaf_node is None or leaf_env_state is None:
                logger.error(f"{sim_log_prefix} Selection failed. Skipping simulation.")
                simulation_paths.append(search_path)  # Store path even if failed
                simulation_values.append(None)  # Mark as failed
                continue

            # 2. Check if Leaf is Terminal
            if leaf_env_state.is_game_over():
                value = self._evaluate_terminal_leaf(leaf_node, leaf_env_state)
                simulation_paths.append(search_path)
                simulation_values.append(value)
                # Backpropagate immediately for terminal nodes
                self._backpropagate(leaf_node, value)
                continue  # Move to next simulation

            # 3. Leaf Needs Evaluation (Cache Check / Request Collection)
            leaf_state_obs = leaf_env_state.get_observation()
            state_key = get_state_key(leaf_state_obs)

            if state_key in evaluation_cache:
                # Cache Hit: Use cached value for this simulation path
                _, value = evaluation_cache[state_key]
                logger.debug(
                    f"{sim_log_prefix} Cache Hit for StateKey={state_key}, Value={value:.3f}"
                )
                simulation_paths.append(search_path)
                simulation_values.append(value)
                # Expand node if needed (using cached policy) - crucial!
                if not leaf_node.is_expanded():
                    policy_np, _ = evaluation_cache[state_key]
                    self._expand(leaf_node, leaf_env_state, policy_np)
                # Backpropagate immediately using cached value
                self._backpropagate(leaf_node, value)

            else:
                # Cache Miss: Record request and store path/node for later processing
                logger.debug(
                    f"{sim_log_prefix} Cache Miss. Recording Request for StateKey={state_key}"
                )
                if state_key not in pending_predict_requests:
                    pending_predict_requests[state_key] = leaf_state_obs

                # Store the leaf node, its env state, and its path, grouped by state_key
                if state_key not in nodes_awaiting_eval:
                    nodes_awaiting_eval[state_key] = []
                # Store a copy of the leaf env state
                nodes_awaiting_eval[state_key].append(
                    (leaf_node, leaf_env_state.copy(), search_path)
                )
                # Mark this simulation path as incomplete for now
                simulation_paths.append(search_path)
                simulation_values.append(
                    None
                )  # Placeholder, will be filled after prediction

        # --- End Simulation Loop (Phase 1) ---
        logger.debug(
            f"{log_prefix} Phase 1 Complete. Unique prediction requests: {len(pending_predict_requests)}"
        )

        # Convert pending requests dict to list format expected by caller
        predict_requests_list = list(pending_predict_requests.items())

        # Return the root, requests, and state needed for Phase 2
        # We bundle the state needed for phase 2 into a dictionary
        phase2_state = {
            "nodes_awaiting_eval": nodes_awaiting_eval,
            "simulation_paths": simulation_paths,
            "simulation_values": simulation_values,
            "evaluation_cache": evaluation_cache,  # Pass cache for reuse if needed
            "initial_state_obs": initial_state_obs,  # For potential re-use/logging
            "train": train,  # Pass training flag
            "current_step": current_step,  # Pass step count
            # Add env for policy target calculation fallback if needed
            "env": env,
        }

        return self.root, predict_requests_list, phase2_state

    def complete_search_and_get_action(
        self,
        root_node: MCTSNode,
        network_results: Dict[str, PredictResult],
        phase2_state: Dict,
    ) -> Tuple[Optional[ActionType], Optional[np.ndarray]]:
        """
        Completes the MCTS search using the provided network results and selects an action.

        Args:
            root_node: The MCTS root node from phase 1.
            network_results: Dictionary mapping state_key to (policy_np, value).
            phase2_state: Dictionary containing state saved from phase 1.

        Returns:
            A tuple containing:
            - chosen_action: The selected action (or None if search fails).
            - policy_target: The calculated policy target (or None if search fails).
        """
        self.root = root_node  # Restore root
        nodes_awaiting_eval = phase2_state["nodes_awaiting_eval"]
        simulation_paths = phase2_state["simulation_paths"]
        simulation_values = phase2_state["simulation_values"]
        evaluation_cache = phase2_state["evaluation_cache"]
        train = phase2_state["train"]
        current_step = phase2_state["current_step"]
        env = phase2_state["env"]  # Retrieve env
        log_prefix = f"AZ MCTS Sync Phase 2"

        logger.debug(f"--- {log_prefix} Start ---")

        # --- Phase 2: Process Network Results and Complete Simulations ---
        num_failed_evals = 0
        for state_key, results in network_results.items():
            if state_key not in nodes_awaiting_eval:
                logger.warning(
                    f"{log_prefix} Received network result for unexpected state key {state_key}"
                )
                continue

            policy_np, value = results
            evaluation_cache[state_key] = results  # Update cache

            # Process all leaf nodes waiting for this state_key
            for (
                leaf_node,
                leaf_env_state,
                search_path,
            ) in nodes_awaiting_eval[state_key]:
                # Find the index of this simulation path to update its value
                sim_index = -1
                # Need a reliable way to find the index. Comparing paths might be fragile.
                # Let's search for the path ending in this specific leaf_node instance.
                for idx, path in enumerate(simulation_paths):
                    if (
                        path
                        and path[-1] is leaf_node
                        and simulation_values[idx] is None
                    ):
                        sim_index = idx
                        break

                if sim_index != -1:
                    simulation_values[sim_index] = value  # Store the evaluated value

                    # Expand the node using the received policy and stored env state
                    if not leaf_node.is_expanded():
                        logger.debug(
                            f"{log_prefix} Expanding node {leaf_node} for state {state_key} (after predict)."
                        )
                        self._expand(leaf_node, leaf_env_state, policy_np)
                    else:
                        logger.trace(f"{log_prefix} Node {leaf_node} already expanded.")

                    # Backpropagate the received value
                    self._backpropagate(leaf_node, value)
                else:
                    logger.error(
                        f"{log_prefix} Could not find simulation path for leaf node {leaf_node} (state {state_key})."
                    )
                    num_failed_evals += 1

        # --- Apply Dirichlet Noise (if training and first simulation was processed) ---
        # We need to know if the root was expanded, which happens during the first successful eval.
        # Check if root has children *after* processing results.
        if train and self.root.is_expanded() and self.config.dirichlet_epsilon > 0:
            self._apply_dirichlet_noise()

        # --- Phase 3: Select Action ---
        if not self.root.children:
            logger.error(
                f"{log_prefix} Root node has no children after search. Cannot select action."
            )
            # What action to return? Random legal action? None?
            # Let's return None, None to indicate failure.
            # Need env and state to get legal actions for random choice. This info isn't here.
            return None, None

        visit_counts = np.array(
            [child.visit_count for child in self.root.children.values()]
        )
        actions = list(self.root.children.keys())

        # Log visit counts before action selection
        if self.config.debug_mode or np.sum(visit_counts) == 0:
            visit_dict = {str(a): v for a, v in zip(actions, visit_counts)}
            logger.debug(f"{log_prefix} Final Visit Counts: {visit_dict}")
            logger.debug(f"{log_prefix} Total Root Visits: {self.root.visit_count}")

        if train:
            temperature = (
                1.0 if current_step < self.config.temperature_decay_steps else 0.1
            )
            if temperature > 1e-6:
                visit_counts_temp = visit_counts ** (1.0 / temperature)
                sum_visits_temp = np.sum(visit_counts_temp)
                if sum_visits_temp > 0:
                    action_probs = visit_counts_temp / sum_visits_temp
                    action_probs /= action_probs.sum()  # Normalize
                    try:
                        chosen_action_index = np.random.choice(
                            len(actions), p=action_probs
                        )
                    except ValueError as e:
                        logger.warning(
                            f"{log_prefix} Error choosing action with temp (probs sum {np.sum(action_probs)}): {e}. Choosing greedily."
                        )
                        chosen_action_index = np.argmax(visit_counts)
                else:
                    logger.warning(
                        f"{log_prefix} Sum of visits with temp is zero. Choosing greedily."
                    )
                    chosen_action_index = np.argmax(visit_counts)
            else:  # Temp is zero or near-zero
                chosen_action_index = np.argmax(visit_counts)
        else:  # Not training, choose greedily
            chosen_action_index = np.argmax(visit_counts)

        chosen_action = actions[chosen_action_index]

        # Calculate policy target using the agent's method (requires agent instance or env+network)
        # We passed env via phase2_state, so we can use the MCTS internal helper
        policy_target = self._calculate_policy_target(
            self.root, actions, visit_counts, env
        )  # Pass env

        logger.debug(f"{log_prefix} Chosen Action: {chosen_action}")
        if policy_target is not None:
            logger.debug(
                f"{log_prefix} Policy Target (Top 5): { {i: p for i, p in enumerate(policy_target) if p > 0.01} }"
            )  # Log non-zero targets

        return chosen_action, policy_target

    def _apply_dirichlet_noise(self):
        """Applies Dirichlet noise to the root node's children's priors."""
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

    def _select_leaf(
        self, root_node: MCTSNode, env: BaseEnvironment, initial_state: StateType
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
        sim_env = env.copy()
        sim_env.set_state(initial_state)
        search_path = [node]
        sim_log_prefix = "  Sim Select:"  # Simplified prefix for helper

        selection_active = True
        loop_count = 0  # Add loop counter for debugging deep selections

        while selection_active and node.is_expanded() and not sim_env.is_game_over():
            loop_count += 1
            parent_visits = node.visit_count
            logger.trace(
                f"{sim_log_prefix} Loop {loop_count}: Node={node}, ParentVisits={parent_visits}"
            )  # Log node and visits

            if not node.children:
                logger.error(f"{sim_log_prefix} Node is expanded but has no children!")
                return None, None, search_path  # Error case

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
