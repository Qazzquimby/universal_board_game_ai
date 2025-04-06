import math
import random
import time
import contextlib
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
    Set,
    Generator,
    Union,
    Literal,
)  # Added Generator, Union, Literal

import torch
from loguru import logger
import numpy as np

from core.config import MuZeroConfig
from environments.base import ActionType, StateType, BaseEnvironment
from models.networks import AlphaZeroNet, MuZeroNet

PredictResult = Tuple[np.ndarray, float]


# --- Type Aliases for Generator Yield Values ---
# Yielded when the generator needs a network prediction
PredictRequestYield = Tuple[Literal["predict_request"], str, StateType]
# Yielded when the generator has resumed after an internal step (cache hit, rollout)
ResumedYield = Tuple[Literal["resumed"]]
# Yielded when the MCTS search is complete
SearchCompleteYield = Tuple[
    Literal["search_complete"], "MCTSNode"
]  # Forward reference MCTSNode

# Union of all possible yield types from search_generator
GeneratorYield = Union[PredictRequestYield, ResumedYield, SearchCompleteYield]


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
        for action, prior in action_priors.items():
            # Ensure action is hashable if it's not already (e.g., list)
            action_key = tuple(action) if isinstance(action, list) else action
            if action_key not in self.children:
                self.children[action_key] = MCTSNode(parent=self, prior=prior)


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
        exploration_constant: float = 1.0,  # c_puct in AlphaZero
        num_simulations: int = 100,
        network: Optional[AlphaZeroNet] = None,  # Type hint Optional
        discount_factor: float = 1.0,
        profiler: Optional[MCTSProfiler] = None,
        # inference_batch_size removed, batching will be handled externally
        # TODO: Add dirichlet_epsilon and dirichlet_alpha for root noise
    ):
        super().__init__(
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,
            num_simulations=num_simulations,
        )
        self.network = network
        self.profiler = profiler
        # self.dirichlet_epsilon = dirichlet_epsilon
        # self.dirichlet_alpha = dirichlet_alpha

    def _expand(
        self,
        node: MCTSNode,
        env_state: BaseEnvironment,  # Pass env state for legal actions
        policy_priors_np: Optional[np.ndarray] = None,  # Optional policy from network
    ) -> None:
        """
        Expand the leaf node. Uses network policy priors if provided, otherwise uniform.

        Args:
            node: The node to expand.
            env_state: The environment state at the node (used for legal actions).
            policy_priors_np: Optional numpy array of policy probabilities from the network.
        """
        if node.is_expanded() or env_state.is_game_over():
            return

        legal_actions = env_state.get_legal_actions()
        if not legal_actions:
            return  # Cannot expand if no legal actions

        action_priors_dict = {}
        if self.network and policy_priors_np is not None:
            # --- Use network priors ---
            for action in legal_actions:
                action_key = tuple(action) if isinstance(action, list) else action
                # Use network's mapping function
                action_index = self.network.get_action_index(action_key)

                if action_index is not None and 0 <= action_index < len(
                    policy_priors_np
                ):
                    prior = policy_priors_np[action_index]
                    action_priors_dict[action_key] = prior
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
                f"  Expand (Network): Applied Priors (Top 5): { {a: f'{p:.3f}' for a, p in sorted_priors[:5]} }"
            )
        else:
            # --- Fallback to uniform priors (networkless or policy not provided) ---
            num_legal = len(legal_actions)
            uniform_prior = 1.0 / num_legal if num_legal > 0 else 1.0
            action_priors_dict = {
                (tuple(action) if isinstance(action, list) else action): uniform_prior
                for action in legal_actions
            }
            logger.debug(
                f"  Expand (Uniform): Legal Actions={legal_actions}, Prior={uniform_prior:.3f}"
            )

        # Call the base node expansion method
        node.expand(action_priors_dict)

    # Note: _rollout is effectively replaced by the batched network value prediction
    # in the modified search loop when a network is present.
    # We keep the base class _rollout for the networkless case.
    # The AlphaZero-specific _rollout using network.predict is removed below.

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

    # Renamed from search to search_generator
    def search_generator(
        self, env: BaseEnvironment, state: StateType
    ) -> Generator[GeneratorYield, Optional[PredictResult], None]:
        """
        Generator-based MCTS search. Yields network requests and expects results via send().

        Yields:
            ('predict_request', state_key, state_obs, node): When network evaluation is needed.
            ('search_complete', root_node): When all simulations are done.

        Receives (via send):
            (policy_np, value): The result of a network prediction for a yielded request.

        Args:
            env: The *real* environment instance (used for copying and initial state).
            state: The current state dictionary from the real environment.

        Returns:
            Nothing directly, yields results. The final root is yielded in 'search_complete'.
        """
        self.reset_root()
        initial_state_obs = state  # Keep the original state dict

        # --- State for Generator-Based Batching ---
        # Cache for network results within this single search instance
        # Maps unique_state_key to (policy_np, value)
        evaluation_cache: Dict[str, Tuple[np.ndarray, float]] = {}

        # Helper to create a unique key for a state (moved inside for encapsulation)
        def get_state_key(s: StateType) -> str:
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

        search_timer = Timer() if self.profiler else contextlib.nullcontext()
        with search_timer:
            log_prefix = f"AZ MCTS Search Gen (Network={self.network is not None}, Sims={self.num_simulations})"
            logger.debug(f"--- {log_prefix} Start: State={initial_state_obs} ---")

            # TODO: Add Dirichlet noise to root priors if training and network exists

            for sim_num in range(self.num_simulations):
                sim_log_prefix = f"  Sim {sim_num+1}/{self.num_simulations}:"
                logger.debug(f"{sim_log_prefix} Starting...")

                # --- Run one simulation path ---
                node = self.root
                sim_env = env.copy()
                sim_env.set_state(initial_state_obs)  # Start from the root state
                search_path = [node]

                # 1. Selection: Find a leaf node using PUCT (_score_child).
                selection_active = True
                while (
                    selection_active
                    and node.is_expanded()
                    and not sim_env.is_game_over()
                ):
                    logger.debug(
                        f"{sim_log_prefix} Selection Loop: Node expanded={node.is_expanded()}, Game Over={sim_env.is_game_over()}"
                    )
                    parent_visits = node.visit_count

                    if not node.children:
                        # This should not happen if node.is_expanded() is true.
                        # If it does, it indicates an issue during expansion or node state.
                        logger.error(
                            f"{sim_log_prefix} Select: Node is expanded but has no children!"
                        )
                        selection_active = False  # Exit loop
                        continue  # Skip rest of loop iteration

                    # Calculate scores for all children determined during expansion
                    child_scores = {
                        act: self._score_child(child, parent_visits)
                        for act, child in node.children.items()
                    }

                    best_action = max(child_scores, key=child_scores.get)
                    logger.debug(
                        f"{sim_log_prefix} Select: ParentVisits={parent_visits}, LegalScores={ {a: f'{s:.3f}' for a, s in child_scores.items()} }, Chosen={best_action}"
                    )

                    try:
                        # Step the environment with the chosen legal action
                        logger.debug(
                            f"{sim_log_prefix} Select: Stepping env with {best_action}"
                        )
                        sim_env.step(best_action)
                        # Update node using the filtered dictionary
                        logger.debug(
                            f"{sim_log_prefix} Select: Moving to child node for action {best_action}"
                        )
                        node = node.children[best_action]
                        search_path.append(node)
                    except (
                        ValueError,
                        KeyError,
                    ) as e:  # ValueError for invalid step, KeyError for dict lookup (shouldn't happen)
                        logger.error(
                            f"{sim_log_prefix} Select: Error stepping env or finding child for action {best_action}. Error: {e}"
                        )
                        node = None  # Signal error
                        selection_active = False  # Exit loop
                        # break # Replaced by selection_active = False

                # --- End of Selection Loop ---
                logger.debug(
                    f"{sim_log_prefix} Selection loop finished. Final node in path: {node}"
                )

                if node is None:
                    logger.error(
                        f"{sim_log_prefix} Skipping simulation due to error during selection."
                    )
                    continue  # Skip to next simulation if error occurred

                # --- Reached a leaf node (or terminal state) ---
                leaf_node = node
                leaf_env_state = sim_env  # Environment state at the leaf

                value = 0.0  # Default value
                policy_np = None  # Default policy

                # 2. Evaluation: Determine value and potentially expand.
                if leaf_env_state.is_game_over():
                    # Game ended during selection. Determine the outcome.
                    player_at_leaf = (
                        leaf_env_state.get_current_player()
                    )  # Whose turn it *would* be
                    winner = leaf_env_state.get_winning_player()
                    if winner is None:
                        value = 0.0
                    elif winner == player_at_leaf:
                        value = 1.0  # Player at leaf wins
                    else:
                        value = -1.0  # Player at leaf loses
                    logger.debug(
                        f"  Terminal Found during Select: Winner={winner}, PlayerAtNode={player_at_leaf}, Value={value}"
                    )
                    # Backpropagate terminal value immediately
                    self._backpropagate(leaf_node, value)
                    continue  # Move to next simulation

                # --- Non-Terminal Leaf Node ---
                if self.network:
                    # --- Network-Based Evaluation (Yielding Logic) ---
                    leaf_state_obs = leaf_env_state.get_observation()
                    state_key = get_state_key(leaf_state_obs)

                    if state_key in evaluation_cache:
                        # Cache Hit
                        policy_np, value = evaluation_cache[state_key]
                        logger.debug(
                            f"{sim_log_prefix} Eval: Cache Hit for StateKey={state_key}, Value={value:.3f}"
                        )
                        if not leaf_node.is_expanded():
                            logger.debug(
                                f"{sim_log_prefix} Eval: Expanding node (cache hit)."
                            )
                            self._expand(leaf_node, leaf_env_state, policy_np)
                        logger.debug(
                            f"{sim_log_prefix} Eval: Backpropagating value {value:.3f} (cache hit)."
                        )
                        self._backpropagate(leaf_node, value)
                        resumed_yield: ResumedYield = ("resumed",)
                        yield resumed_yield
                    else:
                        # Cache Miss: Yield request and wait for result
                        logger.debug(
                            f"{sim_log_prefix} Eval: Cache Miss. Yielding Request for StateKey={state_key}"
                        )
                        # Yield only the info needed by the caller to get the prediction
                        predict_request_yield: PredictRequestYield = (
                            "predict_request",
                            state_key,
                            leaf_state_obs,
                        )
                        # Add type hint for received value
                        received_result: Optional[
                            PredictResult
                        ] = yield predict_request_yield
                        # --- Resumed after yield ---
                        # Expecting received_result to be a tuple (policy_np, value)
                        # Check if None explicitly first
                        if received_result is None:
                            logger.error(
                                f"Generator received None result for state {state_key}. Skipping backprop."
                            )
                            continue
                        # Check tuple structure and types more carefully
                        if (
                            not isinstance(received_result, tuple)
                            or len(received_result) != 2
                        ):
                            logger.error(
                                f"Generator received invalid result structure for state {state_key}. Type: {type(received_result)}, Value: {received_result}. Skipping backprop."
                            )
                            continue

                        policy_np, value = received_result  # Unpack the received tuple
                        # Add extra check for types inside the tuple
                        if not isinstance(policy_np, np.ndarray) or not isinstance(
                            value, (float, int)
                        ):  # Allow int just in case
                            logger.error(
                                f"Generator received tuple with unexpected types for state {state_key}: ({type(policy_np)}, {type(value)}). Skipping backprop."
                            )
                            continue
                        # Convert value to float if it was int
                        value = float(value)

                        logger.debug(
                            f"{sim_log_prefix} Eval: Received Result for StateKey={state_key}, Value={value:.3f}"
                        )
                        # Cache the received result
                        evaluation_cache[state_key] = (policy_np, value)
                        # Expand node using the received policy if not already expanded
                        if not leaf_node.is_expanded():
                            logger.debug(
                                f"{sim_log_prefix} Eval: Expanding node (after predict)."
                            )
                            self._expand(leaf_node, leaf_env_state, policy_np)
                        # Backpropagate the received value
                        logger.debug(
                            f"{sim_log_prefix} Eval: Backpropagating value {value:.3f} (after predict)."
                        )
                        self._backpropagate(leaf_node, value)
                        # Instead of continue, signal resumption
                        resumed_yield: ResumedYield = ("resumed",)
                        yield resumed_yield
                        # continue # REMOVED

                else:
                    # --- Networkless Evaluation (Expand + Rollout) ---
                    logger.debug(
                        f"{sim_log_prefix} Eval: Networkless - Expanding and Rolling out."
                    )
                    # Expand using uniform priors
                    if not leaf_node.is_expanded():
                        logger.debug(
                            f"{sim_log_prefix} Eval: Expanding node (networkless)."
                        )
                        self._expand(
                            leaf_node, leaf_env_state, policy_priors_np=None
                        )  # Pass None policy

                    # Perform random rollout using base class method
                    logger.debug(f"{sim_log_prefix} Eval: Starting rollout.")
                    value = super()._rollout(leaf_env_state)  # Call MCTS._rollout
                    logger.debug(
                        f"{sim_log_prefix} Eval: Rollout finished, value={value:.3f}."
                    )
                    # Backpropagate rollout result immediately
                    logger.debug(
                        f"{sim_log_prefix} Eval: Backpropagating value {value:.3f} (networkless)."
                    )
                    self._backpropagate(leaf_node, value)
                    # Instead of continue, signal resumption
                    resumed_yield: ResumedYield = ("resumed",)
                    yield resumed_yield
                    # continue # REMOVED

                # --- Internal batch processing block removed ---

            # --- End of Simulation Loop ---

        # --- End of Simulation Loop ---
        logger.debug(f"{log_prefix} All simulations finished.")

        if self.profiler and isinstance(search_timer, Timer):
            self.profiler.record_search_time(search_timer.elapsed_ms)
            # Log cache stats if network was used
            if self.network:
                logger.debug(f"{log_prefix} End: Cache size={len(evaluation_cache)}")

        # Log final root node state
        if self.root.children:
            child_visits = {a: c.visit_count for a, c in self.root.children.items()}
            logger.debug(
                f"{log_prefix} End: Root VisitCount={self.root.visit_count}, Child Visits={child_visits}"
            )
        else:
            logger.debug(
                f"{log_prefix} End: Root VisitCount={self.root.visit_count}, No children expanded."
            )

        # Signal completion and yield the final root node
        # Explicitly cast to satisfy type checker if needed, though structure matches
        search_complete_yield: SearchCompleteYield = ("search_complete", self.root)
        yield search_complete_yield

    # Removed duplicated search method - search_generator is the active one


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
                        self.children[action_key] = MuZeroMCTSNode(
                            parent=self, prior=float(prior)
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

    def _select_child(self, node: MuZeroMCTSNode) -> Tuple[ActionType, MuZeroMCTSNode]:
        """Selects the child node with the highest PUCT score."""
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
