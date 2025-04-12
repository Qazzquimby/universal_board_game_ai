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
import torch.nn as nn

from core.config import AlphaZeroConfig
from environments.base import ActionType, StateType, BaseEnvironment
from models.networks import AlphaZeroNet

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
                logger.trace(
                    f"  Node.expand: Creating child for action {action_key} with prior {prior:.4f}"
                )
                self.children[action_key] = MCTSNode(parent=self, prior=prior)
                children_added += 1
            else:
                logger.warning(
                    f"  Node.expand: Child for action {action_key} already exists."
                    f" Was expand called multiple times?"
                )

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
            logger.debug(
                f"  Select: ParentVisits={parent_visits}, Scores={ {a: f'{s:.3f}' for a, s in child_scores.items()} }, Chosen={best_action}"
            )
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
            sim_env.set_state(state)

            # 1. Selection: Find a leaf node using UCB1.
            leaf_node, leaf_env_state = self._select(self.root, sim_env)

            # 2. Evaluation: Get the value of the leaf node.
            if not leaf_env_state.is_game_over():
                if not leaf_node.is_expanded():
                    self._expand(leaf_node, leaf_env_state)
                    value = self._rollout(leaf_env_state)
                else:
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

                logger.debug(
                    f"  Terminal Found during Select: Winner={winner}, PlayerAtNode={player_at_leaf}, Value={value}"
                )
            # 3. Backpropagation: Update nodes along the path from the leaf to the root.
            self._backpropagate(leaf_node, value)

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

            policy_len = len(policy_priors_np) if policy_priors_np is not None else -1
            logger.trace(
                f"  Expand Check: Action={action_key}, Index={action_index}, PolicyLen={policy_len}"
            )

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

            sorted_priors = sorted(
                action_priors_dict.items(), key=lambda item: item[1], reverse=True
            )
            state_obs = env_state.get_observation()
            logger.debug(
                f"  Expand (Network): State={state_obs.get('board', state_obs.get('piles', 'N/A'))}, Player={state_obs['current_player']}"
            )
            logger.debug(f"  Expand (Network): Legal Actions={legal_actions}")
            logger.debug(
                f"  Expand (Network={type(self.network).__name__}): Applied Priors (Top 5): { {a: f'{p:.3f}' for a, p in sorted_priors[:5]} }"
            )

        node.expand(action_priors_dict)

        # If priors were provided and non-zero, children should have been created.
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

    # Add helper method to AlphaZeroMCTS or MCTS
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

    def get_network_request(
        self, previous_response: Optional[Tuple[np.ndarray, float]] = None
    ) -> Optional[StateType]:
        if previous_response:
            policy_np, value = previous_response
            self._expand(self.current_leaf_node, self.current_leaf_env, policy_np)
            if (
                self.current_leaf_node == self.root
                and self.training
                and self.config.dirichlet_epsilon > 0
            ):
                self._apply_dirichlet_noise()
            self._backpropagate(self.current_leaf_node, value)
            self.current_leaf_node = None
            self.current_leaf_env = None

        while self.sim_count < self.num_simulations:
            self.sim_count += 1
            leaf_node, leaf_env, _ = self._select_leaf(self.root, self.base_env)
            if leaf_env.is_game_over():
                value = self.get_terminal_value(leaf_env)
                self._backpropagate(leaf_node, value)
                continue

            self.current_leaf_node = leaf_node
            self.current_leaf_env = leaf_env
            return leaf_env.get_observation()

        return None

    def start_search(
        self, env: BaseEnvironment, state: StateType, train: bool = False
    ) -> None:
        self.reset_root()
        self.base_env = env.copy()
        self.base_env.set_state(state)
        self.sim_count = 0
        self.training = train
        self.current_leaf_node = None
        self.current_leaf_env = None

    def get_result(self) -> Tuple[Optional[ActionType], Optional[np.ndarray]]:
        if not self.root.children:
            logger.error("Root node has no children after search.")
            assert False

        visit_counts = np.array(
            [child.visit_count for child in self.root.children.values()]
        )
        actions = list(self.root.children.keys())

        if np.sum(visit_counts) == 0:
            assert False
        elif self.training:
            visit_counts_temp = visit_counts ** (1.0 / 1.0)
            action_probs = visit_counts_temp / np.sum(visit_counts_temp)
            chosen_action_index = np.random.choice(len(actions), p=action_probs)
            chosen_action = actions[chosen_action_index]
        else:
            chosen_action_index = np.argmax(visit_counts)
            chosen_action = actions[chosen_action_index]

        policy_target = self._calculate_policy_target(
            self.root, actions, visit_counts, self.base_env
        )
        return chosen_action, policy_target

    # Modify signature to accept env directly
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
            }
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
