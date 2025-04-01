import math
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

from environments.base import ActionType, StateType, BaseEnvironment
from models.networks import AlphaZeroNet


class MCTSNode:
    """MCTS node adapted for AlphaZero."""

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
                # Create child node with the prior probability
                self.children[action_key] = MCTSNode(parent=self, prior=prior)


class MCTS:
    """Core MCTS algorithm implementation (UCB1 + Random Rollout)."""

    def __init__(
        self,
        exploration_constant: float = 1.41,  # UCB1 exploration constant
        discount_factor: float = 1.0,  # Discount factor for rollout rewards
        num_simulations: int = 100,
    ):
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.num_simulations = num_simulations
        self.root = MCTSNode()

    def reset_root(self):
        """Resets the root node."""
        self.root = MCTSNode()

    def _ucb_score(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate UCB1 score."""
        if node.visit_count == 0:
            return float("inf")
        # Ensure parent_visits is at least 1 to avoid math.log(0)
        safe_parent_visits = max(1, parent_visits)
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(safe_parent_visits) / node.visit_count
        )
        # Maximize (-opponent_value + exploration) which is equivalent to (-node.value + exploration)
        # node.value is from the perspective of the player whose turn it is AT the node.
        # The parent selecting wants to maximize its own value, which is the negative of the child node's value.
        return -node.value + exploration_term

    def _select(
        self, node: MCTSNode, env: BaseEnvironment
    ) -> Tuple[MCTSNode, BaseEnvironment]:
        """Select child node with highest UCB score until a leaf node is reached."""
        while node.is_expanded() and not env.is_game_over():
            parent_visits = node.visit_count
            # Select the action corresponding to the child with the highest UCB score
            # Need to handle the case where parent_visits is 0 (shouldn't happen if root is visited once?)
            # Let's ensure root is visited at least once implicitly by the loop structure.
            best_item = max(
                node.children.items(),
                key=lambda item: self._ucb_score(item[1], parent_visits),
            )
            action, node = best_item
            # Action here is the key from children dict, should be hashable
            env.step(action)  # Update the environment state as we traverse
        return node, env

    def _expand(self, node: MCTSNode, env: BaseEnvironment):
        """Expand the leaf node by creating children for all legal actions."""
        if node.is_expanded() or env.is_game_over():
            return  # Already expanded or terminal

        legal_actions = env.get_legal_actions()
        # Use uniform prior for standard MCTS expansion
        # The MCTSNode prior defaults to 1.0, which is fine here.
        action_priors = {
            action: 1.0 / len(legal_actions) if legal_actions else 1.0
            for action in legal_actions
        }
        node.expand(action_priors)  # Use the expand method with uniform priors

    def _rollout(self, env: BaseEnvironment) -> float:
        """Simulate game from current state using random policy."""
        # Determine the player whose perspective the rollout value should represent.
        # This is the player whose turn it was *at the start of the rollout*.
        player_at_rollout_start = env.get_current_player()

        sim_env = env.copy()  # Simulate on a copy
        while not sim_env.is_game_over():
            legal_actions = sim_env.get_legal_actions()
            if not legal_actions:
                break  # Should not happen if is_game_over is checked, but safety first
            action = random.choice(legal_actions)
            sim_env.step(action)

        winner = sim_env.get_winning_player()
        if winner is None:  # Draw
            return 0.0
        # Return +1 if the player who started the rollout won, -1 otherwise
        return 1.0 if winner == player_at_rollout_start else -1.0

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the rollout value up the tree."""
        current_node = node
        # Value is from the perspective of the player whose turn it was *at the start of the rollout*.
        # As we go up, the perspective flips for the parent.
        while current_node is not None:
            current_node.visit_count += 1
            # Update total value (W). The value should be relative to the player whose turn it is *at the parent node*.
            # Since 'value' is from the child's perspective (start of rollout), we negate it at each step up.
            current_node.total_value += value
            value = -value  # Flip value for the parent (opponent's perspective)
            current_node = current_node.parent

    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode:
        """Run MCTS search from the given state using UCB1 and random rollouts."""
        # Set the root to a new node corresponding to the current state
        self.reset_root()  # Start fresh search for each call

        for _ in range(self.num_simulations):
            # Start from the root node and a copy of the environment set to the initial state
            node = self.root
            sim_env = env.copy()
            sim_env.set_state(state)  # Ensure simulation starts from the correct state

            # 1. Selection: Traverse the tree using UCB1 until a leaf node is reached
            node, sim_env = self._select(node, sim_env)

            # 2. Expansion: If the game is not over and the node hasn't been expanded, expand it.
            value = 0.0  # Default value
            if not sim_env.is_game_over():
                if not node.is_expanded():
                    self._expand(node, sim_env)
                    # After expansion, usually perform a rollout from one of the new children.
                    # Or, more commonly, rollout from the expanded node itself *before* selecting a child.
                    # Let's rollout from the state reached *after* selection (sim_env).
                    value = self._rollout(sim_env)
                else:
                    # If node was already expanded but selection ended here (e.g., terminal state reached during selection),
                    # we still need a value. Rollout is one option, or use terminal state value.
                    # Since selection stops *before* entering a terminal state in the loop condition,
                    # this 'else' block might be less common. Let's assume rollout is the default.
                    value = self._rollout(sim_env)  # Rollout from the state reached

            else:
                # Game ended during selection phase. Determine the outcome.
                winner = sim_env.get_winning_player()
                # Value should be from the perspective of the player whose turn it *would* have been.
                player_at_terminal_node = sim_env.get_current_player()

                if winner is None:  # Draw
                    value = 0.0
                # Perspective adjustment for backpropagation (similar to AlphaZero logic):
                # Value from the perspective of the player who *just moved* to reach this state.
                # Use the num_players property from the environment
                player_who_just_moved = (
                    sim_env.get_current_player() + sim_env.num_players - 1
                ) % sim_env.num_players
                if winner == player_who_just_moved:  # Player who moved won
                    value = 1.0
                else:  # Player who moved lost (or draw handled above)
                    value = -1.0

            # 3. Backpropagation: Update visit counts and values up the tree
            self._backpropagate(node, value)

        return self.root


# --- AlphaZero MCTS Subclass ---


class AlphaZeroMCTS(MCTS):
    """MCTS algorithm adapted for AlphaZero (PUCT + Network Evaluation)."""

    def __init__(
        self,
        exploration_constant: float = 1.0,  # c_puct in AlphaZero
        num_simulations: int = 100,
        network: AlphaZeroNet = None,  # Network is required
        discount_factor: float = 1.0,  # Usually 1.0 for AlphaZero MCTS value (consistency)
        # TODO: Add dirichlet_epsilon and dirichlet_alpha for root noise
    ):
        if network is None:
            raise ValueError("AlphaZeroMCTS requires a network.")

        # Call parent init, but exploration constant is now c_puct
        super().__init__(
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,  # Keep for consistency, though not used in backprop value calc
            num_simulations=num_simulations,
        )
        self.network = network
        # self.dirichlet_epsilon = dirichlet_epsilon
        # self.dirichlet_alpha = dirichlet_alpha

    # --- Overridden methods from base MCTS ---

    # Override selection to use PUCT score
    def _select(
        self, node: MCTSNode, env: BaseEnvironment
    ) -> Tuple[MCTSNode, BaseEnvironment]:
        """Select child node with highest PUCT score until a leaf node is reached."""
        while node.is_expanded() and not env.is_game_over():
            parent_visits = node.visit_count
            # Select the action corresponding to the child with the highest PUCT score
            best_item = max(
                node.children.items(),
                key=lambda item: self._puct_score(item[1], parent_visits),
            )
            action, node = best_item
            env.step(action)  # Update the environment state as we traverse
        return node, env

    # --- Helper methods specific to AlphaZeroMCTS ---

    def _puct_score(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate the PUCT score for a node."""
        # Use self.exploration_constant as c_puct
        if node.visit_count == 0:
            # If node hasn't been visited, U is based only on prior and parent visits
            # Ensure parent_visits > 0 for sqrt
            safe_parent_visits = max(1, parent_visits)
            u_score = (
                self.exploration_constant * node.prior * math.sqrt(safe_parent_visits)
            )
            return u_score  # Q score is 0 initially
        else:
            # Q(s,a) + U(s,a)
            # Q(s,a) must be from the perspective of the player selecting at the PARENT node.
            # node.value (q_score) is from the perspective of the player AT the node (the opponent).
            # So, we use -node.value for the parent's perspective.
            q_score_parent_perspective = -node.value
            u_score = (
                self.exploration_constant
                * node.prior
                * math.sqrt(parent_visits)
                / (1 + node.visit_count)
            )
            return q_score_parent_perspective + u_score

    # Removed _get_policy_for_legal_actions and _create_action_index_map
    # Action mapping is now handled by the network's get_action_index method

    # --- Overridden methods from base MCTS ---

    # Override expansion to use network priors and network's action mapping
    def _expand(self, node: MCTSNode, env: BaseEnvironment):
        """Expand the leaf node using prior probabilities from the network."""
        if node.is_expanded() or env.is_game_over():
            return

        current_state_obs = env.get_observation()
        legal_actions = env.get_legal_actions()

        if not legal_actions:
            print(
                "Warning: Expanding node with no legal actions despite not being game over."
            )
            return  # Cannot expand

        # Get policy priors from the network
        # TODO: Add device handling
        flat_state = self.network._flatten_state(current_state_obs)
        self.network.eval()
        with torch.no_grad():
            policy_logits_tensor, _ = self.network(flat_state)

        policy_logits = policy_logits_tensor.squeeze(0).cpu().numpy()

        # --- Map policy outputs to legal actions using network's mapping ---
        action_priors = {}
        policy_sum_legal = 0.0
        raw_priors = {}

        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action
            # Use network's method to get the index
            idx = self.network.get_action_index(action_key)

            if idx is not None and 0 <= idx < len(policy_logits):
                # Use exp(logit) as raw prior score before normalization
                raw_prior = float(np.exp(policy_logits[idx]))
                raw_priors[action_key] = raw_prior
                policy_sum_legal += raw_prior
            else:
                print(
                    f"Warning: Action {action_key} could not be mapped to a valid index by network. Assigning zero prior."
                )
                raw_priors[action_key] = 0.0  # Assign zero prior if mapping fails

        # Normalize the raw priors (effectively softmax over legal actions)
        if policy_sum_legal > 1e-8:
            for action, raw_prior in raw_priors.items():
                action_priors[action] = raw_prior / policy_sum_legal
        elif legal_actions:
            print(
                "Warning: Sum of exp(logits) for legal actions is near zero. Using uniform priors."
            )
            prior = 1.0 / len(legal_actions)
            action_priors = {action: prior for action in legal_actions}
        else:
            action_priors = {}  # No legal actions to assign priors to

        # Add Dirichlet noise to root node priors during training self-play
        # if node.parent is None and self.dirichlet_epsilon > 0:
        #     action_priors = self._add_dirichlet_noise(action_priors, legal_actions)

        # Expand the node with legal actions and their priors
        node.expand(action_priors)

    # Override rollout with network evaluation
    def _rollout(self, env: BaseEnvironment) -> float:
        """Evaluate the leaf node state using the network's value head."""
        if env.is_game_over():
            # Handle terminal state reached during selection (should be handled in search loop)
            print("Warning: _rollout called on a terminal state.")
            return 0.0  # Or determine actual outcome

        current_state_obs = env.get_observation()
        # TODO: Add device handling
        flat_state = self.network._flatten_state(current_state_obs)
        self.network.eval()
        with torch.no_grad():
            _, value_tensor = self.network(flat_state)

        value = value_tensor.squeeze(0).cpu().item()

        # Value is from the perspective of the current player in 'env'
        return value

    # Override search loop to use network evaluation instead of rollout
    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode:
        """Run MCTS search using PUCT selection and network evaluation."""
        self.reset_root()

        for _ in range(self.num_simulations):
            node = self.root
            sim_env = env.copy()
            sim_env.set_state(state)

            # 1. Selection (using PUCT)
            node, sim_env = self._select(node, sim_env)

            # 2. Expansion & Evaluation
            value = 0.0
            if not sim_env.is_game_over():
                # Expand node using network policy priors (calls overridden _expand)
                if not node.is_expanded():
                    self._expand(node, sim_env)
                # Evaluate the reached state using network value head (calls overridden _rollout)
                value = self._rollout(sim_env)
                # Value is from the perspective of the player whose turn it is in sim_env
            else:
                # Game ended during selection. Determine outcome.
                winner = sim_env.get_winning_player()
                # Value from perspective of player who *just moved* to reach this state.
                player_who_just_moved = (
                    sim_env.get_current_player() + sim_env.num_players - 1
                ) % sim_env.num_players
                if winner is None:
                    value = 0.0
                elif winner == player_who_just_moved:
                    value = 1.0
                else:
                    value = -1.0

            # 3. Backpropagation (uses base class method)
            self._backpropagate(node, value)

        return self.root

    # --- Helper methods specific to AlphaZeroMCTS ---

    def _puct_score(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate the PUCT score for a node."""
        # Use self.exploration_constant as c_puct
        if node.visit_count == 0:
            # If node hasn't been visited, U is based only on prior and parent visits
            # Ensure parent_visits > 0 for sqrt
            safe_parent_visits = max(1, parent_visits)
            u_score = (
                self.exploration_constant * node.prior * math.sqrt(safe_parent_visits)
            )
            return u_score  # Q score is 0 initially
        else:
            # Q(s,a) + U(s,a)
            q_score = node.value  # Mean action value (Q)
            u_score = (
                self.exploration_constant
                * node.prior
                * math.sqrt(parent_visits)
                / (1 + node.visit_count)
            )
            return q_score + u_score

    # Removed _get_policy_for_legal_actions and _create_action_index_map
    # Action mapping is now handled by the network's get_action_index method
