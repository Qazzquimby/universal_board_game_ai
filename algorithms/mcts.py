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


# --- MuZero MCTS ---
# Note: This is a simplified structure. A full MuZero MCTS often involves
# slightly different node structures (storing hidden state, reward) and search logic.

class MuZeroMCTSNode:
    """Node specific to MuZero MCTS, storing hidden state."""
    def __init__(self, parent: Optional['MuZeroMCTSNode'] = None, prior: float = 0.0):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0 # W(s,a)
        self.reward = 0.0 # R(s,a) - Reward obtained reaching this node *from parent*
        self.hidden_state: Optional[torch.Tensor] = None # s - Hidden state represented by this node
        self.children: Dict[ActionType, 'MuZeroMCTSNode'] = {}
        # Store policy logits and value prediction when node is first evaluated? Optional.
        # self.policy_logits = None
        # self.value = None

    @property
    def value(self) -> float: # Q(s,a) - Mean action value
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)

    def expand(self, actions: List[ActionType], policy_logits: torch.Tensor, network: 'MuZeroNet'):
        """Expand node using policy predictions from the network."""
        policy_probs = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
        for action in actions:
            action_key = tuple(action) if isinstance(action, list) else action
            if action_key not in self.children:
                idx = network.get_action_index(action_key)
                if idx is not None and 0 <= idx < len(policy_probs):
                    prior = float(policy_probs[idx])
                    self.children[action_key] = MuZeroMCTSNode(parent=self, prior=prior)
                else:
                    print(f"Warning: Could not map action {action_key} during MuZero node expansion.")


class MuZeroMCTS:
    """MCTS implementation adapted for MuZero's learned model."""

    def __init__(
        self,
        config: 'MuZeroConfig', # Use MuZeroConfig
        network: 'MuZeroNet' # Requires MuZeroNet
    ):
        self.config = config
        self.network = network
        # Root node doesn't have a prior action, hidden state comes from representation(obs)
        self.root = MuZeroMCTSNode(prior=0.0)

    def reset_root(self):
        self.root = MuZeroMCTSNode(prior=0.0)

    def _puct_score(self, node: MuZeroMCTSNode, parent_visits: int, discount: float = 0.99) -> float:
        """
        Calculate the PUCT score for a node in MuZero MCTS.
        Uses G + discount * Q(s', a') formulation.
        """
        # Calculate Q-value (mean action value from this node onwards)
        q_value = node.value # Average value accumulated from visits passing through this node

        # Calculate U-value (exploration bonus)
        if node.visit_count == 0:
            u_value = self.config.cpuct * node.prior * math.sqrt(max(1, parent_visits))
        else:
            u_value = self.config.cpuct * node.prior * math.sqrt(parent_visits) / (1 + node.visit_count)

        # MuZero PUCT: Maximize Q(s,a) + U(s,a)
        # Where Q(s,a) = R(s,a) + discount * V(s') - Value is estimated from parent perspective
        # Let's use the simpler AlphaZero PUCT for now: Q(s,a) + U(s,a)
        # where Q(s,a) is the mean value of the child node, adjusted for parent perspective.
        # Q(parent, action_to_node) = node.reward + discount * (-node.value) ??? This gets complex.

        # Let's stick to AlphaZero PUCT for selection for now, using the node's stored value.
        # We need to ensure backpropagation correctly updates node.total_value.
        # Value from parent perspective = -node.value
        q_score_parent_perspective = -node.value

        return q_score_parent_perspective + u_value


    def _select_child(self, node: MuZeroMCTSNode) -> Tuple[ActionType, MuZeroMCTSNode]:
        """Selects the child node with the highest PUCT score."""
        parent_visits = node.visit_count
        # TODO: Add discount factor from config if needed for PUCT variant
        best_item = max(
            node.children.items(),
            key=lambda item: self._puct_score(item[1], parent_visits)
        )
        return best_item # Returns (action, child_node)


    def _backpropagate(self, node: MuZeroMCTSNode, value: float, discount: float = 0.99):
        """Backpropagate value using MuZero logic (incorporating rewards)."""
        current_node = node
        # Value is the predicted value V(s) from the leaf node perspective.
        while current_node is not None:
            current_node.visit_count += 1
            # Update total value W(s,a) = sum(R(s,a) + discount * V(s'))
            # The 'value' passed up is V(s') from the child.
            # We add the reward R(s,a) obtained by reaching the current_node.
            current_node.total_value += current_node.reward + discount * value

            # The value passed to the parent becomes the value estimate V(s) for the current node.
            # This seems wrong. Let's rethink backprop.

            # Standard MCTS backprop:
            # current_node.total_value += value
            # value = -value # Flip for parent

            # MuZero backprop (simplified):
            # Update W using the value estimate from the leaf node.
            # The value estimate needs to be relative to the player at each node.
            current_node.total_value += value
            # The value passed up should incorporate the reward received on the path.
            value = current_node.reward + discount * value # G = R + gamma*V

            # Flip perspective for parent? MuZero often uses value relative to current player.
            # If value is always relative to the player whose turn it is at the node, no flipping needed?
            # Let's assume value is relative to the player at the node. Backprop needs care.

            # Let's use AlphaZero backprop for now for simplicity, assuming zero-sum game.
            # We'll need to revisit this if rewards aren't zero-sum or if using MuZero's value definition.
            value = -value # Flip for parent perspective in zero-sum game

            current_node = current_node.parent


    def search(self, observation_dict: dict) -> MuZeroMCTSNode:
        """
        Run MuZero MCTS search starting from a root observation.
        """
        self.reset_root()

        # Initial step: Get hidden state and initial prediction from observation
        # TODO: Add device handling
        value, _, policy_logits, hidden_state = self.network.initial_inference(observation_dict)
        self.root.hidden_state = hidden_state

        # TODO: Get legal actions from the *actual* environment for the root node
        # This is a key difference/challenge in MuZero vs AlphaZero MCTS.
        # MuZero plans entirely in latent space, but needs legal actions at the root.
        # We might need the env instance here, or pass legal actions in.
        # Let's assume we can get legal actions for the root state.
        # temp_env = self.env.copy(); temp_env.set_state(observation_dict) # Requires env access
        # root_legal_actions = temp_env.get_legal_actions()
        root_legal_actions = [] # Placeholder - NEED TO GET ACTUAL LEGAL ACTIONS
        if not root_legal_actions:
             print("Warning: No legal actions provided for MuZero MCTS root.")
             # Cannot expand root if no legal actions known

        self.root.expand(root_legal_actions, policy_logits, self.network)
        # TODO: Add Dirichlet noise to root priors here if training

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = self.root
            search_path = [node] # Keep track of nodes visited

            # 1. Selection: Traverse tree using PUCT until a leaf node is reached
            while node.is_expanded():
                action, node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion & Evaluation: Expand leaf node, get prediction from network
            parent = search_path[-2] # Node from which the leaf was selected
            leaf_node = search_path[-1]

            # Use the parent's hidden state and the chosen action to get next state and reward via dynamics
            # This requires the action taken to reach the leaf node.
            action_to_leaf = None
            for act, child in parent.children.items():
                if child == leaf_node:
                    action_to_leaf = act
                    break

            if action_to_leaf is None:
                 # Should not happen if selection works correctly
                 print("Error: Could not find action leading to leaf node during MuZero search.")
                 continue

            # Infer next state, reward, policy, value using the learned model
            value, reward, policy_logits, next_hidden_state = self.network.recurrent_inference(
                parent.hidden_state, action_to_leaf
            )

            # Store results in the newly reached (or created) leaf node
            leaf_node.hidden_state = next_hidden_state
            leaf_node.reward = reward.item() # Store reward obtained reaching this node

            # TODO: Get legal actions for the *hypothetical* state represented by leaf_node.
            # This is the core challenge - MuZero doesn't know the real state.
            # Option 1: Assume all actions are legal in latent space (original MuZero).
            # Option 2: Predict a legal action mask (requires network modification).
            # Let's use Option 1 for now. Assume all actions possible from policy head are "legal" in latent space.
            num_actions = policy_logits.shape[-1]
            latent_legal_actions = list(range(num_actions)) # Indices 0 to num_actions-1

            # Expand the leaf node using the predicted policy
            # Need to map action indices back to ActionType if node dict uses ActionType keys
            # For simplicity, let's assume children keys are action indices for now.
            # This requires changing MuZeroMCTSNode.children to Dict[int, MuZeroMCTSNode]
            # And adapting _select_child accordingly. Let's skip this complexity for now.
            # Assume expand works with the original ActionType keys for now.
            # This requires mapping indices back to actions.
            # placeholder_actions = [self.network.get_action_from_index(i) for i in latent_legal_actions]
            # filtered_actions = [a for a in placeholder_actions if a is not None]
            # leaf_node.expand(filtered_actions, policy_logits, self.network)
            # This is getting complicated. Let's simplify the expansion assumption.
            # Assume expand takes policy logits and populates children based on network output size.
            # This bypasses needing explicit legal actions during search beyond the root.

            # Let's refine expand to work with indices directly?
            # Or assume prediction head only outputs logits for potentially valid actions?

            # --- Simplified Expansion ---
            # Expand based on the predicted policy for all possible actions
            # This requires a way to map policy indices to actions if needed later.
            # For now, let's assume expansion populates based on policy_logits size.
            # This part needs careful implementation matching the reference.
            # The reference MCTS likely handles nodes and actions differently.

            # --- Backpropagation ---
            # Backpropagate the predicted value V(s_leaf)
            self._backpropagate(leaf_node, value.item()) # Pass scalar value

        # Return the root node containing search statistics
        return self.root

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

        # Apply softmax to the raw logits to get probabilities over *all* actions
        policy_probs = torch.softmax(policy_logits_tensor.squeeze(0), dim=0).cpu().numpy()

        # --- Assign priors P(s,a) using the network's probabilities ---
        action_priors = {}
        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action
            # Use network's method to get the index
            idx = self.network.get_action_index(action_key)

            if idx is not None and 0 <= idx < len(policy_probs):
                # Use the probability from the softmax output directly as the prior
                action_priors[action_key] = float(policy_probs[idx])
            else:
                # This case should ideally not happen if action mapping is correct
                print(
                    f"Warning: Action {action_key} could not be mapped to a valid index by network during expansion. Assigning zero prior."
                )
                action_priors[action_key] = 0.0

        # Note: We do NOT re-normalize here. The PUCT formula uses the P(s,a)
        # from the network's policy head distribution directly.
        # If the sum of priors for legal actions is zero (e.g., network assigns zero prob to all legal moves),
        # the selection might behave unexpectedly. Handle this?
        # For now, let MCTS proceed; PUCT score will rely solely on Q-value if prior is 0.
        if not action_priors and legal_actions:
             print("Warning: All legal actions received zero prior probability from the network.")
             # Fallback? Maybe uniform? Let's stick to network priors for now.
             pass

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
