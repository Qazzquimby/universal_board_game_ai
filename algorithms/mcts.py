# Standard library imports
import math
import random
from typing import List, Tuple, Any, Optional, Dict

# Third-party imports
import numpy as np
import torch

# Local application imports
from environments.env_interface import ActionType, StateType, BaseEnvironment
from models.networks import AlphaZeroNet # Import the network


class MCTSNode:
    """MCTS node adapted for AlphaZero."""

    def __init__(self, parent: Optional['MCTSNode'] = None, prior: float = 1.0):
        self.parent = parent
        self.prior = prior  # P(s,a) - Prior probability from the network
        self.visit_count = 0
        self.total_value = 0.0  # W(s,a) - Total action value accumulated
        self.children: Dict[ActionType, MCTSNode] = {}

    @property
    def value(self) -> float: # Q(s,a) - Mean action value
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
    """MCTS algorithm adapted for AlphaZero."""

    def __init__(
        self,
        exploration_constant: float = 1.0, # c_puct in AlphaZero
        discount_factor: float = 1.0, # Usually 1.0 for AlphaZero MCTS value
        num_simulations: int = 100,
        network: Optional[AlphaZeroNet] = None # Network for policy/value
    ):
        self.exploration_constant = exploration_constant # c_puct
        self.discount_factor = discount_factor # Not typically used in AZ backprop, but kept for potential future use
        self.num_simulations = num_simulations
        self.network = network
        self.root = MCTSNode()
        # TODO: Add Dirichlet noise for root exploration during training

    def reset_root(self):
        """Resets the root node."""
        self.root = MCTSNode()

    def _puct_score(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate the PUCT score for a node."""
        if node.visit_count == 0:
            # If node hasn't been visited, U is based only on prior and parent visits
            u_score = self.exploration_constant * node.prior * math.sqrt(parent_visits)
            return u_score # Q score is 0 initially
        else:
            # Q(s,a) + U(s,a)
            q_score = node.value # Mean action value
            u_score = self.exploration_constant * node.prior * math.sqrt(parent_visits) / (1 + node.visit_count)
            return q_score + u_score

    def _select(self, node: MCTSNode, env: BaseEnvironment) -> Tuple[MCTSNode, BaseEnvironment]:
        """Select child node with highest PUCT score until a leaf node is reached."""
        while node.is_expanded() and not env.is_game_over():
            parent_visits = node.visit_count
            # Select the action corresponding to the child with the highest PUCT score
            action, node = max(
                (
                    (action, child) for action, child in node.children.items()
                ),
                key=lambda item: self._puct_score(item[1], parent_visits)
            )
            # Action here is the key from children dict, should be hashable
            env.step(action) # Update the environment state as we traverse
        return node, env

    def _expand_and_evaluate(self, node: MCTSNode, env: BaseEnvironment) -> float:
        """
        Expand the leaf node using the network, evaluate the position, and return the value.
        """
        if self.network is None:
            raise ValueError("MCTS requires a network for AlphaZero-style expansion and evaluation.")

        current_state_obs = env.get_observation()
        legal_actions = env.get_legal_actions()

        if not legal_actions: # Should not happen if not env.is_game_over() was checked before calling
             print("Warning: Expanding node with no legal actions despite not being game over.")
             return 0.0 # Or handle based on game rules (e.g., loss for current player)


        # Get policy priors and value from the network
        # Ensure network is on correct device and input is formatted correctly
        # TODO: Add device handling
        policy_logits_tensor, value_tensor = self.network(current_state_obs) # Use the forward method

        # Process policy logits
        policy_logits = policy_logits_tensor.squeeze(0).cpu().detach().numpy()
        # Apply softmax? Network might already do it. Assume logits for now.
        # policy_probs = torch.softmax(policy_logits_tensor, dim=1).squeeze(0).cpu().detach().numpy()

        # Process value
        value = value_tensor.squeeze(0).cpu().item() # Get scalar value

        # --- Map policy outputs to legal actions ---
        # This requires a consistent mapping between the network's policy output vector
        # and the environment's legal actions. This is a CRITICAL part.
        # We need helper functions in the network or environment to handle this mapping.
        # For now, let's assume a placeholder function `_get_policy_for_legal_actions` exists.

        action_priors = self._get_policy_for_legal_actions(policy_logits, legal_actions, env)

        # Expand the node with legal actions and their priors
        node.expand(action_priors)

        # Return the network's value estimate for this state (from the perspective of the current player in env)
        return value

    def _get_policy_for_legal_actions(self, policy_output: np.ndarray, legal_actions: List[ActionType], env: BaseEnvironment) -> Dict[ActionType, float]:
        """
        Placeholder: Maps raw network policy output to priors for legal actions.
        This needs a concrete implementation based on the environment and network structure.
        """
        # Example: Assume policy_output is a vector for all possible actions
        # and we need to mask illegal actions and re-normalize.
        action_priors = {}
        policy_sum_legal = 0.0

        # Create a mapping from action to index (this should be efficient)
        # This mapping logic MUST match AlphaZeroNet._calculate_policy_size
        action_to_index_map = self._create_action_index_map(env) # Needs implementation

        if not action_to_index_map:
             print("Warning: Action-to-index map is empty. Cannot assign policy priors.")
             # Fallback: Uniform priors for legal actions
             if legal_actions:
                 prior = 1.0 / len(legal_actions)
                 return {action: prior for action in legal_actions}
             else:
                 return {}


        legal_indices = []
        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action
            idx = action_to_index_map.get(action_key)
            if idx is not None and 0 <= idx < len(policy_output):
                legal_indices.append(idx)
                # Use softmax on logits for legal actions only? Or use pre-softmax values?
                # AlphaZero paper uses softmax over all actions, then selects legal ones.
                # Let's assume policy_output are logits.
                action_priors[action_key] = float(np.exp(policy_output[idx])) # Use float() to ensure type
                policy_sum_legal += action_priors[action_key]
            else:
                 print(f"Warning: Action {action_key} not found in map or index out of bounds.")


        # Normalize the priors for legal actions
        if policy_sum_legal > 1e-6: # Avoid division by zero
            for action in action_priors:
                action_priors[action] /= policy_sum_legal
        elif legal_actions:
             # Fallback to uniform if all legal actions had zero or negative logits
             print("Warning: Sum of exp(logits) for legal actions is near zero. Using uniform priors.")
             prior = 1.0 / len(legal_actions)
             action_priors = {action: prior for action in legal_actions}
        else:
             action_priors = {} # No legal actions

        # Add Dirichlet noise here if training

        return action_priors

    def _create_action_index_map(self, env: BaseEnvironment) -> Dict[ActionType, int]:
        """
        Placeholder: Creates a map from hashable action representation to policy index.
        Needs specific implementation per environment type.
        """
        # Example for FourInARow (assuming policy is flattened board)
        if hasattr(env, 'board_size') and isinstance(env, BaseEnvironment):
             size = env.board_size
             # Assuming action is (row, col)
             return {(r, c): r * size + c for r in range(size) for c in range(size)}
        # Example for Nim (more complex, depends on max items/piles)
        elif hasattr(env, 'initial_piles') and isinstance(env, BaseEnvironment):
             # This needs a robust mapping based on max possible removals per pile
             # Let's use a simple sequential index for now, assuming a fixed max state space
             # This is NOT robust and needs proper implementation matching the network output size.
             print("Warning: Using placeholder action indexing for Nim. Needs proper implementation.")
             max_piles = len(env.initial_piles)
             # Estimate max items based on initial state (highly approximate)
             max_items = max(env.initial_piles) if env.initial_piles else 1
             mapping = {}
             idx = 0
             for p_idx in range(max_piles):
                 for n_items in range(1, max_items + 1): # Assume max removal = max items
                     mapping[(p_idx, n_items)] = idx
                     idx += 1
             # Check if calculated size matches network's expected size
             expected_size = self.network._calculate_policy_size(env) if self.network else -1
             if idx != expected_size:
                 print(f"Warning: Nim action map size ({idx}) doesn't match expected network policy size ({expected_size}).")
             return mapping

        else:
             print("Warning: Cannot create action index map for this environment type.")
             return {}


    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the value estimate up the tree."""
        current_node = node
        # Value is from the perspective of the player whose turn it was *at the evaluated node*.
        # As we go up, the perspective flips for the parent.
        while current_node is not None:
            current_node.visit_count += 1
            # Update total value (W). The value should be relative to the player whose turn it is *at the parent node*.
            # Since 'value' is from the child's perspective, we add it directly if the parent is the same player,
            # or negate it if the parent is the opponent. In zero-sum games, we negate at each step up.
            current_node.total_value += value
            value = -value  # Flip value for the parent (opponent's perspective)
            current_node = current_node.parent

    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode:
        """Run MCTS search using the neural network."""
        if self.network is None:
            raise ValueError("MCTS search requires a network for AlphaZero.")

        # Set the root to a new node corresponding to the current state
        # We might want to reuse the tree if possible, but for now, start fresh.
        self.reset_root() # Start fresh search for each call

        # The root node represents the state *before* any action is taken in this search.
        # We need an initial evaluation of the root state if it's not expanded.
        # However, the expansion happens during the first simulation's selection phase.

        for _ in range(self.num_simulations):
            # Start from the root node and a copy of the environment set to the initial state
            node = self.root
            sim_env = env.copy()
            sim_env.set_state(state) # Ensure simulation starts from the correct state

            # 1. Selection: Traverse the tree using PUCT until a leaf node is reached
            node, sim_env = self._select(node, sim_env)

            # 2. Expansion & Evaluation: If the game is not over at the leaf node,
            #    expand it using the network and get the value estimate.
            value = 0.0 # Default value
            if not sim_env.is_game_over():
                # Expand the node, get policy priors for children, and evaluate the current state (sim_env)
                value = self._expand_and_evaluate(node, sim_env)
                # Value is from the perspective of the player whose turn it is in sim_env (the node just expanded)
            else:
                # Game ended during selection phase. Determine the outcome.
                winner = sim_env.get_winning_player()
                # The value should be from the perspective of the player whose turn it *would* have been
                # at this terminal node (which is sim_env.get_current_player()).
                player_at_terminal_node = sim_env.get_current_player()

                if winner is None:  # Draw
                    value = 0.0
                # If the player whose turn it is at the terminal node is the winner
                elif winner == player_at_terminal_node:
                    # This seems counter-intuitive for backpropagation.
                    # Let's define value from the perspective of the *parent* node's player.
                    # The parent player made the move leading to this terminal state.
                    # If the parent's move resulted in a win for the parent (winner != player_at_terminal_node), value is +1.
                    # If the parent's move resulted in a loss for the parent (winner == player_at_terminal_node), value is -1.
                    value = -1.0 # Parent lost
                else: # Opponent won (parent won)
                    value = 1.0 # Parent won

                # Note: The backpropagation negates this value for the node itself.
                # Let's rethink: Value should be from the perspective of the player whose turn it is *at the node being evaluated*.
                # If node is terminal:
                #   If draw: value = 0
                #   If player P (whose turn it would be) won: value = +1 (impossible state, P already won)
                #   If player P lost: value = -1 (opponent won on previous turn)
                # Let's use the perspective of the player who *just moved* to reach this state.
                # The player who just moved is `1 - sim_env.get_current_player()` (for 2 players).
                player_who_just_moved = (sim_env.get_current_player() + sim_env.num_players -1) % sim_env.num_players

                if winner is None: # Draw
                    value = 0.0
                elif winner == player_who_just_moved: # Player who moved won
                    value = 1.0
                else: # Player who moved lost
                    value = -1.0
                # This value is now from the perspective of the player who made the move into this state.
                # Backpropagation needs to handle alternating perspectives correctly.

            # 3. Backpropagation: Update visit counts and values up the tree
            self._backpropagate(node, value)

        return self.root
