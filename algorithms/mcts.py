import math
import random
import time
import contextlib
from typing import List, Tuple, Optional, Dict

import torch

from core.config import MuZeroConfig
from environments.base import ActionType, StateType, BaseEnvironment
from models.networks import AlphaZeroNet, MuZeroNet

SHOULD_PROFILE = True


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


# Todo, please replace the if debug prints below with loguru


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
        debug: bool = False,
    ):
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.num_simulations = num_simulations
        self.debug = debug
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
            best_action = max(child_scores, key=child_scores.get)
            if self.debug:
                print(
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
        while not sim_env.is_game_over():
            legal_actions = sim_env.get_legal_actions()
            if not legal_actions:
                print("WARN: Game not over, but no legal actions")
                break
            action = random.choice(legal_actions)
            sim_env.step(action)

        winner = sim_env.get_winning_player()
        if winner is None:
            value = 0.0
        elif winner == player_at_rollout_start:
            value = 1.0
        else:
            value = -1.0
        if self.debug:
            print(
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
            if self.debug:
                print(
                    f"  Backprop: Node={current_node}, ValueToAdd={value_for_current_node:.3f}, OldW={current_node.total_value:.3f}, OldN={current_node.visit_count - 1}",
                    end="",
                )
            current_node.total_value += value_for_current_node
            if self.debug:
                print(
                    f", NewW={current_node.total_value:.3f}, NewN={current_node.visit_count}"
                )

            # Flip the value perspective for the parent node.
            value_for_current_node *= -1
            current_node = current_node.parent

    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode:
        """Run MCTS search from the given state using UCB1 and random rollouts."""
        self.reset_root()

        total_network_time_ms = 0.0

        if SHOULD_PROFILE:
            search_timer = Timer()
        else:
            search_timer = contextlib.nullcontext()

        with search_timer:
            if self.debug:
                print(f"--- MCTS Search Start: State={state} ---")

            for sim_num in range(self.num_simulations):
                if self.debug:
                    print(f" Simulation {sim_num+1}/{self.num_simulations}")
                # Start from the root node and a copy of the environment set to the initial state
                sim_env = env.copy()
                sim_env.set_state(
                    state
                )  # Ensure simulation starts from the correct state

                # 1. Selection: Find a leaf node using UCB1.
                leaf_node, leaf_env_state = self._select(self.root, sim_env)

                # 2. Evaluation: Get the value of the leaf node.
                if not leaf_env_state.is_game_over():
                    if not leaf_node.is_expanded():
                        # Expand with network timing if profiling
                        if SHOULD_PROFILE:
                            with Timer() as net_timer:
                                self._expand(leaf_node, leaf_env_state)
                            total_network_time_ms += net_timer.elapsed_ms
                        else:
                            self._expand(leaf_node, leaf_env_state)

                        # Rollout with network timing if profiling
                        if SHOULD_PROFILE:
                            with Timer() as net_timer:
                                value = self._rollout(leaf_env_state)
                            total_network_time_ms += net_timer.elapsed_ms
                        else:
                            value = self._rollout(leaf_env_state)
                else:
                    # Game ended during selection. Determine the outcome.
                    # The value must be from the perspective of the player whose turn it was AT THE LEAF node.
                    player_at_leaf = sim_env.get_current_player()
                    winner = sim_env.get_winning_player()

                    if winner is None:
                        value = 0.0
                    elif winner == player_at_leaf:
                        value = 1.0
                    else:
                        value = -1.0
                    if self.debug:
                        print(
                            f"  Terminal Found: Winner={winner}, PlayerAtNode={player_at_leaf}, Value={value}"
                        )
                # 3. Backpropagation: Update nodes along the path from the leaf to the root.
                self._backpropagate(leaf_node, value)

        if SHOULD_PROFILE and isinstance(search_timer, Timer):
            search_time_ms = search_timer.elapsed_ms
            if search_time_ms > 1e-6:
                non_network_time_ms = search_time_ms - total_network_time_ms
                network_perc = (total_network_time_ms / search_time_ms) * 100
                print(
                    f"[Profile AZ MCTS Search]: Total={search_time_ms:.2f}ms, "
                    f"Network={total_network_time_ms:.2f}ms ({network_perc:.1f}%), "
                    f"Other={non_network_time_ms:.2f}ms "
                    f"({self.num_simulations} sims)"
                )
            else:
                print(
                    f"[Profile AZ MCTS Search]: Total={search_time_ms:.2f}ms (negligible)"
                )

        return self.root


class AlphaZeroMCTS(MCTS):
    """MCTS algorithm adapted for AlphaZero (PUCT + Network Evaluation)."""

    def __init__(
        self,
        exploration_constant: float = 1.0,  # c_puct in AlphaZero
        num_simulations: int = 100,
        network: AlphaZeroNet = None,
        discount_factor: float = 1.0,  # Usually 1.0 for AlphaZero MCTS value (consistency)
        # TODO: Add dirichlet_epsilon and dirichlet_alpha for root noise
    ):
        super().__init__(
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,  # Keep for consistency, though not used in backprop value calc
            num_simulations=num_simulations,
        )
        self.profile = profile
        self.network = network
        # self.dirichlet_epsilon = dirichlet_epsilon
        # self.dirichlet_alpha = dirichlet_alpha

    def _expand(self, node: MCTSNode, env: BaseEnvironment):
        """
        Expand the leaf node using policy predictions from the network
        for legal actions.
        """
        if node.is_expanded() or env.is_game_over():
            return

        if not self.network:
            # Fallback to uniform priors if no network is provided (though not typical for AZ)
            super()._expand(node, env)
            return

        # Get network prediction for the current state
        state_obs = env.get_observation()
        policy_np, _ = self.network.predict(
            state_obs
        )  # Value prediction not needed here

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return  # Cannot expand if no legal actions

        action_priors = {}
        for action in legal_actions:
            action_key = tuple(action) if isinstance(action, list) else action
            action_index = self.network.get_action_index(action_key)

            if action_index is not None and 0 <= action_index < len(policy_np):
                prior = policy_np[action_index]
                action_priors[action_key] = prior
            else:
                # This might happen if the network's policy size doesn't perfectly
                # match all possible actions or if mapping fails. Assign zero prior.
                # print(f"Warning: Could not map legal action {action_key} to policy index during expansion.")
                action_priors[action_key] = 0.0

        if self.debug:
            sorted_priors = sorted(
                action_priors.items(), key=lambda item: item[1], reverse=True
            )
            print(
                f"  Expand: State={state_obs.get('board', state_obs.get('piles', 'N/A'))}, Player={state_obs['current_player']}"
            )
            print(f"  Expand: Legal Actions={legal_actions}")
            print(
                f"  Expand: Network Priors (Top 5): { {a: f'{p:.3f}' for a, p in sorted_priors[:5]} }"
            )

        node.expand(action_priors)

    def _rollout(self, env: BaseEnvironment) -> float:
        """
        Evaluate the leaf node state using the network's value prediction.
        Overrides the random rollout simulation of the base MCTS.
        """
        if not self.network:
            # Fallback to random rollout if no network (shouldn't happen in typical AZ)
            return super()._rollout(env)

        # Get the value prediction from the network for the current state
        state_obs = env.get_observation()
        _, value = self.network.predict(state_obs)

        # The network's value prediction is from the perspective of the current player
        # at the leaf node, which is what backpropagation expects.
        if self.debug:
            print(
                f"  Evaluate (NN): State={state_obs.get('board', state_obs.get('piles', 'N/A'))}, Player={state_obs['current_player']}, Value={value:.3f}"
            )

        return value

    def _score_child(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate the PUCT score for a node."""
        # Use self.exploration_constant as c_puct
        if node.visit_count == 0:
            # If node hasn't been visited, U is based only on prior and parent visits
            # Ensure parent_visits > 0 for sqrt
            q_value_for_parent = 0
        else:
            # Q(s,a)[value] + U(s,a)[exploration]
            # Q(s,a) must be from the perspective of the player selecting at the PARENT node.
            # node.value (q_score) is from the perspective of the player AT the node (the opponent).
            # So, we use -node.value for the parent's perspective.
            q_value_for_parent = -node.value
        u_score = (
            self.exploration_constant
            * node.prior
            * math.sqrt(parent_visits)
            / (1 + node.visit_count)
        )
        return q_value_for_parent + u_score


# --- MuZero MCTS ---
# Note: This is a simplified structure. A full MuZero MCTS often involves
# slightly different node structures (storing hidden state, reward) and search logic.


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
        # Store policy logits and value prediction when node is first evaluated? Optional.
        # self.policy_logits = None
        # self.value = None

    @property
    def value(self) -> float:  # Q(s,a) - Mean action value
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)

    def expand(self, policy_logits: torch.Tensor, network: MuZeroNet):
        """
        Expand node using policy predictions from the network.
        Creates child nodes for all actions in the policy output, using the predicted priors.
        Does NOT filter by legality from the environment rules.
        """
        policy_probs = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
        # Assumes policy_logits covers the entire potential action space defined by the network
        # We need to map indices back to ActionType to use as keys.
        # Requires an inverse mapping function in the network.
        if not hasattr(network, "get_action_from_index"):
            print(
                "ERROR: MuZeroNet needs get_action_from_index method for MCTS expansion."
            )
            return  # Cannot expand without inverse mapping

        for action_index, prior in enumerate(policy_probs):
            if prior > 1e-6:  # Small threshold
                action = network.get_action_from_index(action_index)
                if action is not None:
                    action_key = tuple(action) if isinstance(action, list) else action
                    if action_key not in self.children:
                        self.children[action_key] = MuZeroMCTSNode(
                            parent=self, prior=float(prior)
                        )
                else:
                    # This might happen if policy size > actual max actions
                    # print(f"Warning: Could not map action index {action_index} back to action.")
                    pass


class MuZeroMCTS:
    """MCTS implementation adapted for MuZero's learned model."""

    def __init__(
        self,
        config: MuZeroConfig,
        network: MuZeroNet,
    ):
        self.config = config
        self.network = network
        # Root node doesn't have a prior action, hidden state comes from representation(obs)
        self.root = MuZeroMCTSNode(prior=0.0)

    def reset_root(self):
        self.root = MuZeroMCTSNode(prior=0.0)

    def _puct_score(
        self, node: MuZeroMCTSNode, parent_visits: int, discount: float = 0.99
    ) -> float:
        """
        Calculate the PUCT score for a node in MuZero MCTS.
        Uses G + discount * Q(s', a') formulation.
        """
        # Calculate Q-value (mean action value from this node onwards)
        q_value = (
            node.value
        )  # Average value accumulated from visits passing through this node

        # Calculate U-value (exploration bonus)
        if node.visit_count == 0:
            u_value = self.config.cpuct * node.prior * math.sqrt(max(1, parent_visits))
        else:
            u_value = (
                self.config.cpuct
                * node.prior
                * math.sqrt(parent_visits)
                / (1 + node.visit_count)
            )

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
            key=lambda item: self._puct_score(item[1], parent_visits),
        )
        return best_item  # Returns (action, child_node)

    def _backpropagate(
        self, node: MuZeroMCTSNode, value: float, discount: float = 0.99
    ):
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
            value = current_node.reward + discount * value  # G = R + gamma*V

            # Flip perspective for parent? MuZero often uses value relative to current player.
            # If value is always relative to the player whose turn it is at the node, no flipping needed?
            # Let's assume value is relative to the player at the node. Backprop needs care.

            # Let's use AlphaZero backprop for now for simplicity, assuming zero-sum game.
            # We'll need to revisit this if rewards aren't zero-sum or if using MuZero's value definition.
            value = -value  # Flip for parent perspective in zero-sum game

            current_node = current_node.parent

    # TODO: This search implementation is a placeholder and needs significant refinement
    # based on the MuZero paper / reference implementation, especially regarding:
    # - Handling legal actions (either assuming all policy outputs are legal in latent space,
    #   or predicting a mask).
    # - Correct backpropagation incorporating rewards and value estimates.
    # - Potentially different node selection criteria (e.g., PUCT variant).
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
        # --- Profiling Setup ---
        total_network_time_ms = 0.0
        search_timer = Timer() if SHOULD_PROFILE else contextlib.nullcontext()
        # --- End Profiling Setup ---

        with search_timer:
            # Initial step: Get hidden state and initial prediction from observation
            if SHOULD_PROFILE:
                with Timer() as net_timer:
                    (
                        value,
                        sim_num,
                        policy_logits,
                        hidden_state,
                    ) = self.network.initial_inference(observation_dict)
                total_network_time_ms += net_timer.elapsed_ms
            else:
                (
                    value,
                    sim_num,
                    policy_logits,
                    hidden_state,
                ) = self.network.initial_inference(observation_dict)
        self.root.hidden_state = hidden_state

        # Use the provided legal actions for the root expansion
        if not legal_actions:
            print("Warning: No legal actions provided for MuZero MCTS root.")
            # If root has no legal actions, MCTS cannot proceed meaningfully.
            # Return the unexpanded root. The agent needs to handle this.
            return self.root

        self.root.expand(legal_actions, policy_logits, self.network)
        # TODO: Add Dirichlet noise to root priors here if training

        # Run simulations
        for sim_num in range(self.config.num_simulations):
            node = self.root
            search_path = [node]  # Keep track of nodes visited

            # 1. Selection: Traverse tree using PUCT until a leaf node is reached
            while node.is_expanded():
                action, node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion & Evaluation: Expand leaf node, get prediction from network
            parent = search_path[-2]  # Node from which the leaf was selected
            leaf_node = search_path[-1]

            # Use the parent's hidden state and the chosen action to get next state and reward via dynamics
            # This requires the action index taken to reach the leaf node.
            # Since children keys are ActionType, we need to find the action.
            action_taken = None
            for act, child in parent.children.items():
                if child == leaf_node:
                    action_taken = act
                    break

            if action_taken is None:
                # Should not happen if selection works correctly
                print(
                    "Error: Could not find action leading to leaf node during MuZero search."
                )
                continue

            # Infer next state, reward, policy, value using the learned model
            # Profile recurrent inference if enabled
            if SHOULD_PROFILE:
                with Timer() as net_timer:
                    (
                        value,
                        reward,
                        policy_logits,
                        next_hidden_state,
                    ) = self.network.recurrent_inference(
                        parent.hidden_state, action_taken
                    )
                total_network_time_ms += net_timer.elapsed_ms
            else:
                (
                    value,
                    reward,
                    policy_logits,
                    next_hidden_state,
                ) = self.network.recurrent_inference(parent.hidden_state, action_taken)

            # Store results in the newly reached (or created) leaf node
            leaf_node.hidden_state = next_hidden_state
            leaf_node.reward = reward.item()  # Store reward obtained reaching this node

            # Expand the leaf node using the predicted policy from the network.
            # The expand method handles creating children based on the policy logits.
            # It does NOT filter based on real environment legality here.
            leaf_node.expand(policy_logits, self.network)

            # --- Backpropagation ---
            # Backpropagate the predicted value V(s_leaf)
            self._backpropagate(leaf_node, value.item())  # Pass scalar value

        # Return the root node containing search statistics
        return self.root
