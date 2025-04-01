import random
from pathlib import Path
from collections import deque
from typing import List

import torch
import torch.optim as optim  # Import optimizer
import torch.nn.functional as F
import numpy as np

from core.agent_interface import Agent
from environments.base import BaseEnvironment, StateType, ActionType
from algorithms.mcts import AlphaZeroMCTS  # Use the specialized MCTS subclass
from models.networks import AlphaZeroNet
from core.config import AlphaZeroConfig, DATA_DIR


class AlphaZeroAgent(Agent):
    """Agent implementing the AlphaZero algorithm."""

    def __init__(self, env: BaseEnvironment, config: AlphaZeroConfig):
        """
        Initialize the AlphaZero agent.

        Args:
            env: The environment instance.
            config: Configuration object with AlphaZero parameters.
        """
        self.env = env
        self.config = config

        # Initialize the neural network
        self.network = AlphaZeroNet(
            env,
            hidden_layer_size=config.hidden_layer_size,
            num_hidden_layers=config.num_hidden_layers,
        )
        # TODO: Add device handling (CPU/GPU)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.network.to(self.device)

        # Initialize AlphaZeroMCTS
        self.mcts = AlphaZeroMCTS(
            exploration_constant=config.cpuct,  # Use cpuct for PUCT formula
            num_simulations=config.num_simulations,
            network=self.network,  # Pass the network to MCTS
            # TODO: Add dirichlet noise parameters from config if/when implemented
        )

        # Experience buffer for training (stores (state, policy_target, value))
        # TODO: Make buffer size configurable
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        # Temporary storage for the current game's history before assigning final outcome
        self._current_episode_history = []

        # Optimizer (managed internally by the agent)
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # TODO: Add scheduler if needed, managed here too

    def act(self, state: StateType, train: bool = False) -> ActionType:
        """
        Choose an action using MCTS guided by the neural network.
        If train=True, also stores the state and MCTS policy target for learning.

        Args:
            state: The current environment state observation.
            train: If True, use temperature sampling for exploration during training.
                   If False, choose the most visited node (greedy).

        Returns:
            The chosen action.
        """
        # Ensure network is in evaluation mode for inference
        self.network.eval()

        # --- Optional Debug Print: Network Prediction ---
        if self.config.debug_mode:
            try:
                policy_np, value_np = self.network.predict(state)
                print(
                    f"[DEBUG Act] State: {state.get('board', state.get('piles', 'N/A'))}"
                )
                print(f"[DEBUG Act] To Play: {state['current_player']}")
                print(f"[DEBUG Act] Network Value Prediction: {value_np:.4f}")
                legal_actions = self.env.get_legal_actions()
                action_probs = {}
                for action in legal_actions:
                    idx = self.network.get_action_index(action)
                    if idx is not None and 0 <= idx < len(policy_np):
                        action_probs[action] = policy_np[idx]
                sorted_probs = sorted(
                    action_probs.items(), key=lambda item: item[1], reverse=True
                )
                top_k = 5
                print(f"[DEBUG Act] Network Policy Priors (Top {top_k} Legal):")
                for action, prob in sorted_probs[:top_k]:
                    print(f"  - {action}: {prob:.4f}")
            except Exception as e:
                print(f"[DEBUG Act] Error during network predict debug: {e}")

        # Run MCTS search from the current state
        # The modified MCTS search will use the network
        root_node = self.mcts.search(self.env, state)

        if not root_node.children:
            print(
                "Warning: MCTS root has no children after search. Choosing random action."
            )
            temp_env = self.env.copy()
            temp_env.set_state(state)
            legal_actions = temp_env.get_legal_actions()
            return random.choice(legal_actions) if legal_actions else None

        # Select action based on visit counts
        visit_counts = np.array(
            [child.visit_count for child in root_node.children.values()]
        )
        actions = list(root_node.children.keys())

        if train:
            # Temperature sampling for exploration during training
            # TODO: Implement temperature logic (e.g., temp=1 for first N moves, then 0)
            temperature = 1.0  # Placeholder
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            action_probs = visit_counts_temp / np.sum(visit_counts_temp)
            chosen_action_index = np.random.choice(len(actions), p=action_probs)
        else:
            # Greedy selection for evaluation/play
            chosen_action_index = np.argmax(visit_counts)

        chosen_action = actions[chosen_action_index]

        # --- Optional Debug Print: MCTS Results ---
        if self.config.debug_mode:
            print(f"[DEBUG Act] MCTS Visit Counts:")
            sorted_visits = sorted(
                zip(actions, visit_counts), key=lambda item: item[1], reverse=True
            )
            for action, visits in sorted_visits:
                print(f"  - {action}: {visits}")
            print(f"[DEBUG Act] Chosen Action (Train={train}): {chosen_action}")

        # --- Store data for training if in training mode ---
        if train:
            # Calculate policy target based on visit counts
            policy_target = self._calculate_policy_target(
                root_node, actions, visit_counts
            )
            # Store state, chosen action, and policy target temporarily.
            # The final outcome (value) will be added later in finish_episode.
            # Ensure state is copied or immutable if necessary (current state dicts seem okay)
            self._current_episode_history.append(
                (state, chosen_action, policy_target)  # Store as tuple
            )

        return chosen_action

    def _calculate_policy_target(self, root_node, actions, visit_counts) -> np.ndarray:
        """Calculates the policy target vector based on MCTS visit counts."""
        policy_size = self.network._calculate_policy_size(self.env)
        policy_target = np.zeros(policy_size, dtype=np.float32)
        total_visits = np.sum(visit_counts)

        if total_visits > 0:
            # Use the network's action mapping
            for i, action in enumerate(actions):
                action_key = tuple(action) if isinstance(action, list) else action
                # Get index from the network
                action_idx = self.network.get_action_index(action_key)

                if action_idx is not None and 0 <= action_idx < policy_size:
                    policy_target[action_idx] = visit_counts[i] / total_visits
                else:
                    # This warning might still occur if get_action_index fails for a valid action
                    print(
                        f"Warning: Action {action_key} could not be mapped to index during policy target calculation."
                    )
        else:
            # Handle case with no visits (should be rare if search runs)
            # Maybe return uniform distribution over legal actions? Or zeros?
            print(
                "Warning: No visits recorded in MCTS root. Policy target will be zeros."
            )

        return policy_target

    def finish_episode(self, final_outcome: float):
        """
        Called at the end of a self-play episode. Assigns the final outcome
        to all steps in the temporary history, adds the necessary data to the
        replay buffer, and returns the complete game history for logging.

        Args:
            final_outcome: The outcome for player 0 (+1 win, -1 loss, 0 draw).

        Returns:
            List[Tuple[StateType, ActionType, np.ndarray, float]]: The full game history,
                where each tuple is (state, action, policy_target, value_target).
        """
        processed_history = []
        num_steps = len(self._current_episode_history)

        for i, (state_at_step, action_taken, policy_target) in enumerate(
            self._current_episode_history
        ):
            # Determine the value target from the perspective of the player at that state
            player_at_step = state_at_step.get("current_player", -1)

            if player_at_step == 0:
                value_target = final_outcome
            elif player_at_step == 1:
                value_target = -final_outcome  # Flip outcome for opponent
            else:
                print(
                    f"Warning: Unknown player {player_at_step} at step {i}. Assigning value target 0.0."
                )
                value_target = 0.0

            # --- Standardize state before adding to buffer ---
            # Ensure board/piles are numpy arrays for consistent network input processing
            buffer_state = (
                state_at_step.copy()
            )  # Avoid modifying original history state
            if "board" in buffer_state and not isinstance(
                buffer_state["board"], np.ndarray
            ):
                buffer_state["board"] = np.array(
                    buffer_state["board"], dtype=np.int8
                )  # Or appropriate dtype
            elif "piles" in buffer_state and not isinstance(
                buffer_state["piles"], np.ndarray
            ):
                # NimEnv currently returns tuple, convert it
                buffer_state["piles"] = np.array(buffer_state["piles"], dtype=np.int32)

            # Add the experience tuple (standardized_state, policy_target, value_target) to replay buffer
            self.replay_buffer.append((buffer_state, policy_target, value_target))

            # Store the original step info (without modification) for the returned history log
            processed_history.append(
                (state_at_step, action_taken, policy_target, value_target)
            )

        # Clear the temporary history *after* processing
        self._current_episode_history = []

        return processed_history

    def _calculate_loss(
        self, policy_logits, value_preds, policy_targets, value_targets
    ):
        """Calculates the combined loss for AlphaZero."""
        # Value loss: Mean Squared Error
        value_loss = F.mse_loss(value_preds, value_targets)

        # Policy loss: Cross-Entropy between predicted policy logits and MCTS policy target
        # Ensure targets are probabilities (sum to 1) and logits are raw scores
        # CrossEntropyLoss expects logits as input and class indices or probabilities as target.
        # Here, policy_targets are probabilities derived from MCTS visits.
        policy_loss = F.cross_entropy(policy_logits, policy_targets)

        # Combine losses (typically simple summation)
        total_loss = value_loss + policy_loss
        return total_loss, value_loss, policy_loss

    # Conforms to Agent interface (no arguments)
    def learn(self):
        """
        Update the neural network by sampling from the internal replay buffer.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            # print(f"Skipping learn step: Buffer size {len(self.replay_buffer)} < Batch size {self.config.batch_size}")
            return None  # Return None if no learning happened

        # 1. Sample a batch from self.replay_buffer.
        batch = random.sample(self.replay_buffer, self.config.batch_size)

        # 2. Format data into batches (states, target policies, target values).
        #    Need to handle device placement (e.g., self.device) if using GPU.
        #    The network's _flatten_state expects a single state dict.
        #    We need to process each state individually first, then stack.
        states_list = [s for s, _, _ in batch]
        policy_targets_list = [
            torch.tensor(p, dtype=torch.float32) for _, p, _ in batch
        ]
        value_targets_list = [
            torch.tensor([v], dtype=torch.float32) for _, _, v in batch
        ]  # Ensure value target is shape [1]

        # Flatten states and stack into a batch tensor
        # TODO: Add device handling (e.g., .to(self.device))
        states_batch = torch.stack(
            [self.network._flatten_state(s) for s in states_list]
        )
        policy_targets_batch = torch.stack(policy_targets_list)
        value_targets_batch = torch.stack(value_targets_list)

        # 3. Perform gradient descent step on the network.
        self.network.train()  # Ensure network is in training mode
        self.optimizer.zero_grad()  # Use the agent's internal optimizer

        # Forward pass - network needs to handle batch input
        policy_logits, value_preds = self.network(states_batch)  # Pass the batch tensor

        # Calculate loss
        total_loss, value_loss, policy_loss = self._calculate_loss(
            policy_logits, value_preds, policy_targets_batch, value_targets_batch
        )

        # Backward pass and optimizer step
        total_loss.backward()
        self.optimizer.step()  # Use the agent's internal optimizer

        # Optional: Log losses
        if self.config.debug_mode:
            print(
                f"[DEBUG Learn] Step Losses: Total={total_loss.item():.4f}, Value={value_loss.item():.4f}, Policy={policy_loss.item():.4f}"
            )

        self.network.eval()  # Switch back to eval mode after training step

        # Return losses for monitoring
        return total_loss.item(), value_loss.item(), policy_loss.item()

    def _get_save_path(self) -> Path:
        """Constructs the save file path for the network weights."""
        env_type_name = type(self.env).__name__
        filename = f"alphazero_net_{env_type_name}.pth"
        return DATA_DIR / filename

    def save(self) -> None:
        """Save the neural network weights."""
        filepath = self._get_save_path()
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.network.state_dict(), filepath)
            print(f"AlphaZero network saved to {filepath}")
        except Exception as e:
            print(f"Error saving AlphaZero network to {filepath}: {e}")

    def load(self) -> bool:
        """Load the neural network weights."""
        filepath = self._get_save_path()
        try:
            if filepath.exists():
                # Load state dict, handling potential device mismatch
                # map_location = self.device # Use if device handling is added
                map_location = torch.device("cpu")  # Load to CPU first
                self.network.load_state_dict(
                    torch.load(filepath, map_location=map_location)
                )
                # self.network.to(self.device) # Move to target device if needed
                self.network.eval()  # Set to evaluation mode after loading
                print(f"AlphaZero network loaded from {filepath}")
                return True
            else:
                print(f"Network weights file not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error loading AlphaZero network from {filepath}: {e}")
            return False

    def reset(self) -> None:
        """Reset agent state (e.g., MCTS tree, episode history)."""
        self.mcts.reset_root()
        self._current_episode_history = []  # Clear temp history on reset

    # --- Helper methods (potentially needed later) ---
    # TODO: Move action mapping logic here or into network/utils
    # def _map_action_to_policy_index(self, action: ActionType) -> Optional[int]:
    #     """Maps an environment action to the corresponding index in the policy vector."""
    #     # This needs to be implemented based on how actions are structured
    #     # and how the policy head is defined in AlphaZeroNet.
    #     # Example for a grid game: return action[0] * self.env.board_size + action[1]
    #     # Example for Nim: Needs careful mapping based on max pile size etc.
    #     # Needs to match the logic in AlphaZeroNet._calculate_policy_size
    #     print(f"Warning: _map_action_to_policy_index not implemented for action {action}")
    #     return None
    #
    # def _map_policy_index_to_action(self, index: int) -> Optional[ActionType]:
    #     """Maps a policy vector index back to an environment action."""
    #     # Inverse of the above mapping.
    #     print(f"Warning: _map_policy_index_to_action not implemented for index {index}")
    #     return None
