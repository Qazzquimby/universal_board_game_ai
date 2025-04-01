from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import MuZeroConfig
from environments.base import BaseEnvironment, ActionType


class AlphaZeroNet(nn.Module):
    """
    Simple MLP network for AlphaZero.
    Takes a flattened representation of the environment state.
    Outputs policy logits and a value prediction.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        hidden_layer_size: int = 128,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.env = env

        # Determine input size from environment observation space
        # This assumes the observation contains numerical data that can be flattened.
        # A more robust approach might involve specific preprocessing per environment.
        input_size = self._calculate_input_size(env)

        # Determine policy output size from environment action space
        policy_size = self._calculate_policy_size(env)

        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layer_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.ReLU())

        # Shared network trunk
        self.shared_net = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Linear(hidden_layer_size, policy_size)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_layer_size, 1),
            nn.Tanh(),  # Value prediction between -1 and 1
        )

        print(f"Initialized AlphaZeroNet:")
        print(f"  Input size: {input_size}")
        print(f"  Policy size: {policy_size}")
        print(f"  Hidden layers: {num_hidden_layers} x {hidden_layer_size}")

    def _flatten_state(self, state_dict: dict) -> torch.Tensor:
        """Flattens the state dictionary into a single tensor."""
        flat_tensors = []
        # Process known keys first
        if "board" in state_dict:
            board = state_dict["board"]
            if isinstance(board, np.ndarray):
                flat_tensors.append(torch.from_numpy(board).float().flatten())
            elif torch.is_tensor(board):
                flat_tensors.append(board.float().flatten())
        elif "piles" in state_dict:
            piles = state_dict["piles"]
            if isinstance(piles, tuple):
                flat_tensors.append(torch.tensor(piles, dtype=torch.float32))
            elif torch.is_tensor(piles):
                flat_tensors.append(piles.float())
            elif isinstance(piles, np.ndarray):
                flat_tensors.append(torch.from_numpy(piles).float())

        # Add current player as a feature
        flat_tensors.append(
            torch.tensor([state_dict.get("current_player", -1)], dtype=torch.float32)
        )

        # Concatenate all parts
        if not flat_tensors:
            raise ValueError(
                "Could not extract numerical data from state for network input."
            )

        return torch.cat(flat_tensors)

    def _calculate_input_size(self, env: BaseEnvironment) -> int:
        """Calculates the flattened input size based on a sample observation."""
        obs = env.reset()
        flat_state = self._flatten_state(obs)
        return flat_state.numel()

    def _calculate_policy_size(self, env: BaseEnvironment) -> int:
        """
        Calculates the size of the policy output layer.
        This needs a consistent way to map all possible actions to indices.
        """
        # Simple approach: Use the number of legal actions in the initial state? Risky.
        # Better: Define a maximum possible action space size.
        # For grid games: board_size * board_size
        # For Nim: Max_piles * Max_items_per_pile (approximate, needs refinement)
        # Let's use a placeholder method that needs environment-specific implementation
        # or configuration.
        if hasattr(env, "num_actions"):  # Like in connect4
            return env.num_actions
        elif hasattr(env, "board_size"):  # Fallback for grid games
            return env.board_size * env.board_size
        elif hasattr(env, "initial_piles"):  # Specific logic for Nim
            # Calculate size based on (pile_index, num_removed) pairs
            # Requires knowing the maximum number removable from any pile,
            # which depends on the initial state. Let's use the max initial pile size.
            max_removable = max(env.initial_piles) if env.initial_piles else 1
            num_piles = len(env.initial_piles)
            return num_piles * max_removable
        else:
            raise ValueError(
                f"Cannot determine policy size for environment type: {type(env).__name__}"
            )

    def get_action_index(self, action: ActionType) -> Optional[int]:
        """Maps an environment action to the corresponding index in the policy vector."""
        env_type = type(self.env).__name__.lower()

        if env_type == "connect4":
            return action  # column
        elif env_type == "nimenv":
            # Action is (pile_index, num_to_remove)
            pile_idx, num_removed = action
            if hasattr(self.env, "initial_piles"):
                max_removable = (
                    max(self.env.initial_piles) if self.env.initial_piles else 1
                )
                # Check bounds
                if (
                    0 <= pile_idx < len(self.env.initial_piles)
                    and 1 <= num_removed <= max_removable
                ):
                    # Map (pile_idx, num_removed) to a flat index
                    # Example: index = pile_idx * max_removable + (num_removed - 1)
                    return pile_idx * max_removable + (num_removed - 1)
                else:
                    print(
                        f"Warning: Invalid Nim action {action} for index calculation."
                    )
                    return None  # Indicate failure
            else:
                print("Warning: Cannot get initial_piles from env in get_action_index")
                return None  # Indicate failure
        else:
            print(f"Warning: Action indexing not implemented for env type {env_type}")
            return None  # Indicate failure

    # Optional: Inverse mapping (might be useful later)
    # def get_action_from_index(self, index: int) -> Optional[ActionType]:
    #     """Maps a policy vector index back to an environment action."""
    #     # Implementation would depend on the logic in get_action_index
    #     pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass. Accepts either a single flattened state tensor
        or a batch of flattened state tensors.

        Args:
            x: A tensor representing the flattened state(s).
               Shape [input_size] for single instance, or [batch_size, input_size] for batch.

        Returns:
            A tuple containing (policy_logits, value).
            Shapes: policy_logits [batch_size, policy_size], value [batch_size, 1]
        """
        # Add batch dimension if processing a single instance
        if x.dim() == 1:
            x = x.unsqueeze(0)

        shared_output = self.shared_net(x)
        policy_logits = self.policy_head(shared_output)
        value = self.value_head(shared_output)
        return policy_logits, value

    def predict(self, state_dict: dict) -> Tuple[np.ndarray, float]:
        """
        Convenience method for inference. Runs forward pass and returns numpy arrays.
        Handles moving tensors to CPU and detaching from graph.

        Args:
            state_dict: The environment state observation dictionary.

        Returns:
            A tuple containing (policy_probabilities_numpy, value_numpy).
        """
        self.eval()  # Set model to evaluation mode
        flat_state = self._flatten_state(state_dict)  # Flatten the input dict
        # TODO: Add device handling if needed: flat_state = flat_state.to(self.device)
        with torch.no_grad():
            # Pass the flattened tensor to forward
            policy_logits, value = self.forward(flat_state)

            # Apply softmax to get probabilities
            policy_probs = F.softmax(policy_logits, dim=1)

            # Move to CPU, detach, and convert to numpy
            policy_np = policy_probs.squeeze(0).cpu().numpy()
            value_np = value.squeeze(0).cpu().item()  # Get scalar value

        return policy_np, value_np


# --- MuZero Network ---


class MuZeroNet(nn.Module):
    """
    Network implementing the three core functions of MuZero:
    - Representation (h): Observation -> Hidden State
    - Dynamics (g): (Hidden State, Action) -> (Next Hidden State, Reward)
    - Prediction (f): Hidden State -> (Policy, Value)
    """

    def __init__(
        self,
        env: BaseEnvironment,
        config: MuZeroConfig
        # TODO: Allow separate hidden sizes for different parts?
    ):
        super().__init__()
        self.env = env
        self.config = config

        # --- Determine Sizes ---
        # Input size for representation network (h)
        input_size_h = self._calculate_input_size(env)
        # Action encoding size for dynamics network (g) - e.g., one-hot
        # For simplicity, let's assume action is passed as a scalar index for now
        action_encoding_size = 1  # Or env.num_actions if using one-hot
        # Input size for dynamics network (g) = hidden_state + action_encoding
        input_size_g = config.hidden_state_size + action_encoding_size
        # Input size for prediction network (f) = hidden_state
        input_size_f = config.hidden_state_size
        # Policy output size
        policy_size = self._calculate_policy_size(env)
        # Reward output size (scalar for now, could be categorical later)
        reward_size = 1
        # Value output size (scalar for now, could be categorical later)
        value_size = 1

        # --- Define Networks ---
        # h: Representation Network (Obs -> Hidden State)
        # Using simple MLP for now
        self.representation_net = nn.Sequential(
            nn.Linear(
                input_size_h, config.hidden_state_size
            ),  # Example intermediate layer
            nn.ReLU(),
            nn.Linear(config.hidden_state_size, config.hidden_state_size),
            # TODO: Add normalization? (See reference code scaling)
        )

        # g: Dynamics Network ((Hidden State, Action) -> (Next Hidden State, Reward))
        # Separate heads for state and reward
        self.dynamics_state_net = nn.Sequential(
            nn.Linear(
                input_size_g, config.hidden_state_size
            ),  # Example intermediate layer
            nn.ReLU(),
            nn.Linear(config.hidden_state_size, config.hidden_state_size),
            # TODO: Add normalization?
        )
        self.dynamics_reward_net = nn.Sequential(
            nn.Linear(
                input_size_g, config.hidden_state_size
            ),  # Can share trunk or be separate
            nn.ReLU(),
            nn.Linear(config.hidden_state_size, reward_size),
            # TODO: Add Tanh if reward is scaled to [-1, 1]? Or support for categorical?
        )

        # f: Prediction Network (Hidden State -> (Policy, Value))
        # Separate heads for policy and value
        self.prediction_policy_net = nn.Sequential(
            nn.Linear(
                input_size_f, config.hidden_state_size
            ),  # Example intermediate layer
            nn.ReLU(),
            nn.Linear(config.hidden_state_size, policy_size),
        )
        self.prediction_value_net = nn.Sequential(
            nn.Linear(
                input_size_f, config.hidden_state_size
            ),  # Can share trunk or be separate
            nn.ReLU(),
            nn.Linear(config.hidden_state_size, value_size),
            nn.Tanh(),  # Value prediction between -1 and 1
        )

        print(f"Initialized MuZeroNet:")
        print(f"  Representation Input size: {input_size_h}")
        print(f"  Dynamics Input size: {input_size_g}")
        print(f"  Prediction Input size: {input_size_f}")
        print(f"  Hidden State size: {config.hidden_state_size}")
        print(f"  Policy size: {policy_size}")

    # --- Helper methods (copied/adapted from AlphaZeroNet) ---

    def _flatten_state(self, state_dict: dict) -> torch.Tensor:
        """Flattens the state dictionary into a single tensor."""
        # Reusing the same flattening logic as AlphaZero for representation input
        flat_tensors = []
        if "board" in state_dict:
            board = state_dict["board"]
            if isinstance(board, np.ndarray):
                flat_tensors.append(torch.from_numpy(board).float().flatten())
            elif torch.is_tensor(board):
                flat_tensors.append(board.float().flatten())
        elif "piles" in state_dict:
            piles = state_dict["piles"]
            if isinstance(piles, tuple):
                flat_tensors.append(torch.tensor(piles, dtype=torch.float32))
            elif torch.is_tensor(piles):
                flat_tensors.append(piles.float())
            elif isinstance(piles, np.ndarray):
                flat_tensors.append(torch.from_numpy(piles).float())

        flat_tensors.append(
            torch.tensor([state_dict.get("current_player", -1)], dtype=torch.float32)
        )
        if not flat_tensors:
            raise ValueError(
                "Could not extract numerical data from state for network input."
            )
        return torch.cat(flat_tensors)

    def _calculate_input_size(self, env: BaseEnvironment) -> int:
        """Calculates the flattened input size based on a sample observation."""
        obs = env.reset()
        flat_state = self._flatten_state(obs)
        return flat_state.numel()

    def _calculate_policy_size(self, env: BaseEnvironment) -> int:
        """Calculates the size of the policy output layer."""
        # Reusing the same logic as AlphaZero
        if hasattr(env, "num_actions"):
            return env.num_actions
        elif hasattr(env, "initial_piles"):
            max_removable = max(env.initial_piles) if env.initial_piles else 1
            num_piles = len(env.initial_piles)
            return num_piles * max_removable
        else:
            raise ValueError(
                f"Cannot determine policy size for environment type: {type(env).__name__}"
            )

    def get_action_index(self, action: ActionType) -> Optional[int]:
        """Maps an environment action to the corresponding index in the policy vector."""
        # Reusing the same logic as AlphaZero
        env_type = type(self.env).__name__.lower()
        if env_type == "connect4":
            return action  # column index
        elif env_type == "nimenv":
            pile_idx, num_removed = action
            if hasattr(self.env, "initial_piles"):
                max_removable = (
                    max(self.env.initial_piles) if self.env.initial_piles else 1
                )
                if (
                    0 <= pile_idx < len(self.env.initial_piles)
                    and 1 <= num_removed <= max_removable
                ):
                    return pile_idx * max_removable + (num_removed - 1)
                else:
                    return None
            else:
                return None
        else:
            return None

    def _encode_action(self, action: ActionType) -> torch.Tensor:
        """Encodes an action for input to the dynamics function."""
        # Simple encoding: use the action index directly as a scalar tensor
        # TODO: Consider one-hot encoding if it improves performance
        action_idx = self.get_action_index(action)
        if action_idx is None:
            raise ValueError(f"Cannot encode invalid or unmappable action: {action}")
        # Return as shape [1] or [batch_size, 1] if handling batches
        return torch.tensor([action_idx], dtype=torch.float32)

    # --- Core MuZero Functions ---

    def representation(self, observation_batch: torch.Tensor) -> torch.Tensor:
        """h(o) -> s"""
        # Add batch dim if needed
        if observation_batch.dim() == 1:
            observation_batch = observation_batch.unsqueeze(0)
        hidden_state = self.representation_net(observation_batch)
        # TODO: Add scaling/normalization here if needed (like in reference)
        return hidden_state

    def dynamics(
        self, hidden_state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """g(s, a) -> s', r"""
        # Ensure action_batch has the correct shape (e.g., [batch_size, 1] for scalar encoding)
        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(-1)
        # Concatenate state and action encoding
        dynamics_input = torch.cat((hidden_state_batch, action_batch), dim=1)

        next_hidden_state = self.dynamics_state_net(dynamics_input)
        reward = self.dynamics_reward_net(dynamics_input)  # Use same input

        # TODO: Add scaling/normalization for next_hidden_state if needed
        return next_hidden_state, reward

    def prediction(
        self, hidden_state_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """f(s) -> p, v"""
        policy_logits = self.prediction_policy_net(hidden_state_batch)
        value = self.prediction_value_net(hidden_state_batch)
        return policy_logits, value

    # --- Inference Wrappers (Similar to AlphaZero) ---

    def initial_inference(
        self, observation_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combines representation and prediction for the first step of MCTS.
        Args:
            observation_dict: The raw environment observation.
        Returns:
            Tuple: (value, reward_logits (0), policy_logits, hidden_state)
                   Reward is 0 at the initial step.
        """
        self.eval()
        with torch.no_grad():
            flat_obs = self._flatten_state(observation_dict)
            # TODO: Add device handling
            hidden_state = self.representation(flat_obs)
            policy_logits, value = self.prediction(hidden_state)

            # Create a dummy reward tensor (scalar 0 for now)
            # TODO: Adapt if using categorical reward support
            reward = torch.zeros_like(value)

        return value, reward, policy_logits, hidden_state

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: ActionType
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combines dynamics and prediction for subsequent steps in MCTS.
        Args:
            hidden_state: The current hidden state tensor.
            action: The action taken (from the environment's action space).
        Returns:
            Tuple: (value, reward_logits, policy_logits, next_hidden_state)
        """
        self.eval()
        with torch.no_grad():
            action_tensor = self._encode_action(action)
            # TODO: Add device handling
            # Ensure hidden_state and action_tensor are on the same device and have batch dim if needed
            if hidden_state.dim() == 1:
                hidden_state = hidden_state.unsqueeze(0)
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)

            next_hidden_state, reward = self.dynamics(hidden_state, action_tensor)
            policy_logits, value = self.prediction(next_hidden_state)

        return value, reward, policy_logits, next_hidden_state
