import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional # Import Optional

from environments.base import BaseEnvironment, ActionType # Import ActionType

class AlphaZeroNet(nn.Module):
    """
    Simple MLP network for AlphaZero.
    Takes a flattened representation of the environment state.
    Outputs policy logits and a value prediction.
    """
    def __init__(self, env: BaseEnvironment, hidden_layer_size: int = 128, num_hidden_layers: int = 2):
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
            nn.Tanh() # Value prediction between -1 and 1
        )

        print(f"Initialized AlphaZeroNet:")
        print(f"  Input size: {input_size}")
        print(f"  Policy size: {policy_size}")
        print(f"  Hidden layers: {num_hidden_layers} x {hidden_layer_size}")


    def _flatten_state(self, state_dict: dict) -> torch.Tensor:
        """Flattens the state dictionary into a single tensor."""
        flat_tensors = []
        # Process known keys first
        if 'board' in state_dict:
            board = state_dict['board']
            if isinstance(board, np.ndarray):
                flat_tensors.append(torch.from_numpy(board).float().flatten())
            elif torch.is_tensor(board):
                flat_tensors.append(board.float().flatten())
        elif 'piles' in state_dict:
            piles = state_dict['piles']
            if isinstance(piles, tuple):
                flat_tensors.append(torch.tensor(piles, dtype=torch.float32))
            elif torch.is_tensor(piles):
                 flat_tensors.append(piles.float())
            elif isinstance(piles, np.ndarray):
                 flat_tensors.append(torch.from_numpy(piles).float())

        # Add current player as a feature
        flat_tensors.append(torch.tensor([state_dict.get('current_player', -1)], dtype=torch.float32))

        # Concatenate all parts
        if not flat_tensors:
             raise ValueError("Could not extract numerical data from state for network input.")

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
        if hasattr(env, 'num_actions'): # Like in FourInARow
            return env.num_actions
        elif hasattr(env, 'board_size'): # Fallback for grid games
             return env.board_size * env.board_size
        elif hasattr(env, 'initial_piles'): # Specific logic for Nim
             # Calculate size based on (pile_index, num_removed) pairs
             # Requires knowing the maximum number removable from any pile,
             # which depends on the initial state. Let's use the max initial pile size.
             max_removable = max(env.initial_piles) if env.initial_piles else 1
             num_piles = len(env.initial_piles)
             return num_piles * max_removable
        else:
            raise ValueError(f"Cannot determine policy size for environment type: {type(env).__name__}")

    def get_action_index(self, action: ActionType) -> Optional[int]:
        """Maps an environment action to the corresponding index in the policy vector."""
        env_type = type(self.env).__name__

        if env_type == "FourInARow":
            # Action is (row, col)
            row, col = action
            # Ensure board_size is accessible, maybe store it during init?
            # Assuming env has board_size attribute accessible via self.env
            if hasattr(self.env, 'board_size'):
                 return row * self.env.board_size + col
            else:
                 print("Warning: Cannot get board_size from env in get_action_index")
                 return None # Indicate failure
        elif env_type == "NimEnv":
            # Action is (pile_index, num_to_remove)
            pile_idx, num_removed = action
            if hasattr(self.env, 'initial_piles'):
                 max_removable = max(self.env.initial_piles) if self.env.initial_piles else 1
                 # Check bounds
                 if 0 <= pile_idx < len(self.env.initial_piles) and 1 <= num_removed <= max_removable:
                      # Map (pile_idx, num_removed) to a flat index
                      # Example: index = pile_idx * max_removable + (num_removed - 1)
                      return pile_idx * max_removable + (num_removed - 1)
                 else:
                      print(f"Warning: Invalid Nim action {action} for index calculation.")
                      return None # Indicate failure
            else:
                 print("Warning: Cannot get initial_piles from env in get_action_index")
                 return None # Indicate failure
        else:
            print(f"Warning: Action indexing not implemented for env type {env_type}")
            return None # Indicate failure

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
        self.eval() # Set model to evaluation mode
        flat_state = self._flatten_state(state_dict) # Flatten the input dict
        # TODO: Add device handling if needed: flat_state = flat_state.to(self.device)
        with torch.no_grad():
            # Pass the flattened tensor to forward
            policy_logits, value = self.forward(flat_state)

            # Apply softmax to get probabilities
            policy_probs = F.softmax(policy_logits, dim=1)

            # Move to CPU, detach, and convert to numpy
            policy_np = policy_probs.squeeze(0).cpu().numpy()
            value_np = value.squeeze(0).cpu().item() # Get scalar value

        return policy_np, value_np
