import random
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

from core.agent_interface import Agent
from environments.base import BaseEnvironment, StateType, ActionType
from algorithms.mcts import MCTS
from models.networks import AlphaZeroNet
from core.config import AlphaZeroConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


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

        # Initialize MCTS
        # We will modify MCTS to accept the network and use PUCT
        self.mcts = MCTS(
            exploration_constant=config.cpuct,  # Use cpuct for PUCT formula
            num_simulations=config.num_simulations,
            discount_factor=1.0,  # Typically 1.0 for AlphaZero MCTS value
            network=self.network,  # Pass the network to MCTS
        )

        # Placeholder for experience buffer (for training)
        self.replay_buffer = []  # TODO: Implement a proper replay buffer

    def act(self, state: StateType, train: bool = False) -> ActionType:
        """
        Choose an action using MCTS guided by the neural network.

        Args:
            state: The current environment state observation.
            train: If True, use temperature sampling for exploration during training.
                   If False, choose the most visited node (greedy).

        Returns:
            The chosen action.
        """
        # Ensure network is in evaluation mode for inference
        self.network.eval()

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

        # TODO: Store the search statistics (state, policy target, value) for training
        # policy_target = np.zeros(self.network._calculate_policy_size(self.env))
        # for i, action in enumerate(actions):
        #     # Map action to policy index - Requires a consistent mapping function
        #     action_idx = self._map_action_to_policy_index(action) # Needs implementation
        #     if action_idx is not None:
        #          policy_target[action_idx] = visit_counts[i] / np.sum(visit_counts)
        # self.replay_buffer.append((state, policy_target, None)) # Value comes later

        return chosen_action

    def learn(self, episode_history: List[Tuple[StateType, ActionType, float, bool]]):
        """
        Update the neural network based on experience.
        (Placeholder - Training logic will be more complex involving the replay buffer)
        """
        # TODO: Implement the actual learning step:
        # 1. Sample data from the replay buffer.
        # 2. Format data into batches (states, target policies, target values).
        # 3. Perform gradient descent step on the network.
        print(
            "AlphaZeroAgent.learn() called - Placeholder, training not implemented yet."
        )
        pass

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
        """Reset the MCTS search tree."""
        self.mcts.reset_root()

    # --- Helper methods (potentially needed later) ---
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
