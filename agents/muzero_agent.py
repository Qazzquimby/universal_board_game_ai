import random
from pathlib import Path
from collections import deque
from typing import List, Tuple

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from core.agent_interface import Agent
from environments.base import BaseEnvironment, StateType, ActionType
from algorithms.mcts import MuZeroMCTS # Use the MuZero MCTS
from models.networks import MuZeroNet # Use the MuZero Network
from core.config import MuZeroConfig # Use the MuZero Config

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class MuZeroAgent(Agent):
    """Agent implementing the MuZero algorithm (placeholder)."""

    def __init__(self, env: BaseEnvironment, config: MuZeroConfig):
        """
        Initialize the MuZero agent.

        Args:
            env: The environment instance (needed for action space info, initial obs).
            config: Configuration object with MuZero parameters.
        """
        self.env = env
        self.config = config

        # Initialize the MuZero network
        self.network = MuZeroNet(env, config)
        # TODO: Add device handling

        # Initialize MuZeroMCTS
        self.mcts = MuZeroMCTS(config, self.network)

        # Experience buffer for training (stores sequences of (obs, action, reward))
        # TODO: Implement proper trajectory storage and sampling
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        self._current_episode_trajectory = [] # Store (obs, action, reward) tuples

        # Optimizer (managed internally)
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def act(self, state: StateType, train: bool = False) -> ActionType:
        """
        Choose an action using MuZero MCTS guided by the learned model.
        If train=True, uses temperature sampling.

        Args:
            state: The current environment state observation dictionary.
            train: If True, use temperature sampling for exploration.

        Returns:
            The chosen action.
        """
        self.network.eval()

        # Run MuZero MCTS search starting from the current observation
        # MuZeroMCTS internally handles representation, dynamics, prediction
        # Get legal actions from the *real* environment state for the root node
        # Create a temporary env to get actions without modifying the main one?
        # Or assume self.env reflects the state correctly. Let's assume self.env is correct.
        # A cleaner way might be to pass env to act, but let's use self.env for now.
        current_legal_actions = self.env.get_legal_actions()

        # Pass observation and legal actions to MCTS search
        root_node = self.mcts.search(state, current_legal_actions)

        # --- Action Selection (similar to AlphaZero) ---
        if not root_node.children:
            print("Warning: MuZero MCTS root has no children after search. Choosing random action.")
            # Need actual legal actions from the real environment state
            legal_actions = self.env.get_legal_actions() # Assumes env reflects current state
            return random.choice(legal_actions) if legal_actions else None

        visit_counts = np.array([child.visit_count for child in root_node.children.values()])
        actions = list(root_node.children.keys())

        if train:
            # TODO: Implement temperature logic
            temperature = 1.0
            visit_counts_temp = visit_counts**(1.0 / temperature)
            action_probs = visit_counts_temp / np.sum(visit_counts_temp)
            chosen_action_index = np.random.choice(len(actions), p=action_probs)
        else:
            chosen_action_index = np.argmax(visit_counts)

        chosen_action = actions[chosen_action_index]

        # TODO: Store search statistics (policy target) if needed for training?
        # MuZero training often relies on re-analyzing trajectories later.
        # For now, just store the basic trajectory info.

        return chosen_action


    def observe(self, obs: StateType, action: ActionType, reward: float, done: bool):
         """Stores the transition in the current trajectory."""
         # Store observation *before* action, the action taken, and reward *after* action
         self._current_episode_trajectory.append({'obs': obs, 'action': action, 'reward': reward, 'done': done})

    def finish_episode(self):
        """Called at the end of an episode to store the trajectory."""
        if self._current_episode_trajectory:
             self.replay_buffer.append(self._current_episode_trajectory)
        self._current_episode_trajectory = []


    def learn(self):
        """
        Update the MuZero network by sampling trajectories from the replay buffer
        and performing BPTT (Backpropagation Through Time) using the learned model.
        (Placeholder - Actual implementation is complex)
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return # Not enough data

        # 1. Sample a batch of trajectories from replay_buffer.
        #    batch_trajectories = random.sample(self.replay_buffer, self.config.batch_size)

        # 2. For each trajectory in the batch:
        #    a. Select a starting point within the trajectory.
        #    b. Get the initial observation `o_t`.
        #    c. Compute initial hidden state `s_t = h(o_t)`.
        #    d. Unroll the dynamics model for `k` steps:
        #       For i = 1 to k:
        #         - Predict policy `p_t+i` and value `v_t+i` from `s_t+i-1`: `p_t+i, v_t+i = f(s_t+i-1)`
        #         - Get the actual action `a_t+i` from the stored trajectory.
        #         - Predict next state `s_t+i` and reward `r_t+i` using dynamics: `s_t+i, r_t+i = g(s_t+i-1, a_t+i)`
        #    e. Calculate losses:
        #       - Policy loss: Compare predicted `p` with MCTS policy target (requires re-analysis or storing targets).
        #       - Value loss: Compare predicted `v` with observed future rewards (n-step return).
        #       - Reward loss: Compare predicted `r` with actual observed reward from trajectory.
        #    f. Sum losses over the unroll steps.

        # 3. Average losses across the batch.
        # 4. Perform gradient descent step using self.optimizer.

        print(f"MuZeroAgent.learn() called - Placeholder. Buffer size: {len(self.replay_buffer)}")
        # TODO: Implement the actual MuZero loss calculation and training step.
        # This will involve calling network.representation, network.dynamics, network.prediction
        # within the unrolling loop and comparing with targets (MCTS policy, n-step returns, actual rewards).
        pass


    def _get_save_path(self) -> Path:
        """Constructs the save file path for the network weights."""
        env_type_name = type(self.env).__name__
        filename = f"muzero_net_{env_type_name}.pth"
        return DATA_DIR / filename

    def save(self) -> None:
        """Save the neural network weights."""
        filepath = self._get_save_path()
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.network.state_dict(), filepath)
            print(f"MuZero network saved to {filepath}")
        except Exception as e:
            print(f"Error saving MuZero network to {filepath}: {e}")

    def load(self) -> bool:
        """Load the neural network weights."""
        filepath = self._get_save_path()
        try:
            if filepath.exists():
                map_location = torch.device('cpu')
                self.network.load_state_dict(torch.load(filepath, map_location=map_location))
                self.network.eval()
                print(f"MuZero network loaded from {filepath}")
                return True
            else:
                print(f"Network weights file not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error loading MuZero network from {filepath}: {e}")
            return False

    def reset(self) -> None:
        """Reset agent state (e.g., MCTS tree, current trajectory)."""
        self.mcts.reset_root()
        self._current_episode_trajectory = []
