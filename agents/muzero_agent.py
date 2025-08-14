# imports from alphazero_agent
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random
import copy

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from loguru import logger

# Muzero specific imports
from agents.alphazero_agent import (
    AlphaZeroAgent,
)  # For now to inherit from, will need refactoring
from environments.base import BaseEnvironment, StateType, ActionType, StateWithKey
from algorithms.mcts import (
    MCTSNode,  # Will need a MuZeroNode
    UCB1Selection,
    ExpansionStrategy,
    EvaluationStrategy,
    StandardBackpropagation,
    Edge,
    BackpropagationStrategy,
    SelectionStrategy,
)
from models.networks import AutoGraphNet  # Will need a MuZeroNet
from core.config import (
    # MuZeroConfig, # Will need this
    AlphaZeroConfig,  # placeholder
    DATA_DIR,
    TrainingConfig,
)
from experiments.architectures.shared import INFERENCE_DEVICE, TRAINING_DEVICE


# TODO: MuZero requires a different MCTS node that stores a hidden state, not the full environment state.
# class MuZeroNode(MCTSNode):
#     def __init__(self, hidden_state, reward=0.0, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden_state = hidden_state
#         self.reward = reward

# TODO: MuZero needs its own network architecture.
# class MuZeroNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.representation_model = ... h()
#         # self.dynamics_model = ... g()
#         # self.prediction_model = ... f()

#     def predict(self, hidden_state): # f(s) -> p, v
#         ...

#     def represent(self, observation): # h(o) -> s
#         ...

#     def dynamics(self, hidden_state, action): # g(s, a) -> r, s'
#         ...

# TODO: Replay buffer will store full games/trajectories.
# The dataset will sample from these trajectories.
class MuZeroReplayBufferDataset(Dataset):
    def __init__(self, buffer: deque, k_unroll_steps: int):
        self.buffer = buffer
        self.k_unroll_steps = k_unroll_steps

    def __len__(self):
        # ...
        return 0

    def __getitem__(self, idx):
        # Sample a game from buffer
        # Sample a starting point in the game
        # Return the observation, and targets for k_unroll_steps
        pass


class MuZeroAgent(
    AlphaZeroAgent
):  # TODO: Should not inherit from AlphaZeroAgent, needs refactoring
    """Agent implementing the MuZero algorithm."""

    def __init__(
        self,
        # ... similar to AlphaZeroAgent, but with MuZeroConfig and MuZeroNet
        network: nn.Module,
        optimizer,
        env: BaseEnvironment,
        config: AlphaZeroConfig,  # TODO: MuZeroConfig
        training_config: TrainingConfig,
    ):
        # super().__init__() # cannot call parent
        # Many properties from AlphaZeroAgent can be reused.
        # The key difference is the MCTS search which is model-based.
        self.network = network  # A MuZeroNet
        self.optimizer = optimizer
        self.env = env
        self.config = config
        self.training_config = training_config
        self.device = INFERENCE_DEVICE
        # ... replay buffers etc.

    def search(self, env: BaseEnvironment, train: bool = False):
        """
        MuZero's MCTS search. This is different from AlphaZero's as it uses a learned model.
        This method should override any parent search method.
        """
        # 1. Get initial hidden state from representation network `h`.
        # initial_observation = env.get_state_with_key()
        # root_hidden_state = self.network.represent(initial_observation)
        # self.root = MuZeroNode(hidden_state=root_hidden_state)

        # 2. Run simulations
        # for _ in range(self.config.num_simulations):
        #     # 2a. Selection: traverse tree using UCB1 on (Q-value, visit_count)
        #     # This part does *not* use the environment. It uses the dynamics model `g`.
        #     ...
        #
        #     # 2b. Expansion: at a leaf, use prediction network `f` to get policy and value.
        #     # Expand node with new children.
        #     ...
        #
        #     # 2c. Backpropagation: update visit counts and values up the tree.
        #     ...

        # This is a placeholder
        logger.warning("MuZero search is not implemented. Using random action.")
        return

    def train_network(self):
        """
        MuZero's learning step. Samples trajectories from replay buffer and updates network.
        """
        # 1. Sample batch from replay buffer.
        # train_loader, val_loader = self._get_train_val_loaders() # Needs MuZeroReplayBufferDataset

        # 2. For each sampled trajectory in batch:
        #   a. Get initial hidden state: s0 = h(o1)
        #   b. Unroll for k steps: For t = 1..k:
        #      i. Predict reward, policy, value from current hidden state: r_t, p_t, v_t = f(s_{t-1})
        #      ii. Calculate loss: (value vs target, policy vs MCTS policy, reward vs target)
        #      iii. Get next hidden state: s_t = g(s_{t-1}, a_t)
        #
        #   c. Backpropagate total loss.
        logger.warning("MuZero learn is not implemented.")
        return None

    def process_finished_episode(
        self,
        game_history: List[
            Tuple[StateType, ActionType, np.ndarray, float]
        ],  # History will also include rewards
        final_outcome: float,
    ):
        """
        Processes a finished episode to generate training data.
        For MuZero, this means storing the full trajectory of (observation, action, reward, mcts_policy).
        The value target is calculated during training using N-step returns.
        """
        # This will be different from AlphaZero. We just store the history.
        # The value is computed during training.
        # return EpisodeResult(buffer_experiences=game_history, logged_history=game_history)
        pass  # placeholder
