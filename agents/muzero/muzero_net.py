from typing import Tuple, Dict

import torch
import torch.nn as nn

from environments.base import BaseEnvironment, StateType, ActionType
from models.networks import BaseTokenizingNet


class MuZeroNet(BaseTokenizingNet):
    """
    A MuZero-style network.
    It consists of three main parts:
    - A representation function (h) that maps an observation to a hidden state.
    - A dynamics function (g) that predicts the next hidden state from a previous state and action.
    - A prediction function (f) that predicts the policy and value from a hidden state.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(env=env, embedding_dim=embedding_dim)

        # --- Representation (h) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # --- Dynamics (g) ---
        # It takes hidden_state + action_embedding -> next_hidden_state
        # TODO: This is a placeholder. A more complex model is likely needed.
        action_embedding_dim = embedding_dim
        hidden_state_dim = embedding_dim * 2
        self.dynamics_network = nn.Sequential(
            nn.Linear(embedding_dim + action_embedding_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, embedding_dim),
        )
        # todo, not meeting requirements
        # get state mean and std vectors to generate specific states
        # also need a head to generate a list of action tokens from a specific hidden state

        # --- Prediction (f) ---
        self.value_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())
        # TODO: MuZero's policy head may need to generate actions, not just score them.
        # This is a placeholder that reuses the AlphaZero policy head structure.
        policy_input_dim = embedding_dim + embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def representation(self, state: StateType) -> torch.Tensor:
        """h(o) -> s. Maps a raw observation to a hidden state."""
        # TODO: Implement representation function.
        # This will be similar to the first part of AlphaZeroNet.forward/predict,
        # using _state_to_tokens and the transformer_encoder to get the game_token_output.
        # It should return a hidden state tensor.
        raise NotImplementedError("MuZero representation function not implemented.")

    def dynamics(self, hidden_state: torch.Tensor, action: ActionType) -> torch.Tensor:
        """g(s, a) -> s'. Predicts the next hidden state."""
        # MuZero also predicts reward, but we are skipping it for now.
        # TODO: Implement dynamics function.
        # This involves getting an action token for the action, concatenating it
        # with the hidden state, and passing it through the dynamics_network.
        raise NotImplementedError("MuZero dynamics function not implemented.")

    def prediction(
        self, hidden_state: torch.Tensor
    ) -> Tuple[Dict[ActionType, float], float]:
        """f(s) -> p, v. Predicts policy and value from a hidden state."""
        # TODO: Implement prediction function.
        # This will be similar to the latter half of AlphaZeroNet.predict, but it
        # needs a way to get legal actions from a hidden state, which is a key
        # challenge in MuZero. For now, this is a placeholder.
        raise NotImplementedError("MuZero prediction function not implemented.")

    def init_zero(self):
        """Initializes the weights of the policy and value heads to zero."""
        # TODO: Adapt for dynamics head as well.
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
