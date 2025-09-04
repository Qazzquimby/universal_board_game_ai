from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, ActionType
from models.networks import BaseTokenizingNet


class Vae:
    def __init__(self, mu: torch.Tensor, log_var: torch.Tensor):
        self.mu = mu  # average value
        self.log_var = log_var

    def make_determinization(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn_like(std)
        return self.mu + eps * std


class MuZeroNet(BaseTokenizingNet):
    def __init__(
        self,
        env: BaseEnvironment,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(env=env, embedding_dim=embedding_dim)

        # get_hidden_state (representation, h)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.fc_hidden_state_mu = nn.Linear(embedding_dim, embedding_dim)
        self.fc_hidden_state_log_var = nn.Linear(embedding_dim, embedding_dim)

        # get_next_hidden_state (dynamics, g)
        # It takes hidden_state + action_embedding -> next_hidden_state_vae
        # May need a more powerful model.
        action_embedding_dim = embedding_dim
        hidden_state_dim = embedding_dim * 2
        self.dynamics_network = nn.Sequential(
            nn.Linear(embedding_dim + action_embedding_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, embedding_dim),
        )

        # get_policy_and_value (prediction, f)
        self.value_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())

        # The policy head scores (hidden_state, action) pairs.
        policy_input_dim = embedding_dim + embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # get_actions_from_hidden_state
        # A head to generate actions from a hidden state.
        # todo, needs to generate a dynamic length list of tokens
        # todo, with what data to train this?

    def get_hidden_state_vae(self, state: StateType) -> Vae:
        """
        Representation function (h): Encodes a state into a stochastic hidden state.
        """
        state = self._apply_transforms(state)
        tokens = self._state_to_tokens(state)

        game_token = self.game_token.expand(1, -1, -1)
        sequence = torch.cat([game_token, tokens], dim=1)

        transformer_output = self.transformer_encoder(sequence)
        game_token_output = transformer_output[:, 0, :]

        mu = self.fc_hidden_state_mu(game_token_output)
        log_var = self.fc_hidden_state_log_var(game_token_output)
        return Vae(mu=mu, log_var=log_var)

    def get_next_hidden_state_vae(
        self, hidden_state: torch.Tensor, action: ActionType
    ) -> torch.Tensor:
        """
        Dynamics function (g): Transitions to the next hidden state given a current
        hidden state and an action.
        """
        action_token = self._action_to_token(action).unsqueeze(0)

        dynamics_input = torch.cat([hidden_state, action_token], dim=1)
        next_hidden_state = self.dynamics_network(dynamics_input)
        return next_hidden_state

    def get_actions_for_hidden_state(
        self, hidden_state: torch.Tensor
    ) -> List[ActionType]:
        pass  # todo Generate a dynamic sized list of tokens given the hidden state

    def get_policy_and_value(
        self, hidden_state: torch.Tensor
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Prediction function (f): Predicts policy and value from a hidden state.
        """
        value = self.value_head(hidden_state).squeeze().cpu().item()

        # TODO actions should be stored alongside the hidden state and passed as a param

        # Score candidate actions to get a policy
        action_tokens = torch.stack(
            [self._action_to_token(a) for a in candidate_actions]
        )
        state_embedding_expanded = hidden_state.expand(len(candidate_actions), -1)
        policy_input = torch.cat([state_embedding_expanded, action_tokens], dim=1)
        scores = self.policy_head(policy_input).squeeze(-1)
        policy_probs = F.softmax(scores, dim=0)

        policy_dict = {
            action: prob.item() for action, prob in zip(candidate_actions, policy_probs)
        }

        return policy_dict, value

    def init_zero(self):
        # todo Initialize all weights to 0. Update as needed
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
