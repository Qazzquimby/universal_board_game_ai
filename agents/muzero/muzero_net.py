from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, ActionType
from models.networks import BaseTokenizingNet

# Don't delete
# Hidden state vae is a distribution for sampling hidden state tensors from, for stochasticity.
# A hidden state tensor can be used to predict a variable length list of encoded action tokens with size action_dim. These should be stored with the hidden state tensor.


class Vae:
    def __init__(self, mu: torch.Tensor, log_var: torch.Tensor):
        self.mu = mu  # average value
        self.log_var = log_var

    def take_sample(self) -> torch.Tensor:
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
        dynamics_hidden_dim = embedding_dim * 2
        self.dynamics_network_base = nn.Sequential(
            nn.Linear(embedding_dim + action_embedding_dim, dynamics_hidden_dim),
            nn.ReLU(),
        )
        self.fc_next_hidden_state_mu = nn.Linear(dynamics_hidden_dim, embedding_dim)
        self.fc_next_hidden_state_log_var = nn.Linear(
            dynamics_hidden_dim, embedding_dim
        )

        # get_policy_and_value (prediction, f)
        self.value_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())

        # The policy head scores (hidden_state, action) pairs.
        policy_input_dim = embedding_dim + embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # A head to generate actions from a hidden state, token by token.
        self.action_decoder_lstm = nn.LSTMCell(embedding_dim, embedding_dim)
        self.hidden_to_lstm_h = nn.Linear(embedding_dim, embedding_dim)
        self.hidden_to_lstm_c = nn.Linear(embedding_dim, embedding_dim)
        self.start_action_token = nn.Parameter(torch.randn(1, embedding_dim))

        # A head to generate encoded action tokens from the LSTM's hidden state.
        self.action_generation_head = nn.Linear(embedding_dim, embedding_dim)
        # A head to predict whether to stop generating actions.
        self.action_generation_stop_head = nn.Linear(embedding_dim, 1)

    def get_hidden_state_vae(self, state: StateType) -> Vae:
        """
        Representation function (h): Encodes a state into a stochastic hidden state distribution.
        """
        state = self._apply_transforms(state)
        tokens = self._state_to_tokens(state)

        game_token = self.game_token.expand(1, -1, -1)
        sequence = torch.cat([game_token, tokens], dim=1)

        transformer_output = self.transformer_encoder(sequence)
        game_token_output = transformer_output[:, 0, :]
        mu = self.fc_hidden_state_mu(game_token_output)
        log_var = self.fc_hidden_state_log_var(game_token_output)
        return Vae(mu, log_var)

    def get_next_hidden_state_vae(
        self, hidden_state: torch.Tensor, action_token: torch.Tensor
    ) -> Vae:
        """
        Dynamics function (g): Predicts the distribution of the next hidden state
        given a current hidden state and an encoded action token.
        """
        dynamics_input = torch.cat([hidden_state, action_token], dim=1)
        base_output = self.dynamics_network_base(dynamics_input)

        mu = self.fc_next_hidden_state_mu(base_output)
        log_var = self.fc_next_hidden_state_log_var(base_output)
        return Vae(mu, log_var)

    def get_actions_for_hidden_state(
        self,
        hidden_state: torch.Tensor,
        max_actions: int = 10,
    ) -> List[torch.Tensor]:
        """
        Generates a list of candidate encoded actions from a hidden state.
        """
        action_tokens = []
        # Use eval mode for generation to disable dropout etc.
        self.eval()
        with torch.no_grad():
            # Project hidden state to initial LSTM state
            h = self.hidden_to_lstm_h(hidden_state)
            c = self.hidden_to_lstm_c(hidden_state)

            # Start with the learnable start-of-action token
            input_token_emb = self.start_action_token

            for _ in range(max_actions):
                h, c = self.action_decoder_lstm(input_token_emb, (h, c))

                stop_logit = self.action_generation_stop_head(h)
                if torch.sigmoid(stop_logit) > 0.5:
                    break

                # Generate the next action token
                next_action_token = self.action_generation_head(h)
                action_tokens.append(next_action_token)

                # The generated token is the input for the next step
                input_token_emb = next_action_token

        return action_tokens

    def get_policy_and_value(
        self, hidden_state: torch.Tensor, candidate_actions: List[torch.Tensor]
    ) -> Tuple[Dict[int, float], float]:
        """
        Prediction function (f): Predicts policy and value from a hidden state
        and a list of candidate encoded actions.
        """
        value = self.value_head(hidden_state).squeeze().cpu().item()

        if not candidate_actions:
            return {}, value

        # Score candidate actions to get a policy
        # candidate_actions is a list of tensors, each of shape (1, embedding_dim)
        action_tokens = torch.cat(candidate_actions, dim=0)

        state_embedding_expanded = hidden_state.expand(len(candidate_actions), -1)
        policy_input = torch.cat([state_embedding_expanded, action_tokens], dim=1)
        scores = self.policy_head(policy_input).squeeze(-1)
        policy_probs = F.softmax(scores, dim=0)

        # Policy dict maps action index to probability
        policy_dict = {i: prob.item() for i, prob in enumerate(policy_probs)}

        return policy_dict, value

    def init_zero(self):
        # todo Initialize all weights to 0. Update as needed
        # Stop deleting my comments and replacing them with docstrings with different meanings.
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
