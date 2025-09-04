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

        # Wrong
        # You generate encoded action token tensors. You do not try to generate game-rule-friendly action descriptions.
        # Generate tokens of length action_dim, period.
        # action_spec = self.network_spec["action_space"]
        # self.action_component_names = action_spec["components"]
        # self.action_component_heads = nn.ModuleDict()
        # for comp_name in self.action_component_names:
        #     cardinality = self.network_spec["cardinalities"][comp_name]
        #     # Vocab size: values (cardinality+1) + padding (1) + stop token (1)
        #     vocab_size = cardinality + 3
        #     self.action_component_heads[comp_name] = nn.Linear(
        #         embedding_dim, vocab_size
        #     )

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
        self, hidden_state: torch.Tensor, action: ActionType
    ) -> Vae:
        """
        Dynamics function (g): Predicts the distribution of the next hidden state
        given a current hidden state and an action.
        """
        action_token = self._action_to_token(action).unsqueeze(0)

        dynamics_input = torch.cat([hidden_state, action_token], dim=1)
        base_output = self.dynamics_network_base(dynamics_input)

        mu = self.fc_next_hidden_state_mu(base_output)
        log_var = self.fc_next_hidden_state_log_var(base_output)
        return Vae(mu, log_var)

    def get_actions_for_hidden_state(
        self,
        hidden_state: torch.Tensor,
        num_actions_to_sample: int = 10,
    ) -> List[ActionType]:
        """
        Generates a list of candidate actions from a hidden state by sequentially
        generating action components until a stop token is produced.
        """
        actions = []
        # Use eval mode for generation to disable dropout etc.
        self.eval()
        with torch.no_grad():
            # Project hidden state to initial LSTM state
            h = self.hidden_to_lstm_h(hidden_state)
            c = self.hidden_to_lstm_c(hidden_state)

            for _ in range(num_actions_to_sample):
                action_components = []
                # Start with the learnable start-of-action token
                input_token_emb = self.start_action_token
                current_h, current_c = h.clone(), c.clone()

                for comp_name in self.action_component_names:
                    current_h, current_c = self.action_decoder_lstm(
                        input_token_emb, (current_h, current_c)
                    )

                    head = self.action_component_heads[comp_name]
                    logits = head(current_h)

                    # Sample from the distribution. Could use argmax for deterministic generation.
                    dist = torch.distributions.Categorical(logits=logits)
                    sampled_token_idx = dist.sample()

                    cardinality = self.network_spec["cardinalities"][comp_name]
                    # Values are 1..cardinality+1. 0 is pad. stop is cardinality+2
                    stop_token_idx = cardinality + 2

                    if sampled_token_idx == stop_token_idx:
                        break  # End of action

                    # Convert from vocabulary index back to original component value
                    # Vocab index is val + 1. So val is index - 1.
                    component_value = sampled_token_idx.item() - 1
                    action_components.append(component_value)

                    # Prepare input for the next step
                    input_token_emb = self.embedding_layers[comp_name](
                        sampled_token_idx
                    )

                if action_components:
                    # Format action based on number of components
                    if len(action_components) == 1:
                        actions.append(action_components[0])
                    else:
                        actions.append(tuple(action_components))

        return list(set(actions))  # Return unique actions generated

    # action torch.Tensor, not action type. It is an *encoded* action.
    def get_policy_and_value(
        self, hidden_state: torch.Tensor, candidate_actions: List[torch.Tensor]
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Prediction function (f): Predicts policy and value from a hidden state.
        """
        value = self.value_head(hidden_state).squeeze().cpu().item()

        if not candidate_actions:
            return {}, value

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
        # Stop deleting my comments and replacing them with docstrings with different meanings.
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
