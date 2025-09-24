from dataclasses import dataclass
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, ActionType
from models.networks import BaseTokenizingNet


@dataclass
class MuZeroNetworkOutput:
    pred_policies: torch.Tensor
    pred_values: torch.Tensor
    pred_hidden_states: torch.Tensor
    target_hidden_states: torch.Tensor
    pred_actions: torch.Tensor


# Don't delete
# Hidden state vae is a distribution for sampling hidden state tensors from, for stochasticity.
# A hidden state tensor can be used to predict a variable length list of encoded action tokens with size action_dim. These should be stored with the hidden state tensor.


class Vae:
    def __init__(self, mu: torch.Tensor, log_var: torch.Tensor):
        self.mu = mu  # average value
        self.log_var = log_var

    def take_sample(self) -> torch.Tensor:
        # std = torch.exp(0.5 * self.log_var)
        # eps = torch.randn_like(std)
        # return self.mu + eps * std
        return self.mu


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
        Representation function (h): Encodes a batch of states into a stochastic hidden state distribution.
        """
        tokens = self._state_to_tokens(state)

        batch_size = tokens.shape[0]
        game_token = self.game_token.expand(batch_size, -1, -1)
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

    def _get_policy_scores(
        self, hidden_states: torch.Tensor, action_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Scores (hidden_state, action) pairs.
        """
        policy_input = torch.cat([hidden_states, action_tokens], dim=1)
        scores = self.policy_head(policy_input).squeeze(-1)
        return scores

    def get_value_batched(
        self,
        hidden_state_batch: torch.Tensor,
    ):
        value_preds = self.value_head(hidden_state_batch).squeeze(-1)
        return value_preds

    def get_policy_batched(
        self,
        hidden_state_batch: torch.Tensor,
        candidate_action_tokens_batch: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        device = self.get_device()
        batch_size = hidden_state_batch.shape[0]

        # todo reorganize, better document
        flat_action_tokens = []
        batch_indices_for_policy = []
        num_actions_per_item = []
        for i in range(batch_size):
            actions = candidate_action_tokens_batch[i]
            num_actions_per_item.append(len(actions))
            if not actions:
                continue
            # candidate_action_tokens_batch[i] is a list of tensors.
            action_tokens = torch.stack(actions)
            action_tokens = action_tokens.squeeze(1)
            flat_action_tokens.append(action_tokens)
            batch_indices_for_policy.extend([i] * len(actions))

        if not batch_indices_for_policy:
            return torch.empty(batch_size, 0, device=device)

        flat_action_tokens_tensor = torch.cat(flat_action_tokens, dim=0)
        batch_indices_tensor = torch.tensor(
            batch_indices_for_policy, device=device, dtype=torch.long
        )

        state_embs_for_policy = hidden_state_batch[batch_indices_tensor]

        scores = self._get_policy_scores(
            hidden_states=state_embs_for_policy, action_tokens=flat_action_tokens_tensor
        )

        scores_by_item = []
        start_idx = 0
        for num_actions in num_actions_per_item:
            if num_actions > 0:
                scores_by_item.append(scores[start_idx : start_idx + num_actions])
                start_idx += num_actions
            else:
                scores_by_item.append(torch.empty(0, device=device))

        policy_preds = nn.utils.rnn.pad_sequence(
            scores_by_item, batch_first=True, padding_value=-torch.inf
        )

        return policy_preds

    def get_policy_and_value(
        self, hidden_state: torch.Tensor, candidate_actions: List[torch.Tensor]
    ) -> Tuple[Dict[int, float], float]:
        """
        Prediction function (f): Predicts policy and value from a hidden state
        and a list of candidate encoded actions.
        """
        hidden_state_batch = hidden_state
        # The candidate_actions are already tokenized.
        candidate_action_tokens_batch = [candidate_actions]

        policy_logits, value_preds = self._predict_policy_and_value_from_tokens(
            hidden_state_batch, candidate_action_tokens_batch
        )

        value = value_preds.squeeze().cpu().item()

        policy_dict = {}
        if candidate_actions:
            # We have a batch of 1, so squeeze out the batch dimension.
            policy_logits_single = policy_logits.squeeze(0)
            policy_probs = F.softmax(policy_logits_single, dim=0)

            # Policy dict maps action index to probability
            policy_dict = {i: prob.item() for i, prob in enumerate(policy_probs)}

        return policy_dict, value

    def forward(
        self,
        state_batch: Dict[str, "DataFrame"],
        action_history_batch: torch.Tensor,
        legal_actions_batch: List[List[List[ActionType]]],
        target_states_batch: List[Dict[str, "DataFrame"]],
    ) -> MuZeroNetworkOutput:
        num_unroll_steps = action_history_batch.shape[1]
        batch_size = action_history_batch.shape[0]

        unrolled_pred_policies = []
        unrolled_pred_values = []
        unrolled_pred_actions = []
        unrolled_pred_dynamics_hidden_vaes = []
        unrolled_pred_representation_hidden_vaes = []

        hidden_state_vae = self.get_hidden_state_vae(state_batch)
        hidden_state_tensor = hidden_state_vae.take_sample()
        if hidden_state_tensor.shape[0] == 1 and batch_size > 1:
            assert False
            # hidden_state_tensor = hidden_state_tensor.expand(batch_size, -1)

        for i in range(num_unroll_steps + 1):
            # POLICY
            # Uses actual actions rather than predicted actions to keep unroll close
            #  to reality. Action generation is trained separately rather than
            #  downstream of this
            legal_actions_for_step = [
                seq[i] if i < len(seq) else [] for seq in legal_actions_batch
            ]
            candidate_action_tokens_batch = [
                list(self._actions_to_tokens(actions))
                for actions in legal_actions_for_step
            ]  # todo avoid list of lists
            pred_policy = self.get_policy_batched(
                hidden_state_batch=hidden_state_tensor,
                candidate_action_tokens_batch=candidate_action_tokens_batch,
            )
            unrolled_pred_policies.append(pred_policy)

            # VALUE
            pred_value = self.get_value_batched(hidden_state_batch=hidden_state_tensor)
            unrolled_pred_values.append(pred_value)

            # ACTIONS
            pred_actions = self.get_actions_for_hidden_state(
                hidden_state=hidden_state_tensor
            )
            unrolled_pred_actions.append(pred_actions)

            if i < num_unroll_steps:
                # REPRESENTATION # todo make sure these states line up, not offby1
                pred_representation_hidden_vae = self.get_hidden_state_vae(
                    target_states_batch[i]
                )
                unrolled_pred_representation_hidden_vaes.append(
                    pred_representation_hidden_vae
                )

                # DYNAMICS
                action_token_batch = self._actions_to_tokens(
                    action_history_batch[:, i].tolist()
                )
                # todo are the types right here? 1 hidden state but batch of actions?
                pred_dynamics_hidden_vae = self.get_next_hidden_state_vae(
                    hidden_state_tensor, action_token_batch
                )
                unrolled_pred_dynamics_hidden_vaes.append(pred_dynamics_hidden_vae)

                hidden_state_tensor = pred_dynamics_hidden_vae.take_sample()

        # Collate and return all predictions.
        max_actions = max(
            (p.shape[1] for p in unrolled_pred_policies if p.numel() > 0), default=0
        )
        padded_policies = [
            F.pad(p, (0, max_actions - p.shape[1]), "constant", value=-torch.inf)
            for p in unrolled_pred_policies
        ]
        pred_policies = torch.stack(padded_policies, dim=1)
        pred_values = torch.stack(unrolled_pred_values, dim=1)

        if num_unroll_steps > 0:
            pred_hidden_states = torch.stack(unrolled_pred_dynamics_hidden_vaes, dim=1)
            target_hidden_states = torch.stack(
                unrolled_pred_representation_hidden_vaes, dim=1
            )
        else:
            pred_hidden_states = torch.empty(
                batch_size, 0, hidden_state_tensor.shape[-1], device=self.get_device()
            )
            target_hidden_states = torch.empty(
                batch_size, 0, hidden_state_tensor.shape[-1], device=self.get_device()
            )

        pred_actions = torch.stack(unrolled_pred_actions)

        return MuZeroNetworkOutput(
            pred_policies=pred_policies,
            pred_values=pred_values,
            pred_hidden_states=pred_hidden_states,
            target_hidden_states=target_hidden_states,
            pred_actions=pred_actions,
        )

    def init_zero(self):
        # todo Initialize all weights to 0. Update as needed
        # Stop deleting my comments and replacing them with docstrings with different meanings.
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
