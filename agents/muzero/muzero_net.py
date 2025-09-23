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

    def _predict_policy_and_value_from_tokens(
        self,
        hidden_state_batch: torch.Tensor,
        candidate_action_tokens_batch: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction function (f) core logic: Predicts policy logits and value from a batch of hidden states
        and a batch of candidate action tokens.
        """
        device = self.get_device()
        batch_size = hidden_state_batch.shape[0]

        # --- Value Head ---
        value_preds = self.value_head(hidden_state_batch).squeeze(-1)

        # --- Policy Head ---
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
            return torch.empty(batch_size, 0, device=device), value_preds

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

        policy_logits = nn.utils.rnn.pad_sequence(
            scores_by_item, batch_first=True, padding_value=-torch.inf
        )

        return policy_logits, value_preds

    def get_policy_and_value_batched(
        self,
        hidden_state_batch: torch.Tensor,
        candidate_actions_batch: List[List[ActionType]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction function (f) for training: Predicts policy logits and value from a batch of hidden states
        and candidate actions for each state.
        """
        # Tokenize actions for each item in the batch
        candidate_action_tokens_batch = [
            list(self._actions_to_tokens(actions))
            for actions in candidate_actions_batch
        ]

        return self._predict_policy_and_value_from_tokens(
            hidden_state_batch, candidate_action_tokens_batch
        )

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

    def _calculate_action_prediction_loss(
        self, hidden_state: torch.Tensor, real_actions_per_item: List[List[ActionType]]
    ):
        """Calculate action prediction loss using teacher forcing."""
        if not any(real_actions_per_item):
            return torch.tensor(0.0, device=self.get_device())

        # Not all items in the batch might have actions, so we filter.
        batch_indices_with_actions = [
            i for i, actions in enumerate(real_actions_per_item) if actions
        ]
        hidden_states_with_actions = hidden_state[batch_indices_with_actions]

        # Tokenize and pad the actions for LSTM processing.
        tokenized_actions = [
            torch.stack(list(self._actions_to_tokens(actions)))
            for actions in real_actions_per_item
            if actions
        ]
        padded_tokens = nn.utils.rnn.pad_sequence(tokenized_actions, batch_first=True)
        seq_lengths = torch.tensor(
            [len(seq) for seq in tokenized_actions], device=self.get_device()
        )

        batch_size, max_seq_len, embedding_dim = padded_tokens.shape
        h = self.hidden_to_lstm_h(hidden_states_with_actions)
        c = self.hidden_to_lstm_c(hidden_states_with_actions)
        start_tokens = self.start_action_token.expand(batch_size, -1)

        # Prepare inputs for LSTM: start_token + ground truth tokens
        lstm_inputs = torch.cat([start_tokens.unsqueeze(1), padded_tokens], dim=1)

        total_loss = 0
        for t in range(max_seq_len + 1):
            active_mask = seq_lengths > t
            if not torch.any(active_mask):
                # Process stop token for sequences that just ended.
                ended_mask = seq_lengths == t
                if torch.any(ended_mask):
                    stop_logits = self.action_generation_stop_head(h[ended_mask])
                    loss = F.binary_cross_entropy_with_logits(
                        stop_logits, torch.ones_like(stop_logits)
                    )
                    total_loss += loss
                break

            input_token_t = lstm_inputs[active_mask, t, :]
            h_active, c_active = h[active_mask], c[active_mask]
            h_next, c_next = self.action_decoder_lstm(
                input_token_t, (h_active, c_active)
            )
            h = h.clone()
            c = c.clone()
            h[active_mask], c[active_mask] = h_next, c_next

            # Predict stop token (should be 'not stop')
            stop_logits = self.action_generation_stop_head(h_next)
            loss = F.binary_cross_entropy_with_logits(
                stop_logits, torch.zeros_like(stop_logits)
            )
            total_loss += loss

            # Predict next action token if not at the end of sequence
            if t < max_seq_len:
                pred_token = self.action_generation_head(h_next)
                target_token = padded_tokens[active_mask, t, :]
                loss = F.mse_loss(pred_token, target_token)
                total_loss += loss

        return total_loss / batch_size if batch_size > 0 else 0

    def forward(
        self,
        state_batch: Dict[str, "DataFrame"],
        action_history_batch: torch.Tensor,
        legal_actions_batch: List[List[List[ActionType]]],
        target_states_batch: List[Dict[str, "DataFrame"]],
    ):
        num_unroll_steps = action_history_batch.shape[1]
        batch_size = action_history_batch.shape[0]

        # 1. Representation (h):
        hidden_state_vae = self.get_hidden_state_vae(state_batch)
        hidden_state_tensor = hidden_state_vae.take_sample()

        if hidden_state_tensor.shape[0] == 1 and batch_size > 1:
            hidden_state_tensor = hidden_state_tensor.expand(batch_size, -1)

        unrolled_policy_logits = []
        unrolled_values = []
        unrolled_predicted_hidden_states = []
        unrolled_target_hidden_states = []

        total_action_pred_loss = 0.0

        for i in range(num_unroll_steps + 1):
            # Get candidate actions for policy head
            legal_actions_for_step = [
                seq[i] if i < len(seq) else [] for seq in legal_actions_batch
            ]
            policy_logits, value = self.get_policy_and_value_batched(
                hidden_state_tensor, legal_actions_for_step
            )
            unrolled_policy_logits.append(policy_logits)
            unrolled_values.append(value)

            # Train action generator
            action_loss = self._calculate_action_prediction_loss(
                hidden_state_tensor, legal_actions_for_step
            )
            total_action_pred_loss += action_loss

            if i < num_unroll_steps:
                # 3. Dynamics (g):
                action_tokens = self._actions_to_tokens(
                    action_history_batch[:, i].tolist()
                )
                next_h_vae = self.get_next_hidden_state_vae(
                    hidden_state_tensor, action_tokens
                )
                hidden_state_tensor = next_h_vae.take_sample()
                unrolled_predicted_hidden_states.append(hidden_state_tensor)

                # Target hidden state for consistency loss
                target_h_vae = self.get_hidden_state_vae(target_states_batch[i])
                target_h_tensor = target_h_vae.take_sample().detach()
                unrolled_target_hidden_states.append(target_h_tensor)

        # Collate and return all predictions.
        max_actions = max(
            (p.shape[1] for p in unrolled_policy_logits if p.numel() > 0), default=0
        )
        padded_policies = [
            F.pad(p, (0, max_actions - p.shape[1]), "constant", value=-torch.inf)
            for p in unrolled_policy_logits
        ]
        policies_tensor = torch.stack(padded_policies, dim=1)
        values_tensor = torch.stack(unrolled_values, dim=1)

        if num_unroll_steps > 0:
            predicted_hidden_states_tensor = torch.stack(
                unrolled_predicted_hidden_states, dim=1
            )
            target_hidden_states_tensor = torch.stack(
                unrolled_target_hidden_states, dim=1
            )
        else:
            predicted_hidden_states_tensor = torch.empty(
                batch_size, 0, hidden_state_tensor.shape[-1], device=self.get_device()
            )
            target_hidden_states_tensor = torch.empty(
                batch_size, 0, hidden_state_tensor.shape[-1], device=self.get_device()
            )

        return (
            policies_tensor,
            values_tensor,
            predicted_hidden_states_tensor,
            target_hidden_states_tensor,
            total_action_pred_loss,
        )

    def init_zero(self):
        # todo Initialize all weights to 0. Update as needed
        # Stop deleting my comments and replacing them with docstrings with different meanings.
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
