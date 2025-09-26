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
    pred_dynamics_mu: torch.Tensor
    pred_dynamics_log_var: torch.Tensor
    target_representation_mu: torch.Tensor
    target_representation_log_var: torch.Tensor
    pred_actions: List[List[List[torch.Tensor]]]
    candidate_action_tokens: List[List[List[torch.Tensor]]]


# Don't delete
# Hidden state vae is a distribution for sampling hidden state tensors from, for stochasticity.
# A hidden state tensor can be used to predict a variable length list of encoded action tokens with size action_dim. These should be stored with the hidden state tensor.


def vae_take_sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # std = torch.exp(0.5 * log_var)
    # eps = torch.randn_like(std)
    # return mu + eps * std
    return mu


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

    def get_hidden_state_vae(
        self, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return mu, log_var

    def get_next_hidden_state_vae(
        self, hidden_state: torch.Tensor, action_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamics function (g): Predicts the distribution of the next hidden state
        given a current hidden state and an encoded action token.
        """
        dynamics_input = torch.cat([hidden_state, action_token], dim=1)
        base_output = self.dynamics_network_base(dynamics_input)

        mu = self.fc_next_hidden_state_mu(base_output)
        log_var = self.fc_next_hidden_state_log_var(base_output)
        return mu, log_var

    def get_actions_for_hidden_state(
        self,
        hidden_state: torch.Tensor,
        max_actions: int = 10,
    ) -> List[List[torch.Tensor]]:
        """
        Generates lists of candidate encoded actions from a batch of hidden states.
        """
        batch_size = hidden_state.shape[0]
        # A list of lists to hold the generated action token sequences for each batch item.
        batched_action_tokens = [[] for _ in range(batch_size)]
        # A mask to track which sequences in the batch are still being generated.
        active_mask = torch.ones(
            batch_size, dtype=torch.bool, device=hidden_state.device
        )

        # Use eval mode for generation to disable dropout etc.
        self.eval()
        with torch.no_grad():
            # Project hidden state to initial LSTM state
            h = self.hidden_to_lstm_h(hidden_state)
            c = self.hidden_to_lstm_c(hidden_state)

            # Start with the learnable start-of-action token, expanded for the batch.
            input_token_emb = self.start_action_token.expand(batch_size, -1)

            for _ in range(max_actions):
                if not active_mask.any():
                    break  # All sequences have stopped.

                h, c = self.action_decoder_lstm(input_token_emb, (h, c))

                stop_logits = self.action_generation_stop_head(h).squeeze(-1)
                # Sequences that should stop are those that are active and meet the stop condition.
                should_stop = (torch.sigmoid(stop_logits) > 0.5) & active_mask

                # Generate the next action token for all sequences in the batch.
                next_action_tokens = self.action_generation_head(h)

                # For sequences that are still active and not stopping, append the new token.
                for i in range(batch_size):
                    if active_mask[i] and not should_stop[i]:
                        batched_action_tokens[i].append(
                            next_action_tokens[i].unsqueeze(0)
                        )

                # Update the active mask: turn off sequences that just stopped.
                active_mask &= ~should_stop

                # The generated token is the input for the next step.
                input_token_emb = next_action_tokens
        return batched_action_tokens

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
        hidden_state: torch.Tensor,
        candidate_action_tokens: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        # state and tokens are batches
        device = self.get_device()

        (scores, num_actions_per_item) = self.get_action_scores_and_counts(
            device=device,
            candidate_action_tokens_batch=candidate_action_tokens,
            hidden_state_batch=hidden_state,
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

    def get_action_scores_and_counts(
        self,
        device,
        candidate_action_tokens_batch: List[List[torch.Tensor]],
        hidden_state_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        batch_size = hidden_state_batch.shape[0]
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
            raise ValueError("No actions in batch")

        flat_action_tokens_tensor = torch.cat(flat_action_tokens, dim=0)
        batch_indices_tensor = torch.tensor(
            batch_indices_for_policy, device=device, dtype=torch.long
        )

        state_embs_for_policy = hidden_state_batch[batch_indices_tensor]
        scores = self._get_policy_scores(
            hidden_states=state_embs_for_policy, action_tokens=flat_action_tokens_tensor
        )

        return scores, num_actions_per_item

    def get_policy_and_value(
        self, hidden_state: torch.Tensor, legal_action_tokens: List[torch.Tensor]
    ) -> Tuple[Dict[int, float], float]:
        """
        Prediction function (f): Predicts policy and value from a hidden state
        and a list of candidate encoded actions.
        """
        candidate_action_tokens_batch = [legal_action_tokens]

        # POLICY
        pred_policy = self.get_policy_batched(
            hidden_state=hidden_state,
            candidate_action_tokens=candidate_action_tokens_batch,
        )
        policy_dict = {}
        if legal_action_tokens:
            # We have a batch of 1, so squeeze out the batch dimension.
            policy_logits_single = pred_policy.squeeze(0)
            policy_probs = F.softmax(policy_logits_single, dim=0)

            # Policy dict maps action index to probability
            policy_dict = {i: prob.item() for i, prob in enumerate(policy_probs)}

        # VALUE
        pred_value = self.get_value_batched(hidden_state_batch=hidden_state)
        value = pred_value.squeeze().cpu().item()

        return policy_dict, value

    def forward(
        self,
        initial_state: Dict[str, "DataFrame"],
        action_history: torch.Tensor,
        legal_actions: List[List[List[ActionType]]],
        unrolled_state: List[Dict[str, "DataFrame"]],
    ) -> MuZeroNetworkOutput:
        # All data variables are batches
        batch_size = action_history.shape[0]
        assert len(legal_actions) == batch_size

        num_unroll_steps = action_history.shape[1]
        assert len(unrolled_state) == num_unroll_steps
        assert len(legal_actions[0]) == num_unroll_steps + 1

        unrolled_pred_policies = []
        unrolled_pred_values = []
        unrolled_pred_actions = []
        unrolled_candidate_action_tokens = []
        unrolled_pred_dynamics_mu = []
        unrolled_pred_dynamics_log_var = []
        unrolled_pred_representation_mu = []
        unrolled_pred_representation_log_var = []

        hidden_state_mu, hidden_state_log_var = self.get_hidden_state_vae(initial_state)
        current_hidden_state = vae_take_sample(hidden_state_mu, hidden_state_log_var)
        assert current_hidden_state.shape[0] == batch_size

        for i in range(num_unroll_steps + 1):
            # POLICY
            # Uses actual actions rather than predicted actions to keep unroll close
            #  to reality. Action generation is trained separately rather than
            #  downstream of this
            legal_actions_for_step = [
                seq[i] if i < len(seq) else [] for seq in legal_actions
            ]
            candidate_action_tokens = [
                list(self._actions_to_tokens(actions))
                for actions in legal_actions_for_step
            ]  # todolater avoid list of lists
            unrolled_candidate_action_tokens.append(candidate_action_tokens)
            pred_policy = self.get_policy_batched(
                hidden_state=current_hidden_state,
                candidate_action_tokens=candidate_action_tokens,
            )
            unrolled_pred_policies.append(pred_policy)

            # VALUE
            pred_value = self.get_value_batched(hidden_state_batch=current_hidden_state)
            unrolled_pred_values.append(pred_value)

            # ACTIONS
            pred_actions = self.get_actions_for_hidden_state(
                hidden_state=current_hidden_state
            )
            unrolled_pred_actions.append(pred_actions)

            if i < num_unroll_steps:
                # REPRESENTATION
                (
                    pred_representation_mu,
                    pred_representation_log_var,
                ) = self.get_hidden_state_vae(unrolled_state[i])
                unrolled_pred_representation_mu.append(pred_representation_mu)
                unrolled_pred_representation_log_var.append(pred_representation_log_var)

                # DYNAMICS
                action_tokens = self._actions_to_tokens(action_history[:, i].tolist())
                (
                    pred_dynamics_mu,
                    pred_dynamics_log_var,
                ) = self.get_next_hidden_state_vae(current_hidden_state, action_tokens)
                unrolled_pred_dynamics_mu.append(pred_dynamics_mu)
                unrolled_pred_dynamics_log_var.append(pred_dynamics_log_var)

                current_hidden_state = vae_take_sample(
                    pred_dynamics_mu, pred_dynamics_log_var
                )

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
            pred_representation_mu = torch.stack(unrolled_pred_representation_mu, dim=1)
            pred_representation_log_var = torch.stack(
                unrolled_pred_representation_log_var, dim=1
            )
            pred_dynamics_mu = torch.stack(unrolled_pred_dynamics_mu, dim=1)
            pred_dynamics_log_var = torch.stack(unrolled_pred_dynamics_log_var, dim=1)
        else:
            empty_hidden_state_part = torch.empty(
                batch_size, 0, current_hidden_state.shape[-1], device=self.get_device()
            )
            pred_representation_mu = empty_hidden_state_part
            pred_representation_log_var = empty_hidden_state_part
            pred_dynamics_mu = empty_hidden_state_part
            pred_dynamics_log_var = empty_hidden_state_part

        # Transpose from step x batch to batch x step.
        pred_actions_transposed = [list(x) for x in zip(*unrolled_pred_actions)]
        candidate_action_tokens_transposed = [
            list(x) for x in zip(*unrolled_candidate_action_tokens)
        ]

        return MuZeroNetworkOutput(
            pred_policies=pred_policies,
            pred_values=pred_values,
            pred_dynamics_mu=pred_dynamics_mu,
            pred_dynamics_log_var=pred_dynamics_log_var,
            target_representation_mu=pred_representation_mu,
            target_representation_log_var=pred_representation_log_var,
            pred_actions=pred_actions_transposed,
            candidate_action_tokens=candidate_action_tokens_transposed,
        )

    def init_zero(self):
        # todo Initialize all weights to 0. Update as needed
        # Stop deleting my comments and replacing them with docstrings with different meanings.
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
