from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, ActionType
from models.networks import BaseTokenizingNet


@dataclass
class UnrollStepOutput:
    pred_policy: torch.Tensor
    pred_value: torch.Tensor
    next_hidden_state: Optional[torch.Tensor]
    target_representation_mu: Optional[torch.Tensor]
    target_representation_log_var: Optional[torch.Tensor]
    pred_dynamics_mu: Optional[torch.Tensor]
    pred_dynamics_log_var: Optional[torch.Tensor]


@dataclass
class MuZeroNetworkOutput:
    pred_policies: torch.Tensor
    pred_values: torch.Tensor
    pred_dynamics_mu: torch.Tensor
    pred_dynamics_log_var: torch.Tensor
    target_representation_mu: torch.Tensor
    target_representation_log_var: torch.Tensor


# Don't delete
# Hidden state vae is a distribution for sampling hidden state tensors from, for stochasticity.
# A hidden state tensor can be used to predict a variable length list of encoded action tokens with size action_dim. These should be stored with the hidden state tensor.


def pad_action_sets(
    action_sets: List[List[List[torch.Tensor]]], embedding_dim: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    # actions are batch, step, action, dim
    if not action_sets:
        return torch.empty(0, 0, 0, 0, device=device), torch.empty(
            0, 0, 0, dtype=torch.bool, device=device
        )

    batch_size = len(action_sets)
    if batch_size == 0:
        return torch.empty(0, 0, 0, embedding_dim, device=device), torch.empty(
            0, 0, 0, dtype=torch.bool, device=device
        )

    max_steps = 0
    max_actions = 0
    for batch in action_sets:
        if len(batch) > max_steps:
            max_steps = len(batch)
        for step in batch:
            if len(step) > max_actions:
                max_actions = len(step)

    padded_tensor = torch.zeros(
        batch_size, max_steps, max_actions, embedding_dim, device=device
    )
    mask = torch.zeros(
        batch_size, max_steps, max_actions, dtype=torch.bool, device=device
    )

    for batch_index, batch in enumerate(action_sets):
        for step_index, actions in enumerate(batch):
            if actions:
                num_actions = len(actions)
                action_tensor = torch.cat(actions, dim=0)
                padded_tensor[batch_index, step_index, :num_actions] = action_tensor
                mask[batch_index, step_index, :num_actions] = True

    return padded_tensor, mask


def take_sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


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
        self.embedding_dim = embedding_dim

        # state_to_root_node_hidden_info_sampler (representation, h)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.state_transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.enc_to_hidden_state_sampler_mu = nn.Linear(embedding_dim, embedding_dim)
        self.enc_to_hidden_state_sampler_log_var = nn.Linear(
            embedding_dim, embedding_dim
        )

        # sampled_root_latent_node_to_successor_enc (dynamics, g)
        # It takes latent_state + action_embedding -> successor_enc
        action_embedding_dim = embedding_dim
        dynamics_hidden_dim = embedding_dim * 2
        self.sampled_root_latent_node_to_successor_enc = nn.Sequential(
            nn.Linear(embedding_dim + action_embedding_dim, dynamics_hidden_dim),
            nn.ReLU(),
        )

        # inner_latent_to_successor_sampler (more dynamics, g)
        # Takes inner latent state
        self.inner_state_to_successor_sampler_mu = nn.Linear(
            embedding_dim, embedding_dim
        )
        self.inner_state_to_successor_sampler_log_var = nn.Linear(
            embedding_dim, embedding_dim
        )

        # get_policy_and_value (prediction, f)
        self.latent_to_value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Tanh(),
        )

        # The policy head scores (hidden_state, action) pairs.
        policy_input_dim = embedding_dim + embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def get_hidden_state_vae(
        self,
        state_tokens: torch.Tensor,
        state_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Representation function (h): Encodes a batch of states into a stochastic hidden state distribution.
        """
        batch_size = state_tokens.shape[0]

        # Prepend game token
        game_token = self.game_token.expand(batch_size, -1, -1)
        sequence = torch.cat([game_token, state_tokens], dim=1)

        if state_padding_mask is not None:
            # The mask needs to be extended for the game token.
            # Game token is not masked, so we add False for it.
            game_token_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=state_padding_mask.device
            )
            padding_mask = torch.cat([game_token_mask, state_padding_mask], dim=1)
        else:
            padding_mask = None

        transformer_output = self.state_transformer_encoder(
            sequence, src_key_padding_mask=padding_mask
        )
        game_token_output = transformer_output[:, 0, :]  # (batch_size, dim)

        mu = self.enc_to_hidden_state_sampler_mu(game_token_output)
        log_var = self.enc_to_hidden_state_sampler_log_var(game_token_output)
        return mu, log_var

    def get_next_hidden_state_vae(
        self, hidden_state: torch.Tensor, action_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamics function (g): Predicts the distribution of the next hidden state
        given a current hidden state and an encoded action token.
        """
        dynamics_input = torch.cat([hidden_state, action_token], dim=1)
        base_output = self.sampled_root_latent_node_to_successor_enc(dynamics_input)

        mu = self.enc_root_and_action_to_successor_sampler_mu(base_output)
        log_var = self.enc_root_and_action_to_successor_sampler_log_var(base_output)
        return mu, log_var

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
        value_preds = self.latent_to_value_head(hidden_state_batch).squeeze(-1)
        return value_preds

    def get_policy_batched(
        self,
        hidden_state: torch.Tensor,
        candidate_action_tokens: List[torch.Tensor],
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
        candidate_action_tokens_batch: List[torch.Tensor],
        hidden_state_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        batch_size = hidden_state_batch.shape[0]
        action_tokens_for_all_batches = []
        batch_indices_for_policy = []
        num_actions_per_item = []
        for i in range(batch_size):
            action_tokens = candidate_action_tokens_batch[i]
            num_actions = action_tokens.shape[0]
            num_actions_per_item.append(num_actions)
            if not action_tokens.numel():
                continue
            # todo maybe can use padded tensor for candidate... to avoid loop
            action_tokens_for_all_batches.append(action_tokens)
            batch_indices_for_policy.extend([i] * num_actions)

        if not batch_indices_for_policy:
            raise ValueError("No actions in batch")

        flat_action_tokens_tensor = torch.cat(action_tokens_for_all_batches, dim=0)
        batch_indices_tensor = torch.tensor(
            batch_indices_for_policy, device=device, dtype=torch.long
        )

        state_embs_for_policy = hidden_state_batch[batch_indices_tensor]
        scores = self._get_policy_scores(
            hidden_states=state_embs_for_policy, action_tokens=flat_action_tokens_tensor
        )

        return scores, num_actions_per_item

    def get_policy(
        self, hidden_state: torch.Tensor, legal_action_tokens: torch.Tensor
    ) -> Dict[int, float]:
        candidate_action_tokens_batch = [legal_action_tokens]

        pred_policy = self.get_policy_batched(
            hidden_state=hidden_state,
            candidate_action_tokens=candidate_action_tokens_batch,
        )
        policy_dict = {}
        if legal_action_tokens.numel():
            # We have a batch of 1, so squeeze out the batch dimension.
            policy_logits_single = pred_policy.squeeze(0)
            policy_probs = F.softmax(policy_logits_single, dim=0)

            # Policy dict maps action index to probability
            policy_dict = {i: prob.item() for i, prob in enumerate(policy_probs)}

        return policy_dict

    def get_value(self, hidden_state: torch.Tensor) -> float:
        pred_value = self.get_value_batched(hidden_state_batch=hidden_state)
        value = pred_value.squeeze().cpu().item()
        return value

    def _unroll_step(
        self,
        i,
        num_unroll_steps,
        current_hidden_state,
        batch_size,
        candidate_action_tokens,
        candidate_action_tokens_mask,
        action_tokens_history,
        unrolled_states_tokens,
        unrolled_states_padding_mask,
    ) -> "UnrollStepOutput":
        # POLICY
        candidate_tokens_for_step = []
        for batch_idx in range(batch_size):
            mask = candidate_action_tokens_mask[batch_idx, i]
            tokens = candidate_action_tokens[batch_idx, i][mask]
            candidate_tokens_for_step.append(tokens)

        pred_policy = self.get_policy_batched(
            hidden_state=current_hidden_state,
            candidate_action_tokens=candidate_tokens_for_step,
        )

        # VALUE
        pred_value = self.get_value_batched(hidden_state_batch=current_hidden_state)

        next_hidden_state = None
        target_representation_mu = None
        target_representation_log_var = None
        pred_dynamics_mu = None
        pred_dynamics_log_var = None

        if i < num_unroll_steps:
            # TARGET REPRESENTATION
            (
                target_representation_mu,
                target_representation_log_var,
            ) = self.get_hidden_state_vae(
                unrolled_states_tokens[:, i], unrolled_states_padding_mask[:, i]
            )

            # DYNAMICS
            action_tokens = action_tokens_history[:, i]
            (
                pred_dynamics_mu,
                pred_dynamics_log_var,
            ) = self.get_next_hidden_state_vae(current_hidden_state, action_tokens)

            next_hidden_state = take_sample(pred_dynamics_mu, pred_dynamics_log_var)

        return UnrollStepOutput(
            pred_policy=pred_policy,
            pred_value=pred_value,
            next_hidden_state=next_hidden_state,
            target_representation_mu=target_representation_mu,
            target_representation_log_var=target_representation_log_var,
            pred_dynamics_mu=pred_dynamics_mu,
            pred_dynamics_log_var=pred_dynamics_log_var,
        )

    def _collate_unrolled_outputs(
        self,
        unrolled_pred_policies,
        unrolled_pred_values,
        unrolled_target_representation_mu,
        unrolled_target_representation_log_var,
        unrolled_pred_dynamics_mu,
        unrolled_pred_dynamics_log_var,
        num_unroll_steps,
        batch_size,
        current_hidden_state_for_shape,
    ):
        # Collate and return all predictions.
        max_actions = max(
            (p.shape[1] for p in unrolled_pred_policies if p.numel() > 0), default=0
        )
        padded_policies = [
            F.pad(p, (0, max_actions - p.shape[1]), "constant", value=-torch.inf)
            if p.numel() > 0
            else torch.full(
                (p.shape[0], max_actions),
                -torch.inf,
                device=p.device,
                dtype=p.dtype,
            )
            for p in unrolled_pred_policies
        ]
        pred_policies = torch.stack(padded_policies, dim=1)

        pred_values = torch.stack(unrolled_pred_values, dim=1)

        if num_unroll_steps > 0:
            target_representation_mu = torch.stack(
                unrolled_target_representation_mu, dim=1
            )
            target_representation_log_var = torch.stack(
                unrolled_target_representation_log_var, dim=1
            )
            pred_dynamics_mu = torch.stack(unrolled_pred_dynamics_mu, dim=1)
            pred_dynamics_log_var = torch.stack(unrolled_pred_dynamics_log_var, dim=1)
        else:
            empty_hidden_state_part = torch.empty(
                batch_size,
                0,
                current_hidden_state_for_shape.shape[-1],
                device=self.get_device(),
            )
            target_representation_mu = empty_hidden_state_part
            target_representation_log_var = empty_hidden_state_part
            pred_dynamics_mu = empty_hidden_state_part
            pred_dynamics_log_var = empty_hidden_state_part

        return (
            pred_policies,
            pred_values,
            pred_dynamics_mu,
            pred_dynamics_log_var,
            target_representation_mu,
            target_representation_log_var,
        )

    def forward(
        self,
        initial_state_tokens: torch.Tensor,
        initial_state_padding_mask: torch.Tensor,
        action_tokens_history: torch.Tensor,
        candidate_action_tokens: torch.Tensor,
        candidate_action_tokens_mask: torch.Tensor,
        unrolled_states_tokens: torch.Tensor,
        unrolled_states_padding_mask: torch.Tensor,
    ) -> MuZeroNetworkOutput:
        # All data variables are batches
        batch_size = action_tokens_history.shape[0]
        num_unroll_steps = action_tokens_history.shape[1]

        unrolled_pred_policies = []
        unrolled_pred_values = []
        unrolled_pred_dynamics_mu = []
        unrolled_pred_dynamics_log_var = []
        unrolled_target_representation_mu = []
        unrolled_target_representation_log_var = []

        hidden_state_mu, hidden_state_log_var = self.get_hidden_state_vae(
            initial_state_tokens, initial_state_padding_mask
        )
        current_hidden_state = take_sample(hidden_state_mu, hidden_state_log_var)

        for i in range(num_unroll_steps + 1):
            step_output = self._unroll_step(
                i,
                num_unroll_steps,
                current_hidden_state,
                batch_size,
                candidate_action_tokens,
                candidate_action_tokens_mask,
                action_tokens_history,
                unrolled_states_tokens,
                unrolled_states_padding_mask,
            )
            unrolled_pred_policies.append(step_output.pred_policy)
            unrolled_pred_values.append(step_output.pred_value)

            if i < num_unroll_steps:
                unrolled_target_representation_mu.append(
                    step_output.target_representation_mu
                )
                unrolled_target_representation_log_var.append(
                    step_output.target_representation_log_var
                )
                unrolled_pred_dynamics_mu.append(step_output.pred_dynamics_mu)
                unrolled_pred_dynamics_log_var.append(step_output.pred_dynamics_log_var)

                current_hidden_state = step_output.next_hidden_state

        (
            pred_policies,
            pred_values,
            pred_dynamics_mu,
            pred_dynamics_log_var,
            target_representation_mu,
            target_representation_log_var,
        ) = self._collate_unrolled_outputs(
            unrolled_pred_policies,
            unrolled_pred_values,
            unrolled_target_representation_mu,
            unrolled_target_representation_log_var,
            unrolled_pred_dynamics_mu,
            unrolled_pred_dynamics_log_var,
            num_unroll_steps,
            batch_size,
            current_hidden_state,
        )

        return MuZeroNetworkOutput(
            pred_policies=pred_policies,
            pred_values=pred_values,
            pred_dynamics_mu=pred_dynamics_mu,
            pred_dynamics_log_var=pred_dynamics_log_var,
            target_representation_mu=target_representation_mu,
            target_representation_log_var=target_representation_log_var,
        )

    def init_zero(self):
        # todo Initialize all weights to 0. Update as needed
        # Stop deleting my comments and replacing them with docstrings with different meanings.
        nn.init.constant_(self.latent_to_value_head[0].weight, 0)
        nn.init.constant_(self.latent_to_value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
