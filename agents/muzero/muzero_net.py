from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Tuple, Optional

import einops
from jaxtyping import Float, Bool
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType
from models.networks import BaseTokenizingNet
from algorithms.sampling import take_sample


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


class RootStateObservationToRevealedLatentSampler(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.state_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.enc_to_latent_mu = nn.Linear(embedding_dim, embedding_dim)
        self.enc_to_latent_log_var = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        state_tokens: Float[torch.Tensor, "batch seq_len emb_dim"],
        state_padding_mask: Optional[Float[torch.Tensor, "batch seq_len"]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch emb_dim"], Float[torch.Tensor, "batch emb_dim"]
    ]:
        batch_size = state_tokens.shape[0]

        # Prepend game token
        game_token = self.game_token.expand(batch_size, -1, -1)
        sequence = torch.cat([game_token, state_tokens], dim=1)

        if state_padding_mask is not None:
            # The mask needs to be extended for the game token.
            game_token_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=state_padding_mask.device
            )
            padding_mask = torch.cat([game_token_mask, state_padding_mask], dim=1)
        else:
            padding_mask = None

        transformer_output = self.state_transformer_encoder(
            sequence,
            src_key_padding_mask=~padding_mask
            # our mask uses True for real, False for pad. They want the opposite.
        )
        game_token_output = transformer_output[:, 0, :]
        mu = self.enc_to_latent_mu(game_token_output)
        log_var = self.enc_to_latent_log_var(game_token_output)
        return mu, log_var

    if typing.TYPE_CHECKING:
        __call__ = forward


class StateLatentAndActionsToPolicyLogits(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        policy_input_dim = embedding_dim + embedding_dim
        self.state_latent_and_action_to_policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        state_latent: Float[torch.Tensor, "batch emb_dim"],
        action_token: Float[torch.Tensor, "batch action emb_dim"],
    ) -> Float[torch.Tensor, "batch action"]:
        # repeat state to match action token height with einops
        state_latent = einops.repeat(
            state_latent, "batch emb -> batch action emb", action=action_token.shape[1]
        )
        policy_input = torch.cat([state_latent, action_token], dim=-1)
        prior_logits = self.state_latent_and_action_to_policy_head(policy_input)
        return prior_logits.squeeze(-1)

    if typing.TYPE_CHECKING:
        __call__ = forward


class StateLatentAndActionToSuccessorLatentSampler(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        dynamics_hidden_dim = embedding_dim * 2
        action_embedding_dim = embedding_dim

        self.state_and_action_to_successor_mu = nn.Linear(
            dynamics_hidden_dim, embedding_dim
        )
        self.state_and_action_to_successor_log_var = nn.Linear(
            dynamics_hidden_dim, embedding_dim
        )

        self.state_and_action_to_successor_base = nn.Sequential(
            nn.Linear(embedding_dim + action_embedding_dim, dynamics_hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        state_latent: Float[torch.Tensor, "batch emb_dim"],
        action_token: Float[torch.Tensor, "batch action emb_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "batch action emb_dim"],
        Float[torch.Tensor, "batch action emb_dim"],
    ]:
        state_latent = einops.repeat(
            state_latent,
            "batch emb -> batch action emb",
            action=action_token.shape[1],
        )

        dynamics_input = torch.cat([state_latent, action_token], dim=-1)
        base_output = self.state_and_action_to_successor_base(dynamics_input)

        mu = self.state_and_action_to_successor_mu(base_output)
        log_var = self.state_and_action_to_successor_log_var(base_output)
        return mu, log_var

    if typing.TYPE_CHECKING:
        __call__ = forward


class StateLatentToValue(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.latent_to_value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self, state_latent: Float[torch.Tensor, "batch emb_dim"]
    ) -> Float[torch.Tensor, "batch"]:
        value_pred = self.latent_to_value_head(state_latent).squeeze(-1)
        return value_pred

    if typing.TYPE_CHECKING:
        __call__ = forward


class StateLatentToActionsAndPriors(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_actions: int = 5):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

        self.fc_vectors = nn.Linear(
            self.embedding_dim, self.num_actions * self.embedding_dim
        )
        self.fc_weight_logits = nn.Linear(self.embedding_dim, self.num_actions)

    def forward(self, state_latent: Float[torch.Tensor, "batch emb_dim"]):
        actions_pred = self.fc_vectors(state_latent).view(
            -1, self.num_actions, self.embedding_dim
        )
        logits = self.fc_weight_logits(state_latent)
        weights_pred = F.softmax(logits, dim=-1)
        return actions_pred, weights_pred

    if typing.TYPE_CHECKING:
        __call__ = forward


class MuZeroNet(BaseTokenizingNet):
    def __init__(
        self,
        env: BaseEnvironment,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        num_actions_for_inner_nodes: int = 5,
    ):
        super().__init__(env=env, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

        self.root_state_observation_to_revealed_latent_sampler = (
            RootStateObservationToRevealedLatentSampler(
                embedding_dim=self.embedding_dim,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
            )
        )

        self.state_latent_and_action_to_successor_latent_sampler = (
            StateLatentAndActionToSuccessorLatentSampler(
                embedding_dim=self.embedding_dim
            )
        )

        self.state_latent_to_actions_and_priors = StateLatentToActionsAndPriors(
            embedding_dim=self.embedding_dim,
            num_actions=num_actions_for_inner_nodes,
        )

        self.state_latent_and_actions_to_policy_logits = (
            StateLatentAndActionsToPolicyLogits(embedding_dim=self.embedding_dim)
        )

        self.state_latent_to_value = StateLatentToValue(
            embedding_dim=self.embedding_dim
        )

    def get_root_state_observation_to_revealed_latent_sampler_params(
        self, state: StateType
    ) -> Tuple[
        Float[torch.Tensor, "batch emb_dim"], Float[torch.Tensor, "batch emb_dim"]
    ]:
        state_tokens = self.tokenize_state(state)
        (
            mu,
            log_var,
        ) = self.root_state_observation_to_revealed_latent_sampler(state_tokens)
        return mu, log_var

    def get_state_latent_to_successor_latent_sampler_params(
        self,
        state_latent: Float[torch.Tensor, "batch emb_dim"],
        action_token: Float[torch.Tensor, "batch emb_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "batch emb_dim"], Float[torch.Tensor, "batch emb_dim"]
    ]:
        mu, log_var = self.state_latent_and_action_to_successor_latent_sampler(
            state_latent=state_latent, action_token=action_token
        )
        return mu, log_var

    def get_state_latent_to_actions(
        self, state_latent: Float[torch.Tensor, "batch emb_dim"]
    ) -> Tuple[
        Float[torch.Tensor, "batch emb_dim"], Float[torch.Tensor, "batch emb_dim"]
    ]:
        state_latent_batch = state_latent
        (
            action_tokens,
            action_weights,
        ) = self.state_latent_to_actions_and_priors(state_latent_batch)
        return action_tokens, action_weights

    def forward(
        self,
        initial_state_tokens: Float[torch.Tensor, "batch seq emb"],
        initial_state_padding_mask: Bool[torch.Tensor, "batch seq"],
        actual_action_token_history: Float[torch.Tensor, "batch inner_unroll emb"],
        candidate_action_tokens: Float[torch.Tensor, "batch unroll action emb"],
        candidate_action_tokens_mask: Bool[torch.Tensor, "batch unroll action"],
        unrolled_states_tokens: Float[torch.Tensor, "batch inner_unroll seq emb"],
        unrolled_states_padding_mask: Bool[torch.Tensor, "batch inner_unroll seq"],
        step_mask: Bool[torch.Tensor, "batch unroll"],
    ) -> MuZeroNetworkOutput:

        (
            batch_size,
            num_unroll_steps,
            max_actions,
            emb_dim,
        ) = candidate_action_tokens.shape
        num_inner_unroll_steps = num_unroll_steps - 1

        unrolled_pred_policies = []
        unrolled_pred_values = []
        unrolled_pred_successor_mu = []
        unrolled_pred_successor_log_var = []
        unrolled_target_representation_mu = []
        unrolled_target_representation_log_var = []

        (
            revelation_mu,
            revelation_log_var,
        ) = self.root_state_observation_to_revealed_latent_sampler(
            initial_state_tokens, initial_state_padding_mask
        )
        assert not revelation_mu.isnan().any() and not revelation_log_var.isnan().any()
        current_latent = take_sample(revelation_mu, revelation_log_var)

        for step_index in range(num_unroll_steps):
            valid_steps = step_mask[:, step_index]
            # POLICY
            actual_action_tokens = candidate_action_tokens[:, step_index]
            prior_logits = self.state_latent_and_actions_to_policy_logits(
                state_latent=current_latent, action_token=actual_action_tokens
            )
            _batch, _action = prior_logits.shape

            action_mask = candidate_action_tokens_mask[:, step_index]
            assert prior_logits.shape == action_mask.shape
            prior_logits = prior_logits.masked_fill(~action_mask, float("-inf"))

            if valid_steps.any():
                assert action_mask[valid_steps].any(dim=1).all()

            prior = torch.zeros_like(prior_logits)
            if valid_steps.any():
                i = valid_steps.nonzero(as_tuple=True)[0]
                prior[i] = torch.softmax(prior_logits[i], dim=1)

            unrolled_pred_policies.append(prior)

            # VALUE
            pred_value = self.state_latent_to_value(current_latent)
            unrolled_pred_values.append(pred_value)

            if step_index < num_inner_unroll_steps:
                # DYNAMICS, predict next latent
                # add action dimension
                actual_action_tokens = actual_action_token_history[
                    :, step_index
                ].unsqueeze(1)
                (
                    successor_mu,
                    successor_log_var,
                ) = self.state_latent_and_action_to_successor_latent_sampler(
                    state_latent=current_latent, action_token=actual_action_tokens
                )
                successor_mu = successor_mu.squeeze(1)
                successor_log_var = successor_log_var.squeeze(1)
                unrolled_pred_successor_mu.append(successor_mu)
                unrolled_pred_successor_log_var.append(successor_log_var)

                next_hidden_state = take_sample(successor_mu, successor_log_var)
                current_latent = next_hidden_state

                # TARGET REPRESENTATION
                # Get the latent from observed state->latent to
                # use as a target for latent->latent
                (
                    target_representation_mu,
                    target_representation_log_var,
                ) = self.root_state_observation_to_revealed_latent_sampler(
                    unrolled_states_tokens[:, step_index],
                    unrolled_states_padding_mask[:, step_index],
                )
                unrolled_target_representation_mu.append(target_representation_mu)
                unrolled_target_representation_log_var.append(
                    target_representation_log_var
                )

        pred_policies = torch.stack(unrolled_pred_policies, dim=1)
        pred_values = torch.stack(unrolled_pred_values, dim=1)

        if num_unroll_steps > 0:
            successor_mu = torch.stack(unrolled_pred_successor_mu, dim=1)
            successor_log_var = torch.stack(unrolled_pred_successor_log_var, dim=1)
            target_representation_mu = torch.stack(
                unrolled_target_representation_mu, dim=1
            )
            target_representation_log_var = torch.stack(
                unrolled_target_representation_log_var, dim=1
            )
        else:
            empty_hidden_state_part = torch.empty(
                batch_size,
                0,
                current_latent.shape[-1],
                device=self.get_device(),
            )
            successor_mu = empty_hidden_state_part
            successor_log_var = empty_hidden_state_part
            target_representation_mu = empty_hidden_state_part
            target_representation_log_var = empty_hidden_state_part

        return MuZeroNetworkOutput(
            pred_policies=pred_policies,
            pred_values=pred_values,
            pred_dynamics_mu=successor_mu,
            pred_dynamics_log_var=successor_log_var,
            target_representation_mu=target_representation_mu,
            target_representation_log_var=target_representation_log_var,
        )

    if typing.TYPE_CHECKING:
        __call__ = forward
