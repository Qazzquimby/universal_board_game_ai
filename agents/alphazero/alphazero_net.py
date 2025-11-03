from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, StateWithKey, ActionType
from models.networks import BaseTokenizingNet


class AlphaZeroNet(BaseTokenizingNet):
    """
    A generic AlphaZero-style network that uses a transformer to process
    the game state represented as a dictionary of DataFrames.
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

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # --- Heads ---
        self.value_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())
        policy_input_dim = embedding_dim + embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.cache = {}

    def _get_state_embedding_and_value(
        self, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes a single state to get its embedding and value."""
        tokens = self.tokenize_state(state)  # (1, num_tokens, dim)

        # Prepend game token
        game_token = self.game_token.expand(1, -1, -1)
        sequence = torch.cat([game_token, tokens], dim=1)

        transformer_output = self.transformer_encoder(sequence)
        game_token_output = transformer_output[:, 0, :]  # (1, dim)

        value = self.value_head(game_token_output)
        return game_token_output, value

    def predict_single(
        self, state_with_key: StateWithKey, legal_actions: List[ActionType]
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Performs a forward pass for a single state during MCTS search.
        Returns a dictionary of action priors and a value estimate.
        """
        self.eval()
        with torch.no_grad():
            state = self.apply_transforms(state_with_key.state)
            game_embedding, value_tensor = self._get_state_embedding_and_value(state)
            value = value_tensor.squeeze().cpu().item()

            if not legal_actions:
                return {}, value

            # Score legal actions
            action_tokens = self.tokenize_actions(legal_actions)
            state_embedding_expanded = game_embedding.expand(len(legal_actions), -1)
            policy_input = torch.cat([state_embedding_expanded, action_tokens], dim=1)
            scores = self.policy_head(policy_input).squeeze(-1)
            policy_probs = F.softmax(scores, dim=0)

            # policy_dict = {
            #     action: prob.item() for action, prob in zip(legal_actions, policy_probs)
            # }
            # I'm changing this to an enumeration to handle dynamic action spaces
            # todo this will break anything that expects the key to be the actual action in connect4
            policy_dict = {
                action_index: prob.item()
                for action_index, prob in enumerate(policy_probs)
            }

            return policy_dict, value

    def forward(
        self,
        state_tokens: torch.Tensor,
        state_padding_mask: torch.Tensor,
        action_tokens: torch.Tensor,
        action_batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for a batch of states during training.
        The input is pre-tokenized state and action tensors.
        """
        batch_size = state_tokens.size(0)
        device = state_tokens.device

        game_token = self.game_token.expand(batch_size, -1, -1)
        sequence = torch.cat([game_token, state_tokens], dim=1)

        game_token_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=state_tokens.device
        )
        full_padding_mask = torch.cat([game_token_mask, state_padding_mask], dim=1)

        transformer_output = self.transformer_encoder(
            sequence, src_key_padding_mask=full_padding_mask
        )
        game_token_output = transformer_output[:, 0, :]  # (batch, dim)

        # --- Value Head ---
        value_preds = self.value_head(game_token_output).squeeze(-1)

        # --- Policy Head ---
        if action_tokens.numel() == 0:
            # No legal actions in the entire batch.
            return torch.empty(batch_size, 0, device=device), value_preds

        # Select the corresponding state embeddings for each action
        state_embs_for_policy = game_token_output[action_batch_indices]

        # Combine state and action embeddings and get scores
        policy_input = torch.cat([state_embs_for_policy, action_tokens], dim=1)
        scores = self.policy_head(policy_input).squeeze(-1)

        # The output `scores` is a flat tensor. We need to pad them into a
        # (batch_size, max_actions) tensor for the loss function.
        action_lengths = torch.bincount(action_batch_indices, minlength=batch_size)
        max_actions = action_lengths.max().item()

        policy_logits = torch.full((batch_size, max_actions), -torch.inf, device=device)

        # Create a mask for scattering the scores into the padded tensor
        action_indices = torch.cat(
            [torch.arange(L, device=device) for L in action_lengths.tolist()]
        )
        policy_logits[action_batch_indices, action_indices] = scores

        return policy_logits, value_preds

    def init_zero(self):
        """
        Initializes the weights of the policy and value heads to zero.
        """
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
