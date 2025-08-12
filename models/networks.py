from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, StateWithKey


class AlphaZeroNet(nn.Module):
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
        super().__init__()
        self.env = env
        self.embedding_dim = embedding_dim
        self.network_spec = env.get_network_spec()

        # --- Embeddings ---
        self.embedding_layers = nn.ModuleDict()
        for feature, cardinality in self.network_spec["cardinalities"].items():
            # Add 1 to cardinality for a "None" or "padding" token
            self.embedding_layers[feature] = nn.Embedding(
                cardinality + 1, embedding_dim
            )

        self.action_embedding = nn.Embedding(env.num_action_types, embedding_dim)

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

    def _state_to_tokens(self, state: StateType) -> torch.Tensor:
        """Converts a game state (dict of DataFrames) into a tensor of token embeddings."""
        device = self.get_device()
        all_tokens = []

        for table_name, table_spec in self.network_spec["tables"].items():
            df = state.get(table_name)
            if df is None or df.is_empty():
                continue

            columns = table_spec["columns"]
            cardinalities = self.network_spec["cardinalities"]

            for row_data in df.rows():
                token_embedding = torch.zeros(1, self.embedding_dim, device=device)
                for i, col_name in enumerate(columns):
                    val = row_data[i]

                    if val is None:
                        # Use max cardinality index for None
                        val = cardinalities[col_name]
                    elif isinstance(val, bool):
                        val = int(val)

                    val_tensor = torch.tensor([val], device=device, dtype=torch.long)
                    token_embedding += self.embedding_layers[col_name](val_tensor)
                all_tokens.append(token_embedding)

        if not all_tokens:
            return torch.empty(1, 0, self.embedding_dim, device=device)

        return torch.cat(all_tokens, dim=0).unsqueeze(0)  # (1, num_tokens, dim)

    def _get_state_embedding_and_value(
        self, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes a single state to get its embedding and value."""
        tokens = self._state_to_tokens(state)  # (1, num_tokens, dim)

        # Prepend game token
        game_token = self.game_token.expand(1, -1, -1)
        sequence = torch.cat([game_token, tokens], dim=1)

        transformer_output = self.transformer_encoder(sequence)
        game_token_output = transformer_output[:, 0, :]  # (1, dim)

        value = self.value_head(game_token_output)
        return game_token_output, value

    def predict(self, state_with_key: StateWithKey) -> Tuple[np.ndarray, float]:
        """
        Performs a forward pass for a single state during MCTS search.
        """
        self.eval()
        with torch.no_grad():
            state = state_with_key.state
            game_embedding, value_tensor = self._get_state_embedding_and_value(state)
            value = value_tensor.squeeze().cpu().item()

            legal_actions = self.env.get_legal_actions()
            if not legal_actions:
                policy = np.zeros(self.env.num_action_types)
                return policy, value

            # Score legal actions
            action_indices = [
                self.env.map_action_to_policy_index(a) for a in legal_actions
            ]
            action_indices_tensor = torch.tensor(
                action_indices, device=self.get_device(), dtype=torch.long
            )
            action_embs = self.action_embedding(action_indices_tensor)

            state_embedding_expanded = game_embedding.expand(len(legal_actions), -1)
            policy_input = torch.cat([state_embedding_expanded, action_embs], dim=1)
            scores = self.policy_head(policy_input).squeeze(-1)

            # Create full policy vector
            policy_logits = torch.full(
                (self.env.num_action_types,), -torch.inf, device=self.get_device()
            )
            policy_logits[action_indices] = scores
            policy_probs = F.softmax(policy_logits, dim=0)

            return policy_probs.cpu().numpy(), value

    def forward(
        self, state_batch: List[StateType]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for a batch of states during training.
        """
        device = self.get_device()

        # --- State Processing ---
        token_sequences = [
            self._state_to_tokens(state).squeeze(0) for state in state_batch
        ]
        padded_tokens = nn.utils.rnn.pad_sequence(
            token_sequences, batch_first=True, padding_value=0
        )
        src_key_padding_mask = padded_tokens.sum(dim=-1) == 0

        batch_size = padded_tokens.shape[0]
        game_tokens = self.game_token.expand(batch_size, -1, -1)
        sequences = torch.cat([game_tokens, padded_tokens], dim=1)
        game_token_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=device
        )
        full_mask = torch.cat([game_token_mask, src_key_padding_mask], dim=1)

        transformer_output = self.transformer_encoder(
            src=sequences, src_key_padding_mask=full_mask
        )
        game_token_output = transformer_output[:, 0, :]  # (batch, dim)

        # --- Value Head ---
        value_preds = self.value_head(game_token_output).squeeze(-1)

        # --- Policy Head ---
        num_actions = self.env.num_action_types
        action_indices = torch.arange(num_actions, device=device)
        action_embs = self.action_embedding(action_indices)

        game_token_expanded = game_token_output.unsqueeze(1).expand(
            -1, num_actions, -1
        )
        action_embs_expanded = action_embs.unsqueeze(0).expand(batch_size, -1, -1)
        policy_input = torch.cat([game_token_expanded, action_embs_expanded], dim=-1)

        policy_input_flat = policy_input.view(batch_size * num_actions, -1)
        policy_logits_flat = self.policy_head(policy_input_flat)
        policy_logits = policy_logits_flat.view(batch_size, num_actions)

        return policy_logits, value_preds

    def get_device(self):
        return next(self.parameters()).device

    def init_zero(self):
        """
        Initializes the weights of the policy and value heads to zero.
        """
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
