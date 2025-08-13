from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, StateWithKey, ActionType


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
        transforms = self.network_spec.get("transforms", {})

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

                    transform = transforms.get(col_name)
                    if transform:
                        val = transform(val, state)

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
        self, state_batch: dict, legal_actions: List[List[ActionType]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for a batch of states during training.
        The input is a dictionary of tensors, where each tensor represents a
        flattened column of data from all states in the batch.
        `legal_actions` is a list of lists, where each inner list contains the
        legal actions for a state in the batch.
        """
        device = self.get_device()
        all_tokens = []
        all_batch_indices = []

        # --- State Processing: Convert input tensors to token embeddings ---
        for table_name, table_spec in self.network_spec["tables"].items():
            columns = table_spec["columns"]
            first_col_key = f"{table_name}_{columns[0]}"
            if first_col_key not in state_batch:
                continue

            num_rows = state_batch[first_col_key].shape[0]
            if num_rows == 0:
                continue

            table_token_embeddings = torch.zeros(
                num_rows, self.embedding_dim, device=device
            )
            for col_name in columns:
                feature_key = f"{table_name}_{col_name}"
                values = state_batch[feature_key]
                table_token_embeddings += self.embedding_layers[col_name](values)

            all_tokens.append(table_token_embeddings)
            all_batch_indices.append(state_batch[f"{table_name}_batch_idx"])

        # Determine batch size. Try from batch indices first, then fallback to a 'game' tensor.
        batch_size = 0
        if all_batch_indices:
            batch_indices_tensor = torch.cat(all_batch_indices, dim=0)
            if batch_indices_tensor.numel() > 0:
                batch_size = int(batch_indices_tensor.max().item() + 1)

        if batch_size == 0 and "game_current_player" in state_batch:
            batch_size = state_batch["game_current_player"].shape[0]

        if batch_size == 0:
            policy_logits = torch.empty(0, self.env.num_action_types, device=device)
            value_preds = torch.empty(0, device=device)
            return policy_logits, value_preds

        if all_tokens:
            token_tensor = torch.cat(all_tokens, dim=0)
            token_sequences = []
            for i in range(batch_size):
                mask = batch_indices_tensor == i
                token_sequences.append(token_tensor[mask])
        else:
            token_sequences = [
                torch.empty(0, self.embedding_dim, device=device)
                for _ in range(batch_size)
            ]

        padded_tokens = nn.utils.rnn.pad_sequence(
            token_sequences, batch_first=True, padding_value=0
        )
        src_key_padding_mask = padded_tokens.sum(dim=-1) == 0

        game_tokens = self.game_token.expand(batch_size, -1, -1)
        sequences = torch.cat([game_tokens, padded_tokens], dim=1)
        game_token_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([game_token_mask, src_key_padding_mask], dim=1)

        transformer_output = self.transformer_encoder(
            src=sequences, src_key_padding_mask=full_mask
        )
        game_token_output = transformer_output[:, 0, :]  # (batch, dim)

        # --- Value Head ---
        value_preds = self.value_head(game_token_output).squeeze(-1)

        # --- Policy Head ---
        policy_logits = torch.full(
            (batch_size, self.env.num_action_types),
            -torch.inf,
            device=device,
        )

        batch_indices_for_policy = []
        action_indices_for_policy = []

        for i in range(batch_size):
            if not legal_actions[i]:
                continue

            num_legal_actions = len(legal_actions[i])
            batch_indices_for_policy.extend([i] * num_legal_actions)
            action_indices_for_policy.extend(
                [self.env.map_action_to_policy_index(a) for a in legal_actions[i]]
            )

        if batch_indices_for_policy:
            batch_indices_tensor = torch.tensor(
                batch_indices_for_policy, device=device, dtype=torch.long
            )
            action_indices_tensor = torch.tensor(
                action_indices_for_policy, device=device, dtype=torch.long
            )

            # Select the corresponding state embeddings
            state_embs_for_policy = game_token_output[batch_indices_tensor]

            # Get action embeddings
            action_embs_for_policy = self.action_embedding(action_indices_tensor)

            # Combine and get scores
            policy_input = torch.cat(
                [state_embs_for_policy, action_embs_for_policy], dim=1
            )
            scores = self.policy_head(policy_input).squeeze(-1)

            # Scatter scores back into the policy_logits tensor
            policy_logits[batch_indices_tensor, action_indices_tensor] = scores

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
