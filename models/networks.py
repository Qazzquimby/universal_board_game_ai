from typing import List, Tuple, Dict

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
            # Add 1 for padding, and 1 because cardinality is max value, not count.
            self.embedding_layers[feature] = nn.Embedding(
                cardinality + 2, embedding_dim, padding_idx=0
            )

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

    def _apply_transforms(self, state: StateType) -> StateType:
        """Applies transformations to a state dictionary of DataFrames, returning a new state dict."""
        transformed_state = {}
        transforms = self.network_spec.get("transforms", {})

        for table_name, df in state.items():
            if df is None or df.is_empty():
                transformed_state[table_name] = df
                continue

            table_spec = self.network_spec["tables"].get(table_name)
            if not table_spec:
                transformed_state[table_name] = df
                continue

            columns_to_update = {}
            for col_name in table_spec["columns"]:
                if col_name in df.columns:
                    transform = transforms.get(col_name)
                    if transform:
                        original_values = df[col_name]
                        # The transform function needs the original state.
                        transformed_values = [
                            transform(val, state) for val in original_values
                        ]
                        columns_to_update[col_name] = transformed_values

            if columns_to_update:
                transformed_state[table_name] = df.with_columns(columns_to_update)
            else:
                transformed_state[table_name] = df

        return transformed_state

    def _state_to_tokens(self, state: StateType) -> torch.Tensor:
        """
        Converts a game state (dict of DataFrames) into a tensor of token embeddings.
        Assumes that any necessary transformations have already been applied to the state.
        """
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
                        # Use 0 for padding/None index
                        val = 0
                    else:
                        if isinstance(val, bool):
                            val = int(val)
                        val = val + 1

                    val_tensor = torch.tensor([val], device=device, dtype=torch.long)
                    token_embedding += self.embedding_layers[col_name](val_tensor)
                all_tokens.append(token_embedding)

        if not all_tokens:
            return torch.empty(1, 0, self.embedding_dim, device=device)

        return torch.cat(all_tokens, dim=0).unsqueeze(0)  # (1, num_tokens, dim)

    def _action_to_token(self, action: ActionType) -> torch.Tensor:
        """Converts a single action into an embedding token."""
        device = self.get_device()
        action_embedding = torch.zeros(1, self.embedding_dim, device=device)
        action_spec = self.network_spec["action_space"]
        action_components = action_spec["components"]

        # Handle simple actions (like int for Connect4)
        if not isinstance(action, (list, tuple)):
            action = [action]

        for i, comp_name in enumerate(action_components):
            val = action[i]
            # Add 1 to shift values for padding_idx=0
            val_tensor = torch.tensor([val + 1], device=device, dtype=torch.long)
            action_embedding += self.embedding_layers[comp_name](val_tensor)
        return action_embedding.squeeze(0)  # (dim,)

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

    def predict(
        self, state_with_key: StateWithKey, legal_actions: List[ActionType]
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Performs a forward pass for a single state during MCTS search.
        Returns a dictionary of action priors and a value estimate.
        """
        self.eval()
        with torch.no_grad():
            state = self._apply_transforms(state_with_key.state)
            game_embedding, value_tensor = self._get_state_embedding_and_value(state)
            value = value_tensor.squeeze().cpu().item()

            if not legal_actions:
                return {}, value

            # Score legal actions
            action_tokens = torch.stack(
                [self._action_to_token(a) for a in legal_actions]
            )
            state_embedding_expanded = game_embedding.expand(len(legal_actions), -1)
            policy_input = torch.cat([state_embedding_expanded, action_tokens], dim=1)
            scores = self.policy_head(policy_input).squeeze(-1)
            policy_probs = F.softmax(scores, dim=0)

            policy_dict = {
                action: prob.item() for action, prob in zip(legal_actions, policy_probs)
            }
            return policy_dict, value

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
                # Add 1 to shift values for padding_idx=0. Assumes -1 for None in batch.
                values = state_batch[feature_key] + 1
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
            # Cannot process anything, return empty tensors.
            # Note: The dimensions might need adjustment depending on the loss function.
            # Here, we return 1D value preds and 2D policy logits.
            return torch.empty(0, 0, device=device), torch.empty(0, device=device)

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
        # This part becomes more complex because each item in the batch can have a
        # different number of legal actions. We process them all together and then
        # group the results.

        flat_action_tokens = []
        batch_indices_for_policy = []
        for i in range(batch_size):
            if not legal_actions[i]:
                continue
            action_tokens = torch.stack(
                [self._action_to_token(a) for a in legal_actions[i]]
            )
            flat_action_tokens.append(action_tokens)
            batch_indices_for_policy.extend([i] * len(legal_actions[i]))

        if not batch_indices_for_policy:
            # If no legal actions in the entire batch, return empty logits.
            # This needs to be handled by the caller/loss function.
            return torch.empty(batch_size, 0, device=device), value_preds

        flat_action_tokens_tensor = torch.cat(flat_action_tokens, dim=0)
        batch_indices_tensor = torch.tensor(
            batch_indices_for_policy, device=device, dtype=torch.long
        )

        # Select the corresponding state embeddings for each action
        state_embs_for_policy = game_token_output[batch_indices_tensor]

        # Combine state and action embeddings and get scores
        policy_input = torch.cat(
            [state_embs_for_policy, flat_action_tokens_tensor], dim=1
        )
        scores = self.policy_head(policy_input).squeeze(-1)

        # The output `scores` is a flat tensor. The loss function will need to
        # handle this. We'll return the raw scores and let the loss function apply softmax.
        # However, CrossEntropyLoss expects logits in a (N, C) tensor.
        # We will pad the scores for each batch item.
        scores_by_item = []
        start_idx = 0
        for i in range(batch_size):
            num_actions = len(legal_actions[i])
            if num_actions > 0:
                scores_by_item.append(scores[start_idx : start_idx + num_actions])
                start_idx += num_actions
            else:
                scores_by_item.append(torch.empty(0, device=device))

        # Pad the list of score tensors to create a single (batch_size, max_actions) tensor.
        policy_logits = nn.utils.rnn.pad_sequence(
            scores_by_item, batch_first=True, padding_value=-torch.inf
        )

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
