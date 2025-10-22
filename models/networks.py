import torch
import torch.nn as nn

from typing import Dict, List, Tuple

from environments.base import BaseEnvironment, StateType, ActionType, DataFrame


class BaseTokenizingNet(nn.Module):
    """Base network for models that tokenize game state DataFrames."""

    def __init__(self, env: BaseEnvironment, embedding_dim: int):
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
        # Create a batched version of the state with batch_idx = 0
        batched_state = {}
        has_data = False
        for table_name, df in state.items():
            if df is None or df.is_empty():
                batched_state[table_name] = df
                continue

            has_data = True
            new_data = [row + [0] for row in df._data]
            new_columns = df.columns + ["batch_idx"]
            batched_state[table_name] = DataFrame(data=new_data, columns=new_columns)

        if not has_data:
            return torch.empty(1, 0, self.embedding_dim, device=device)

        token_sequences, token_mask = self._batched_state_to_tokens_and_mask(
            batched_state, batch_size=1
        )

        return token_sequences[0].unsqueeze(0)  # (1, num_tokens, dim)

    def _get_transformer_output(
        self, state: StateType, batch_size: int = 1
    ) -> torch.Tensor:
        padded_tokens, padding_mask = self._batched_state_to_tokens_and_mask(
            state, batch_size=batch_size
        )

        game_token = self.game_token.expand(batch_size, -1, -1)
        sequence = torch.cat([game_token, padded_tokens], dim=1)

        game_token_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=padded_tokens.device
        )
        full_padding_mask = torch.cat([game_token_mask, padding_mask], dim=1)

        transformer_output = self.transformer_encoder(
            sequence, src_key_padding_mask=full_padding_mask
        )
        return transformer_output

    def _batched_state_to_tokens_and_mask(
        self, state_batch: Dict[str, DataFrame], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a batch of game states (represented as a dictionary of batched DataFrames)
        padding tokens, mask
        """
        device = self.get_device()
        all_tokens = []
        all_batch_indices = []

        for table_name, df in state_batch.items():
            table_spec = self.network_spec["tables"].get(table_name)
            if df is None or df.is_empty() or not table_spec:
                continue

            columns = table_spec["columns"]
            num_rows = df.height
            table_token_embeddings = torch.zeros(
                num_rows, self.embedding_dim, device=device
            )
            # todo handle numeric columns
            for col_name in columns:
                raw_values = df[col_name]
                # Convert None to -1 for later processing
                final_values = [v if v is not None else -1 for v in raw_values]
                values_tensor = torch.tensor(
                    final_values, dtype=torch.long, device=device
                )
                # Add 1 to shift values for padding_idx=0
                values_tensor += 1
                table_token_embeddings += self.embedding_layers[col_name](values_tensor)

            all_tokens.append(table_token_embeddings)
            if batch_size > 1 or "batch_idx" in df.columns:
                all_batch_indices.append(
                    torch.tensor(df["batch_idx"], dtype=torch.long, device=device)
                )
            else:
                all_batch_indices.append(
                    torch.zeros(df.height, dtype=torch.long, device=device)
                )

        if not all_tokens:
            pass  # TODO handle escape return

        token_tensor = torch.cat(all_tokens, dim=0)
        batch_indices_tensor = torch.cat(all_batch_indices, dim=0)

        token_sequences = []
        for i in range(batch_size):
            mask = batch_indices_tensor == i
            token_sequences.append(token_tensor[mask])

        padded_tokens = nn.utils.rnn.pad_sequence(
            token_sequences, batch_first=True, padding_value=0.0
        )
        # batch_size, max_token_len, embedding_dim

        original_lengths = [len(t) for t in token_sequences]
        max_len = padded_tokens.size(1)
        padding_mask = (
            torch.arange(max_len, device=padded_tokens.device)[None, :]
            >= torch.tensor(original_lengths, device=padded_tokens.device)[:, None]
        )

        return padded_tokens, padding_mask

    def _actions_to_tokens(self, actions: List[ActionType]) -> torch.Tensor:
        """Converts a batch of actions into embedding tokens."""
        device = self.get_device()
        if not actions:
            return torch.empty(0, self.embedding_dim, device=device)

        action_spec = self.network_spec["action_space"]
        action_components = action_spec["components"]
        batch_size = len(actions)
        action_embeddings = torch.zeros(batch_size, self.embedding_dim, device=device)

        # Normalize actions to be list of lists, e.g. [1, 2] -> [[1], [2]]
        if not isinstance(actions[0], (list, tuple)):
            normalized_actions = [[a] for a in actions]
        else:
            normalized_actions = actions

        # Transpose for batch processing: [[a1,b1], [a2,b2]] -> [[a1,a2], [b1,b2]]
        transposed_actions = list(zip(*normalized_actions))

        for i, comp_name in enumerate(action_components):
            vals = transposed_actions[i]
            # Add 1 to shift values for padding_idx=0
            val_tensor = torch.tensor(vals, device=device, dtype=torch.long) + 1
            action_embeddings += self.embedding_layers[comp_name](val_tensor)
        return action_embeddings

    def _action_to_token(self, action: ActionType) -> torch.Tensor:
        """Converts a single action into an embedding token."""
        return self._actions_to_tokens([action]).squeeze(0)

    def get_device(self):
        return next(self.parameters()).device
