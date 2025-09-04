import torch
import torch.nn as nn

from environments.base import BaseEnvironment, StateType, ActionType


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

    def get_device(self):
        return next(self.parameters()).device
