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

    def apply_transforms(self, state: StateType) -> StateType:
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

    def tokenize_state(self, state: StateType) -> torch.Tensor:
        """
        Converts a game state (dict of DataFrames) into a tensor of token embeddings.
        Assumes that any necessary transformations have already been applied to the state.
        """
        assert (
            "batch_idx" not in state["game"]
        ), "Trying to use tokenize_state on batched state. Use tokenize_state_batch instead"
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

        token_sequences, token_mask = self.tokenize_state_batch(
            batched_state, batch_size=1
        )

        return token_sequences[0].unsqueeze(0)  # (1, num_tokens, dim)

    def tokenize_state_batch(
        self, state_batch: Dict[str, DataFrame], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a batch of game states (represented as a dictionary of batched DataFrames)
        into padded token sequences and a padding mask, using a vectorized approach.
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
            # Handle case with no data across the entire batch
            return (
                torch.empty(batch_size, 0, self.embedding_dim, device=device),
                torch.empty(batch_size, 0, dtype=torch.bool, device=device),
            )

        token_tensor = torch.cat(all_tokens, dim=0)
        batch_indices_tensor = torch.cat(all_batch_indices, dim=0)

        # --- Vectorized Padding ---

        # Sort tokens by batch index to group them.
        sorted_indices = torch.argsort(batch_indices_tensor)
        sorted_batch_indices = batch_indices_tensor[sorted_indices]
        sorted_tokens = token_tensor[sorted_indices]

        lengths = torch.bincount(sorted_batch_indices, minlength=batch_size)
        max_len = lengths.max().item()

        # Create sequence indices (i.e., token position within each sequence)
        seq_indices = torch.cat(
            [torch.arange(L, device=device) for L in lengths.tolist()]
        )

        padded_tokens = torch.zeros(
            batch_size, max_len, self.embedding_dim, device=device
        )
        if sorted_tokens.numel() > 0:
            padded_tokens[sorted_batch_indices, seq_indices] = sorted_tokens

        padding_mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]

        return padded_tokens, padding_mask

    def tokenize_actions(self, actions: List[ActionType]) -> torch.Tensor:
        """Converts a batch of actions into embedding tokens using vectorized operations."""
        device = self.get_device()
        if not actions:
            return torch.empty(0, self.embedding_dim, device=device)

        batch_size = len(actions)
        action_spec = self.network_spec["action_space"]

        # Backwards compatibility for simple action spaces (e.g. Connect4)
        if "components" in action_spec and isinstance(action_spec["components"], list):
            action_components = action_spec["components"]
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(1)
            actions_tensor += 1  # For padding_idx

            action_embeddings = torch.zeros(
                batch_size, self.embedding_dim, device=device
            )
            for i, comp_name in enumerate(action_components):
                component_values = actions_tensor[:, i]
                action_embeddings += self.embedding_layers[comp_name](component_values)
            return action_embeddings

        # Handle complex, multi-type actions (e.g. Gobblet)
        action_embeddings = torch.zeros(batch_size, self.embedding_dim, device=device)
        action_types_spec = action_spec["types"]

        for i, action in enumerate(actions):  # todo no loop
            action_type_name: str
            action_dict: dict

            if not isinstance(action, dict):
                action_type_name = type(action).__name__
                action_dict = action.dict()
            else:
                action_dict = action
                # Infer action type from keys
                action_keys = set(action_dict.keys())
                found_type = False
                for type_name, comp_names in action_types_spec.items():
                    if set(comp_names) == action_keys:
                        action_type_name = type_name
                        found_type = True
                        break
                if not found_type:
                    raise ValueError(
                        f"Could not determine action type for dict: {action_dict}"
                    )

            components = action_types_spec[action_type_name]

            # Add component embeddings
            for comp_name in components:
                val = action_dict[comp_name]
                val_tensor = torch.tensor(val + 1, dtype=torch.long, device=device)
                action_embeddings[i] += self.embedding_layers[comp_name](val_tensor)

        return action_embeddings

    def tokenize_action(self, action: ActionType) -> torch.Tensor:
        """Converts a single action into an embedding token."""
        return self.tokenize_actions([action]).squeeze(0)

    def get_device(self):
        return next(self.parameters()).device
