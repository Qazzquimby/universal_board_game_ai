from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from environments.base import BaseEnvironment, StateWithKey


class AutoGraphNet(nn.Module):
    """
    A network architecture that is automatically configured based on the game
    environment's schema. It uses a transformer-based StateModel to process
    a graph-like representation of the game state and a detached PolicyModel
    to score potential actions.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        state_model_params: dict,
        policy_model_params: dict,
    ):
        super().__init__()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The StateModel needs to know the embedding_dim for the PolicyModel to match
        if "embedding_dim" not in state_model_params:
            raise ValueError("state_model_params must include 'embedding_dim'")
        state_embedding_dim = state_model_params["embedding_dim"]

        self.state_model = _StateModel(env, **state_model_params)
        self.policy_model = _PolicyModel(
            env, state_embedding_dim=state_embedding_dim, **policy_model_params
        )

        self.to(self.device)

    def predict(self, state_with_key: StateWithKey) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            # 1. Run state model to get state embeddings and value
            state_tokens, value_tensor = self.state_model(state_with_key)
            value = value_tensor.squeeze(0).cpu().item()

            # 2. Score legal actions using the policy model
            # This assumes self.env is set to the correct state to get legal actions.
            legal_actions = self.env.get_legal_actions()

            if not legal_actions:
                policy = np.zeros(self.env.num_action_types)
                return policy, value

            scores = [
                self.policy_model(state_tokens, action) for action in legal_actions
            ]
            scores_tensor = torch.cat(scores)

            # 3. Construct the full policy vector, masking illegal actions
            policy_logits = torch.full(
                (self.env.num_action_types,), -torch.inf, device=self.device
            )
            legal_action_indices = [
                self.env.map_action_to_policy_index(a) for a in legal_actions
            ]
            policy_logits[legal_action_indices] = scores_tensor.squeeze(-1)

            policy_probs = F.softmax(policy_logits, dim=0)
            return policy_probs.cpu().numpy(), value

    def forward(self, src_batch, src_key_padding_mask_batch):
        state_tokens, value_preds = self.state_model(
            src_batch, src_key_padding_mask_batch
        )

        game_token_embedding = state_tokens[:, 0, :]  # (batch, state_embedding_dim)

        action_indices = torch.arange(self.env.num_action_types, device=self.device)
        action_embs = self.policy_model.action_embedding(action_indices)

        batch_size = game_token_embedding.shape[0]
        num_actions = action_embs.shape[0]

        game_token_expanded = game_token_embedding.unsqueeze(1).expand(
            -1, num_actions, -1
        )
        action_embs_expanded = action_embs.unsqueeze(0).expand(batch_size, -1, -1)

        policy_input = torch.cat([game_token_expanded, action_embs_expanded], dim=-1)

        policy_input_flat = policy_input.view(batch_size * num_actions, -1)
        policy_logits_flat = self.policy_model.policy_head(policy_input_flat)
        policy_logits = policy_logits_flat.view(batch_size, num_actions)

        return policy_logits, value_preds


class _StateModel(nn.Module):
    """Internal StateModel for the AutoGraphNet."""

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Todo Get everything from the state. Don't assert state is 1 grid.
        from environments.base import Grid

        grid_instance = None
        grid_field_name = None
        for key, val in self.env.state.__dict__.items():
            if isinstance(val, Grid):
                grid_instance = val
                grid_field_name = key
                break

        if grid_instance is None:
            raise TypeError(
                "Could not find a `Grid` field in the environment's state model."
            )
        self.grid_field_name = grid_field_name
        self.network_config = grid_instance.get_network_config(self.env)

        self.embedding_layers = nn.ModuleDict()
        self.embedding_dim = embedding_dim
        for pos_dim, size in self.network_config["position_dims"].items():
            self.embedding_layers[pos_dim] = nn.Embedding(size, embedding_dim)
        for feat, info in self.network_config["features"].items():
            self.embedding_layers[feat] = nn.Embedding(
                info["cardinality"], embedding_dim
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.value_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())

    def create_input_tensors_from_state(
        self, state_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_data = state_dict[self.grid_field_name]
        height = self.network_config["position_dims"]["y"]
        width = self.network_config["position_dims"]["x"]
        features = self.network_config["features"]

        entity_tokens = []
        for r in range(height):
            for c in range(width):
                cell = grid_data["cells"][r][c]
                # Sum embeddings for position and features
                pos_y_idx = torch.tensor([r], device=self.device)
                pos_x_idx = torch.tensor([c], device=self.device)
                token = self.embedding_layers["y"](pos_y_idx) + self.embedding_layers[
                    "x"
                ](pos_x_idx)

                for feat_name in features:
                    if cell is None:
                        # Use last index for None/empty
                        feat_idx_val = (
                            self.embedding_layers[feat_name].num_embeddings - 1
                        )
                    else:
                        feat_idx_val = cell[feat_name]
                    feat_idx = torch.tensor([feat_idx_val], device=self.device)
                    token += self.embedding_layers[feat_name](feat_idx)
                entity_tokens.append(token)

        sequence = torch.cat(entity_tokens, dim=0).unsqueeze(0)
        mask = torch.zeros(1, sequence.shape[1], dtype=torch.bool, device=self.device)
        return sequence, mask

    def forward(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(args[0], StateWithKey):
            # Inference path: called with a single StateWithKey object
            state_with_key: StateWithKey = args[0]
            src, src_key_padding_mask = self.create_input_tensors_from_state(
                state_with_key.state
            )
        elif torch.is_tensor(args[0]) and len(args) == 2:
            # Training path: called with tensors
            src, src_key_padding_mask = args
        else:
            raise TypeError(
                f"Unsupported input type for _StateModel.forward: {[type(arg) for arg in args]}"
            )

        batch_size = src.shape[0]

        # Prepend game token to sequence
        game_tokens = self.game_token.expand(batch_size, -1, -1)
        sequences = torch.cat([game_tokens, src], dim=1)
        game_token_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=self.device
        )
        full_mask = torch.cat([game_token_mask, src_key_padding_mask], dim=1)

        transformer_output = self.transformer_encoder(
            src=sequences, src_key_padding_mask=full_mask
        )
        game_token_output = transformer_output[:, 0, :]
        value = self.value_head(game_token_output)

        return transformer_output, value


class _PolicyModel(nn.Module):
    """Internal PolicyModel for the AutoGraphNet."""

    def __init__(
        self, env: BaseEnvironment, state_embedding_dim: int, embedding_dim: int = 64
    ):
        super().__init__()
        self.env = env
        self.action_embedding = nn.Embedding(env.num_action_types, embedding_dim)
        policy_input_dim = state_embedding_dim + embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, state_tokens: torch.Tensor, action) -> torch.Tensor:
        game_token_embedding = state_tokens[:, 0, :]  # (batch, state_embedding_dim)

        action_idx = self.env.map_action_to_policy_index(action)
        action_idx_tensor = torch.tensor(
            [action_idx], device=game_token_embedding.device
        )
        action_emb = self.action_embedding(action_idx_tensor)

        # Replicate game token for batch size of actions if needed
        if action_emb.shape[0] > game_token_embedding.shape[0]:
            game_token_embedding = game_token_embedding.expand(action_emb.shape[0], -1)

        policy_input = torch.cat([game_token_embedding, action_emb], dim=-1)
        score = self.policy_head(policy_input)
        return score
