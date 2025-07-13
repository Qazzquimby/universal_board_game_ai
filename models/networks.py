from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from environments.base import BaseEnvironment, StateWithKey


class GenericPieceTransformer(nn.Module):
    """
    A network that automatically builds a transformer architecture
    by inspecting the environment's Pydantic state model.

    It identifies grids of entities, creates embeddings for their
    properties and positions, and processes them with a transformer.
    This is a generic version of a "piece transformer" network.
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Analyze state structure to find entities and their features.
        self.entity_info = self._analyze_state_structure()
        if not self.entity_info:
            raise ValueError(
                "Could not find a 'Grid' of entities in the environment's state model."
            )

        # 2. Create embedding layers based on the analyzed structure.
        self.embedding_layers = nn.ModuleDict()
        total_embedding_dim = 0

        # Positional embeddings (e.g., for x and y coordinates)
        for pos_dim_name, dim_size in self.entity_info["position_dims"].items():
            self.embedding_layers[pos_dim_name] = nn.Embedding(dim_size, embedding_dim)
            total_embedding_dim += embedding_dim

        # Feature embeddings for entity properties (e.g., player ID)
        for feat_name, feat_info in self.entity_info["features"].items():
            self.embedding_layers[feat_name] = nn.Embedding(
                feat_info["cardinality"], embedding_dim
            )
            total_embedding_dim += embedding_dim

        self.total_embedding_dim = total_embedding_dim

        # 3. Create the Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, self.total_embedding_dim))

        # 4. Create prediction heads for policy and value.
        policy_size = self._calculate_policy_size(env)
        self.policy_head = nn.Linear(self.total_embedding_dim, policy_size)
        self.value_head = nn.Sequential(
            nn.Linear(self.total_embedding_dim, 1),
            nn.Tanh(),
        )
        self.to(self.device)
        print("GenericPieceTransformer initialized successfully.")

    def _analyze_state_structure(self):
        """
        Inspects the environment's Pydantic state model to find a grid of entities.

        This makes a significant assumption that the environment instance has a
        `state` attribute which is a Pydantic model instance, and that this
        state contains a field which is a subclass of `Grid`. This holds for
        `Connect4`, but a more robust mechanism (e.g., explicit registration)
        would be needed for true generality across varied environments.
        """
        if not hasattr(self.env, "state"):
            return None

        from typing import get_origin, get_args
        from pydantic import BaseModel
        from environments.connect4 import Grid

        model_class = type(self.env.state)

        for field_name, field in model_class.model_fields.items():
            field_type = field.annotation
            origin = get_origin(field_type)

            if origin and issubclass(origin, Grid):
                grid_type = field_type
                entity_type_arg = get_args(grid_type)[0]
                # Handle Optional[EntityType]
                optional_args = get_args(entity_type_arg)
                entity_type = optional_args[0] if optional_args else entity_type_arg

                if not issubclass(entity_type, BaseModel):
                    continue

                grid_instance = grid_type()
                position_dims = {"y": grid_instance.height, "x": grid_instance.width}

                features = {}
                for e_fname, e_field in entity_type.model_fields.items():
                    if e_field.annotation == int:
                        if e_fname == "id" and hasattr(self.env, "players"):
                            features[e_fname] = {
                                "cardinality": len(self.env.players) + 1
                            }  # +1 for empty

                if not features:
                    continue

                return {
                    "grid_field_name": field_name,
                    "position_dims": position_dims,
                    "features": features,
                    "entity_type": entity_type,
                }
        return None

    def create_input_tensors_from_state(
        self, state_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a raw state dictionary into a sequence of token embeddings
        and a padding mask, ready for the forward pass.
        """
        grid_field = self.entity_info["grid_field_name"]
        grid_data = state_dict[grid_field]
        height = self.entity_info["position_dims"]["y"]
        width = self.entity_info["position_dims"]["x"]

        entity_tokens = []

        for r in range(height):
            for c in range(width):
                cell_data = grid_data["cells"][r][c]
                feature_embeddings = []

                # Positional embeddings
                pos_y_idx = torch.tensor([r], device=self.device)
                feature_embeddings.append(self.embedding_layers["y"](pos_y_idx))
                pos_x_idx = torch.tensor([c], device=self.device)
                feature_embeddings.append(self.embedding_layers["x"](pos_x_idx))

                # Entity feature embeddings
                for feat_name in self.entity_info["features"]:
                    if cell_data is None:
                        # Use last index for None/empty
                        feat_idx = torch.tensor(
                            [self.embedding_layers[feat_name].num_embeddings - 1],
                            device=self.device,
                        )
                    else:
                        feat_idx = torch.tensor(
                            [cell_data[feat_name]], device=self.device
                        )
                    feature_embeddings.append(
                        self.embedding_layers[feat_name](feat_idx)
                    )

                token = torch.cat(feature_embeddings, dim=-1)
                entity_tokens.append(token)

        sequence = torch.cat(entity_tokens, dim=0).unsqueeze(0)
        mask = torch.zeros(1, sequence.shape[1], dtype=torch.bool, device=self.device)

        return sequence, mask

    def forward(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: Tensor of shape (batch_size, num_entities, total_embedding_dim)
            src_key_padding_mask: Tensor of shape (batch_size, num_entities)
        """
        batch_size = src.shape[0]

        game_tokens = self.game_token.expand(batch_size, -1, -1)
        sequences_with_game_token = torch.cat([game_tokens, src], dim=1)

        game_token_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=self.device
        )
        full_mask = torch.cat([game_token_mask, src_key_padding_mask], dim=1)

        transformer_output = self.transformer_encoder(
            src=sequences_with_game_token, src_key_padding_mask=full_mask
        )

        game_token_output = transformer_output[:, 0, :]

        policy_logits = self.policy_head(game_token_output)
        value = self.value_head(game_token_output)

        return policy_logits, value

    def predict(self, state_with_key: StateWithKey) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            src, mask = self.create_input_tensors_from_state(state_with_key.state)
            policy_logits, value = self.forward(src, mask)
            policy_probs = F.softmax(policy_logits, dim=1)

            policy_np = policy_probs.squeeze(0).cpu().numpy()
            value_np = value.squeeze(0).cpu().item()

        return policy_np, value_np

    def _calculate_policy_size(self, env: BaseEnvironment) -> int:
        if type(env).__name__ == "Connect4" and hasattr(env, "width"):
            return env.width

        if hasattr(env, "num_actions"):
            return env.num_actions
        elif hasattr(env, "board_size"):
            return env.board_size * env.board_size
        elif hasattr(env, "initial_piles"):
            max_removable = max(env.initial_piles) if env.initial_piles else 1
            num_piles = len(env.initial_piles)
            return num_piles * max_removable
        else:
            raise ValueError(
                f"Cannot determine policy size for environment type: {type(env).__name__}"
            )
