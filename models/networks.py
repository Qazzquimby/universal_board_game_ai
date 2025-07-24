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
        self.cache = {}

        self.env = env

        # The StateModel needs to know the embedding_dim for the PolicyModel to match
        if "embedding_dim" not in state_model_params:
            raise ValueError("state_model_params must include 'embedding_dim'")
        state_embedding_dim = state_model_params["embedding_dim"]

        self.state_model = _StateModel(env, **state_model_params)
        self.policy_model = _PolicyModel(
            env, state_embedding_dim=state_embedding_dim, **policy_model_params
        )

    def init_zero(self):
        """
        Initializes the weights of the policy and value heads to zero.
        This results in a uniform policy and a value of 0 for all states,
        which is useful for a clean baseline when comparing to vanilla MCTS.
        """
        # Initialize value head's final layer to zero
        final_value_layer = self.state_model.value_head[0]
        if isinstance(final_value_layer, nn.Linear):
            nn.init.constant_(final_value_layer.weight, 0)
            nn.init.constant_(final_value_layer.bias, 0)

        # Initialize policy head's final layer to zero
        final_policy_layer = self.policy_model.policy_head[-1]
        if isinstance(final_policy_layer, nn.Linear):
            nn.init.constant_(final_policy_layer.weight, 0)
            nn.init.constant_(final_policy_layer.bias, 0)

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
                (self.env.num_action_types,), -torch.inf, device=self.get_device()
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

        action_indices = torch.arange(
            self.env.num_action_types, device=self.get_device()
        )
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

    def get_device(self):
        return next(self.parameters()).device


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

        self.network_config = self.env.get_network_config()

        self.embedding_layers = nn.ModuleDict()
        self.embedding_dim = embedding_dim
        for pos_dim, size in self.network_config["position_dims"].items():
            self.embedding_layers[pos_dim] = nn.Embedding(size, embedding_dim)
        for feat, info in self.network_config["features"].items():
            self.embedding_layers[feat] = nn.Embedding(
                info["cardinality"] + 1, embedding_dim
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.value_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())

    def create_input_tensors_from_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates input tensors from the current state of the environment.
        Relies on the agent to have set the environment to the correct state.
        """
        from environments.base import Networkable

        device = next(self.parameters()).device
        features = self.network_config["features"]
        entity_type = self.network_config["entity_type"]
        all_entities = self.env.get_all_networkable_entities()

        entity_tokens = []
        for entity_instance, pos_data in all_entities:
            # Create position part of token by summing position dimension embeddings
            token = 0
            for pos_dim_name, pos_dim_val in pos_data.items():
                pos_idx = torch.tensor([pos_dim_val], device=device)
                token += self.embedding_layers[pos_dim_name](pos_idx)

            # Create feature part of token by summing feature embeddings
            feature_values = None
            if entity_instance is not None:
                # The entity might be a raw dict (from Pydantic) or already an object
                if isinstance(entity_instance, dict):
                    entity_obj = entity_type(**entity_instance)
                elif isinstance(entity_instance, Networkable):
                    entity_obj = entity_instance
                else:
                    raise TypeError(f"Unexpected entity type: {type(entity_instance)}")
                feature_values = entity_obj.get_feature_values()

            for feat_name in features:
                if feature_values is None:
                    # Use last index for None/empty cell
                    feat_idx_val = self.embedding_layers[feat_name].num_embeddings - 1
                else:
                    feat_idx_val = feature_values[feat_name]

                feat_idx = torch.tensor([feat_idx_val], device=device)
                token += self.embedding_layers[feat_name](feat_idx)
            entity_tokens.append(token)

        if not entity_tokens:
            # Handle case with no entities (e.g., empty board)
            # Create a zero tensor with the correct embedding dimension
            zero_token = torch.zeros(1, self.embedding_dim, device=device)
            entity_tokens.append(zero_token)

        sequence = torch.cat(entity_tokens, dim=0).unsqueeze(0)
        mask = torch.zeros(1, sequence.shape[1], dtype=torch.bool, device=device)
        return sequence, mask

    def forward(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(args[0], StateWithKey):
            # Inference path: called with a single StateWithKey object.
            # The agent is responsible for setting the env state before calling predict.
            src, src_key_padding_mask = self.create_input_tensors_from_state()
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
            batch_size, 1, dtype=torch.bool, device=src.device
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
