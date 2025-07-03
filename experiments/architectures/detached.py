import torch
from torch import nn
from torch.nn import functional as F

from experiments.architectures.shared import BOARD_HEIGHT, BOARD_WIDTH


# This is based off of the transformer model but with a more extensible policy


class StateModel(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.owner_embedding = nn.Embedding(2, embedding_dim)
        self.row_embedding = nn.Embedding(BOARD_HEIGHT, embedding_dim)
        self.col_embedding = nn.Embedding(BOARD_WIDTH, embedding_dim)

        self.game_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            dim_feedforward=embedding_dim * 4,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        fc_out_size = 64
        self.fc_value = nn.Linear(embedding_dim, fc_out_size)
        self.value_head = nn.Linear(fc_out_size, 1)

    def forward(self, owner, coords, src_key_padding_mask):
        piece_mask = src_key_padding_mask
        row_coords = coords[:, :, 0].long()
        col_coords = coords[:, :, 1].long()

        owner_indices = torch.argmax(owner, dim=-1)
        owner_emb = self.owner_embedding(owner_indices)
        row_emb = self.row_embedding(row_coords)
        col_emb = self.col_embedding(col_coords)
        piece_embedded = owner_emb + row_emb + col_emb

        batch_size = owner.shape[0]
        game_tokens = self.game_token.expand(batch_size, -1, -1)
        game_mask = torch.zeros(batch_size, 1, device=owner.device, dtype=torch.bool)

        tokens = torch.cat((piece_embedded, game_tokens), dim=1)
        mask = torch.cat((piece_mask, game_mask), dim=1)
        tokens = self.dropout(tokens)

        transformer_output = self.transformer_encoder(tokens, src_key_padding_mask=mask)
        game_out = transformer_output[:, -1, :]
        value_out = F.relu(self.fc_value(game_out))
        value = torch.tanh(self.value_head(value_out))

        return transformer_output, value


class PolicyModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.action_embedding = nn.Embedding(BOARD_WIDTH, embedding_dim)

        # The policy head will take concatenated state and action embeddings
        # State: game_token. Action: action_embedding
        policy_input_dim = 2 * embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, state_tokens, action_indices):
        # state_tokens: (batch_size, seq_len + 1, embedding_dim)
        # action_indices: (batch_size,)

        game_token = state_tokens[:, -1, :]

        action_emb = self.action_embedding(action_indices)

        policy_input = torch.cat([game_token, action_emb], dim=-1)

        score = self.policy_head(policy_input)
        return score


class DetachedPolicyNet(nn.Module):
    def __init__(self, state_model_params, policy_model_params):
        super().__init__()
        self.state_model = StateModel(**state_model_params)
        self.policy_model = PolicyModel(**policy_model_params)

    def forward(self, owner, coords, src_key_padding_mask):
        # 1. Run state model
        state_tokens, value = self.state_model(owner, coords, src_key_padding_mask)

        # 2. Iterate over possible actions (columns) to get policy logits
        batch_size = owner.shape[0]
        policy_logits = []

        action_indices = torch.arange(BOARD_WIDTH, device=owner.device)

        # todo: should loop over legal moves, not board width
        # todo later: this could later be batched rather than looped
        for i in range(BOARD_WIDTH):
            # Prepare action index for batch
            current_action_indices = action_indices[i].expand(batch_size)

            # 3. Run policy model for this action
            score = self.policy_model(state_tokens, current_action_indices)
            policy_logits.append(score)

        # Concatenate scores to form policy logits
        policy_logits = torch.cat(policy_logits, dim=1)

        return policy_logits, value
