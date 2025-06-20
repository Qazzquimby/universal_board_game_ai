import math
import einops
import torch
from einops import rearrange, repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from experiments.architectures.shared import BOARD_HEIGHT, BOARD_WIDTH


def create_transformer_input(board_tensor):
    """
    Converts a board tensor into a sequence of piece tokens for a transformer.
    Returns two tensors:
    - features: (num_pieces, 2) with [is_my_piece, is_opp_piece]
    - coords: (num_pieces, 2) with [row, col]
    """
    p1_pieces = board_tensor[0]  # My pieces
    p2_pieces = board_tensor[1]  # Opponent's pieces

    piece_features = []
    piece_coords = []

    # My pieces
    my_locs = torch.nonzero(p1_pieces)
    for loc in my_locs:
        r, c = loc[0].item(), loc[1].item()
        piece_features.append([1.0, 0.0])
        piece_coords.append([r, c])

    # Opponent's pieces
    opp_locs = torch.nonzero(p2_pieces)
    for loc in opp_locs:
        r, c = loc[0].item(), loc[1].item()
        piece_features.append([0.0, 1.0])
        piece_coords.append([r, c])

    if not piece_features:  # Handle empty board
        return torch.empty(0, 2, dtype=torch.float), torch.empty(0, 2, dtype=torch.long)

    return torch.tensor(piece_features, dtype=torch.float), torch.tensor(
        piece_coords, dtype=torch.long
    )


def create_cell_transformer_input(board_tensor):
    """
    Converts a board tensor into a sequence of tokens for each cell.
    Returns a tensor of shape (H * W,) with integer class labels:
    0=empty, 1=my_piece, 2=opp_piece.
    """
    p1_board = board_tensor[0]  # My pieces
    p2_board = board_tensor[1]  # Opponent's pieces
    cell_states = torch.zeros(BOARD_HEIGHT, BOARD_WIDTH, dtype=torch.long)
    cell_states[p1_board == 1] = 1
    cell_states[p2_board == 1] = 2
    return cell_states.flatten()


class PieceTransformerNet(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        # Input features: [is_my_piece, is_opp_piece] -> 2 features
        self.input_proj = nn.Linear(2, embedding_dim)

        # Learnable embeddings for each grid position
        self.row_embedding = nn.Embedding(BOARD_HEIGHT, embedding_dim)
        self.col_embedding = nn.Embedding(BOARD_WIDTH, embedding_dim)

        # Special token for global board representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

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

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, src, coords, src_key_padding_mask):
        # src shape: (batch_size, seq_len, feature_dim=2)
        # coords shape: (batch_size, seq_len, 2) -> (row, col)
        # src_key_padding_mask shape: (batch_size, seq_len)

        # Project input features to embedding dimension
        src_embedded = self.input_proj(src)  # (batch_size, seq_len, embedding_dim)

        # Create positional encoding from coordinates
        row_emb = self.row_embedding(coords[:, :, 0])
        col_emb = self.col_embedding(coords[:, :, 1])
        pos_encoding = row_emb + col_emb
        src_with_pos = src_embedded + pos_encoding

        # Prepend CLS token
        batch_size, seq_len, _ = src_with_pos.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, src_with_pos), dim=1)

        # Apply dropout to the combined token sequence
        x = self.dropout(x)

        # Adjust padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, device=src.device, dtype=torch.bool)
        final_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            x, src_key_padding_mask=final_padding_mask
        )

        # Use the output of the CLS token for prediction
        cls_output = transformer_output[:, 0, :]  # (batch_size, embedding_dim)

        out = F.relu(self.fc1(cls_output))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

        return policy_logits, value


class PieceTransformerNet_Sinusoidal(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        if embedding_dim % 4 != 0:
            raise ValueError(
                "embedding_dim must be divisible by 4 for 2D sinusoidal encoding."
            )

        # Input features: [is_my_piece, is_opp_piece] -> 2 features
        self.input_proj = nn.Linear(2, embedding_dim)

        # Special token for global board representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

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

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def _generate_sinusoidal_embeddings(self, coords, embedding_dim):
        # coords: (batch_size, seq_len, 2) -> (row, col)
        d_model_half = embedding_dim // 2
        if d_model_half % 2 != 0:
            raise ValueError("embedding_dim // 2 must be even for sinusoidal encoding.")

        rows = coords[:, :, 0].float()  # (batch_size, seq_len)
        cols = coords[:, :, 1].float()  # (batch_size, seq_len)

        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float()
            * (-math.log(10000.0) / d_model_half)
        ).to(coords.device)

        # Positional encoding for rows
        pe_rows = torch.zeros(
            rows.shape[0], rows.shape[1], d_model_half, device=coords.device
        )
        pe_rows[:, :, 0::2] = torch.sin(rows.unsqueeze(-1) * div_term)
        pe_rows[:, :, 1::2] = torch.cos(rows.unsqueeze(-1) * div_term)

        # Positional encoding for columns
        pe_cols = torch.zeros(
            cols.shape[0], cols.shape[1], d_model_half, device=coords.device
        )
        pe_cols[:, :, 0::2] = torch.sin(cols.unsqueeze(-1) * div_term)
        pe_cols[:, :, 1::2] = torch.cos(cols.unsqueeze(-1) * div_term)

        # Concatenate row and column embeddings
        pos_encoding = torch.cat([pe_rows, pe_cols], dim=-1)
        return pos_encoding

    def forward(self, src, coords, src_key_padding_mask):
        # src shape: (batch_size, seq_len, feature_dim=2)
        # coords shape: (batch_size, seq_len, 2) -> (row, col)
        # src_key_padding_mask shape: (batch_size, seq_len)

        # Project input features to embedding dimension
        src_embedded = self.input_proj(src)  # (batch_size, seq_len, embedding_dim)

        # Create positional encoding from coordinates
        pos_encoding = self._generate_sinusoidal_embeddings(
            coords, src_embedded.shape[-1]
        )
        src_with_pos = src_embedded + pos_encoding

        # Prepend CLS token
        batch_size, seq_len, _ = src_with_pos.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, src_with_pos), dim=1)

        # Apply dropout to the combined token sequence
        x = self.dropout(x)

        # Adjust padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, device=src.device, dtype=torch.bool)
        final_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            x, src_key_padding_mask=final_padding_mask
        )

        # Use the output of the CLS token for prediction
        cls_output = transformer_output[:, 0, :]  # (batch_size, embedding_dim)

        out = F.relu(self.fc1(cls_output))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

        return policy_logits, value


class PieceTransformerNet_Sinusoidal_Learnable(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        if embedding_dim % 4 != 0:
            raise ValueError(
                "embedding_dim must be divisible by 4 for 2D sinusoidal encoding."
            )
        self.embedding_dim = embedding_dim

        # Input features: [is_my_piece, is_opp_piece] -> 2 features
        self.input_proj = nn.Linear(2, embedding_dim)
        self.pos_proj = nn.Linear(embedding_dim, embedding_dim)

        # Special token for global board representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

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

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def _generate_sinusoidal_embeddings(self, coords, embedding_dim):
        # coords: (batch_size, seq_len, 2) -> (row, col)
        d_model_half = embedding_dim // 2
        if d_model_half % 2 != 0:
            raise ValueError("embedding_dim // 2 must be even for sinusoidal encoding.")

        rows = coords[:, :, 0].float()  # (batch_size, seq_len)
        cols = coords[:, :, 1].float()  # (batch_size, seq_len)

        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float()
            * (-math.log(10000.0) / d_model_half)
        ).to(coords.device)

        # Positional encoding for rows
        pe_rows = torch.zeros(
            rows.shape[0], rows.shape[1], d_model_half, device=coords.device
        )
        pe_rows[:, :, 0::2] = torch.sin(rows.unsqueeze(-1) * div_term)
        pe_rows[:, :, 1::2] = torch.cos(rows.unsqueeze(-1) * div_term)

        # Positional encoding for columns
        pe_cols = torch.zeros(
            cols.shape[0], cols.shape[1], d_model_half, device=coords.device
        )
        pe_cols[:, :, 0::2] = torch.sin(cols.unsqueeze(-1) * div_term)
        pe_cols[:, :, 1::2] = torch.cos(cols.unsqueeze(-1) * div_term)

        # Concatenate row and column embeddings
        pos_encoding = torch.cat([pe_rows, pe_cols], dim=-1)
        return pos_encoding

    def forward(self, src, coords, src_key_padding_mask):
        # src shape: (batch_size, seq_len, feature_dim=2)
        # coords shape: (batch_size, seq_len, 2) -> (row, col)
        # src_key_padding_mask shape: (batch_size, seq_len)

        # Project input features to embedding dimension
        src_embedded = self.input_proj(src)  # (batch_size, seq_len, embedding_dim)

        # Create positional encoding from coordinates and make it learnable
        pos_encoding = self._generate_sinusoidal_embeddings(coords, self.embedding_dim)
        pos_encoding = F.relu(self.pos_proj(pos_encoding))
        src_with_pos = src_embedded + pos_encoding

        # Prepend CLS token
        batch_size, seq_len, _ = src_with_pos.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, src_with_pos), dim=1)

        # Apply dropout to the combined token sequence
        x = self.dropout(x)

        # Adjust padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, device=src.device, dtype=torch.bool)
        final_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            x, src_key_padding_mask=final_padding_mask
        )

        # Use the output of the CLS token for prediction
        cls_output = transformer_output[:, 0, :]  # (batch_size, embedding_dim)

        out = F.relu(self.fc1(cls_output))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

        return policy_logits, value


class PieceTransformerNet_ConcatPos(nn.Module):
    # Stronger than learnable pos
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
        pos_embedding_dim=4,
    ):
        super().__init__()

        # Input features: [is_my_piece, is_opp_piece] -> 2 features
        # Positional features: x, y -> pos_embedding_dim * 2
        self.input_proj = nn.Linear(2 + pos_embedding_dim * 2, embedding_dim)

        # Learnable embeddings for each grid position
        self.row_embedding = nn.Embedding(BOARD_HEIGHT, pos_embedding_dim)
        self.col_embedding = nn.Embedding(BOARD_WIDTH, pos_embedding_dim)

        # Special token for global board representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

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

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, src, coords, src_key_padding_mask):
        # src shape: (batch_size, seq_len, feature_dim=2)
        # coords shape: (batch_size, seq_len, 2) -> (row, col)
        # src_key_padding_mask shape: (batch_size, seq_len)

        # Create positional encoding from coordinates
        row_emb = self.row_embedding(coords[:, :, 0])
        col_emb = self.col_embedding(coords[:, :, 1])

        # Concatenate features and positional embeddings
        combined_input = torch.cat([src, row_emb, col_emb], dim=-1)
        src_embedded = self.input_proj(combined_input)

        # Prepend CLS token
        batch_size, seq_len, _ = src_embedded.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, src_embedded), dim=1)

        # Apply dropout to the combined token sequence
        x = self.dropout(x)

        # Adjust padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, device=src.device, dtype=torch.bool)
        final_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            x, src_key_padding_mask=final_padding_mask
        )

        # Use the output of the CLS token for prediction
        cls_output = transformer_output[:, 0, :]  # (batch_size, embedding_dim)

        out = F.relu(self.fc1(cls_output))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

        return policy_logits, value


class PieceTransformer_OnehotLoc(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        owner_ = 2
        row_ = BOARD_HEIGHT
        col_ = BOARD_WIDTH
        input_size = owner_ + row_ + col_
        self.piece_proj = nn.Linear(input_size, embedding_dim)

        # this could be a game input * a game_proj, but I don't really see any game input here
        # so itd be simpler to just make it a learnable parameter.
        self.game_size = 1
        self.game_proj = nn.Linear(self.game_size, embedding_dim)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

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

        self.policy_head = nn.Linear(3 * embedding_dim, BOARD_WIDTH)
        self.value_head = nn.Linear(3 * embedding_dim, 1)

    def forward(self, owner, coords, src_key_padding_mask):
        row_coords = coords[:, :, 0].long()
        col_coords = coords[:, :, 1].long()

        row_onehot = F.one_hot(row_coords, num_classes=BOARD_HEIGHT)
        col_onehot = F.one_hot(col_coords, num_classes=BOARD_WIDTH)
        piece_raw = torch.concat([owner, row_onehot, col_onehot], dim=-1)

        piece_embedded = self.piece_proj(piece_raw)
        # (batch_size, seq_len, embedding_dim)

        ## Game token
        # normally input wouldn't be hardcoded. This is somewhat silly but meant to resemble more normal situations.
        game_input = torch.ones(self.game_size).to(owner.device)
        game_embedded = self.game_proj(game_input)
        game_embedded = repeat(
            game_embedded, "game -> batch game", batch=piece_embedded.shape[0]
        )
        game_embedded = rearrange(game_embedded, "batch emb -> batch 1 emb")

        tokens = torch.concat((piece_embedded, game_embedded), dim=1)
        tokens = self.dropout(tokens)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            tokens,  # src_key_padding_mask=src_key_padding_mask
        )

        cards_out = transformer_output[:, :-1, :]  # (batch_size, embedding_dim)
        cards_max = torch.max(cards_out, dim=1).values
        cards_mean = torch.mean(cards_out, dim=1)
        game_out = transformer_output[:, -1, :]  # (batch_size, embedding_dim)
        full_out = torch.concat((cards_max, cards_mean, game_out), dim=-1)
        # batch, 3*embedding dim

        policy_logits = self.policy_head(full_out)
        value = torch.tanh(self.value_head(full_out))

        return policy_logits, value


class PieceTransformer_OnehotLoc_FixMask(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        owner_ = 2
        row_ = BOARD_HEIGHT
        col_ = BOARD_WIDTH
        input_size = owner_ + row_ + col_
        self.piece_proj = nn.Linear(input_size, embedding_dim)

        # this could be a game input * a game_proj, but I don't really see any game input here
        # so itd be simpler to just make it a learnable parameter.
        self.game_size = 1
        self.game_proj = nn.Linear(self.game_size, embedding_dim)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

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

        self.policy_head = nn.Linear(3 * embedding_dim, BOARD_WIDTH)
        self.value_head = nn.Linear(3 * embedding_dim, 1)

    def forward(self, owner, coords, src_key_padding_mask):
        piece_mask = src_key_padding_mask
        row_coords = coords[:, :, 0].long()
        col_coords = coords[:, :, 1].long()

        row_onehot = F.one_hot(row_coords, num_classes=BOARD_HEIGHT)
        col_onehot = F.one_hot(col_coords, num_classes=BOARD_WIDTH)
        piece_raw = torch.concat([owner, row_onehot, col_onehot], dim=-1)

        piece_embedded = self.piece_proj(piece_raw)
        # (batch_size, seq_len, embedding_dim)

        ## Game token
        # normally input wouldn't be hardcoded. This is somewhat silly but meant to resemble more normal situations.
        batch_size = owner.shape[0]
        game_input = torch.ones(self.game_size).to(owner.device)
        game_embedded = self.game_proj(game_input)
        game_embedded = repeat(
            game_embedded, "game -> batch game", batch=piece_embedded.shape[0]
        )
        game_embedded = rearrange(game_embedded, "batch emb -> batch 1 emb")
        game_mask = torch.zeros(batch_size, 1, device=owner.device, dtype=torch.bool)

        tokens = torch.concat((piece_embedded, game_embedded), dim=1)
        mask = torch.concat((piece_mask, game_mask), dim=1)
        tokens = self.dropout(tokens)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(tokens, src_key_padding_mask=mask)

        cards_out = transformer_output[:, :-1, :]  # (batch_size, embedding_dim)
        cards_max = torch.max(cards_out, dim=1).values
        cards_mean = torch.mean(cards_out, dim=1)
        game_out = transformer_output[:, -1, :]  # (batch_size, embedding_dim)
        full_out = torch.concat((cards_max, cards_mean, game_out), dim=-1)
        # batch, 3*embedding dim

        policy_logits = self.policy_head(full_out)
        value = torch.tanh(self.value_head(full_out))

        return policy_logits, value


class CellTransformerNet(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        num_patches = BOARD_HEIGHT * BOARD_WIDTH

        # Embedding for the state of each cell (patch)
        self.patch_embedding = nn.Embedding(
            3, embedding_dim
        )  # 0: empty, 1: mine, 2: opp

        # Learnable positional embeddings for each patch + CLS token
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dim)
        )

        # Special token for global board representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

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

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, src):
        # src shape: (batch_size, seq_len=42)

        # Get patch embeddings from integer states
        x = self.patch_embedding(src)  # (batch_size, seq_len, embedding_dim)

        # Prepend CLS token
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.positional_embedding
        x = self.dropout(x)

        # Pass through transformer encoder (no mask needed for fixed-size input)
        transformer_output = self.transformer_encoder(x)

        # Use the output of the CLS token for prediction
        cls_output = transformer_output[:, 0, :]  # (batch_size, embedding_dim)

        out = F.relu(self.fc1(cls_output))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

        return policy_logits, value


class Connect4TransformerDataset(Dataset):
    def __init__(self, transformer_inputs, policy_labels, value_labels):
        self.transformer_inputs = transformer_inputs
        self.policy_labels = torch.from_numpy(policy_labels).long()
        self.value_labels = torch.from_numpy(value_labels)

    def __len__(self):
        return len(self.transformer_inputs)

    def __getitem__(self, idx):
        return (
            self.transformer_inputs[idx],
            self.policy_labels[idx],
            self.value_labels[idx],
        )


class Connect4CellTransformerDataset(Dataset):
    def __init__(self, cell_inputs, policy_labels, value_labels):
        self.cell_inputs = cell_inputs  # This will be a list of tensors
        self.policy_labels = torch.from_numpy(policy_labels).long()
        self.value_labels = torch.from_numpy(value_labels)

    def __len__(self):
        return len(self.cell_inputs)

    def __getitem__(self, idx):
        return (
            self.cell_inputs[idx],
            self.policy_labels[idx],
            self.value_labels[idx],
        )


def _process_batch_transformer(
    model, data_batch, device, policy_criterion, value_criterion
):
    (inputs, coords, attention_mask), policy_labels, value_labels = data_batch
    inputs = inputs.to(device)
    coords = coords.to(device)
    attention_mask = attention_mask.to(device)
    policy_labels = policy_labels.to(device)
    value_labels = value_labels.to(device)

    policy_logits, value_preds = model(inputs, coords, attention_mask)

    loss_p = policy_criterion(policy_logits, policy_labels)
    loss_v = value_criterion(value_preds.squeeze(), value_labels)
    loss = loss_p + loss_v

    _, predicted_policies = torch.max(policy_logits, 1)
    policy_acc = (predicted_policies == policy_labels).int().sum().item()
    value_mse = F.mse_loss(value_preds.squeeze(), value_labels).item()

    return loss, loss_p.item(), loss_v.item(), policy_acc, value_mse


def _process_batch_cell_transformer(
    model, data_batch, device, policy_criterion, value_criterion
):
    inputs, policy_labels, value_labels = data_batch
    inputs = inputs.to(device)
    policy_labels = policy_labels.to(device)
    value_labels = value_labels.to(device)

    policy_logits, value_preds = model(inputs)

    loss_p = policy_criterion(policy_logits, policy_labels)
    loss_v = value_criterion(value_preds.squeeze(), value_labels)
    loss = loss_p + loss_v

    _, predicted_policies = torch.max(policy_logits, 1)
    policy_acc = (predicted_policies == policy_labels).int().sum().item()
    value_mse = F.mse_loss(value_preds.squeeze(), value_labels).item()

    return loss, loss_p.item(), loss_v.item(), policy_acc, value_mse


def transformer_collate_fn(batch):
    """
    Custom collate function for transformer.
    Pads sequences to the max length in the batch and creates a padding mask.
    """
    # batch is a list of ((features, coords), policy, value)
    inputs, policies, values = zip(*batch)
    feature_list, coord_list = zip(*inputs)

    # Pad feature sequences
    padded_features = nn.utils.rnn.pad_sequence(
        feature_list, batch_first=True, padding_value=0
    )

    # Pad coordinate sequences
    padded_coords = nn.utils.rnn.pad_sequence(
        coord_list, batch_first=True, padding_value=0
    )

    # Create attention mask (True for padded elements)
    lengths = [len(x) for x in feature_list]
    attention_mask = (
        torch.arange(padded_features.size(1))[None, :] >= torch.tensor(lengths)[:, None]
    )

    batched_policies = torch.stack(list(policies))
    batched_values = torch.stack(list(values))

    return (
        (padded_features, padded_coords, attention_mask),
        batched_policies,
        batched_values,
    )
