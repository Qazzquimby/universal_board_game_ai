import math
from enum import IntEnum

import torch
import torch_geometric.nn as pyg_nn
from einops import rearrange
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch, Data, HeteroData
from experiments.architectures.shared import BOARD_WIDTH


# todo at some point I want to replace these enums with something automatically generated.


class CellToken(IntEnum):
    EMPTY = 0
    MY_PIECE = 1
    OPP_PIECE = 2
    PAD = 3


class CellEdgeType(IntEnum):
    """
    Enum for edge types in the cell graph.
    The values are explicitly set to allow for reverse edge calculation (e.g., SOUTH = NORTH + 1).
    """

    NORTH = 1
    SOUTH = NORTH + 1
    EAST = 3
    WEST = EAST + 1


class CellColumnToken(IntEnum):
    EMPTY = 0
    MY_PIECE = 1
    OPP_PIECE = 2
    COLUMN = 3
    PAD = 4


class CellColumnSpecialToken(IntEnum):
    GAME = 0


class CellColumnEdgeType(IntEnum):
    NORTH = 1
    SOUTH = NORTH + 1
    EAST = 3
    WEST = EAST + 1
    CELL_TO_COLUMN = 5
    COLUMN_TO_CELL = CELL_TO_COLUMN + 1


class CellColumnPieceToken(IntEnum):
    CELL = 0
    COLUMN = 1
    MY_PIECE = 2
    OPP_PIECE = 3
    PAD = 4


class CellColumnPieceSpecialToken(IntEnum):
    GAME = 0


class CellColumnPieceEdgeType(IntEnum):
    NORTH = 1
    SOUTH = NORTH + 1
    EAST = 3
    WEST = EAST + 1
    CELL_TO_COLUMN = 5
    COLUMN_TO_CELL = CELL_TO_COLUMN + 1
    PIECE_TO_CELL = 7
    CELL_TO_PIECE = PIECE_TO_CELL + 1


class PieceColumnSpecialToken(IntEnum):
    GAME = 0


class PieceColumnEdgeType(IntEnum):
    PIECE_TO_COLUMN = 1
    COLUMN_TO_PIECE = PIECE_TO_COLUMN + 1


class HeteroEdgeBias(nn.Module):
    def __init__(self, num_heads, num_edge_types):
        super().__init__()
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.edge_embedding = nn.Embedding(
            num_embeddings=self.num_edge_types + 1, embedding_dim=self.num_heads
        )
        # embedding_dim is num_heads to have a bias per head.

    def forward(
        self,
        edge_index: Tensor,
        edge_type: Tensor,
        batch_vec: Tensor,
        batch_size: int,
        max_seq_len: int,
        graph_node_offsets: Tensor,
    ):
        # edge_index: [2, num_edges]
        # edge_type: [num_edges]
        # batch_vec: [num_nodes] (node index to graph index)
        bias_matrix = torch.zeros(
            batch_size,
            self.num_heads,
            max_seq_len,
            max_seq_len,
            device=edge_index.device,
        )

        edge_bias_values = self.edge_embedding(edge_type)  # [num_edges, num_heads]

        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        batch_indices = batch_vec[src_nodes]

        src_nodes_in_graph = src_nodes - graph_node_offsets[batch_indices]
        dst_nodes_in_graph = dst_nodes - graph_node_offsets[batch_indices]

        bias_matrix[
            batch_indices, :, src_nodes_in_graph, dst_nodes_in_graph
        ] = edge_bias_values
        return bias_matrix


class EdgeBiasedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.head_dim = (
            embed_dim,
            num_heads,
            embed_dim // num_heads,
        )
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)  # q, k, v
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias, key_padding_mask=None):
        batch_len, seq_len, embed_dim = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(batch_len, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_len, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_len, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if edge_bias is not None:
            attn_scores = attn_scores + edge_bias

        if key_padding_mask is not None:
            # Reshape mask for broadcasting: [batch, seq] -> [batch, 1, 1, seq]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_len, seq_len, embed_dim)
        return self.out_proj(output), attn_weights


class EdgeUpdateGate(nn.Module):
    def __init__(self, embed_dim: int, num_edge_types: int):
        super().__init__()
        # Each edge type gets its own learnable update vector.
        self.edge_vector_embedding = nn.Embedding(
            num_embeddings=num_edge_types + 1, embedding_dim=embed_dim
        )

    def forward(
        self,
        attention_weights: Tensor,
        edge_type_matrix: Tensor,
    ):
        # attention_weights: [batch, heads, nodes, nodes]
        # edge_type_matrix:  [batch, nodes, nodes]

        # First, get the update vector for each edge in the graph.
        # edge_vector_matrix shape: [batch, nodes, nodes, embed_dim]
        edge_vector_matrix = self.edge_vector_embedding(edge_type_matrix)

        # We average the attention weights across the heads to get a single
        # importance score for each pair of nodes.
        # avg_attention_weights shape: [batch, nodes, nodes, 1]
        avg_attention_weights = attention_weights.mean(dim=1).unsqueeze(-1)

        # Now, we "gate" the edge vectors with the attention weights.
        # For each node, this calculates a weighted sum of the update vectors
        # from all other nodes.
        # update_vectors shape: [batch, nodes, embed_dim]
        # Transpose edge_vector_matrix to use incoming edge embeddings for updates.
        # For node i, we use attention(i,j) to weight edge_vector(j,i).
        update_vectors = (
            avg_attention_weights * edge_vector_matrix.transpose(1, 2)
        ).sum(dim=2)

        return update_vectors


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_edge_types: int,
        dim_feedforward=512,
        dropout=0.1,
        use_edge_info: bool = True,
    ):
        super().__init__()
        self.use_edge_info = use_edge_info
        self.num_edge_types = num_edge_types

        self.attention = EdgeBiasedMultiHeadAttention(embed_dim, num_heads, dropout)
        self.edge_gate = EdgeUpdateGate(
            embed_dim=embed_dim, num_edge_types=num_edge_types
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias, edge_type_matrix, key_padding_mask=None):
        normalized_x = self.norm1(x)  # batch, nodes, embed_dim

        current_edge_bias = edge_bias if self.use_edge_info else None
        attention_output, attention_weights = self.attention(
            normalized_x, current_edge_bias, key_padding_mask
        )

        if self.use_edge_info:
            edge_update_output = self.edge_gate(attention_weights, edge_type_matrix)
            x = x + self.dropout(attention_output + edge_update_output)
        else:
            x = x + self.dropout(attention_output)

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CellGraphTransformer(nn.Module):
    def __init__(self, num_encoder_layers=4, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()

        self.pad_idx = CellToken.PAD
        # Could embed or 1hot linear
        self.patch_embedding = nn.Embedding(
            len(CellToken), embed_dim, padding_idx=self.pad_idx
        )
        self.game_special_token_idx = 0
        self.num_special_tokens = 1
        self.special_tokens = nn.Parameter(
            torch.randn(1, self.num_special_tokens, embed_dim)
        )

        num_edge_types = len(CellEdgeType)  # N, S, E, W

        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    use_edge_info=(i == 0),
                )
                for i in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        fc1_out_size = 64
        self.fc1 = nn.Linear(embed_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, data: Data):
        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(0, BOARD_WIDTH, device=device), torch.zeros(
                0, 1, device=device
            )

        node_features_list = [data.x[data.batch == i] for i in range(batch_size)]
        padded_features = nn.utils.rnn.pad_sequence(
            node_features_list, batch_first=True, padding_value=self.pad_idx
        )
        lengths = torch.tensor(
            [len(x) for x in node_features_list], device=padded_features.device
        )
        key_padding_mask = (
            torch.arange(padded_features.size(1), device=padded_features.device)[
                None, :
            ]
            >= lengths[:, None]
        )

        x_embedded = self.patch_embedding(padded_features)
        special_tokens = self.special_tokens.expand(batch_size, -1, -1)
        x = torch.cat((special_tokens, x_embedded), dim=1)
        x = self.dropout(x)

        max_len = x.shape[1]
        max_nodes = max_len - self.num_special_tokens

        # Adjust padding mask for game token
        game_token_mask = torch.zeros(
            batch_size, self.num_special_tokens, device=x.device, dtype=torch.bool
        )
        final_key_padding_mask = torch.cat((game_token_mask, key_padding_mask), dim=1)

        ## scalar edge bias matrix
        edge_bias = self.edge_bias_module(
            data.edge_index,
            data.edge_type,
            data.batch,
            batch_size,
            max_nodes,
            data.ptr,
        )
        # Pad for game token. The padding adds a row and column for the game token, which is at index 0.
        edge_bias = F.pad(
            edge_bias,
            (self.num_special_tokens, 0, self.num_special_tokens, 0),
            "constant",
            0,
        )

        ## edge type matrix for gating
        edge_type_matrix = torch.zeros(
            batch_size,
            max_len,
            max_len,
            dtype=torch.long,
            device=x.device,
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_nodes_in_graph = src - data.ptr[batch_indices]
        dst_nodes_in_graph = dst - data.ptr[batch_indices]

        edge_type_matrix[
            batch_indices,
            src_nodes_in_graph + self.num_special_tokens,
            dst_nodes_in_graph + self.num_special_tokens,
        ] = data.edge_type

        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, final_key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[
            :, self.game_special_token_idx, :
        ]  # (batch_size, embed_dim)

        game_token_encoded = F.relu(self.fc1(game_token_out))
        policy_logits = self.policy_head(game_token_encoded)
        value = torch.tanh(self.value_head(game_token_encoded))

        return policy_logits, value


class CellColumnGraphTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embed_dim=128,
        num_heads=8,
        dropout=0.1,
        board_height=6,
        board_width=BOARD_WIDTH,
    ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width

        self.token_enum = CellColumnToken
        self.special_token_enum = CellColumnSpecialToken
        self.edge_type_enum = CellColumnEdgeType
        self.pad_idx = self.token_enum.PAD

        # 0: empty, 1: mine, 2: opp, 3: column token, 4: pad
        self.token_embedding = nn.Embedding(
            len(self.token_enum), embed_dim, padding_idx=self.pad_idx
        )
        self.special_tokens = nn.Parameter(
            torch.randn(1, len(self.special_token_enum), embed_dim)
        )

        num_edge_types = len(self.edge_type_enum)
        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    use_edge_info=(i == 0),
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        fc_out_size = 64
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_fc = nn.Linear(embed_dim, fc_out_size)
        self.value_head = nn.Linear(fc_out_size, 1)

    def forward(self, data: Data):
        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(0, self.board_width, device=device), torch.zeros(
                0, 1, device=device
            )

        node_features_list = [data.x[data.batch == i] for i in range(batch_size)]
        padded_features = nn.utils.rnn.pad_sequence(
            node_features_list, batch_first=True, padding_value=self.pad_idx
        )
        lengths = torch.tensor(
            [len(x) for x in node_features_list], device=padded_features.device
        )
        key_padding_mask = (
            torch.arange(padded_features.size(1), device=padded_features.device)[
                None, :
            ]
            >= lengths[:, None]
        )

        column_masks = [data.column_mask[data.batch == i] for i in range(batch_size)]
        padded_column_mask = nn.utils.rnn.pad_sequence(
            column_masks, batch_first=True, padding_value=False
        )

        x_embedded = self.token_embedding(padded_features)
        special_tokens = self.special_tokens.expand(batch_size, -1, -1)
        x = torch.cat((special_tokens, x_embedded), dim=1)
        x = self.dropout(x)

        max_len = x.shape[1]
        num_special = len(self.special_token_enum)
        max_nodes = max_len - num_special

        game_token_mask = torch.zeros(
            batch_size, num_special, device=x.device, dtype=torch.bool
        )
        final_key_padding_mask = torch.cat((game_token_mask, key_padding_mask), dim=1)

        edge_bias = self.edge_bias_module(
            data.edge_index,
            data.edge_type,
            data.batch,
            batch_size,
            max_nodes,
            data.ptr,
        )
        # Pad for game token. The padding adds a row and column for the game token, which is at index 0.
        edge_bias = F.pad(edge_bias, (num_special, 0, num_special, 0), "constant", 0)

        edge_type_matrix = torch.zeros(
            batch_size, max_len, max_len, dtype=torch.long, device=x.device
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_nodes_in_graph = src - data.ptr[batch_indices]
        dst_nodes_in_graph = dst - data.ptr[batch_indices]
        edge_type_matrix[
            batch_indices,
            src_nodes_in_graph + num_special,
            dst_nodes_in_graph + num_special,
        ] = data.edge_type

        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, final_key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, self.special_token_enum.GAME, :]
        game_token_encoded = F.relu(self.value_fc(game_token_out))
        value = torch.tanh(self.value_head(game_token_encoded))

        padded_column_mask = F.pad(
            padded_column_mask, (num_special, 0), "constant", False
        )  # for game token
        column_tokens_out = transformer_output[padded_column_mask].view(
            batch_size, self.board_width, -1
        )
        policy_logits = self.policy_head(column_tokens_out).squeeze(-1)

        return policy_logits, value


class CellColumnPieceGraphTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embed_dim=128,
        num_heads=8,
        dropout=0.1,
        board_height=6,
        board_width=BOARD_WIDTH,
    ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width

        self.token_enum = CellColumnPieceToken
        self.special_token_enum = CellColumnPieceSpecialToken
        self.edge_type_enum = CellColumnPieceEdgeType
        self.pad_idx = self.token_enum.PAD

        # 0:cell, 1:col, 2:my_piece, 3:opp_piece, 4:pad_piece
        self.token_embedding = nn.Embedding(
            len(self.token_enum), embed_dim, padding_idx=self.pad_idx
        )
        self.special_tokens = nn.Parameter(
            torch.randn(1, len(self.special_token_enum), embed_dim)
        )

        num_edge_types = len(self.edge_type_enum)
        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    use_edge_info=(i == 0),
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        fc_out_size = 64
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_fc = nn.Linear(embed_dim, fc_out_size)
        self.value_head = nn.Linear(fc_out_size, 1)

    def forward(self, data: Data):
        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(0, self.board_width, device=device), torch.zeros(
                0, 1, device=device
            )

        node_features_list = [data.x[data.batch == i] for i in range(batch_size)]
        padded_features = nn.utils.rnn.pad_sequence(
            node_features_list, batch_first=True, padding_value=self.pad_idx
        )
        lengths = torch.tensor(
            [len(x) for x in node_features_list], device=padded_features.device
        )
        key_padding_mask = (
            torch.arange(padded_features.size(1), device=padded_features.device)[
                None, :
            ]
            >= lengths[:, None]
        )

        column_masks = [data.column_mask[data.batch == i] for i in range(batch_size)]
        padded_column_mask = nn.utils.rnn.pad_sequence(
            column_masks, batch_first=True, padding_value=False
        )

        x_embedded = self.token_embedding(padded_features)
        special_tokens = self.special_tokens.expand(batch_size, -1, -1)
        x = torch.cat((special_tokens, x_embedded), dim=1)
        x = self.dropout(x)

        max_len = x.shape[1]
        num_special = len(self.special_token_enum)
        max_nodes = max_len - num_special

        game_token_mask = torch.zeros(
            batch_size, num_special, device=x.device, dtype=torch.bool
        )
        final_key_padding_mask = torch.cat((game_token_mask, key_padding_mask), dim=1)

        edge_bias = self.edge_bias_module(
            data.edge_index,
            data.edge_type,
            data.batch,
            batch_size,
            max_nodes,
            data.ptr,
        )
        # Pad for game token. The padding adds a row and column for the game token, which is at index 0.
        edge_bias = F.pad(edge_bias, (num_special, 0, num_special, 0), "constant", 0)

        edge_type_matrix = torch.zeros(
            batch_size, max_len, max_len, dtype=torch.long, device=x.device
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_nodes_in_graph = src - data.ptr[batch_indices]
        dst_nodes_in_graph = dst - data.ptr[batch_indices]
        edge_type_matrix[
            batch_indices,
            src_nodes_in_graph + num_special,
            dst_nodes_in_graph + num_special,
        ] = data.edge_type

        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, final_key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, self.special_token_enum.GAME, :]
        game_token_encoded = F.relu(self.value_fc(game_token_out))
        value = torch.tanh(self.value_head(game_token_encoded))

        padded_column_mask = F.pad(
            padded_column_mask, (num_special, 0), "constant", False
        )  # for game token
        column_tokens_out = transformer_output[padded_column_mask].view(
            batch_size, self.board_width, -1
        )
        policy_logits = self.policy_head(column_tokens_out).squeeze(-1)

        return policy_logits, value


class PieceColumnGraphTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embed_dim=128,
        num_heads=8,
        dropout=0.1,
        board_height=6,
        board_width=BOARD_WIDTH,
    ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width

        self.owner_embedding = nn.Embedding(2, embed_dim)  # 0:my, 1:opp
        self.row_embedding = nn.Embedding(board_height, embed_dim)
        self.col_embedding = nn.Embedding(board_width, embed_dim)
        self.column_token = nn.Parameter(torch.randn(1, embed_dim))

        self.special_token_enum = PieceColumnSpecialToken
        self.edge_type_enum = PieceColumnEdgeType
        self.special_tokens = nn.Parameter(
            torch.randn(1, len(self.special_token_enum), embed_dim)
        )

        num_edge_types = len(self.edge_type_enum)
        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    use_edge_info=(i == 0),
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        fc_out_size = 64
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_fc = nn.Linear(embed_dim, fc_out_size)
        self.value_head = nn.Linear(fc_out_size, 1)

    def forward(self, data: Data):
        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(0, self.board_width, device=device), torch.zeros(
                0, 1, device=device
            )

        graphs = data.to_data_list()
        node_embed_list = []
        for graph in graphs:
            piece_mask = graph.piece_mask
            owner_indices = graph.x[piece_mask]
            coords = graph.coords
            owner_emb = self.owner_embedding(owner_indices)
            row_emb = self.row_embedding(coords[:, 0])
            col_emb = self.col_embedding(coords[:, 1])
            piece_embedded = owner_emb + row_emb + col_emb
            column_embedded = self.column_token.expand(self.board_width, -1)
            node_embed_list.append(torch.cat([piece_embedded, column_embedded], dim=0))

        padded_embeds = nn.utils.rnn.pad_sequence(
            node_embed_list, batch_first=True, padding_value=0
        )
        lengths = torch.tensor(
            [len(x) for x in node_embed_list], device=padded_embeds.device
        )
        key_padding_mask = (
            torch.arange(padded_embeds.size(1), device=padded_embeds.device)[None, :]
            >= lengths[:, None]
        )

        special_tokens = self.special_tokens.expand(batch_size, -1, -1)
        x = torch.cat((special_tokens, padded_embeds), dim=1)
        x = self.dropout(x)

        max_len = x.shape[1]
        num_special = len(self.special_token_enum)
        max_nodes = max_len - num_special

        game_token_mask = torch.zeros(
            batch_size, num_special, device=x.device, dtype=torch.bool
        )
        final_key_padding_mask = torch.cat((game_token_mask, key_padding_mask), dim=1)

        edge_bias = self.edge_bias_module(
            data.edge_index,
            data.edge_type,
            data.batch,
            batch_size,
            max_nodes,
            data.ptr,
        )
        # Pad for game token.
        edge_bias = F.pad(edge_bias, (num_special, 0, num_special, 0), "constant", 0)

        edge_type_matrix = torch.zeros(
            batch_size, max_len, max_len, dtype=torch.long, device=x.device
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_nodes_in_graph = src - data.ptr[batch_indices]
        dst_nodes_in_graph = dst - data.ptr[batch_indices]
        edge_type_matrix[
            batch_indices,
            src_nodes_in_graph + num_special,
            dst_nodes_in_graph + num_special,
        ] = data.edge_type

        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, final_key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, self.special_token_enum.GAME, :]
        game_token_encoded = F.relu(self.value_fc(game_token_out))
        value = torch.tanh(self.value_head(game_token_encoded))

        column_masks = [g.column_mask for g in graphs]
        padded_column_mask = nn.utils.rnn.pad_sequence(
            column_masks, batch_first=True, padding_value=False
        )
        padded_column_mask = F.pad(
            padded_column_mask, (num_special, 0), "constant", False
        )

        column_tokens_out = transformer_output[padded_column_mask].view(
            batch_size, self.board_width, -1
        )
        policy_logits = self.policy_head(column_tokens_out).squeeze(-1)

        return policy_logits, value


def graph_collate_fn(batch):
    graphs, policies, values = zip(*batch)

    batched_graphs = Batch.from_data_list(graphs)
    batched_policies = torch.stack(list(policies))
    batched_values = torch.stack(list(values))

    return batched_graphs, batched_policies, batched_values


def create_cell_graph(board_tensor):
    h, w = board_tensor.shape[1], board_tensor.shape[2]

    p1_board = board_tensor[0]
    p2_board = board_tensor[1]
    cell_states = torch.full((h, w), fill_value=CellToken.EMPTY, dtype=torch.long)
    cell_states[p1_board == 1] = CellToken.MY_PIECE  # my piece
    cell_states[p2_board == 1] = CellToken.OPP_PIECE  # opp piece
    x = cell_states.flatten()

    edge_indices = []
    edge_types = []

    # Edge types use CellEdgeType. Reverse edges are `edge_type + 1`, relying on enum values.
    directions = [(-1, 0), (0, 1)]  # N, E
    base_edge_types = [CellEdgeType.NORTH, CellEdgeType.EAST]

    for i, (dr, dc) in enumerate(directions):
        edge_type = base_edge_types[i]
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    u = r * w + c
                    v = nr * w + nc
                    edge_indices.append([u, v])
                    edge_types.append(edge_type)
                    edge_indices.append([v, u])
                    edge_types.append(edge_type + 1)  # reverse

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

    return data


def create_cell_column_graph(board_tensor):
    h, w = board_tensor.shape[1], board_tensor.shape[2]

    p1_board = board_tensor[0]
    p2_board = board_tensor[1]
    cell_states = torch.full((h, w), fill_value=CellColumnToken.EMPTY, dtype=torch.long)
    cell_states[p1_board == 1] = CellColumnToken.MY_PIECE
    cell_states[p2_board == 1] = CellColumnToken.OPP_PIECE

    # Node features: cell states followed by column markers
    column_markers = torch.full((w,), CellColumnToken.COLUMN, dtype=torch.long)
    x = torch.cat([cell_states.flatten(), column_markers])

    num_cell_nodes = h * w
    column_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    column_mask[num_cell_nodes:] = True

    edge_indices = []
    edge_types = []

    # Cell-to-cell edges
    directions = [(-1, 0), (0, 1)]  # N, E
    base_edge_types = [CellColumnEdgeType.NORTH, CellColumnEdgeType.EAST]
    for i, (dr, dc) in enumerate(directions):
        edge_type = base_edge_types[i]
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    u = r * w + c
                    v = nr * w + nc
                    edge_indices.append([u, v])
                    edge_types.append(edge_type)
                    edge_indices.append([v, u])
                    edge_types.append(edge_type + 1)  # reverse

    # Cell-to-column edges
    edge_type = CellColumnEdgeType.CELL_TO_COLUMN
    num_cell_nodes = h * w
    for c in range(w):
        col_node_idx = num_cell_nodes + c
        for r in range(h):
            cell_node_idx = r * w + c
            edge_indices.append([cell_node_idx, col_node_idx])
            edge_types.append(edge_type)
            edge_indices.append([col_node_idx, cell_node_idx])
            edge_types.append(edge_type + 1)

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(
        x=x, edge_index=edge_index, edge_type=edge_type, column_mask=column_mask
    )

    return data


def create_cell_column_piece_graph(board_tensor):
    h, w = board_tensor.shape[1], board_tensor.shape[2]
    p1_board = board_tensor[0]
    p2_board = board_tensor[1]

    # Node features
    cell_tokens = torch.full((h * w,), CellColumnPieceToken.CELL, dtype=torch.long)
    column_tokens = torch.full((w,), CellColumnPieceToken.COLUMN, dtype=torch.long)

    p1_locs = torch.nonzero(p1_board)
    p2_locs = torch.nonzero(p2_board)
    num_p1 = p1_locs.shape[0]
    num_p2 = p2_locs.shape[0]

    my_piece_tokens = torch.full(
        (num_p1,), CellColumnPieceToken.MY_PIECE, dtype=torch.long
    )
    opp_piece_tokens = torch.full(
        (num_p2,), CellColumnPieceToken.OPP_PIECE, dtype=torch.long
    )
    piece_tokens = torch.cat([my_piece_tokens, opp_piece_tokens])

    x = torch.cat([cell_tokens, column_tokens, piece_tokens])

    num_cell_nodes = h * w
    num_col_nodes = w
    column_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    column_mask[num_cell_nodes : num_cell_nodes + num_col_nodes] = True

    edge_indices = []
    edge_types = []

    # Cell-to-cell edges (types 1-4)
    directions = [(-1, 0), (0, 1)]  # N, E
    base_edge_types = [CellColumnPieceEdgeType.NORTH, CellColumnPieceEdgeType.EAST]
    for i, (dr, dc) in enumerate(directions):
        edge_type = base_edge_types[i]
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    u = r * w + c
                    v = nr * w + nc
                    edge_indices.append([u, v])
                    edge_types.append(edge_type)
                    edge_indices.append([v, u])
                    edge_types.append(edge_type + 1)

    # Cell-to-column edges (types 5-6)
    num_cell_nodes = h * w
    edge_type = CellColumnPieceEdgeType.CELL_TO_COLUMN
    for c in range(w):
        col_node_idx = num_cell_nodes + c
        for r in range(h):
            cell_node_idx = r * w + c
            edge_indices.append([cell_node_idx, col_node_idx])
            edge_types.append(edge_type)
            edge_indices.append([col_node_idx, cell_node_idx])
            edge_types.append(edge_type + 1)

    # Piece-to-cell edges (types 7-8)
    num_col_nodes = w
    piece_start_idx = num_cell_nodes + num_col_nodes
    all_locs = torch.cat([p1_locs, p2_locs], dim=0)
    edge_type = CellColumnPieceEdgeType.PIECE_TO_CELL
    for i, loc in enumerate(all_locs):
        r, c = loc[0].item(), loc[1].item()
        piece_node_idx = piece_start_idx + i
        cell_node_idx = r * w + c
        edge_indices.append([piece_node_idx, cell_node_idx])
        edge_types.append(edge_type)
        edge_indices.append([cell_node_idx, piece_node_idx])
        edge_types.append(edge_type + 1)

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(
        x=x, edge_index=edge_index, edge_type=edge_type, column_mask=column_mask
    )
    return data


def create_piece_column_graph(board_tensor):
    h, w = board_tensor.shape[1], board_tensor.shape[2]
    p1_board = board_tensor[0]
    p2_board = board_tensor[1]

    p1_locs = torch.nonzero(p1_board)
    p2_locs = torch.nonzero(p2_board)
    num_p1 = p1_locs.shape[0]
    num_p2 = p2_locs.shape[0]

    my_owner_indices = torch.full((num_p1,), 0, dtype=torch.long)
    opp_owner_indices = torch.full((num_p2,), 1, dtype=torch.long)
    owner_indices = torch.cat([my_owner_indices, opp_owner_indices])
    num_pieces = owner_indices.shape[0]

    # For columns, we use a dummy index not used by owners.
    column_indices = torch.full((w,), -1, dtype=torch.long)
    x = torch.cat([owner_indices, column_indices])

    piece_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    piece_mask[:num_pieces] = True
    column_mask = ~piece_mask

    coords = torch.cat([p1_locs, p2_locs], dim=0)

    edge_indices = []
    edge_types = []

    # Piece-to-column edges
    edge_type = PieceColumnEdgeType.PIECE_TO_COLUMN
    for i in range(num_pieces):
        c = coords[i, 1].item()
        piece_node_idx = i
        col_node_idx = num_pieces + c
        edge_indices.append([piece_node_idx, col_node_idx])
        edge_types.append(edge_type)
        edge_indices.append([col_node_idx, piece_node_idx])
        edge_types.append(edge_type + 1)

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        coords=coords,
        piece_mask=piece_mask,
        column_mask=column_mask,
    )
    return data


def create_cell_piece_graph(board_tensor):
    h, w = board_tensor.shape[1], board_tensor.shape[2]
    p1_board = board_tensor[0]
    p2_board = board_tensor[1]

    cell_states = torch.zeros(h, w, dtype=torch.long)

    data = HeteroData()
    data["cell"].x = cell_states.flatten()

    # Piece info
    p1_locs = torch.nonzero(p1_board)
    p2_locs = torch.nonzero(p2_board)

    num_p1_pieces = p1_locs.shape[0]
    num_p2_pieces = p2_locs.shape[0]

    if num_p1_pieces + num_p2_pieces == 0:
        data["piece"].x = torch.empty(0, 1, dtype=torch.long)
        data["piece", "occupies", "cell"].edge_index = torch.empty(
            2, 0, dtype=torch.long
        )
        return data

    piece_types = torch.cat(
        [
            torch.zeros(num_p1_pieces, dtype=torch.long),  # my pieces
            torch.ones(num_p2_pieces, dtype=torch.long),  # opponent pieces
        ]
    ).unsqueeze(1)
    data["piece"].x = piece_types

    piece_locs = torch.cat([p1_locs, p2_locs], dim=0)

    piece_indices = torch.arange(num_p1_pieces + num_p2_pieces)
    cell_indices = piece_locs[:, 0] * w + piece_locs[:, 1]

    edge_index = torch.stack([piece_indices, cell_indices], dim=0)
    data["piece", "occupies", "cell"].edge_index = edge_index

    return data


def create_combined_graph(board_tensor):
    data = create_cell_piece_graph(board_tensor)  # it's a HeteroData

    h, w = board_tensor.shape[1], board_tensor.shape[2]
    directions = {
        "N": (-1, 0),
        "E": (0, 1),
    }

    for name, (dr, dc) in directions.items():
        edges = []
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    u = r * w + c
                    v = nr * w + nc
                    edges.append([u, v])
        if edges:
            data["cell", name, "cell"].edge_index = (
                torch.tensor(edges, dtype=torch.long).t().contiguous()
            )
        else:
            data["cell", name, "cell"].edge_index = torch.empty(2, 0, dtype=torch.long)

    return data


# Training time per epoch should be recorded in the training script, as it includes data loading.
