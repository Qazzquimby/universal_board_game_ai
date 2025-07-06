import math

import torch
import torch_geometric.nn as pyg_nn
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data, HeteroData
from experiments.architectures.shared import BOARD_WIDTH


# TODO highly unsure this is what I want
# I want to be able to say "token A has a directed edge of some type, pointing to token B."
# I want that to adjust both token's attention towards each other, and learn it differently for each edge type and for from and to.
# There may be very few edges, so I don't like the input being a matrix like this.
class HeteroEdgeBias(nn.Module):
    def __init__(self, num_heads, num_edge_types):
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types + 1, num_heads)
        nn.init.zeros_(self.edge_embedding.weight)

    def forward(self, edge_type_matrix):
        # edge_type_matrix: [B, N, N]
        # embedding output: [B, N, N, H]
        # desired output: [B, H, N, N]
        bias = self.edge_embedding(edge_type_matrix)
        bias = rearrange(bias, "B N1 N2 H -> B H N1 N2")
        return bias


class EdgeBiasedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.head_dim = (
            embed_dim,
            num_heads,
            embed_dim // num_heads,
        )
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias, key_padding_mask=None):
        B, N, D = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Inject the edge bias
        attn_scores = attn_scores + edge_bias

        # Add padding mask to prevent attention to padding tokens
        if key_padding_mask is not None:
            # Reshape mask for broadcasting: [B, N] -> [B, 1, 1, N]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(output)


class GraphTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.attn = EdgeBiasedMultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_bias, key_padding_mask=None):
        attn_output = self.attn(self.norm1(x), edge_bias, key_padding_mask)
        x = x + self.dropout1(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)
        return x


# Please focus on this class
# It should have one token for each cell (connect4) containing empty, mine, opponent.
# There should be an edge affecting attention for east and north.
# Reverse connections should also be learned but dont need their own reverse-edge. That is, if A has edge of type 3 to B, A and B should both have a learned an attention bias for the other.
# This is not a message passing gnn. It's a fully connected transformer.


class CellGraphTransformer(nn.Module):
    def __init__(
        self, num_encoder_layers=4, embedding_dim=128, num_heads=4, dropout=0.1
    ):
        super().__init__()

        # Could embed or 1hot linear
        self.patch_embedding = nn.Embedding(
            3, embedding_dim
        )  # 0: empty, 1: mine, 2: opp
        self.game_tokens = nn.Parameter(torch.randn(1, 1, embedding_dim))

        num_edge_types = 2  # N, E
        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, data: Data):
        cell_states = data.x
        edge_type_matrix = data.edge_type_matrix

        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(
                0, BOARD_WIDTH, device=device
            ), torch.zeros(0, 1, device=device)

        x_embedded = self.patch_embedding(cell_states)
        x = rearrange(x_embedded, "(B N) D -> B N D", B=batch_size)

        # Prepend game token
        game_tokens = self.game_tokens.expand(batch_size, -1, -1)
        x = torch.cat((game_tokens, x), dim=1)
        x = self.dropout(x)

        edge_bias = self.edge_bias_module(edge_type_matrix)
        # Pad bias for game token (no bias for game token)
        edge_bias = F.pad(edge_bias, (1, 0, 1, 0))

        key_padding_mask = None  # since static
        for layer in self.layers:
            x = layer(x, edge_bias, key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, 0, :]  # (batch_size, embedding_dim)

        out = F.relu(self.fc1(game_token_out))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

        return policy_logits, value


class CellPieceGraphTransformer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cell_embedding = nn.Embedding(1, embedding_dim)
        self.piece_embedding = nn.Embedding(2, embedding_dim)

        # TODO pretty sure I don't want a conv, I want to add edge attention manually.
        # todo redo this and all later cell graph transformers.
        # https://aistudio.google.com/prompts/12TRLcHI6P9AI7Wuyv1owLas8Klr-8kyz

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = pyg_nn.HeteroConv(
                {
                    ("piece", "occupies", "cell"): pyg_nn.GATv2Conv(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                        concat=False,
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, data):
        x_dict = {
            "cell": self.cell_embedding(data["cell"].x),
            "piece": self.piece_embedding(data["piece"].x.squeeze(-1)),
        }

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        graph_embedding = pyg_nn.global_mean_pool(x_dict["cell"], data["cell"].batch)

        out = F.relu(self.fc1(graph_embedding))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

        return policy_logits, value


class CombinedGraphTransformer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cell_embedding = nn.Embedding(1, embedding_dim)
        self.piece_embedding = nn.Embedding(2, embedding_dim)

        self.convs = nn.ModuleList()
        directions = ["N", "E"]

        for _ in range(num_layers):
            conv_dict = {
                ("piece", "occupies", "cell"): pyg_nn.GATv2Conv(
                    embedding_dim,
                    embedding_dim,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=False,
                ),
            }
            for d in directions:
                conv_dict[("cell", d, "cell")] = pyg_nn.GATv2Conv(
                    embedding_dim,
                    embedding_dim,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=False,
                )

            conv = pyg_nn.HeteroConv(conv_dict, aggr="sum")
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)

        fc1_out_size = 64
        self.fc1 = nn.Linear(embedding_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, data):
        x_dict = {
            "cell": self.cell_embedding(data["cell"].x),
            "piece": self.piece_embedding(data["piece"].x.squeeze(-1)),
        }

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        graph_embedding = pyg_nn.global_mean_pool(x_dict["cell"], data["cell"].batch)

        out = F.relu(self.fc1(graph_embedding))
        policy_logits = self.policy_head(out)
        value = torch.tanh(self.value_head(out))

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
    cell_states = torch.zeros(h, w, dtype=torch.long)
    cell_states[p1_board == 1] = 1  # my piece
    cell_states[p2_board == 1] = 2  # opp piece
    x = cell_states.flatten()

    edge_type_matrix = torch.zeros(h * w, h * w, dtype=torch.long)

    # edge type 1: North, 2: East
    directions = [(-1, 0), (0, 1)]  # N, E

    for i, (dr, dc) in enumerate(directions):
        edge_type = i + 1
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    u = r * w + c
                    v = nr * w + nc
                    edge_type_matrix[u, v] = edge_type
                    edge_type_matrix[v, u] = edge_type

    data = Data(x=x, edge_type_matrix=edge_type_matrix)

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
