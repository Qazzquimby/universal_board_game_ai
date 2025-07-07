import math

import torch
import torch_geometric.nn as pyg_nn
from einops import rearrange
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch, Data, HeteroData
from experiments.architectures.shared import BOARD_WIDTH


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
        num_nodes_per_graph: int,
    ):
        # edge_index: [2, num_edges]
        # edge_type: [num_edges]
        # batch_vec: [num_nodes] (node index to graph index)
        # This will need to be redone as soon as graphs aren't all the same size.
        bias_matrix = torch.zeros(
            batch_size,
            self.num_heads,
            num_nodes_per_graph,
            num_nodes_per_graph,
            device=edge_index.device,
        )

        edge_bias_values = self.edge_embedding(edge_type)  # [num_edges, num_heads]

        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        batch_indices = batch_vec[src_nodes]

        # For fixed size graphs, which CellGraphTransformer uses.
        src_nodes_in_graph = src_nodes % num_nodes_per_graph
        dst_nodes_in_graph = dst_nodes % num_nodes_per_graph

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
    ):
        super().__init__()
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
        attention_output, attention_weights = self.attention(
            normalized_x, edge_bias, key_padding_mask
        )
        edge_update_output = self.edge_gate(attention_weights, edge_type_matrix)

        x = x + self.dropout(attention_output + edge_update_output)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CellGraphTransformer(nn.Module):
    def __init__(self, num_encoder_layers=4, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()

        # Could embed or 1hot linear
        self.patch_embedding = nn.Embedding(3, embed_dim)  # 0: empty, 1: mine, 2: opp
        self.game_tokens = nn.Parameter(torch.randn(1, 1, embed_dim))

        num_edge_types = 4  # N, E (and reverse)

        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        fc1_out_size = 64
        self.fc1 = nn.Linear(embed_dim, fc1_out_size)
        self.policy_head = nn.Linear(fc1_out_size, BOARD_WIDTH)
        self.value_head = nn.Linear(fc1_out_size, 1)

    def forward(self, data: Data):
        cell_states = data.x

        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(0, BOARD_WIDTH, device=device), torch.zeros(
                0, 1, device=device
            )

        num_nodes_total = cell_states.size(0)
        num_nodes_per_graph = num_nodes_total // batch_size

        x_embedded = self.patch_embedding(cell_states)
        x = rearrange(
            x_embedded, "(batch seq) emb_dim -> batch seq emb_dim", batch=batch_size
        )
        game_tokens = self.game_tokens.expand(batch_size, -1, -1)
        x = torch.cat((game_tokens, x), dim=1)
        x = self.dropout(x)
        num_nodes_with_token = num_nodes_per_graph + 1  # +1 for game token

        ## scalar edge bias matrix
        edge_bias = self.edge_bias_module(
            data.edge_index, data.edge_type, data.batch, batch_size, num_nodes_per_graph
        )
        edge_bias = F.pad(edge_bias, (1, 0, 1, 0), "constant", 0)  # Pad for game token

        ## edge type matrix for gating
        edge_type_matrix = torch.zeros(
            batch_size,
            num_nodes_with_token,
            num_nodes_with_token,
            dtype=torch.long,
            device=x.device,
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_in_graph = src % num_nodes_per_graph + 1  # +1 to account for game_token
        dst_in_graph = dst % num_nodes_per_graph + 1  # +1 to account for game_token
        edge_type_matrix[batch_indices, src_in_graph, dst_in_graph] = data.edge_type

        key_padding_mask = None
        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, 0, :]  # (batch_size, embed_dim)

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

        # 0: empty, 1: mine, 2: opp, 3: column token
        self.token_embedding = nn.Embedding(4, embed_dim)
        self.game_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        num_edge_types = 6  # N, E, cell->col (and reverse)
        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        fc_out_size = 64
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_fc = nn.Linear(embed_dim, fc_out_size)
        self.value_head = nn.Linear(fc_out_size, 1)

    def forward(self, data: Data):
        cell_states = data.x

        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(0, self.board_width, device=device), torch.zeros(
                0, 1, device=device
            )

        num_nodes_total = cell_states.size(0)
        num_nodes_per_graph = num_nodes_total // batch_size

        x_embedded = self.token_embedding(cell_states)
        x = rearrange(
            x_embedded, "(batch seq) emb_dim -> batch seq emb_dim", batch=batch_size
        )

        game_tokens = self.game_token.expand(batch_size, -1, -1)
        x = torch.cat((game_tokens, x), dim=1)
        x = self.dropout(x)
        num_nodes_with_token = num_nodes_per_graph + 1  # +1 for game token

        ## scalar edge bias matrix
        edge_bias = self.edge_bias_module(
            data.edge_index, data.edge_type, data.batch, batch_size, num_nodes_per_graph
        )
        edge_bias = F.pad(edge_bias, (1, 0, 1, 0), "constant", 0)  # Pad for game token

        edge_type_matrix = torch.zeros(
            batch_size,
            num_nodes_with_token,
            num_nodes_with_token,
            dtype=torch.long,
            device=x.device,
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_in_graph = src % num_nodes_per_graph + 1  # +1 to account for game_token
        dst_in_graph = dst % num_nodes_per_graph + 1  # +1 to account for game_token
        edge_type_matrix[batch_indices, src_in_graph, dst_in_graph] = data.edge_type

        key_padding_mask = None  # since static
        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, 0, :]  # (batch_size, embed_dim)
        game_token_encoded = F.relu(self.value_fc(game_token_out))
        value = torch.tanh(self.value_head(game_token_encoded))

        num_cell_nodes = self.board_height * self.board_width
        # After game token, the node embeddings from the graph data start.
        # These are ordered as cells then columns.
        column_tokens_out = transformer_output[:, 1 + num_cell_nodes :, :]
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
        max_pieces=42,
    ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.max_pieces = max_pieces

        # 0:cell, 1:col, 2:my_piece, 3:opp_piece, 4:pad_piece
        self.token_embedding = nn.Embedding(5, embed_dim)
        self.game_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        num_edge_types = 8  # N/S/E/W, cell-col, piece-cell
        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        fc_out_size = 64
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_fc = nn.Linear(embed_dim, fc_out_size)
        self.value_head = nn.Linear(fc_out_size, 1)

    def forward(self, data: Data):
        node_types = data.x

        batch_size = data.num_graphs
        if batch_size == 0:
            device = self.policy_head.weight.device
            return torch.zeros(0, self.board_width, device=device), torch.zeros(
                0, 1, device=device
            )

        num_nodes_total = node_types.size(0)
        num_nodes_per_graph = num_nodes_total // batch_size

        x_embedded = self.token_embedding(node_types)
        x = rearrange(
            x_embedded, "(batch seq) emb_dim -> batch seq emb_dim", batch=batch_size
        )

        game_tokens = self.game_token.expand(batch_size, -1, -1)
        x = torch.cat((game_tokens, x), dim=1)
        x = self.dropout(x)
        num_nodes_with_token = num_nodes_per_graph + 1  # +1 for game token

        ## scalar edge bias matrix
        edge_bias = self.edge_bias_module(
            data.edge_index, data.edge_type, data.batch, batch_size, num_nodes_per_graph
        )
        edge_bias = F.pad(edge_bias, (1, 0, 1, 0), "constant", 0)  # Pad for game token

        edge_type_matrix = torch.zeros(
            batch_size,
            num_nodes_with_token,
            num_nodes_with_token,
            dtype=torch.long,
            device=x.device,
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_in_graph = src % num_nodes_per_graph + 1  # +1 to account for game_token
        dst_in_graph = dst % num_nodes_per_graph + 1  # +1 to account for game_token
        edge_type_matrix[batch_indices, src_in_graph, dst_in_graph] = data.edge_type

        key_padding_mask = None  # since static
        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, 0, :]  # (batch_size, embed_dim)
        game_token_encoded = F.relu(self.value_fc(game_token_out))
        value = torch.tanh(self.value_head(game_token_encoded))

        num_cell_nodes = self.board_height * self.board_width
        num_col_nodes = self.board_width
        col_start_idx = 1 + num_cell_nodes
        col_end_idx = col_start_idx + num_col_nodes
        column_tokens_out = transformer_output[:, col_start_idx:col_end_idx, :]
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
        max_pieces=42,
    ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.max_pieces = max_pieces

        self.owner_embedding = nn.Embedding(3, embed_dim)  # 0:my, 1:opp, 2:pad
        self.row_embedding = nn.Embedding(board_height, embed_dim)
        self.col_embedding = nn.Embedding(board_width, embed_dim)
        self.column_token = nn.Parameter(torch.randn(1, embed_dim))

        self.game_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        num_edge_types = 2  # piece->col, col->piece
        self.edge_bias_module = HeteroEdgeBias(num_heads, num_edge_types)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
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

        num_nodes_per_graph = self.max_pieces + self.board_width
        num_piece_nodes = self.max_pieces

        owner_indices = data.x.view(batch_size, num_nodes_per_graph)[
            :, :num_piece_nodes
        ]

        # Create piece embeddings
        coords = data.coords.view(batch_size, num_piece_nodes, 2)
        owner_emb = self.owner_embedding(owner_indices)
        row_emb = self.row_embedding(coords[:, :, 0])
        col_emb = self.col_embedding(coords[:, :, 1])
        piece_embedded = owner_emb + row_emb + col_emb

        # Create column embeddings
        column_embedded = self.column_token.expand(
            batch_size, self.board_width, -1
        )

        x = torch.cat((piece_embedded, column_embedded), dim=1)

        game_tokens = self.game_token.expand(batch_size, -1, -1)
        x = torch.cat((game_tokens, x), dim=1)
        x = self.dropout(x)
        num_nodes_with_token = num_nodes_per_graph + 1

        ## scalar edge bias matrix
        edge_bias = self.edge_bias_module(
            data.edge_index, data.edge_type, data.batch, batch_size, num_nodes_per_graph
        )
        edge_bias = F.pad(edge_bias, (1, 0, 1, 0), "constant", 0)

        edge_type_matrix = torch.zeros(
            batch_size,
            num_nodes_with_token,
            num_nodes_with_token,
            dtype=torch.long,
            device=x.device,
        )
        src, dst = data.edge_index
        batch_indices = data.batch[src]
        src_in_graph = src % num_nodes_per_graph + 1
        dst_in_graph = dst % num_nodes_per_graph + 1
        edge_type_matrix[batch_indices, src_in_graph, dst_in_graph] = data.edge_type

        key_padding_mask = None
        for layer in self.layers:
            x = layer(x, edge_bias, edge_type_matrix, key_padding_mask)
        transformer_output = x

        game_token_out = transformer_output[:, 0, :]
        game_token_encoded = F.relu(self.value_fc(game_token_out))
        value = torch.tanh(self.value_head(game_token_encoded))

        col_start_idx = 1 + num_piece_nodes
        col_end_idx = col_start_idx + self.board_width
        column_tokens_out = transformer_output[:, col_start_idx:col_end_idx, :]
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
    cell_states = torch.zeros(h, w, dtype=torch.long)
    cell_states[p1_board == 1] = 1  # my piece
    cell_states[p2_board == 1] = 2  # opp piece
    x = cell_states.flatten()

    edge_indices = []
    edge_types = []

    # edge type 1: North, 3: East. Reverse edges get type+1 (2:S, 4:W)
    directions = [(-1, 0), (0, 1)]  # N, E

    for i, (dr, dc) in enumerate(directions):
        edge_type = 2 * i + 1
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
    cell_states = torch.zeros(h, w, dtype=torch.long)
    cell_states[p1_board == 1] = 1  # my piece
    cell_states[p2_board == 1] = 2  # opp piece

    # Node features: cell states followed by column markers
    column_markers = torch.full((w,), 3, dtype=torch.long)
    x = torch.cat([cell_states.flatten(), column_markers])

    edge_indices = []
    edge_types = []

    # Cell-to-cell edges
    # edge type 1: N, 2: S, 3: E, 4: W
    directions = [(-1, 0), (0, 1)]  # N, E
    for i, (dr, dc) in enumerate(directions):
        edge_type = 2 * i + 1
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
    # edge type 5: cell->col, 6: col->cell
    num_cell_nodes = h * w
    for c in range(w):
        col_node_idx = num_cell_nodes + c
        for r in range(h):
            cell_node_idx = r * w + c
            edge_indices.append([cell_node_idx, col_node_idx])
            edge_types.append(5)
            edge_indices.append([col_node_idx, cell_node_idx])
            edge_types.append(6)

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

    return data


def create_cell_column_piece_graph(board_tensor, max_pieces=42):
    h, w = board_tensor.shape[1], board_tensor.shape[2]
    p1_board = board_tensor[0]
    p2_board = board_tensor[1]

    # Node features
    cell_tokens = torch.full((h * w,), 0, dtype=torch.long)
    column_tokens = torch.full((w,), 1, dtype=torch.long)

    p1_locs = torch.nonzero(p1_board)
    p2_locs = torch.nonzero(p2_board)
    num_p1 = p1_locs.shape[0]
    num_p2 = p2_locs.shape[0]

    piece_tokens = torch.full((max_pieces,), 4, dtype=torch.long)  # 4 is pad
    piece_tokens[:num_p1] = 2  # my piece
    piece_tokens[num_p1 : num_p1 + num_p2] = 3  # opp piece

    x = torch.cat([cell_tokens, column_tokens, piece_tokens])

    edge_indices = []
    edge_types = []

    # Cell-to-cell edges (types 1-4)
    directions = [(-1, 0), (0, 1)]  # N, E
    for i, (dr, dc) in enumerate(directions):
        edge_type = 2 * i + 1
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
    for c in range(w):
        col_node_idx = num_cell_nodes + c
        for r in range(h):
            cell_node_idx = r * w + c
            edge_indices.append([cell_node_idx, col_node_idx])
            edge_types.append(5)
            edge_indices.append([col_node_idx, cell_node_idx])
            edge_types.append(6)

    # Piece-to-cell edges (types 7-8)
    num_col_nodes = w
    piece_start_idx = num_cell_nodes + num_col_nodes
    all_locs = torch.cat([p1_locs, p2_locs], dim=0)
    for i, loc in enumerate(all_locs):
        r, c = loc[0].item(), loc[1].item()
        piece_node_idx = piece_start_idx + i
        cell_node_idx = r * w + c
        edge_indices.append([piece_node_idx, cell_node_idx])
        edge_types.append(7)
        edge_indices.append([cell_node_idx, piece_node_idx])
        edge_types.append(8)

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    return data


def create_piece_column_graph(board_tensor, max_pieces=42):
    h, w = board_tensor.shape[1], board_tensor.shape[2]
    p1_board = board_tensor[0]
    p2_board = board_tensor[1]

    p1_locs = torch.nonzero(p1_board)
    p2_locs = torch.nonzero(p2_board)
    num_p1 = p1_locs.shape[0]
    num_p2 = p2_locs.shape[0]

    owner_indices = torch.full((max_pieces,), 2, dtype=torch.long)  # 2 is pad
    owner_indices[:num_p1] = 0  # my piece
    owner_indices[num_p1 : num_p1 + num_p2] = 1  # opp piece

    # For columns, we use a different type index.
    column_indices = torch.full((w,), 3, dtype=torch.long)
    x = torch.cat([owner_indices, column_indices])

    coords = torch.zeros(max_pieces, 2, dtype=torch.long)
    all_locs = torch.cat([p1_locs, p2_locs], dim=0)
    if all_locs.shape[0] > 0:
        coords[: all_locs.shape[0]] = all_locs

    edge_indices = []
    edge_types = []

    # Piece-to-column edges
    num_piece_nodes = max_pieces
    for i in range(num_p1 + num_p2):
        c = coords[i, 1].item()
        piece_node_idx = i
        col_node_idx = num_piece_nodes + c
        edge_indices.append([piece_node_idx, col_node_idx])
        edge_types.append(1)
        edge_indices.append([col_node_idx, piece_node_idx])
        edge_types.append(2)

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, coords=coords)
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


# TODO
# Could you explain why the padding is 1 0 1 0?
# Could you adjust the code to work with varying sized graphs? I'd like to try a model where spaces and pieces are both tokens and the number of pieces varies.
# I do not like the manual indexing. I'd like to be able to map "game token" or "column tokens" to their indexes automatically.
