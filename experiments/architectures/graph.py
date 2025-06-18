import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool, HGTConv

from experiments.architectures.shared import BOARD_HEIGHT, BOARD_WIDTH


def get_connect4_graph_with_edge_attrs(board_tensor):
    """Constructs a PyG Data object with edge attributes for direction."""
    p1_pieces = board_tensor[0].numpy()
    p2_pieces = board_tensor[1].numpy()

    node_features = []
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            is_p1 = p1_pieces[r, c] == 1
            is_p2 = p2_pieces[r, c] == 1
            is_empty = not is_p1 and not is_p2
            node_features.append([float(is_empty), float(is_p1), float(is_p2)])
    x = torch.tensor(node_features, dtype=torch.float)

    edges = []
    edge_attrs = []
    direction_map = {}
    idx = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            direction_map[(dr, dc)] = idx
            idx += 1

    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            node_idx = r * BOARD_WIDTH + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH:
                        neighbor_idx = nr * BOARD_WIDTH + nc
                        edges.append([node_idx, neighbor_idx])

                        direction_idx = direction_map[(dr, dc)]
                        attr = [0] * 8
                        attr[direction_idx] = 1
                        edge_attrs.append(attr)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class DirectionalConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")  # "mean" aggregation.
        self.edge_mlp = nn.Linear(
            in_channels + 8, out_channels
        )  # Process node feature + edge feature
        self.node_mlp = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j is the feature of the neighbor node
        # Concatenate neighbor feature with the edge's directional feature
        input_for_mlp = torch.cat([x_j, edge_attr], dim=1)
        return self.edge_mlp(input_for_mlp)

    def update(self, aggr_out, x):
        # aggr_out is the aggregated messages from neighbors
        # We can also add a skip connection from the original node feature
        return aggr_out + self.node_mlp(x)


class DirectionalGNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        num_layers=2,
        pooling_fn=global_mean_pool,
    ):
        super().__init__()
        self.pooling = pooling_fn
        self.in_proj = nn.Linear(in_channels, hidden_channels)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = DirectionalConv(hidden_channels, hidden_channels)
            self.conv_layers.append(conv)

        self.fc1 = nn.Linear(hidden_channels, 128)
        self.policy_head = nn.Linear(128, BOARD_WIDTH)
        self.value_head = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = self.in_proj(x)
        x = F.relu(x)

        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        graph_embedding = self.pooling(x, batch)

        x = F.relu(self.fc1(graph_embedding))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        return policy_logits, value


def construct_graph_data(board_tensor):
    """Constructs a PyG HeteroData object from a board tensor."""
    p1_pieces = board_tensor[0].numpy()
    p2_pieces = board_tensor[1].numpy()

    node_features = []
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            is_p1 = p1_pieces[r, c] == 1
            is_p2 = p2_pieces[r, c] == 1
            is_empty = not is_p1 and not is_p2
            node_features.append([float(is_empty), float(is_p1), float(is_p2)])

    data = HeteroData()
    data["cell"].x = torch.tensor(node_features, dtype=torch.float)

    directions = {
        "N": (-1, 0),
        "NE": (-1, 1),
        "E": (0, 1),
        "SE": (1, 1),
        "S": (1, 0),
        "SW": (1, -1),
        "W": (0, -1),
        "NW": (-1, -1),
    }

    for name, (dr, dc) in directions.items():
        sources, dests = [], []
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH:
                    sources.append(r * BOARD_WIDTH + c)
                    dests.append(nr * BOARD_WIDTH + nc)
        edge_type = ("cell", name, "cell")
        data[edge_type].edge_index = torch.tensor([sources, dests], dtype=torch.long)

    return data


class Connect4GraphDataset(Dataset):
    def __init__(self, graphs, raw_inputs, policy_labels, value_labels):
        self.graphs = graphs
        self.raw_inputs = torch.from_numpy(raw_inputs)
        self.policy_labels = torch.from_numpy(policy_labels).long()
        self.value_labels = torch.from_numpy(value_labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.policy_labels[idx],
            self.value_labels[idx],
        )


def pyg_collate_fn(batch):
    """Custom collate function to batch graph data and labels correctly."""
    graphs, policies, values = zip(*batch)
    batched_graph = Batch.from_data_list(list(graphs))
    batched_policies = torch.stack(list(policies))
    batched_values = torch.stack(list(values))
    return batched_graph, batched_policies, batched_values


class GraphNet(nn.Module):
    def __init__(
        self,
        hidden_channels=64,
        num_heads=4,
        num_layers=2,
        pooling_fn=global_mean_pool,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.pooling = pooling_fn

        node_types = ["cell"]
        edge_types = [
            ("cell", "N", "cell"),
            ("cell", "NE", "cell"),
            ("cell", "E", "cell"),
            ("cell", "SE", "cell"),
            ("cell", "S", "cell"),
            ("cell", "SW", "cell"),
            ("cell", "W", "cell"),
            ("cell", "NW", "cell"),
        ]
        self.metadata = (node_types, edge_types)

        self.in_proj = nn.Linear(3, hidden_channels)

        self.hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, self.metadata, num_heads)
            self.hgt_layers.append(conv)

        self.fc1 = nn.Linear(hidden_channels, 128)
        self.policy_head = nn.Linear(128, BOARD_WIDTH)
        self.value_head = nn.Linear(128, 1)

    def forward(self, data):
        x_dict = {"cell": self.in_proj(data["cell"].x)}

        for layer in self.hgt_layers:
            x_dict = layer(x_dict, data.edge_index_dict)

        cell_features = x_dict["cell"]

        graph_embedding = self.pooling(cell_features, data["cell"].batch)

        x = F.relu(self.fc1(graph_embedding))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        return policy_logits, value
