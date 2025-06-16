# Symmetricalize by x dim and player
# repeat but with manually creating graph implementations and understanding them well.


import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import wandb
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import (
    GATConv,
    HGTConv,
    MessagePassing,
    global_add_pool,
    global_mean_pool,
)

from environments.connect4 import Connect4

BOARD_HEIGHT = 6
BOARD_WIDTH = 7
MAX_EPOCHS = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "connect_4_states"
    / "pure_mcts"
    / "mcts_generated_states.json"
)


def _process_raw_item(item, env):
    action_history_str = item.get("action_history", "")
    if not action_history_str:
        return None

    env.reset()
    actions = [int(a) for a in action_history_str]
    for action in actions:
        env.step(action)

    current_player_idx = env.get_current_player()
    opponent_player_idx = 1 - current_player_idx

    current_player_piece = current_player_idx + 1
    opponent_player_piece = opponent_player_idx + 1

    board_state = np.array(env.board)
    p1_board = (board_state == current_player_piece).astype(np.float32)
    p2_board = (board_state == opponent_player_piece).astype(np.float32)
    input_tensor = np.stack([p1_board, p2_board])

    policy_label = item["next_action"]

    winner_piece = item["winner"]
    value = 0.0
    if winner_piece is not None and winner_piece != 0:
        if winner_piece == current_player_piece:
            value = 1.0
        else:
            value = -1.0

    return input_tensor, policy_label, value


def create_transformer_input(board_tensor):
    """
    Converts a board tensor into a sequence of piece tokens for a transformer.
    Each token has features: [is_my_piece, is_opp_piece, norm_x, norm_y].
    Returns a tensor of shape (num_pieces, 4).
    """
    p1_pieces = board_tensor[0]  # My pieces
    p2_pieces = board_tensor[1]  # Opponent's pieces

    piece_features = []

    # My pieces
    my_locs = torch.nonzero(p1_pieces)
    for loc in my_locs:
        r, c = loc[0].item(), loc[1].item()
        features = [1.0, 0.0, c / (BOARD_WIDTH - 1), r / (BOARD_HEIGHT - 1)]
        piece_features.append(features)

    # Opponent's pieces
    opp_locs = torch.nonzero(p2_pieces)
    for loc in opp_locs:
        r, c = loc[0].item(), loc[1].item()
        features = [0.0, 1.0, c / (BOARD_WIDTH - 1), r / (BOARD_HEIGHT - 1)]
        piece_features.append(features)

    if not piece_features:  # Handle empty board
        return torch.empty(0, 4, dtype=torch.float)

    return torch.tensor(piece_features, dtype=torch.float)


def load_and_process_data():
    print("Loading and processing data...")
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)

    inputs = []
    policy_labels = []
    value_labels = []

    env = Connect4(width=BOARD_WIDTH, height=BOARD_HEIGHT)

    for item in tqdm(raw_data):
        processed_item = _process_raw_item(item, env)
        if processed_item:
            input_tensor, policy_label, value_label = processed_item
            inputs.append(input_tensor)
            policy_labels.append(policy_label)
            value_labels.append(value_label)

    return (
        np.array(inputs),
        np.array(policy_labels),
        np.array(value_labels, dtype=np.float32),
    )


class Connect4Dataset(Dataset):
    def __init__(self, inputs, policy_labels, value_labels):
        self.inputs = torch.from_numpy(inputs)
        self.policy_labels = torch.from_numpy(policy_labels).long()
        self.value_labels = torch.from_numpy(value_labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.policy_labels[idx], self.value_labels[idx]


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


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        input_size = 2 * BOARD_HEIGHT * BOARD_WIDTH
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, BOARD_WIDTH)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * BOARD_HEIGHT * BOARD_WIDTH, 128)
        self.policy_head = nn.Linear(128, BOARD_WIDTH)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv_in = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 64)
        self.fc1 = nn.Linear(64 * BOARD_HEIGHT * BOARD_WIDTH, 128)
        self.policy_head = nn.Linear(128, BOARD_WIDTH)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value


class PieceTransformerNet(nn.Module):
    def __init__(
        self,
        num_encoder_layers=4,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        # Input features: [is_my_piece, is_opp_piece, norm_x, norm_y] -> 4 features
        self.input_proj = nn.Linear(4, embedding_dim)

        # Special token for global board representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.fc1 = nn.Linear(embedding_dim, 128)
        self.policy_head = nn.Linear(128, BOARD_WIDTH)
        self.value_head = nn.Linear(128, 1)

    def forward(self, src, src_key_padding_mask):
        # src shape: (batch_size, seq_len, feature_dim=4)
        # src_key_padding_mask shape: (batch_size, seq_len)

        # Project input features to embedding dimension
        src = self.input_proj(src)  # (batch_size, seq_len, embedding_dim)

        # Prepend CLS token
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)

        # Adjust padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, device=src.device, dtype=torch.bool)
        src_key_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # Use the output of the CLS token for prediction
        cls_output = transformer_output[:, 0, :]  # (batch_size, embedding_dim)

        x = F.relu(self.fc1(cls_output))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        return policy_logits, value


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
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + 8, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels),
        )
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
    # Does very poorly, possibly due to missing attention mechanism
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


def transformer_collate_fn(batch):
    """
    Custom collate function for transformer.
    Pads sequences to the max length in the batch and creates a padding mask.
    """
    inputs, policies, values = zip(*batch)

    # Pad input sequences
    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)

    # Create attention mask (True for padded elements)
    lengths = [len(x) for x in inputs]
    attention_mask = (
        torch.arange(padded_inputs.size(1))[None, :] >= torch.tensor(lengths)[:, None]
    )

    batched_policies = torch.stack(list(policies))
    batched_values = torch.stack(list(values))

    return (padded_inputs, attention_mask), batched_policies, batched_values


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


def _process_batch(model, data_batch, device, policy_criterion, value_criterion):
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


def _process_batch_transformer(
    model, data_batch, device, policy_criterion, value_criterion
):
    (inputs, attention_mask), policy_labels, value_labels = data_batch
    inputs = inputs.to(device)
    attention_mask = attention_mask.to(device)
    policy_labels = policy_labels.to(device)
    value_labels = value_labels.to(device)

    policy_logits, value_preds = model(inputs, attention_mask)

    loss_p = policy_criterion(policy_logits, policy_labels)
    loss_v = value_criterion(value_preds.squeeze(), value_labels)
    loss = loss_p + loss_v

    _, predicted_policies = torch.max(policy_logits, 1)
    policy_acc = (predicted_policies == policy_labels).int().sum().item()
    value_mse = F.mse_loss(value_preds.squeeze(), value_labels).item()

    return loss, loss_p.item(), loss_v.item(), policy_acc, value_mse


def train_and_evaluate(
    model,
    model_name,
    train_loader,
    test_loader,
    learning_rate=LEARNING_RATE,
    process_batch_fn=_process_batch,
):
    print(f"\n--- Training {model_name} on {DEVICE} ---")
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    results = []
    best_test_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_train_loss, total_train_policy_acc, total_train_value_mse = 0, 0, 0
        total_train_policy_loss, total_train_value_loss = 0, 0

        for data_batch in train_loader:
            optimizer.zero_grad()
            loss, loss_p_item, loss_v_item, policy_acc, value_mse = process_batch_fn(
                model, data_batch, DEVICE, policy_criterion, value_criterion
            )
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_policy_loss += loss_p_item
            total_train_value_loss += loss_v_item
            total_train_policy_acc += policy_acc
            total_train_value_mse += value_mse

        model.eval()
        total_test_loss, total_test_policy_acc, total_test_value_mse = 0, 0, 0
        total_test_policy_loss, total_test_value_loss = 0, 0
        with torch.no_grad():
            for data_batch in test_loader:
                (
                    loss,
                    loss_p_item,
                    loss_v_item,
                    policy_acc,
                    value_mse,
                ) = process_batch_fn(
                    model, data_batch, DEVICE, policy_criterion, value_criterion
                )
                total_test_loss += loss.item()
                total_test_policy_loss += loss_p_item
                total_test_value_loss += loss_v_item
                total_test_policy_acc += policy_acc
                total_test_value_mse += value_mse

        train_batches = len(train_loader)
        train_loss = total_train_loss / train_batches
        train_policy_loss = total_train_policy_loss / train_batches
        train_value_loss = total_train_value_loss / train_batches
        train_acc = total_train_policy_acc / len(train_loader.dataset)
        train_mse = total_train_value_mse / train_batches

        test_batches = len(test_loader)
        test_loss = total_test_loss / test_batches
        test_policy_loss = total_test_policy_loss / test_batches
        test_value_loss = total_test_value_loss / test_batches
        test_acc = total_test_policy_acc / len(test_loader.dataset)
        test_mse = total_test_value_mse / test_batches

        log_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_policy_loss": train_policy_loss,
            "train_value_loss": train_value_loss,
            "train_acc": train_acc,
            "train_mse": train_mse,
            "test_loss": test_loss,
            "test_policy_loss": test_policy_loss,
            "test_value_loss": test_value_loss,
            "test_acc": test_acc,
            "test_mse": test_mse,
        }

        wandb.log(log_info)
        results.append(log_info)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    training_time = time.time() - start_time
    final_epoch = len(results)
    final_results = results[-1]
    print(f"Converged after {final_epoch} epochs.")
    print(
        f"Final Results | Test Loss: {final_results['test_loss']:.4f}, Test Acc: {final_results['test_acc']:.4f}, Test MSE: {final_results['test_mse']:.4f}"
    )
    return pd.DataFrame(results), training_time


def main():
    run_group_id = f"run_{int(time.time())}"
    inputs, policy_labels, value_labels = load_and_process_data()

    X_train, X_test, p_train, p_test, v_train, v_test = train_test_split(
        inputs, policy_labels, value_labels, test_size=0.2, random_state=42
    )

    train_dataset = Connect4Dataset(X_train, p_train, v_train)
    test_dataset = Connect4Dataset(X_test, p_test, v_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models_to_train = {}  # {"MLP": MLPNet(), "CNN": CNNNet(), "ResNet": ResNet()}

    all_results = {}
    for name, model in models_to_train.items():
        wandb.init(
            project="connect4_arch_comparison",
            name=name,
            group=run_group_id,
            config={
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "architecture": name,
            },
        )
        model.to(DEVICE)
        results_df, training_time = train_and_evaluate(
            model, name, train_loader, test_loader
        )
        all_results[name] = {"df": results_df, "time": training_time}
        wandb.finish()

    # --- Transformer Experiment ---
    print("\n--- Pre-processing data for Transformer ---")
    transformer_train_inputs = [
        create_transformer_input(torch.from_numpy(board))
        for board in tqdm(X_train, desc="Processing Transformer inputs")
    ]
    transformer_test_inputs = [
        create_transformer_input(torch.from_numpy(board))
        for board in tqdm(X_test, desc="Processing Transformer inputs")
    ]

    transformer_train_dataset = Connect4TransformerDataset(
        transformer_train_inputs, p_train, v_train
    )
    transformer_test_dataset = Connect4TransformerDataset(
        transformer_test_inputs, p_test, v_test
    )

    transformer_train_loader = DataLoader(
        transformer_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=transformer_collate_fn,
    )
    transformer_test_loader = DataLoader(
        transformer_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=transformer_collate_fn,
    )

    transformer_exp = {
        "name": "PieceTransformer",
        "model_class": PieceTransformerNet,
        "params": {"num_encoder_layers": 4, "embedding_dim": 128, "num_heads": 8},
        "lr": 0.001,
    }

    wandb.init(
        project="connect4_arch_comparison",
        name=transformer_exp["name"],
        group=run_group_id,
        reinit=True,
        config={
            "learning_rate": transformer_exp["lr"],
            "batch_size": BATCH_SIZE,
            "architecture": transformer_exp["model_class"].__name__,
            **transformer_exp["params"],
        },
    )
    model = transformer_exp["model_class"](**transformer_exp["params"]).to(DEVICE)
    results_df, training_time = train_and_evaluate(
        model=model,
        model_name=transformer_exp["name"],
        train_loader=transformer_train_loader,
        test_loader=transformer_test_loader,
        learning_rate=transformer_exp["lr"],
        process_batch_fn=_process_batch_transformer,
    )
    all_results[transformer_exp["name"]] = {"df": results_df, "time": training_time}
    wandb.finish()

    # --- GNN Experiments ---
    print("\n--- Pre-processing data for GNNs ---")
    cell_train_graphs = [
        construct_graph_data(torch.from_numpy(board))
        for board in tqdm(X_train, desc="Processing Hetero graphs")
    ]
    cell_test_graphs = [
        construct_graph_data(torch.from_numpy(board))
        for board in tqdm(X_test, desc="Processing Hetero graphs")
    ]

    dir_train_graphs = [
        get_connect4_graph_with_edge_attrs(torch.from_numpy(board))
        for board in tqdm(X_train, desc="Processing Directional graphs")
    ]
    dir_test_graphs = [
        get_connect4_graph_with_edge_attrs(torch.from_numpy(board))
        for board in tqdm(X_test, desc="Processing Directional graphs")
    ]

    gnn_experiments = [
        # {
        #     "name": "CellGNN_HGT",
        #     "model_class": GraphNet,
        #     "train_graphs": cell_train_graphs,
        #     "test_graphs": cell_test_graphs,
        #     "params": {
        #         "hidden_channels": 128,
        #         "num_heads": 8,
        #         "num_layers": 4,
        #         "pooling_fn": global_add_pool,
        #     },
        #     "lr": 0.001,
        # },
        {
            "name": "DirectionalGNN_EdgeAttr_with_edge_mlp",
            "model_class": DirectionalGNN,
            "train_graphs": dir_train_graphs,
            "test_graphs": dir_test_graphs,
            "params": {
                "hidden_channels": 128,
                "num_layers": 4,
                "pooling_fn": global_add_pool,
            },
            "lr": 0.001,
        },
    ]

    for exp in gnn_experiments:
        train_dataset = Connect4GraphDataset(
            exp["train_graphs"], X_train, p_train, v_train
        )
        test_dataset = Connect4GraphDataset(exp["test_graphs"], X_test, p_test, v_test)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=pyg_collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=pyg_collate_fn,
        )

        wandb.init(
            project="connect4_arch_comparison",
            name=exp["name"],
            group=run_group_id,
            reinit=True,
            config={
                "learning_rate": exp["lr"],
                "batch_size": BATCH_SIZE,
                "architecture": exp["model_class"].__name__,
                **exp["params"],
            },
        )
        model = exp["model_class"](**exp["params"]).to(DEVICE)
        results_df, training_time = train_and_evaluate(
            model=model,
            model_name=exp["name"],
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=exp["lr"],
        )
        all_results[exp["name"]] = {"df": results_df, "time": training_time}
        wandb.finish()

    print("\n--- Final Results Summary ---")
    for name, results in all_results.items():
        print(f"\nModel: {name}")
        print(f"Training Time: {results['time']:.2f} seconds")
        print(results["df"].iloc[-1].to_string())


if __name__ == "__main__":
    main()
