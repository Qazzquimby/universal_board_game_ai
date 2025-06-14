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
from torch_geometric.nn import GATConv, HGTConv, global_add_pool, global_mean_pool

from environments.connect4 import Connect4

BOARD_HEIGHT = 6
BOARD_WIDTH = 7
MAX_EPOCHS = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "connect_4_states"
    / "pure_mcts"
    / "mcts_generated_states.json"
)


def load_and_process_data():
    print("Loading and processing data...")
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)

    inputs = []
    policy_labels = []
    value_labels = []

    env = Connect4(width=BOARD_WIDTH, height=BOARD_HEIGHT)

    for item in tqdm(raw_data):
        action_history_str = item.get("action_history", "")
        if not action_history_str:
            continue

        env.reset()
        actions = [int(a) for a in action_history_str]
        for action in actions:
            env.step(action)

        current_player_idx = env.get_current_player()  # 0 or 1
        opponent_player_idx = 1 - current_player_idx

        current_player_piece = current_player_idx + 1  # 1 or 2
        opponent_player_piece = opponent_player_idx + 1  # 2 or 1

        board_state = np.array(env.board)
        p1_board = (board_state == current_player_piece).astype(np.float32)
        p2_board = (board_state == opponent_player_piece).astype(np.float32)
        input_tensor = np.stack([p1_board, p2_board])
        inputs.append(input_tensor)

        policy_labels.append(item["next_action"])

        winner_piece = item["winner"]  # Assumes winner is 1 or 2
        value = 0.0
        if winner_piece is not None and winner_piece != 0:
            if winner_piece == current_player_piece:
                value = 1.0
            else:
                value = -1.0
        value_labels.append(value)

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
            self.raw_inputs[idx],
            self.policy_labels[idx],
            self.value_labels[idx],
        )


def pyg_collate_fn(batch):
    """Custom collate function to batch graph data and labels correctly."""
    graphs, raw_inputs, policies, values = zip(*batch)
    batched_graph = Batch.from_data_list(list(graphs))
    batched_raw_inputs = torch.stack(list(raw_inputs))
    batched_policies = torch.stack(list(policies))
    batched_values = torch.stack(list(values))
    return batched_graph, batched_raw_inputs, batched_policies, batched_values


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

    def forward(self, data, raw_boards=None):
        x_dict = {"cell": self.in_proj(data["cell"].x)}

        for layer in self.hgt_layers:
            x_dict = layer(x_dict, data.edge_index_dict)

        cell_features = x_dict["cell"]

        graph_embedding = self.pooling(cell_features, data["cell"].batch)

        x = F.relu(self.fc1(graph_embedding))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        return policy_logits, value


def construct_column_graph_data(board_tensor):
    p1_pieces = board_tensor[0].numpy()
    p2_pieces = board_tensor[1].numpy()

    data = HeteroData()

    # Cell nodes
    cell_features = []
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            is_p1 = p1_pieces[r, c] == 1
            is_p2 = p2_pieces[r, c] == 1
            is_empty = not is_p1 and not is_p2
            cell_features.append([float(is_empty), float(is_p1), float(is_p2)])
    data["cell"].x = torch.tensor(cell_features, dtype=torch.float)

    # Column nodes (with placeholder features)
    data["column"].x = torch.zeros((BOARD_WIDTH, 1))

    # Edges
    adj_sources, adj_dests = [], []
    cell_to_col_sources, cell_to_col_dests = [], []

    directions = {
        "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
        "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
    }
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            cell_idx = r * BOARD_WIDTH + c
            # Cell -> Column edges
            cell_to_col_sources.append(cell_idx)
            cell_to_col_dests.append(c)
            # Cell -> Cell adjacency edges
            for dr, dc in directions.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH:
                    neighbor_idx = nr * BOARD_WIDTH + nc
                    adj_sources.append(cell_idx)
                    adj_dests.append(neighbor_idx)

    data[("cell", "adjacent_to", "cell")].edge_index = torch.tensor([adj_sources, adj_dests], dtype=torch.long)
    data[("cell", "in_column", "column")].edge_index = torch.tensor([cell_to_col_sources, cell_to_col_dests], dtype=torch.long)
    data[("column", "rev_in_column", "cell")].edge_index = data[("cell", "in_column", "column")].edge_index.flip([0])

    return data


class ColumnGraphNet(nn.Module):
    def __init__(self, hidden_channels=64, num_heads=4, num_layers=2):
        super().__init__()
        node_types = ["cell", "column"]
        edge_types = [
            ("cell", "adjacent_to", "cell"),
            ("cell", "in_column", "column"),
            ("column", "rev_in_column", "cell"),
        ]
        self.metadata = (node_types, edge_types)

        self.cell_in_proj = nn.Linear(3, hidden_channels)
        self.col_in_proj = nn.Linear(1, hidden_channels)

        self.hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, self.metadata, num_heads)
            self.hgt_layers.append(conv)

        # Policy head operates on column node embeddings
        self.policy_fc = nn.Linear(hidden_channels, 64)
        self.policy_head = nn.Linear(64, 1)  # 1 logit per column

        # Value head operates on a global embedding of cell nodes
        self.value_fc = nn.Linear(hidden_channels, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, data, raw_boards=None):
        x_dict = {
            "cell": self.cell_in_proj(data["cell"].x),
            "column": self.col_in_proj(data["column"].x),
        }

        for layer in self.hgt_layers:
            x_dict = layer(x_dict, data.edge_index_dict)

        # Policy from column embeddings
        column_embeds = x_dict["column"]
        p = F.relu(self.policy_fc(column_embeds))
        policy_logits_flat = self.policy_head(p).squeeze(-1)
        policy_logits = policy_logits_flat.view(-1, BOARD_WIDTH)

        # Value from global cell embedding
        cell_embeds = x_dict["cell"]
        graph_embedding = global_add_pool(cell_embeds, data["cell"].batch)
        v = F.relu(self.value_fc(graph_embedding))
        value = torch.tanh(self.value_head(v))

        return policy_logits, value


class HybridGNN(nn.Module):
    def __init__(self, hidden_channels=64, num_heads=4, num_layers=2):
        super().__init__()
        node_types = ["cell", "column"]
        edge_types = [
            ("cell", "adjacent_to", "cell"),
            ("cell", "in_column", "column"),
            ("column", "rev_in_column", "cell"),
        ]
        self.metadata = (node_types, edge_types)
        self.hidden_channels = hidden_channels

        self.cell_in_proj = nn.Linear(3, hidden_channels)
        self.col_in_proj = nn.Linear(1, hidden_channels)

        self.hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, self.metadata, num_heads)
            self.hgt_layers.append(conv)

        mlp_input_size = 2 * BOARD_HEIGHT * BOARD_WIDTH

        # Policy head combines GNN column features and raw board state
        self.policy_fc = nn.Linear(hidden_channels + mlp_input_size, 64)
        self.policy_head = nn.Linear(64, 1)

        # Value head combines GNN global features and raw board state
        self.value_fc = nn.Linear(hidden_channels + mlp_input_size, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, data, raw_boards):
        x_dict = {
            "cell": self.cell_in_proj(data["cell"].x),
            "column": self.col_in_proj(data["column"].x),
        }

        for layer in self.hgt_layers:
            x_dict = layer(x_dict, data.edge_index_dict)

        flat_boards = raw_boards.view(raw_boards.size(0), -1)

        # Policy
        column_embeds = x_dict["column"]
        expanded_flat_boards = flat_boards.repeat_interleave(BOARD_WIDTH, dim=0)
        policy_input = torch.cat([column_embeds, expanded_flat_boards], dim=1)
        p = F.relu(self.policy_fc(policy_input))
        policy_logits_flat = self.policy_head(p).squeeze(-1)
        policy_logits = policy_logits_flat.view(-1, BOARD_WIDTH)

        # Value
        cell_embeds = x_dict["cell"]
        graph_embedding = global_add_pool(cell_embeds, data["cell"].batch)
        value_input = torch.cat([graph_embedding, flat_boards], dim=1)
        v = F.relu(self.value_fc(value_input))
        value = torch.tanh(self.value_head(v))

        return policy_logits, value


def train_and_evaluate(
    model, model_name, train_loader, test_loader, learning_rate=LEARNING_RATE
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
        train_batches = 0

        for data_batch in train_loader:
            if len(data_batch) == 4:  # GNN loader
                inputs, raw_boards, policy_labels, value_labels = data_batch
                inputs = inputs.to(DEVICE)
                raw_boards = raw_boards.to(DEVICE)
            else:  # Standard loader
                inputs, policy_labels, value_labels = data_batch
                raw_boards = None

            inputs = inputs.to(DEVICE)
            policy_labels = policy_labels.to(DEVICE)
            value_labels = value_labels.to(DEVICE)

            optimizer.zero_grad()

            if raw_boards is not None:
                policy_logits, value_preds = model(inputs, raw_boards)
            else:
                policy_logits, value_preds = model(inputs)

            loss_p = policy_criterion(policy_logits, policy_labels)
            loss_v = value_criterion(value_preds.squeeze(), value_labels)
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_policy_loss += loss_p.item()
            total_train_value_loss += loss_v.item()
            _, predicted_policies = torch.max(policy_logits, 1)
            total_train_policy_acc += (
                (predicted_policies == policy_labels).int().sum().item()
            )
            total_train_value_mse += F.mse_loss(
                value_preds.squeeze(), value_labels
            ).item()
            train_batches += 1

        model.eval()
        total_test_loss, total_test_policy_acc, total_test_value_mse = 0, 0, 0
        total_test_policy_loss, total_test_value_loss = 0, 0
        test_batches = 0
        with torch.no_grad():
            for data_batch in test_loader:
                if len(data_batch) == 4:  # GNN loader
                    inputs, raw_boards, policy_labels, value_labels = data_batch
                    inputs = inputs.to(DEVICE)
                    raw_boards = raw_boards.to(DEVICE)
                else:  # Standard loader
                    inputs, policy_labels, value_labels = data_batch
                    raw_boards = None

                inputs = inputs.to(DEVICE)
                policy_labels = policy_labels.to(DEVICE)
                value_labels = value_labels.to(DEVICE)

                if raw_boards is not None:
                    policy_logits, value_preds = model(inputs, raw_boards)
                else:
                    policy_logits, value_preds = model(inputs)
                loss_p = policy_criterion(policy_logits, policy_labels)
                loss_v = value_criterion(value_preds.squeeze(), value_labels)
                loss = loss_p + loss_v

                total_test_loss += loss.item()
                total_test_policy_loss += loss_p.item()
                total_test_value_loss += loss_v.item()
                _, predicted_policies = torch.max(policy_logits, 1)
                total_test_policy_acc += (
                    (predicted_policies == policy_labels).int().sum().item()
                )
                total_test_value_mse += F.mse_loss(
                    value_preds.squeeze(), value_labels
                ).item()
                test_batches += 1

        train_loss = total_train_loss / train_batches
        train_policy_loss = total_train_policy_loss / train_batches
        train_value_loss = total_train_value_loss / train_batches
        train_acc = total_train_policy_acc / len(train_loader.dataset)
        train_mse = total_train_value_mse / train_batches

        test_loss = total_test_loss / test_batches
        test_policy_loss = total_test_policy_loss / test_batches
        test_value_loss = total_test_value_loss / test_batches
        test_acc = total_test_policy_acc / len(test_loader.dataset)
        test_mse = total_test_value_mse / test_batches

        wandb.log(
            {
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
        )

        results.append(
            {
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
        )

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

    models_to_train = {"MLP": MLPNet(), "CNN": CNNNet(), "ResNet": ResNet()}

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

    # --- GNN Experiments ---
    print("\n--- Pre-processing data for GNNs ---")
    cell_train_graphs = [
        construct_graph_data(torch.from_numpy(board))
        for board in tqdm(X_train, desc="Processing Cell graphs")
    ]
    cell_test_graphs = [
        construct_graph_data(torch.from_numpy(board))
        for board in tqdm(X_test, desc="Processing Cell graphs")
    ]
    col_train_graphs = [
        construct_column_graph_data(torch.from_numpy(board))
        for board in tqdm(X_train, desc="Processing Column graphs")
    ]
    col_test_graphs = [
        construct_column_graph_data(torch.from_numpy(board))
        for board in tqdm(X_test, desc="Processing Column graphs")
    ]

    gnn_experiments = [
        {
            "name": "CellGNN_ValueFocused",
            "model_class": GraphNet,
            "train_graphs": cell_train_graphs,
            "test_graphs": cell_test_graphs,
            "params": {
                "hidden_channels": 128,
                "num_heads": 8,
                "num_layers": 4,
                "pooling_fn": global_add_pool,
            },
            "lr": 0.001,
        },
        {
            "name": "ColumnGNN_PolicyFocused",
            "model_class": ColumnGraphNet,
            "train_graphs": col_train_graphs,
            "test_graphs": col_test_graphs,
            "params": {"hidden_channels": 128, "num_heads": 4, "num_layers": 4},
            "lr": 0.001,
        },
        {
            "name": "HybridGNN_Baseline",
            "model_class": HybridGNN,
            "train_graphs": col_train_graphs,
            "test_graphs": col_test_graphs,
            "params": {"hidden_channels": 64, "num_heads": 4, "num_layers": 2},
            "lr": 0.001,
        },
        {
            "name": "HybridGNN_Complex",
            "model_class": HybridGNN,
            "train_graphs": col_train_graphs,
            "test_graphs": col_test_graphs,
            "params": {"hidden_channels": 128, "num_heads": 4, "num_layers": 4},
            "lr": 0.001,
        },
    ]

    for exp in gnn_experiments:
        train_dataset = Connect4GraphDataset(
            exp["train_graphs"], X_train, p_train, v_train
        )
        test_dataset = Connect4GraphDataset(exp["test_graphs"], X_test, p_test, v_test)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pyg_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pyg_collate_fn
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
            model, exp["name"], train_loader, test_loader, learning_rate=exp["lr"]
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
