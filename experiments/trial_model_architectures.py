import json
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

from environments.connect4 import Connect4

BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 256
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
        self.inputs = torch.from_numpy(inputs).to(DEVICE)
        self.policy_labels = torch.from_numpy(policy_labels).long().to(DEVICE)
        self.value_labels = torch.from_numpy(value_labels).to(DEVICE)

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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


def train_and_evaluate(model, model_name, train_loader, test_loader):
    print(f"\n--- Training {model_name} on {DEVICE} ---")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    results = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss, total_train_policy_acc, total_train_value_mse = 0, 0, 0
        train_batches = 0

        for inputs, policy_labels, value_labels in train_loader:
            optimizer.zero_grad()
            policy_logits, value_preds = model(inputs)

            loss_p = policy_criterion(policy_logits, policy_labels)
            loss_v = value_criterion(value_preds.squeeze(), value_labels)
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
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
        test_batches = 0
        with torch.no_grad():
            for inputs, policy_labels, value_labels in test_loader:
                policy_logits, value_preds = model(inputs)
                loss_p = policy_criterion(policy_logits, policy_labels)
                loss_v = value_criterion(value_preds.squeeze(), value_labels)
                loss = loss_p + loss_v

                total_test_loss += loss.item()
                _, predicted_policies = torch.max(policy_logits, 1)
                total_test_policy_acc += (
                    (predicted_policies == policy_labels).int().sum().item()
                )
                total_test_value_mse += F.mse_loss(
                    value_preds.squeeze(), value_labels
                ).item()
                test_batches += 1

        train_loss = total_train_loss / train_batches
        train_acc = total_train_policy_acc / len(train_loader.dataset)
        train_mse = total_train_value_mse / train_batches

        test_loss = total_test_loss / test_batches
        test_acc = total_test_policy_acc / len(test_loader.dataset)
        test_mse = total_test_value_mse / test_batches

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train MSE: {train_mse:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test MSE: {test_mse:.4f}"
        )

        results.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_mse": train_mse,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_mse": test_mse,
            }
        )

    return pd.DataFrame(results)


def main():
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
        model.to(DEVICE)
        results_df = train_and_evaluate(model, name, train_loader, test_loader)
        all_results[name] = results_df

    print("\n--- Final Results Summary ---")
    for name, df in all_results.items():
        print(f"\nModel: {name}")
        print(df.iloc[-1].to_string())


if __name__ == "__main__":
    main()
