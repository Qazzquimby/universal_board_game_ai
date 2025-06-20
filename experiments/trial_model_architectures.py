import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

import wandb
from environments.connect4 import Connect4
from experiments.architectures.basic import Connect4Dataset
from experiments.architectures.graph import (
    get_connect4_graph_with_edge_attrs,
    construct_graph_data,
    Connect4GraphDataset,
    pyg_collate_fn,
)
from experiments.architectures.shared import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MAX_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
    DATA_PATH,
)
from experiments.architectures.transformers import (
    create_transformer_input,
    create_cell_transformer_input,
    PieceTransformerNet_Sinusoidal_Learnable,
    PieceTransformerNet_ConcatPos,
    CellTransformerNet,
    Connect4TransformerDataset,
    Connect4CellTransformerDataset,
    _process_batch_transformer,
    _process_batch_cell_transformer,
    transformer_collate_fn,
    PieceTransformer_OnehotLoc,
    PieceTransformerNet,
    PieceTransformer_OnehotLoc_FixMask,
)

TINY_RUN = False


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


def _augment_and_append(
    input_tensor, policy_label, value, inputs, policy_labels, value_labels
):
    """Appends original and augmented data points to the lists."""
    # Original
    inputs.append(input_tensor)
    policy_labels.append(policy_label)
    value_labels.append(value)

    # Symmetrical (horizontal flip)
    sym_input = np.flip(input_tensor, axis=2).copy()
    sym_policy = (BOARD_WIDTH - 1) - policy_label
    inputs.append(sym_input)
    policy_labels.append(sym_policy)
    value_labels.append(value)

    # Player-swapped
    # Note: input_tensor[0] is current player, input_tensor[1] is opponent
    swapped_input = input_tensor[[1, 0], :, :].copy()
    swapped_value = -value
    inputs.append(swapped_input)
    policy_labels.append(policy_label)
    value_labels.append(swapped_value)

    # Symmetrical and player-swapped
    sym_swapped_input = np.flip(swapped_input, axis=2).copy()
    inputs.append(sym_swapped_input)
    policy_labels.append(sym_policy)
    value_labels.append(swapped_value)


def load_and_process_data():
    print("Loading and processing data...")
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)

    if TINY_RUN:
        raw_data = raw_data[:100]

    inputs = []
    policy_labels = []
    value_labels = []

    env = Connect4(width=BOARD_WIDTH, height=BOARD_HEIGHT)

    for item in tqdm(raw_data):
        processed_item = _process_raw_item(item, env)
        if processed_item:
            input_tensor, policy_label, value = processed_item
            _augment_and_append(
                input_tensor,
                policy_label,
                value,
                inputs,
                policy_labels,
                value_labels,
            )

    print(f"Data augmentation complete. Total samples: {len(inputs)}")
    return (
        np.array(inputs),
        np.array(policy_labels),
        np.array(value_labels, dtype=np.float32),
    )


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

        if not TINY_RUN:
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

    models_to_train = {
        # "MLP": MLPNet(),
        # "CNN": CNNNet(),
        # "ResNet": ResNet(),
    }

    all_results = {}
    for name, model in models_to_train.items():
        if not TINY_RUN:
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
        if not TINY_RUN:
            wandb.finish()

    # --- Transformer Experiment ---
    transformer_experiments = [
        # {
        #     "name": "PieceTransformer_v2",
        #     "model_class": PieceTransformerNet,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.001,
        # },
        # {
        #     "name": "PieceTransformer_Sinusoidal",
        #     "model_class": PieceTransformerNet_Sinusoidal,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.001,
        # },
        # {
        #     "name": "PieceTransformer_Sinusoidal_Learnable",
        #     "model_class": PieceTransformerNet_Sinusoidal_Learnable,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.001,
        # },
        # {
        #     "name": "PieceTransformer_ConcatPos",
        #     "model_class": PieceTransformerNet_ConcatPos,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #         "pos_embedding_dim": 4,
        #     },
        #     "lr": 0.001,
        # },
        ###
        # {
        #     "name": "PieceTransformer_onehotloc",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.001,
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_small",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 32,
        #         "num_heads": 2,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.001,
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_nodropout",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.0,
        #     },
        #     "lr": 0.001,
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_highlr",
        # Performance is terrible
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.01,
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_lowlr",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.0001,
        # },
        {
            "name": "PieceTransformer_onehotloc_highdropout",
            "model_class": PieceTransformer_OnehotLoc,
            "params": {
                "num_encoder_layers": 4,
                "embedding_dim": 128,
                "num_heads": 8,
                "dropout": 0.25,
            },
            "lr": 0.001,
        },
        {
            "name": "PieceTransformer_onehotloc_fixingmask",
            "model_class": PieceTransformer_OnehotLoc_FixMask,
            "params": {
                "num_encoder_layers": 4,
                "embedding_dim": 128,
                "num_heads": 8,
                "dropout": 0.1,
            },
            "lr": 0.001,
        },
    ]

    if transformer_experiments:
        print("\n--- Pre-processing data for Transformer ---")
        transformer_train_inputs = [
            create_transformer_input(torch.from_numpy(board))
            for board in tqdm(X_train, desc="Processing Train Transformer inputs")
        ]
        transformer_test_inputs = [
            create_transformer_input(torch.from_numpy(board))
            for board in tqdm(X_test, desc="Processing Test Transformer inputs")
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
    for exp in transformer_experiments:
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
            train_loader=transformer_train_loader,
            test_loader=transformer_test_loader,
            learning_rate=exp["lr"],
            process_batch_fn=_process_batch_transformer,
        )
        all_results[exp["name"]] = {"df": results_df, "time": training_time}
        wandb.finish()

    # --- Cell Transformer Experiment ---
    cell_transformer_experiments = [
        # {
        #     "name": "CellTransformer",
        #     "model_class": CellTransformerNet,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 128,
        #         "num_heads": 8,
        #         "dropout": 0.1,
        #     },
        #     "lr": 0.001,
        # },
    ]
    if cell_transformer_experiments:
        print("\n--- Pre-processing data for Cell Transformer ---")
        cell_train_inputs = [
            create_cell_transformer_input(torch.from_numpy(board))
            for board in tqdm(X_train, desc="Processing Cell Transformer inputs")
        ]
        cell_test_inputs = [
            create_cell_transformer_input(torch.from_numpy(board))
            for board in tqdm(X_test, desc="Processing Cell Transformer inputs")
        ]

        cell_train_dataset = Connect4CellTransformerDataset(
            cell_train_inputs, p_train, v_train
        )
        cell_test_dataset = Connect4CellTransformerDataset(
            cell_test_inputs, p_test, v_test
        )

        cell_train_loader = DataLoader(
            cell_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        cell_test_loader = DataLoader(
            cell_test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    for exp in cell_transformer_experiments:
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
            train_loader=cell_train_loader,
            test_loader=cell_test_loader,
            learning_rate=exp["lr"],
            process_batch_fn=_process_batch_cell_transformer,
        )
        all_results[exp["name"]] = {"df": results_df, "time": training_time}
        wandb.finish()

    # --- GNN Experiments ---
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
        # {  # very weak, no attention
        #     "name": "DirectionalGNN_EdgeAttr",
        #     "model_class": DirectionalGNN,
        #     "train_graphs": dir_train_graphs,
        #     "test_graphs": dir_test_graphs,
        #     "params": {
        #         "hidden_channels": 128,
        #         "num_layers": 4,
        #         "pooling_fn": global_add_pool,
        #     },
        #     "lr": 0.001,
        # },
    ]
    if gnn_experiments:
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
