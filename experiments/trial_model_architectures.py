import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

import wandb
from experiments.architectures.detached import DetachedPolicyNet
from experiments.architectures.graph_transformers import (
    CellGraphTransformer,
    CellColumnGraphTransformer,
    graph_collate_fn,
    create_cell_graph,
    create_cell_column_graph,
    create_cell_piece_graph,
    create_combined_graph,
    CellColumnPieceGraphTransformer,
    create_cell_column_piece_graph,
    create_piece_column_graph,
    PieceColumnGraphTransformer,
)
from experiments.data_utils import load_and_process_data
from experiments.architectures.basic import AZDataset, AZGraphDataset, CNNNet, MLPNet
from experiments.architectures.shared import (
    LEARNING_RATE,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
)
from experiments.architectures.transformers import (
    create_transformer_input,
    _process_batch_transformer,
    transformer_collate_fn,
    PieceTransformer_EncoderSum_SimpleOut_ParamGameToken,
    PieceTransformerNet,
)

TINY_RUN = False
MAX_TRAINING_TIME_SECONDS = 2 * 3600  # 2h


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
    if process_batch_fn is None:
        process_batch_fn = _process_batch
    print(f"\n--- Training {model_name} on {DEVICE} ---")
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    results = []
    best_test_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    while time.time() - start_time < MAX_TRAINING_TIME_SECONDS:
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

        elapsed_time = time.time() - start_time
        log_info = {
            "epoch": epoch + 1,
            "elapsed_time": elapsed_time,
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

        epoch += 1

    if time.time() - start_time >= MAX_TRAINING_TIME_SECONDS:
        print(f"Max training time of {MAX_TRAINING_TIME_SECONDS} seconds reached.")

    training_time = time.time() - start_time
    final_epoch = len(results)
    if not results:
        print("No epochs were completed.")
        return pd.DataFrame(), training_time
    final_results = results[-1]
    print(f"Finished training after {final_epoch} epochs.")
    print(
        f"Final Results | Test Loss: {final_results['test_loss']:.4f}, Test Acc: {final_results['test_acc']:.4f}, Test MSE: {final_results['test_mse']:.4f}"
    )
    return pd.DataFrame(results), training_time


def _run_and_log_experiment(
    exp,
    model,
    run_group_id,
    all_results,
    train_loader,
    test_loader,
    process_batch_fn=_process_batch,
    data_processing_time=0.0,
):
    name = exp["name"]
    params = exp.get("params", {})
    lr = exp.get("lr", LEARNING_RATE)

    if not TINY_RUN:
        wandb.init(
            project="connect4_arch_comparison",
            name=name,
            group=run_group_id,
            reinit=True,
            config={
                "learning_rate": lr,
                "batch_size": BATCH_SIZE,
                "architecture": model.__class__.__name__,
                **params,
            },
        )

    results_df, training_time = train_and_evaluate(
        model=model,
        model_name=name,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=lr,
        process_batch_fn=process_batch_fn,
    )
    all_results[name] = {
        "df": results_df,
        "training_time": training_time,
        "data_processing_time": data_processing_time,
    }
    if not TINY_RUN:
        wandb.log(
            {
                "training_time": training_time,
                "data_processing_time": data_processing_time,
                "train_data_ratio": training_time / data_processing_time
                if data_processing_time > 0
                else 0,
            }
        )
        wandb.finish()


@dataclass
class TestData:
    X_train: Tensor
    X_test: Tensor
    policy_train: Tensor
    policy_test: Tensor
    value_train: Tensor
    value_test: Tensor


RUN_GROUP_ID = f"run_{int(time.time())}"


def get_dataloaders(
    data: TestData,
    name: str,
    input_creator: callable = None,
    collate_function: callable = None,
    dataset_class: type = AZDataset,
) -> tuple[DataLoader, DataLoader]:
    if input_creator:
        train_inputs = [
            input_creator(torch.from_numpy(board))
            for board in tqdm(data.X_train, desc=f"Processing {name} Train inputs")
        ]
        test_inputs = [
            input_creator(torch.from_numpy(board))
            for board in tqdm(data.X_test, desc=f"Processing {name} Test inputs")
        ]
    else:
        train_inputs = data.X_train
        test_inputs = data.X_test

    train_dataset = AZGraphDataset(train_inputs, data.policy_train, data.value_train)
    test_dataset = AZGraphDataset(test_inputs, data.policy_test, data.value_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_function
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_function
    )
    return train_loader, test_loader


def run_experiments(
    all_results: dict,
    data: TestData,
    name: str,
    experiments: list[dict],
    process_batch_fn: callable,
    input_creator: callable = None,
    collate_function: callable = None,
):
    """
    Run experiments for a specific architecture defined by `name` and `input_creator`.
    """
    if not experiments:
        print("No transformer experiments defined.")
        return

    print(f"\n--- Pre-processing data for {name} ---")
    get_loader_per_experiment = isinstance(input_creator, str)

    data_processing_time = 0.0
    if not get_loader_per_experiment:
        start_time = time.time()
        train_loader, test_loader = get_dataloaders(
            data=data,
            name=name,
            input_creator=input_creator,
            collate_function=collate_function,
        )
        data_processing_time = time.time() - start_time

    for experiment in experiments:
        params = experiment.get("params", {})

        current_data_proc_time = data_processing_time
        if get_loader_per_experiment:
            start_time = time.time()
            input_creator_func = experiment[input_creator]
            train_loader, test_loader = get_dataloaders(
                data=data,
                name=name,
                input_creator=input_creator_func,
                collate_function=collate_function,
                dataset_class=AZGraphDataset,
            )
            current_data_proc_time = time.time() - start_time

        assert train_loader and test_loader

        model = experiment["model_class"](**params).to(DEVICE)
        _run_and_log_experiment(
            exp=experiment,
            model=model,
            run_group_id=RUN_GROUP_ID,
            all_results=all_results,
            train_loader=train_loader,
            test_loader=test_loader,
            process_batch_fn=process_batch_fn,
            data_processing_time=current_data_proc_time,
        )


def run_basic_experiments(all_results: dict, data: TestData):
    experiments = [
        # {"name": "MLP", "model_class": MLPNet, "params": {}},
        # {"name": "CNN", "model_class": CNNNet, "params": {}},
        # {"name": "ResNet", "model_class": ResNet, "params": {}},
    ]
    run_experiments(
        all_results=all_results,
        data=data,
        name="Basic",
        input_creator=None,
        experiments=experiments,
        process_batch_fn=_process_batch,
    )


def run_piece_transformer_experiments(all_results: dict, data: TestData):
    experiments = [
        {
            "name": "DetachedPolicy_v1",
            "model_class": DetachedPolicyNet,
            "params": {
                "state_model_params": {
                    "embedding_dim": 128,
                    "num_heads": 8,
                    "num_encoder_layers": 4,
                    "dropout": 0.1,
                },
                "policy_model_params": {
                    "embedding_dim": 128,
                },
            },
        },
        {
            "name": "PieceTransformer_v2",
            "model_class": PieceTransformerNet,
        },
        # {
        #     "name": "PieceTransformer_Sinusoidal",
        #     "model_class": PieceTransformerNet_Sinusoidal,
        # },
        # {
        #     "name": "PieceTransformer_Sinusoidal_Learnable",
        #     "model_class": PieceTransformerNet_Sinusoidal_Learnable,
        # },
        # {
        #     "name": "PieceTransformer_ConcatPos",
        #     "model_class": PieceTransformerNet_ConcatPos,
        #     "params": {
        #         "pos_embedding_dim": 4,
        #     },
        # },
        ###
        # {
        #     "name": "PieceTransformer_onehotloc",
        #     "model_class": PieceTransformer_OnehotLoc,
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_small",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "num_encoder_layers": 4,
        #         "embedding_dim": 32,
        #         "num_heads": 2,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_nodropout",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "dropout": 0.0,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_highlr",
        # Performance is terrible
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "lr": 0.01,
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_lowlr",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "lr": 0.0001,
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_highdropout",
        #     "model_class": PieceTransformer_OnehotLoc,
        #     "params": {
        #         "dropout": 0.25,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_onehotloc_simpleout",
        #     "model_class": PieceTransformer_OnehotLoc_SimpleOut,
        # },
        # {
        #     "name": "PieceTransformer_encodersum_simpleout",
        #     "model_class": PieceTransformer_OnehotLoc_SimpleOut,
        # },
        # {
        #     "name": "PieceTransformer_OnehotLoc_BottleneckOut",
        #     "model_class": PieceTransformer_OnehotLoc_BottleneckOut,
        # },
        # {
        #     "name": "PieceTransformer_v2_2layers",
        #     "model_class": PieceTransformerNet,
        #     "params": {
        #         "num_encoder_layers": 2,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_v2_6layers",
        #     "model_class": PieceTransformerNet,
        #     "params": {
        #         "num_encoder_layers": 6,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_v2_64dim",
        #     "model_class": PieceTransformerNet,
        #     "params": {
        #         "embedding_dim": 64,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_v2_256dim",
        #     "model_class": PieceTransformerNet,
        #     "params": {
        #         "embedding_dim": 256,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_v2_4heads",
        #     "model_class": PieceTransformerNet,
        #     "params": {
        #         "num_heads": 4,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_v16heads",
        #     "model_class": PieceTransformerNet,
        #     "params": {
        #         "num_heads": 16,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_v2_0.002lr",
        #     "model_class": PieceTransformerNet,
        #     "lr": 0.002,
        # },
        # {
        #     "name": "PieceTransformer_v2_0.0005lr",
        #     "model_class": PieceTransformerNet,
        #     "lr": 0.0005,
        # },
        # {
        #     "name": "PieceTransformer_EncoderSum_SimpleOut",
        #     "model_class": PieceTransformer_EncoderSum_SimpleOut,
        # },
        # {
        #     "name": "PieceTransformer_EncoderSum_SimpleOut_ParamGameToken",
        #     "model_class": PieceTransformer_EncoderSum_SimpleOut_ParamGameToken,
        # },
        # {
        #     "name": "PieceTransformer_EncoderSum_SimpleOut_ParamGameToken_Small",
        #     "model_class": PieceTransformer_EncoderSum_SimpleOut_ParamGameToken,
        #     "params": {
        #         "embedding_dim": 32,
        #         "num_heads": 2,
        #     },
        # },
        # {
        #     "name": "PieceTransformer_EncoderSum_SimpleOut_ParamGameToken_Small_Highlr",
        #     "model_class": PieceTransformer_EncoderSum_SimpleOut_ParamGameToken,
        #     "params": {
        #         "embedding_dim": 32,
        #         "num_heads": 2,
        #     },
        #     "lr": 0.005,
        # },
    ]

    run_experiments(
        all_results=all_results,
        data=data,
        name="Piece Transformer",
        input_creator=create_transformer_input,
        experiments=experiments,
        collate_function=transformer_collate_fn,
        process_batch_fn=_process_batch_transformer,
    )


def run_graph_transformer_experiments(all_results: dict, data: TestData):
    experiments = [
        # {
        #     "name": "CellGraphTransformer",
        #     "model_class": CellGraphTransformer,
        #     "input_creator": create_cell_graph,
        # },
        {  # very strong
            "name": "CellColumnGraphTransformer",
            "model_class": CellColumnGraphTransformer,
            "input_creator": create_cell_column_graph,
        },
        # {
        #     "name": "CellColumnPieceGraphTransformer",
        #     "model_class": CellColumnPieceGraphTransformer,
        #     "input_creator": create_cell_column_piece_graph,
        # },
        # {
        #     "name": "PieceColumnGraphTransformer",
        #     "model_class": PieceColumnGraphTransformer,
        #     "input_creator": create_piece_column_graph,
        # },
    ]

    run_experiments(
        all_results=all_results,
        data=data,
        name="Graph Transformer",
        experiments=experiments,
        process_batch_fn=None,  # ?
        input_creator="input_creator",
        collate_function=graph_collate_fn,
    )


def main():
    inputs, policy_labels, value_labels = load_and_process_data(TINY_RUN)
    _X_train, _X_test, _p_train, _p_test, _v_train, _v_test = train_test_split(
        inputs, policy_labels, value_labels, test_size=0.2, random_state=42
    )
    data = TestData(
        X_train=_X_train,
        X_test=_X_test,
        policy_train=_p_train,
        policy_test=_p_test,
        value_train=_v_train,
        value_test=_v_test,
    )
    all_results = {}
    run_basic_experiments(all_results=all_results, data=data)

    run_piece_transformer_experiments(all_results=all_results, data=data)

    run_graph_transformer_experiments(all_results=all_results, data=data)

    print("\n--- Final Results Summary ---")
    for name, results in all_results.items():
        print(f"\nModel: {name}")
        training_time = results["training_time"]
        data_processing_time = results["data_processing_time"]
        ratio = (
            training_time / data_processing_time
            if data_processing_time > 0
            else float("inf")
        )
        print(f"Data Processing Time: {data_processing_time:.2f} seconds")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Train/Data Ratio: {ratio:.2f}")
        print(results["df"].iloc[-1].to_string())


if __name__ == "__main__":
    main()
