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
import optuna

from core.config import TRAINING_DEVICE
from agents.alphazero.alphazero_net import AlphaZeroNet
from agents.base_learning_agent import (
    _get_batched_state,
)
from environments.base import DataFrame
from environments.connect4 import Connect4
from experiments.architectures.graph_transformers import (
    graph_collate_fn,
    CellColumnGraphTransformer,
    create_cell_column_graph,
)
from experiments.data_utils import load_and_process_data
from experiments.architectures.basic import (
    AZDataset,
    AZIrregularInputsDataset,
    MLPNet,
    CNNNet,
    ResNet,
)
from experiments.architectures.shared import (
    LEARNING_RATE,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
)
from experiments.architectures.transformers import (
    create_transformer_input,
    _process_batch_transformer,
    transformer_collate_fn,
    PieceTransformerNet,
)


@dataclass
class AZTrialCollation:
    batched_state: dict
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    legal_actions_batch: tuple


TINY_RUN = False
USE_OPTUNA = False
OPTUNA_N_TRIALS = 20
# MAX_TRAINING_TIME_SECONDS = 2 * 3600  # 2h
MAX_TRAINING_TIME_SECONDS = 1 * 3600

if TINY_RUN:
    MAX_TRAINING_TIME_SECONDS = 20
    OPTUNA_N_TRIALS = 3


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


def _process_batch_az(model, data_batch, device, policy_criterion, value_criterion):
    # data_batch is an AZTrialCollation object
    batch_size = len(data_batch.legal_actions_batch)
    state_tokens, state_padding_mask = model.tokenize_state_batch(
        data_batch.batched_state, batch_size=batch_size
    )
    state_tokens = state_tokens.to(device)
    state_padding_mask = state_padding_mask.to(device)

    # Tokenize actions
    flat_legal_actions = []
    batch_indices_for_policy = []
    for i, actions in enumerate(data_batch.legal_actions_batch):
        if actions:
            flat_legal_actions.extend(actions)
            batch_indices_for_policy.extend([i] * len(actions))

    action_tokens = model.tokenize_actions(flat_legal_actions).to(device)
    action_batch_indices = torch.tensor(
        batch_indices_for_policy, dtype=torch.long, device=device
    )

    policy_targets = data_batch.policy_targets.to(device)
    value_labels = data_batch.value_targets.to(device)

    policy_logits, value_preds = model(
        state_tokens=state_tokens,
        state_padding_mask=state_padding_mask,
        action_tokens=action_tokens,
        action_batch_indices=action_batch_indices,
    )

    # Assuming policy_targets are distributions, convert to indices for CrossEntropyLoss
    policy_target_indices = torch.argmax(policy_targets, dim=1)

    loss_p = policy_criterion(policy_logits, policy_target_indices)
    loss_v = value_criterion(value_preds.squeeze(), value_labels.squeeze(-1))
    loss = loss_p + loss_v

    _, predicted_policies = torch.max(policy_logits, 1)
    policy_acc = (predicted_policies == policy_target_indices).int().sum().item()
    value_mse = F.mse_loss(value_preds.squeeze(), value_labels.squeeze(-1)).item()

    return loss, loss_p.item(), loss_v.item(), policy_acc, value_mse


def train_and_evaluate(
    model,
    model_name,
    train_loader,
    test_loader,
    learning_rate=LEARNING_RATE,
    weight_decay=0.0,
    process_batch_fn=_process_batch,
    trial: "optuna.Trial" = None,
):
    if process_batch_fn is None:
        process_batch_fn = _process_batch
    print(f"\n--- Training {model_name} on {TRAINING_DEVICE} ---")
    start_time = time.time()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    results = []
    best_test_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    pbar = tqdm(train_loader, desc=f"{model_name}")
    while time.time() - start_time < MAX_TRAINING_TIME_SECONDS:
        model.train()
        total_train_loss, total_train_policy_acc, total_train_value_mse = 0, 0, 0
        total_train_policy_loss, total_train_value_loss = 0, 0

        for data_batch in train_loader:
            optimizer.zero_grad()
            loss, loss_p_item, loss_v_item, policy_acc, value_mse = process_batch_fn(
                model, data_batch, TRAINING_DEVICE, policy_criterion, value_criterion
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
                    model,
                    data_batch,
                    TRAINING_DEVICE,
                    policy_criterion,
                    value_criterion,
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

        if not TINY_RUN and not trial:
            wandb.log(log_info)
        results.append(log_info)

        if trial:
            trial.report(test_loss, epoch)
            if trial.should_prune():
                pbar.close()
                raise optuna.exceptions.TrialPruned()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        epoch += 1
        pbar.update(1)

    if time.time() - start_time >= MAX_TRAINING_TIME_SECONDS:
        print(f"Max training time of {MAX_TRAINING_TIME_SECONDS} seconds reached.")

    training_time = time.time() - start_time
    final_epoch = len(results)
    if not results:
        print("No epochs were completed.")
        return pd.DataFrame(), training_time, float("inf")
    final_results = results[-1]
    print(f"Finished training after {final_epoch} epochs.")
    print(
        f"Final Results | Test Loss: {final_results['test_loss']:.4f}, Test Acc: {final_results['test_acc']:.4f}, Test MSE: {final_results['test_mse']:.4f}"
    )
    return pd.DataFrame(results), training_time, final_results["test_loss"]


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
    weight_decay = exp.get("weight_decay", 0.0)

    if not TINY_RUN:
        wandb.init(
            project="connect4_arch_comparison",
            name=name,
            group=run_group_id,
            reinit=True,
            config={
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "batch_size": BATCH_SIZE,
                "architecture": model.__class__.__name__,
                **params,
            },
        )

    results_df, training_time, _ = train_and_evaluate(
        model=model,
        model_name=name,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=lr,
        weight_decay=weight_decay,
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

    train_dataset = dataset_class(train_inputs, data.policy_train, data.value_train)
    test_dataset = dataset_class(test_inputs, data.policy_test, data.value_test)

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
    dataset_class: type = AZDataset,
    collate_function: callable = None,
):
    """
    Run experiments for a specific architecture defined by `name` and `input_creator`.
    """
    if not experiments:
        print(f"No {name} experiments defined.")
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
            dataset_class=dataset_class,
        )
        data_processing_time = time.time() - start_time

    for experiment in experiments:
        current_data_proc_time = data_processing_time
        if get_loader_per_experiment:
            start_time = time.time()
            input_creator_func = experiment[input_creator]
            train_loader, test_loader = get_dataloaders(
                data=data,
                name=name,
                input_creator=input_creator_func,
                collate_function=collate_function,
                dataset_class=AZIrregularInputsDataset,
            )
            current_data_proc_time = time.time() - start_time
        assert train_loader and test_loader

        if USE_OPTUNA and "param_space" in experiment:
            print(f"\n--- Running Optuna search for {experiment['name']} ---")

            def objective(trial):
                param_generator = experiment["param_space"]
                hyperparams = param_generator(trial)

                lr = hyperparams.pop("learning_rate", LEARNING_RATE)
                weight_decay = hyperparams.pop("weight_decay", 0.0)
                model_params = {**experiment.get("params", {}), **hyperparams}

                model = experiment["model_class"](**model_params).to(TRAINING_DEVICE)
                _, _, loss = train_and_evaluate(
                    model=model,
                    model_name=f"{experiment['name']}_trial_{trial.number}",
                    train_loader=train_loader,
                    test_loader=test_loader,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    process_batch_fn=process_batch_fn,
                    trial=trial,
                )
                return loss

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

            print(f"Best trial for {experiment['name']}:")
            print(f"  Value: {study.best_value}")
            print(f"  Params: {study.best_params}")

            best_params = study.best_params
            experiment["lr"] = best_params.pop("learning_rate", LEARNING_RATE)
            experiment["weight_decay"] = best_params.pop("weight_decay", 0.0)
            experiment.setdefault("params", {}).update(best_params)

        params = experiment.get("params", {})
        model = experiment["model_class"](**params).to(TRAINING_DEVICE)
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
        # {
        #     "name": "ResNet",
        #     "model_class": ResNet,
        #     "params": {},
        #     "param_space": lambda trial: {
        #         "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-2),
        #         "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        #     },
        # },
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
    def piece_transformer_param_space(trial: optuna.Trial):
        embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        if embedding_dim % num_heads != 0:
            raise optuna.exceptions.TrialPruned()
        return {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
            "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        }

    experiments = [
        # {
        #     "name": "DetachedPolicy_v1",
        #     "model_class": DetachedPolicyNet,
        #     "params": {
        #         "state_model_params": {
        #             "embedding_dim": 128,
        #             "num_heads": 8,
        #             "num_encoder_layers": 4,
        #             "dropout": 0.1,
        #         },
        #         "policy_model_params": {
        #             "embedding_dim": 128,
        #         },
        #     },
        # },
        ##
        # {
        #     "name": "PieceTransformer_v2",
        #     "model_class": PieceTransformerNet,
        #     "param_space": piece_transformer_param_space,
        # },
        ##
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
        dataset_class=AZIrregularInputsDataset,
        experiments=experiments,
        collate_function=transformer_collate_fn,
        process_batch_fn=_process_batch_transformer,
    )


c4_env_for_conversion = Connect4()


def create_az_input(board_tensor: torch.Tensor):
    pieces_data = []
    # board_tensor[0] is for the current player, which we'll set as player 0
    p1_pieces = torch.nonzero(board_tensor[0])
    for r, c in p1_pieces:
        pieces_data.append([r.item(), c.item(), 0])

    # board_tensor[1] is for the opponent, player 1
    p2_pieces = torch.nonzero(board_tensor[1])
    for r, c in p2_pieces:
        pieces_data.append([r.item(), c.item(), 1])

    pieces_df = DataFrame(data=pieces_data, columns=["row", "col", "player_id"])

    winner = Connect4.check_for_winner_from_pieces(
        pieces_df, width=Connect4.width, height=Connect4.height
    )
    is_draw = (winner is None) and (
        pieces_df.height == Connect4.width * Connect4.height
    )
    done = (winner is not None) or is_draw

    game_df = DataFrame(
        data=[[0, done, winner]],  # current_player=0
        columns=["current_player", "done", "winner"],
    )

    state_to_set = {"pieces": pieces_df, "game": game_df}
    c4_env_for_conversion.set_state(state_to_set)

    state_dict = c4_env_for_conversion._get_state()
    legal_actions = c4_env_for_conversion.get_legal_actions()
    return (state_dict, legal_actions)


def alphazero_collate_fn(batch):
    inputs, policy_targets, value_targets = zip(*batch)
    state_dicts, legal_actions_batch = zip(*inputs)

    batched_state = _get_batched_state(state_dicts=state_dicts)

    policy_dist_targets = []
    for i, legal_actions in enumerate(legal_actions_batch):
        policy_target_action = policy_targets[i].item()
        policy_dist = torch.zeros(len(legal_actions), dtype=torch.float32)
        if legal_actions and policy_target_action in legal_actions:
            action_idx = legal_actions.index(policy_target_action)
            policy_dist[action_idx] = 1.0
        policy_dist_targets.append(policy_dist)

    padded_policy_targets = nn.utils.rnn.pad_sequence(
        policy_dist_targets, batch_first=True, padding_value=0.0
    )
    value_targets = torch.stack(list(value_targets), 0)

    return AZTrialCollation(
        batched_state=batched_state,
        policy_targets=padded_policy_targets,
        value_targets=value_targets,
        legal_actions_batch=legal_actions_batch,
    )


def run_alphazero_experiments(all_results: dict, data: TestData):
    c4_env = Connect4()

    def alphazero_param_space(trial: optuna.Trial):
        embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        if embedding_dim % num_heads != 0:
            raise optuna.exceptions.TrialPruned()
        return {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
            "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        }

    experiments = [
        {
            "name": "AlphaZeroNet_v1",
            "model_class": AlphaZeroNet,
            "params": {
                "env": c4_env,
            },
            "param_space": alphazero_param_space,
        },
    ]

    if experiments:
        run_experiments(
            all_results=all_results,
            data=data,
            name="AlphaZero",
            input_creator=create_az_input,
            dataset_class=AZIrregularInputsDataset,
            experiments=experiments,
            collate_function=alphazero_collate_fn,
            process_batch_fn=_process_batch_az,
        )


def run_graph_transformer_experiments(all_results: dict, data: TestData):
    experiments = [
        # {
        #     "name": "Auto_V1",
        #     "model_class": AutoGraphNet,
        #     "input_creator": make_state_with_key,
        # }
        # {
        #     "name": "CellGraphTransformer",
        #     "model_class": CellGraphTransformer,
        #     "input_creator": create_cell_graph,
        # },
        # {  # very strong
        #     "name": "CellColumnGraphTransformer",
        #     "model_class": CellColumnGraphTransformer,
        #     "input_creator": create_cell_column_graph,
        # },
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


def create_actions_are_tokens_input(board_tensor):
    h, w = board_tensor.shape[1], board_tensor.shape[2]
    p1_pieces = torch.nonzero(board_tensor[0])
    p2_pieces = torch.nonzero(board_tensor[1])
    num_p1 = p1_pieces.shape[0]
    num_p2 = p2_pieces.shape[0]

    if num_p1 + num_p2 == 0:
        coords = torch.empty(0, 2, dtype=torch.long)
        owner = torch.empty(0, 2, dtype=torch.float)
    else:
        coords = torch.cat([p1_pieces, p2_pieces], dim=0)
        owner_p1 = torch.tensor([[1.0, 0.0]]).repeat(num_p1, 1)
        owner_p2 = torch.tensor([[0.0, 1.0]]).repeat(num_p2, 1)
        owner = torch.cat([owner_p1, owner_p2], dim=0)

    legal_moves = [c for c in range(w) if board_tensor[:, 0, c].sum() == 0]
    return owner, coords, torch.tensor(legal_moves, dtype=torch.long)


def legal_decoder_collate_fn(batch):
    inputs, policy_labels, value_labels = zip(*batch)
    owners, coords, legal_moves_list = zip(*inputs)

    padded_owners = nn.utils.rnn.pad_sequence(owners, batch_first=True, padding_value=0)
    padded_coords = nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=0)

    lengths = torch.tensor([len(c) for c in coords])
    src_key_padding_mask = (
        torch.arange(padded_owners.size(1))[None, :] >= lengths[:, None]
    )

    padded_legal_moves = nn.utils.rnn.pad_sequence(
        [lm for lm in legal_moves_list], batch_first=True, padding_value=-1
    )
    legal_moves_mask = padded_legal_moves == -1
    padded_legal_moves[legal_moves_mask] = 0

    batched_policies = torch.stack(list(policy_labels))
    batched_values = torch.stack(list(value_labels))

    return (
        padded_owners,
        padded_coords,
        src_key_padding_mask,
        padded_legal_moves,
        legal_moves_mask,
        batched_policies,
        batched_values,
    )


def _process_batch_legal_decoder(
    model, data_batch, device, policy_criterion, value_criterion
):
    (
        owners,
        coords,
        src_key_padding_mask,
        legal_moves,
        legal_moves_mask,
        policy_labels,
        value_labels,
    ) = data_batch

    owners = owners.to(device)
    coords = coords.to(device)
    src_key_padding_mask = src_key_padding_mask.to(device)
    legal_moves = legal_moves.to(device)
    legal_moves_mask = legal_moves_mask.to(device)
    policy_labels = policy_labels.to(device)
    value_labels = value_labels.to(device)

    policy_logits, value_preds = model(
        owners, coords, src_key_padding_mask, legal_moves, legal_moves_mask
    )

    loss_p = policy_criterion(policy_logits, policy_labels)
    loss_v = value_criterion(value_preds.squeeze(), value_labels)
    loss = loss_p + loss_v

    _, predicted_policies = torch.max(policy_logits, 1)
    policy_acc = (predicted_policies == policy_labels).int().sum().item()
    value_mse = F.mse_loss(value_preds.squeeze(), value_labels).item()

    return loss, loss_p.item(), loss_v.item(), policy_acc, value_mse


def run_legal_action_decoder_experiments(all_results: dict, data: TestData):
    experiments = [
        # {
        #     "name": "ActionsAreTokensNet_v1",
        #     "model_class": ActionsAreTokensNet,
        #     "params": {
        #         "state_model_params": {
        #             "embedding_dim": 128,
        #             "num_heads": 8,
        #             "num_encoder_layers": 4,
        #             "dropout": 0.1,
        #         },
        #     },
        # },
    ]

    run_experiments(
        all_results=all_results,
        data=data,
        name="Legal Action Decoder",
        input_creator=create_actions_are_tokens_input,
        dataset_class=AZIrregularInputsDataset,
        experiments=experiments,
        collate_function=legal_decoder_collate_fn,
        process_batch_fn=_process_batch_legal_decoder,
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

    run_alphazero_experiments(all_results=all_results, data=data)

    run_graph_transformer_experiments(all_results=all_results, data=data)

    run_legal_action_decoder_experiments(all_results=all_results, data=data)

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


# todo try legal moves being a type of token (basically the same as column tokens but would link by column embedding rather than graph I think)
# TODO dear future me
# networks.py is pretty disgusting and needs to be rewritten carefully.
# The forward method takes a state_with_key but doesnt even use it.
# the main way to test a model is using this file, but the data intake is totally different and neither is simple
# good luck friend.
