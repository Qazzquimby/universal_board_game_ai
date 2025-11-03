import json
import numpy as np
from tqdm import tqdm

from environments.base import DataFrame
from environments.connect4.connect4 import Connect4
from experiments.architectures.shared import (
    BOARD_WIDTH,
    DATA_PATH,
)


def _process_raw_item(item, env: Connect4):
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

    id_board = []
    for row in env.state.board.cells:
        id_row = []
        for cell in row:
            cell_id = cell.id + 1 if cell is not None else 0
            id_row.append(cell_id)
        id_board.append(id_row)
    board_state = np.array(id_board)
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


def _state_dict_to_numpy(state: dict) -> np.ndarray:
    game_df = state.get("game")
    board_df = state.get("pieces")

    if not game_df or not board_df or game_df.is_empty():
        raise ValueError

    if "current_player" not in game_df.columns:
        raise ValueError
    current_player = game_df._data[0][game_df._col_to_idx["current_player"]]
    opponent_player = 1 - current_player

    p1_board = np.zeros((6, BOARD_WIDTH), dtype=np.float32)
    p2_board = np.zeros((6, BOARD_WIDTH), dtype=np.float32)

    if not board_df.is_empty():
        row_idx = board_df._col_to_idx["row"]
        col_idx = board_df._col_to_idx["col"]
        player_id_idx = board_df._col_to_idx["player_id"]

        for piece in board_df._data:
            row, col, player_id = piece[row_idx], piece[col_idx], piece[player_id_idx]
            if player_id == current_player:
                p1_board[row, col] = 1.0
            elif player_id == opponent_player:
                p2_board[row, col] = 1.0
    return np.stack([p1_board, p2_board])


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


MAX_FILES = 100


def load_and_process_data(tiny_run=False):
    print("Loading and processing data from game logs...")
    log_dir = DATA_PATH / "connect4" / "game_logs"
    assert log_dir.exists()

    log_files = sorted(list(log_dir.glob("**/*.json")))
    assert log_files

    if tiny_run:
        log_files = log_files[-2:]  # Take last 2 files for tiny run
    elif MAX_FILES:
        log_files = log_files[-MAX_FILES:]

    inputs = []
    policy_labels = []
    value_labels = []

    all_steps = []
    for log_file in tqdm(log_files, desc="Scanning log files"):
        with open(log_file, "r") as f:
            try:
                game_data = json.load(f)
                all_steps.extend(game_data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {log_file}. Skipping.")
                continue

    if tiny_run and len(all_steps) > 100:
        all_steps = all_steps[-100:]

    for step_data in tqdm(all_steps, desc="Processing game steps"):
        state_json = step_data.get("state")
        policy_target_list = step_data.get("policy_target")
        value_target = step_data.get("value_target")

        if not all([state_json, policy_target_list, value_target is not None]):
            continue

        state = {
            table_name: DataFrame(
                data=table_data.get("_data"),
                columns=table_data.get("columns"),
            )
            for table_name, table_data in state_json.items()
        }

        input_tensor = _state_dict_to_numpy(state)

        legal_actions_df = state.get("legal_actions")
        assert legal_actions_df and not legal_actions_df.is_empty()
        legal_actions = [row[0] for row in legal_actions_df.rows()]

        policy_target = np.array(policy_target_list)
        assert len(policy_target) == len(legal_actions)

        best_action_idx = np.argmax(policy_target)
        policy_label = legal_actions[best_action_idx]

        _augment_and_append(
            input_tensor,
            policy_label,
            value_target,
            inputs,
            policy_labels,
            value_labels,
        )

    assert inputs

    print(f"Data augmentation complete. Total samples: {len(inputs)}")
    return (
        np.array(inputs),
        np.array(policy_labels),
        np.array(value_labels, dtype=np.float32),
    )
