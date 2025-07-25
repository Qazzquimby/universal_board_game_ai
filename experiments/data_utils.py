import json
import numpy as np
from tqdm import tqdm

from environments.connect4 import Connect4
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


def load_and_process_data(tiny_run=False):
    print("Loading and processing data...")
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)

    if tiny_run:
        raw_data = raw_data[:100]

    inputs = []
    policy_labels = []
    value_labels = []

    env = Connect4()

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
