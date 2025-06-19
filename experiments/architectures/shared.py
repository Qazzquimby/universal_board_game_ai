from pathlib import Path

import torch

BOARD_HEIGHT = 6
BOARD_WIDTH = 7
MAX_EPOCHS = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "connect_4_states"
    / "pure_mcts"
    / "mcts_generated_states.json"
)
