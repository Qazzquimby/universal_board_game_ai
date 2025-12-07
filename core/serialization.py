import datetime
import json
from typing import List, Tuple

import numpy as np
from loguru import logger

from agents.base_learning_agent import LoggedStep
from core.config import DATA_DIR
from environments.base import StateType, ActionType


def _default_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    # Add other custom types here if needed
    # Example: if isinstance(obj, SomeCustomClass): return obj.to_dict()
    try:
        return obj.__dict__  # Fallback for simple objects
    except AttributeError:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )


def get_file_path(
    iteration: int,
    game_index: int,
    env_name: str,
    model_name: str,
):
    log_dir = DATA_DIR / env_name / "game_logs" / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"game_{iteration}_{timestamp}_{game_index:04d}.json"
    file_path = log_dir / file_name
    return file_path


def save_game_log(
    logged_history: List[LoggedStep],
    file_path: str,
):
    """Saves the processed game history to a JSON file."""
    serializable_log = []
    for step in logged_history:
        serializable_log.append(
            {
                "state": step.state,
                "action_index": step.action_index,
                "policy_target": step.policy.tolist(),  # Convert numpy array
                "value_target": step.value,
            }
        )

    with open(file_path, "w") as f:
        json.dump(serializable_log, f, indent=2, default=_default_serializer)
