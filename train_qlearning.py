import sys
from typing import Tuple

from tqdm import tqdm

from agents.qlearning import run_qlearning_training, QLearningAgent
from core.config import AppConfig
from environments.base import BaseEnvironment
from agents.alphazero_agent import AlphaZeroAgent
from factories import get_environment
from utils.plotting import plot_results


if __name__ == "__main__":
    # --- Configuration ---
    config = AppConfig()

    # --- Environment Selection (Optional: Add CLI arg parsing) ---
    if len(sys.argv) > 1:
        env_override = sys.argv[1]  # e.g., python train_qlearning.py Nim

    env = get_environment(config.env)

    run_qlearning_training(
        env=env,
        agent=QLearningAgent(env=env, config=config.q_learning),
        num_episodes=config.training.num_episodes,
        q_config=config.q_learning,
        training_config=config.training,
    )
