import sys
from collections import deque

from loguru import logger
from tqdm import trange

from core.config import AppConfig
from factories import get_environment
from agents.alphazero.alphazero_agent import make_pure_az


def force_overfit(num_experiences: int = 1):
    """
    Runs an experiment to check if the AlphaZero model can overfit to a small
    amount of data.
    """
    config = AppConfig()
    # Use a small batch size for overfitting
    config.alphazero.training_batch_size = min(num_experiences, 4)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("--- Starting Overfitting Experiment ---")
    env = get_environment(config.env)
    agent = make_pure_az(
        env=env, config=config.alphazero, training_config=config.training
    )

    # Load existing game data
    logger.info("Loading existing game data...")
    agent.load_game_logs(config.env.name, buffer_limit=num_experiences)

    all_experiences = list(agent.train_replay_buffer) + list(agent.val_replay_buffer)

    if len(all_experiences) < num_experiences:
        logger.error(
            f"Not enough experiences in logs. Found {len(all_experiences)}, needed {num_experiences}. Exiting."
        )
        return

    # Select a small subset of experiences
    experiences_to_overfit = all_experiences[:num_experiences]
    logger.info(f"Selected {len(experiences_to_overfit)} experience(s) to overfit on.")

    # Manually set the replay buffer to our small dataset
    agent.train_replay_buffer = deque(experiences_to_overfit)
    agent.val_replay_buffer = deque()  # No validation

    # Train for many iterations on the same small dataset
    logger.info("Starting training loop...")
    agent.network.train()

    metrics = agent.train_network(iteration=-1, save_checkpoints=False)
    if metrics:
        train_metrics = metrics.train
        logger.info(
            f"Loss: {train_metrics.loss:.4f}, "
            f"Value Loss: {train_metrics.value_loss:.4f}, "
            f"Policy Loss: {train_metrics.policy_loss:.4f}, "
            f"Policy Acc: {train_metrics.acc:.2%}"
        )

    logger.info("--- Overfitting Experiment Finished ---")


if __name__ == "__main__":
    force_overfit(num_experiences=1)
