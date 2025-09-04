import sys

from loguru import logger

from core.config import AppConfig
from models.training import run_training_loop

if __name__ == "__main__":
    config = AppConfig()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    run_training_loop(config, model_type="alphazero", env_name_override=None)
