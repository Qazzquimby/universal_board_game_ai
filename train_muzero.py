# Context

# A state is made of several dataframes, where each row is converted to a token.
# Actions are another set of tokens
# Muzero will need to generate a set of action tokens in order to MCTS.
# You can *not* assume a single finite superset of actions or a fixed count of actions.
# The game logs contain the states for each turn of a game, in order.

import sys

from loguru import logger

from core.config import AppConfig
from models.training import run_training_loop

if __name__ == "__main__":
    config = AppConfig()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    run_training_loop(config, model_type="muzero", env_name_override=None)
