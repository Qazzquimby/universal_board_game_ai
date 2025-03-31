import unittest
import sys
from pathlib import Path

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the main functions and config
from core.config import AppConfig
from main import run_main
from train_alphazero import run_training


class TestSmoke(unittest.TestCase):
    """
    Smoke tests that run the main logic with minimal settings
    to catch basic crashes and integration errors. Runs in the same process
    for easier debugging.
    """

    def _get_smoke_config(self) -> AppConfig:
        """Creates a config object with minimal settings for smoke tests."""
        config = AppConfig()
        config.smoke_test = True # Set the smoke test flag

        # Evaluation settings
        config.evaluation.num_games = 2
        config.evaluation.elo_iterations = 2

        # MCTS settings
        config.mcts.num_simulations_short = 2
        config.mcts.num_simulations_long = 3 # Keep slightly different for MCTS_200 if included

        # AlphaZero settings
        config.alpha_zero.num_simulations = 2
        config.alpha_zero.replay_buffer_size = 10
        config.alpha_zero.batch_size = 4

        # Training settings
        config.training.num_episodes = 2 # Q-learning episodes
        config.training.num_iterations = 1 # AlphaZero iterations
        config.training.num_episodes_per_iteration = 1 # AlphaZero self-play games

        return config

    def test_main_evaluation_fourinarow(self):
        """Smoke test main evaluation logic with FourInARow."""
        print("\n--- Running Smoke Test: main.py FourInARow ---")
        config = self._get_smoke_config()
        config.env.name = "FourInARow"
        try:
            run_main(config)
        except Exception as e:
            self.fail(f"run_main(FourInARow) failed with exception: {e}")

    def test_main_evaluation_nim(self):
        """Smoke test main evaluation logic with Nim."""
        print("\n--- Running Smoke Test: main.py Nim ---")
        config = self._get_smoke_config()
        config.env.name = "Nim"
        try:
            run_main(config)
        except Exception as e:
            self.fail(f"run_main(Nim) failed with exception: {e}")

    def test_train_alphazero_fourinarow(self):
        """Smoke test AlphaZero training logic with FourInARow."""
        print("\n--- Running Smoke Test: train_alphazero.py FourInARow ---")
        config = self._get_smoke_config()
        # run_training takes env name override separately if needed, but setting config is fine
        config.env.name = "FourInARow"
        try:
            run_training(config)
        except Exception as e:
            self.fail(f"run_training(FourInARow) failed with exception: {e}")

    def test_train_alphazero_nim(self):
        """Smoke test AlphaZero training logic with Nim."""
        print("\n--- Running Smoke Test: train_alphazero.py Nim ---")
        config = self._get_smoke_config()
        config.env.name = "Nim"
        try:
            run_training(config)
        except Exception as e:
            self.fail(f"run_training(Nim) failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
