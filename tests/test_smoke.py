import unittest
import subprocess
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define paths to the scripts relative to project root
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
TRAIN_AZ_SCRIPT = PROJECT_ROOT / "train_alphazero.py"

# Minimal settings for smoke tests
SMOKE_SETTINGS = {
    "SMOKE_TEST_EVAL_GAMES": "2",
    "SMOKE_TEST_EVAL_ELO_ITER": "2",
    "SMOKE_TEST_MCTS_SIMS_SHORT": "2",
    "SMOKE_TEST_MCTS_SIMS_LONG": "3", # Keep slightly different
    "SMOKE_TEST_AZ_SIMS": "2",
    "SMOKE_TEST_AZ_ITERATIONS": "1",
    "SMOKE_TEST_AZ_EPISODES_PER_ITER": "1",
    "SMOKE_TEST_AZ_BUFFER": "10", # Small buffer
    "SMOKE_TEST_AZ_BATCH": "4", # Small batch
    "SMOKE_TEST_QL_EPISODES": "2",
}

class TestSmoke(unittest.TestCase):
    """
    Smoke tests that run the main scripts with minimal settings
    to catch basic crashes and integration errors.
    """

    def _run_script(self, script_path: Path, args: list = None, env_vars: dict = None):
        """Helper method to run a script in a subprocess."""
        command = [sys.executable, str(script_path)]
        if args:
            command.extend(args)

        # Set environment variables for the subprocess
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        print(f"\nRunning command: {' '.join(command)}")
        print(f"With extra environment variables: {env_vars}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=process_env,
            check=False, # Don't raise exception on non-zero exit code
        )

        # Print output for debugging if test fails
        if result.returncode != 0:
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)

        self.assertEqual(result.returncode, 0, f"Script {script_path} failed with args {args}")

    def test_main_evaluation_fourinarow(self):
        """Smoke test main.py with FourInARow."""
        self._run_script(MAIN_SCRIPT, args=["FourInARow"], env_vars=SMOKE_SETTINGS)

    def test_main_evaluation_nim(self):
        """Smoke test main.py with Nim."""
        self._run_script(MAIN_SCRIPT, args=["Nim"], env_vars=SMOKE_SETTINGS)

    def test_train_alphazero_fourinarow(self):
        """Smoke test train_alphazero.py with FourInARow."""
        self._run_script(TRAIN_AZ_SCRIPT, args=["FourInARow"], env_vars=SMOKE_SETTINGS)

    def test_train_alphazero_nim(self):
        """Smoke test train_alphazero.py with Nim."""
        self._run_script(TRAIN_AZ_SCRIPT, args=["Nim"], env_vars=SMOKE_SETTINGS)


if __name__ == "__main__":
    unittest.main()
