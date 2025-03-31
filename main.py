import sys

# Core imports
from core.config import AppConfig

# Factories and Evaluation
from factories import get_environment, get_agents
from evaluation import run_evaluation


if __name__ == "__main__":
    # --- Configuration ---
    # TODO: Add argument parsing here to override config defaults (e.g., using argparse or Hydra)
    config = AppConfig()

    # --- Environment Selection ---
    # Simple selection logic using command-line argument
    if len(sys.argv) > 1:
        config.env.name = sys.argv[1] # e.g., python main.py Nim

    # --- Instantiation via Factories ---
    env = get_environment(config.env)
    agents = get_agents(env, config) # Training happens inside get_agents if needed

    # --- Evaluation ---
    run_evaluation(env, agents, config)

    print("\n--- Main script finished ---")
