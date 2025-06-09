import sys

from core.config import AppConfig
from factories import get_environment, get_agents
from evaluation import run_evaluation


def run_main(config: AppConfig):
    """Runs the main evaluation process."""
    env = get_environment(config.env)
    agents = get_agents(env, config)

    run_evaluation(env, agents, config)

    print("\n--- Main evaluation finished ---")


if __name__ == "__main__":
    config = AppConfig()

    if len(sys.argv) > 1:
        config.env.name = sys.argv[1]  # e.g., python main.py Nim

    run_main(config)
