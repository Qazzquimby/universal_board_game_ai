import sys
import time

from agents.alphazero_agent import AlphaZeroAgent
from agents.mcts_agent import MCTSAgent
from core.config import AppConfig
from evaluation import run_test_games
from factories import get_environment, get_agents
from train_alphazero import run_eval_against_benchmark
from utils.training_reporter import TrainingReporter


def run_comparison(config: AppConfig):
    config.mcts.num_simulations = 400
    num_games = 100
    config.evaluation.full_eval_num_games = num_games
    config.evaluation.periodic_eval_num_games = num_games
    config.wandb.enabled = False

    env = get_environment(config.env)

    print("Loading agents...")
    agents = get_agents(env, config, load_all_az_iterations=True)

    mcts_agent_name = f"MCTS_{config.mcts.num_simulations}"
    az_agent_name = f"AZ_{config.mcts.num_simulations}_iter_1"

    if mcts_agent_name not in agents:
        print(f"Agent {mcts_agent_name} not found.")
        return
    if az_agent_name not in agents:
        print(
            f"Agent {az_agent_name} not found. Available agents: {list(agents.keys())}"
        )
        return

    mcts_agent = agents[mcts_agent_name]
    az_agent_iter_1 = agents[az_agent_name]

    print(
        f"\n--- Comparison of evaluation methods for {mcts_agent_name} vs {az_agent_name} ---"
    )
    print(f"--- Running {num_games} games for each method. ---")

    # Method 1: evaluation.py
    print("\n--- Method 1: Using evaluation.run_test_games ---")
    results1 = run_test_games(
        env=env,
        agent0_name=az_agent_name,
        agent0=az_agent_iter_1,
        agent1_name=mcts_agent_name,
        agent1=mcts_agent,
        config=config,
        num_games=config.evaluation.full_eval_num_games,
    )
    print("\nResults from evaluation.run_test_games:")
    print(results1)

    # Method 2: train_alphazero.py
    print("\n--- Method 2: Using train_alphazero.run_eval_against_benchmark ---")
    reporter = TrainingReporter(config, az_agent_iter_1, time.time())

    assert isinstance(az_agent_iter_1, AlphaZeroAgent)
    assert isinstance(mcts_agent, MCTSAgent)

    eval_results, _ = run_eval_against_benchmark(
        iteration=-1,  # To prevent logging to reporter
        reporter=reporter,
        current_agent=az_agent_iter_1,
        best_agent=mcts_agent,
        best_agent_name=mcts_agent_name,
        config=config,
        env=env,
    )

    print("\nResults from train_alphazero.run_eval_against_benchmark:")
    print(eval_results)

    print("\n--- Comparison Complete ---")


if __name__ == "__main__":
    config = AppConfig()

    if len(sys.argv) > 1:
        config.env.name = sys.argv[1]

    run_comparison(config)
