import time
from typing import Dict, Optional

from tqdm import tqdm

from core.agent_interface import Agent
from core.config import AppConfig
from environments.base import BaseEnvironment


class GameProfiler:
    def __init__(self):
        self.agent1_times = []
        self.agent2_times = []


def _play_one_game(
    env: BaseEnvironment, agent0: Agent, agent1: Agent, profiler: GameProfiler = None
) -> Optional[int]:
    """
    Plays a single game from the current env state.
    Args:
        env: The environment instance (assumed to be reset or in a valid state).
        agent0: The agent playing as player 0.
        agent1: The agent playing as player 1.
    Returns:
        The index of the winning player (0 or 1), or None for a draw.
    """
    done = False

    while not done:
        current_player_idx = env.get_current_player()

        if profiler:
            start = time.perf_counter()
            if current_player_idx == 0:
                action = agent0.act(env=env)
                profiler.agent0_times.append(time.perf_counter() - start)
            else:
                action = agent1.act(env=env)
                profiler.agent1_times.append(time.perf_counter() - start)
        else:
            if current_player_idx == 0:
                action = agent0.act(env=env)
            else:
                action = agent1.act(env=env)
        assert action is not None

        result = env.step(action)
        done = result.done

        if done:
            break

    final_winner = env.get_winning_player()
    return final_winner


def run_test_games(
    env: BaseEnvironment,
    agent1_name: str,
    agent1: Agent,
    agent2_name: str,
    agent2: Agent,
    num_games: int = 100,
):
    """
    Run test games between two agents, ensuring both play as player 1 and player 2.

    Args:
        env: The game environment instance (a copy will be used).
        agent1_name: Name of the first agent.
        agent1: The first agent object.
        agent2_name: Name of the second agent.
        agent2: The second agent object.
        num_games: Total number of games to play (will be split between starting orders).

    Returns:
        dict: Aggregated results {agent1_name: total_wins, agent2_name: total_wins, "draws": total_draws}
    """
    print(f"\n--- Testing {agent1_name} vs {agent2_name} ---")
    results = {agent1_name: 0, agent2_name: 0, "draws": 0}

    profiler = GameProfiler()

    for i in tqdm(range(num_games), desc=f"{agent1_name} (P0) vs {agent2_name} (P1)"):
        game_env = env.copy()
        game_env.reset()
        if i % 2 == 0:
            winner = _play_one_game(game_env, agent1, agent2, profiler=profiler)

            if winner == 0:
                results[agent1_name] += 1
            elif winner == 1:
                results[agent2_name] += 1
            else:
                results["draws"] += 1
        else:
            winner = _play_one_game(game_env, agent2, agent1)
            if winner == 0:
                results[agent2_name] += 1
            elif winner == 1:
                results[agent1_name] += 1
            else:
                results["draws"] += 1

    print(f"--- Results after {num_games} games ({agent1_name} vs {agent2_name}) ---")
    print(f"{agent1_name} total wins: {results[agent1_name]}")
    print(f"{agent2_name} total wins: {results[agent2_name]}")
    print(f"Total draws: {results['draws']}")
    print("-" * (len(f"--- Testing {agent1_name} vs {agent2_name} ---") + 5))

    total_games = sum(results.values())
    if total_games == 0:
        return {
            f"{agent1_name}_win_rate": 0.0,
            f"{agent2_name}_win_rate": 0.0,
            "draw_rate": 0.0,
        }

    rates = {
        f"{agent1_name}_win_rate": results[agent1_name] / total_games,
        f"{agent2_name}_win_rate": results[agent2_name] / total_games,
        "draw_rate": results["draws"] / total_games,
    }
    return rates


def run_evaluation(env: BaseEnvironment, agents: Dict[str, Agent], config: AppConfig):
    """
    Runs the full evaluation suite: round-robin games and Elo calculation.

    Args:
        env: An instance of the environment (copies will be made).
        agents: Dictionary mapping agent names to agent instances.
        config: The configuration object containing evaluation parameters.
    """
    print("\n--- Starting Agent Evaluation ---")
    agent_names = list(agents.keys())

    agent1_name = agent_names[0]
    agent2_name = agent_names[1]
    agent1 = agents[agent1_name]
    agent2 = agents[agent2_name]

    run_test_games(
        env=env,
        agent1_name=agent1_name,
        agent1=agent1,
        agent2_name=agent2_name,
        agent2=agent2,
        num_games=config.evaluation.full_eval_num_games,
    )
