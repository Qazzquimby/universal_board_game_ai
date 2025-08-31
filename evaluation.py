import time
from statistics import mean
from typing import Dict, Optional

from tqdm import tqdm

from core.agent_interface import Agent
from core.config import AppConfig
from environments.base import BaseEnvironment
import wandb


class GameProfiler:
    def __init__(self):
        self.agent0_times = []
        self.agent1_times = []


def _play_one_game(
    env: BaseEnvironment,
    agent0: Agent,
    agent1: Agent,
    profiler: GameProfiler = None,
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

    for agent in (agent0, agent1):
        agent.reset_game()
        if hasattr(agent, "network") and agent.network:
            agent.network.eval()

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
    agent0_name: str,
    agent0: Agent,
    agent1_name: str,
    agent1: Agent,
    config: AppConfig,
    num_games: int = 100,
):
    """
    Run test games between two agents, ensuring both play as player 1 and player 2.

    Args:
        env: The game environment instance (a copy will be used).
        agent0_name: Name of the first agent.
        agent0: The first agent object.
        agent1_name: Name of the second agent.
        agent1: The second agent object.
        num_games: Total number of games to play (will be split between starting orders).

    Returns:
        dict: Aggregated results {agent1_name: total_wins, agent2_name: total_wins, "draws": total_draws}
    """
    print(f"\n--- Testing {agent0_name} vs {agent1_name} ---")
    results = {agent0_name: 0, agent1_name: 0, "draws": 0}

    profiler = GameProfiler()

    config.init_wandb()

    with tqdm(total=num_games, desc=f"{agent0_name} vs {agent1_name}") as pbar:
        for i in range(num_games):
            game_env = env.copy()
            game_env.reset()
            if i % 2 == 0:
                winner = _play_one_game(game_env, agent0, agent1, profiler=profiler)

                if winner == 0:
                    results[agent0_name] += 1
                elif winner == 1:
                    results[agent1_name] += 1
                else:
                    results["draws"] += 1
            else:
                winner = _play_one_game(game_env, agent1, agent0, profiler=profiler)
                if winner == 0:
                    results[agent1_name] += 1
                elif winner == 1:
                    results[agent0_name] += 1
                else:
                    results["draws"] += 1
            pbar.set_postfix(results)
            pbar.update(1)

    log_results = {
        f"{agent0_name} total wins": results[agent0_name],
        f"{agent1_name} total wins": results[agent1_name],
        "total draws": results["draws"],
    }
    if profiler:
        log_results[f"{agent0_name} total seconds"] = sum(profiler.agent0_times)
        log_results[f"{agent1_name} total seconds"] = sum(profiler.agent1_times)
        log_results[f"{agent0_name} avg seconds"] = mean(profiler.agent0_times)
        log_results[f"{agent1_name} avg seconds"] = mean(profiler.agent1_times)

    if config.wandb.enabled:
        wandb.log(log_results)

    print(f"--- Results after {num_games} games ({agent0_name} vs {agent1_name}) ---")
    for key, value in log_results.items():
        print(f"{key}: {value}")
    print("-" * (len(f"--- Testing {agent0_name} vs {agent1_name} ---") + 5))

    return results


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
    agent_stats = {
        name: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for name in agent_names
    }

    if len(agent_names) < 2:
        print("Need at least two agents for evaluation.")
        return

    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            agent0_name = agent_names[i]
            agent1_name = agent_names[j]
            agent0 = agents[agent0_name]
            agent1 = agents[agent1_name]

            results = run_test_games(
                env=env,
                agent0_name=agent0_name,
                agent0=agent0,
                agent1_name=agent1_name,
                agent1=agent1,
                num_games=config.evaluation.full_eval_num_games,
                config=config,
            )
            agent0_wins = results[agent0_name]
            agent1_wins = results[agent1_name]
            draws = results["draws"]

            agent_stats[agent0_name]["wins"] += agent0_wins
            agent_stats[agent0_name]["losses"] += agent1_wins
            agent_stats[agent0_name]["draws"] += draws
            agent_stats[agent0_name]["games"] += agent0_wins + agent1_wins + draws

            agent_stats[agent1_name]["wins"] += agent1_wins
            agent_stats[agent1_name]["losses"] += agent0_wins
            agent_stats[agent1_name]["draws"] += draws
            agent_stats[agent1_name]["games"] += agent0_wins + agent1_wins + draws

    print("\n--- Final Evaluation Summary ---")
    print(
        f"{'Agent':<20} | {'Wins':>5} | {'Losses':>6} | {'Draws':>5} | {'Win Rate':>10}"
    )
    print("-" * 58)
    for agent_name in agent_names:
        stats = agent_stats[agent_name]
        total_games = stats["games"]
        win_rate = stats["wins"] / total_games if total_games > 0 else 0.0
        print(
            f"{agent_name:<20} | {stats['wins']:>5} | {stats['losses']:>6} | {stats['draws']:>5} | {win_rate:>10.2%}"
        )
