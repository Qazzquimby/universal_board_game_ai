from typing import Dict, Optional

from tqdm import tqdm

from core.agent_interface import Agent
from core.config import AppConfig
from environments.base import BaseEnvironment


# --- Helper for Single Game Simulation ---
def _play_one_game(env: BaseEnvironment, agent0: Agent, agent1: Agent) -> Optional[int]:
    """
    Plays a single game from the current env state.
    Args:
        env: The environment instance (assumed to be reset or in a valid state).
        agent0: The agent playing as player 0.
        agent1: The agent playing as player 1.
    Returns:
        The index of the winning player (0 or 1), or None for a draw.
    """
    state = env.get_observation()  # Get initial state from env
    done = False

    while not done:
        current_player_idx = env.get_current_player()
        if current_player_idx == 0:
            action = agent0.act(state)
            agent_name = type(agent0).__name__
        else:
            action = agent1.act(state)
            agent_name = type(agent1).__name__

        if action is None:
            print(
                f"Warning: Agent {agent_name} (Player {current_player_idx}) returned None action. Treating as loss."
            )
            break

        state, _, done = env.step(action)

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

    for i in tqdm(range(num_games), desc=f"{agent1_name} (P0) vs {agent2_name} (P1)"):
        game_env = env.copy()  # Use a fresh copy for each game
        game_env.reset()
        if i % 2 == 0:
            winner = _play_one_game(
                game_env, agent1, agent2
            )  # agent1 is P0, agent2 is P1

            if winner == 0:
                results[agent1_name] += 1
            elif winner == 1:
                results[agent2_name] += 1
            else:
                results["draws"] += 1
        else:
            winner = _play_one_game(
                game_env, agent2, agent1
            )  # agent2 is P0, agent1 is P1

            # Adjust win recording based on who was player 0 in this game
            if winner == 0:  # agent2 (P0) won
                results[agent2_name] += 1
            elif winner == 1:  # agent1 (P1) won
                results[agent1_name] += 1
            else:
                results["draws"] += 1

    print(f"--- Results after {num_games} games ({agent1_name} vs {agent2_name}) ---")
    print(f"{agent1_name} total wins: {results[agent1_name]}")
    print(f"{agent2_name} total wins: {results[agent2_name]}")
    print(f"Total draws: {results['draws']}")
    print("-" * (len(f"--- Testing {agent1_name} vs {agent2_name} ---") + 5))

    # Calculate rates
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
