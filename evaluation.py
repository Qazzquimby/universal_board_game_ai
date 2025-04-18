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
    winner = None

    while not done:
        current_player_idx = env.get_current_player()
        if current_player_idx == 0:
            action = agent0.act(state)
            agent_name = type(agent0).__name__  # For logging
        else:
            action = agent1.act(state)
            agent_name = type(agent1).__name__  # For logging

        if action is None:
            print(
                f"Warning: Agent {agent_name} (Player {current_player_idx}) returned None action. Treating as loss."
            )
            winner = 1 - current_player_idx  # Opponent wins
            break  # End game

        try:
            state, _, done = env.step(action)
            winner = env.get_winning_player()  # Get winner after step
        except ValueError as e:
            print(
                f"Warning: Invalid action {action} during testing by Player {current_player_idx} ({agent_name}). Error: {e}"
            )
            winner = 1 - current_player_idx  # Opponent wins due to invalid move
            done = True  # Ensure loop terminates on error

        if done:
            break  # Exit loop immediately if done is set

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

    # Ensure we have exactly the two agents we expect for the benchmark
    if len(agent_names) != 2 or "AlphaZero" not in agents:
        print(
            f"Warning: Expected 'AlphaZero' and one 'MCTS_X' agent, but found: {agent_names}. Skipping evaluation."
        )
        return

    # Identify the MCTS agent name dynamically
    mcts_agent_name = next(
        (name for name in agent_names if name.startswith("MCTS_")), None
    )
    if not mcts_agent_name:
        print(
            f"Warning: Could not find MCTS agent in agents: {agent_names}. Skipping evaluation."
        )
        return

    agent1_name = "AlphaZero"
    agent2_name = mcts_agent_name
    agent1 = agents[agent1_name]
    agent2 = agents[agent2_name]

    # Run games between AlphaZero and the MCTS benchmark agent
    run_test_games(
        env,
        agent1_name,
        agent1,
        agent2_name,
        agent2,
        num_games=config.evaluation.full_eval_num_games,
    )

    print("\n--- Agent Evaluation Complete ---")
