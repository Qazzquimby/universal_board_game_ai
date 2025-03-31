import itertools
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from tqdm import tqdm

from core.agent_interface import Agent
from core.config import MainConfig
from environments.env_interface import BaseEnvironment


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
            break  # End game

    return winner


# --- Plotting ---
def plot_results(win_history, window_size=100):
    """Plot the training results (win/loss/draw rates)."""
    plt.figure(figsize=(12, 6))  # Adjusted figure size

    # Calculate win/draw/loss rates using a sliding window
    if len(win_history) >= window_size:
        win_rates = []
        draw_rates = []
        loss_rates = []

        for i in range(len(win_history) - window_size + 1):
            window = win_history[i : i + window_size]
            win_rates.append(window.count(1) / window_size)
            draw_rates.append(window.count(0) / window_size)
            loss_rates.append(window.count(-1) / window_size)

        episodes = range(window_size - 1, len(win_history))
        plt.plot(episodes, win_rates, "g-", label=f"Win Rate (Avg over {window_size})")
        plt.plot(
            episodes, draw_rates, "y-", label=f"Draw Rate (Avg over {window_size})"
        )
        plt.plot(
            episodes, loss_rates, "r-", label=f"Loss Rate (Avg over {window_size})"
        )
        plt.legend()

    else:
        # Plot raw outcomes if not enough data for smoothing
        plt.plot(win_history, "b.", label="Episode Outcome (1:Win, 0:Draw, -1:Loss)")
        plt.legend()

    plt.title("Agent Training Performance Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Rate / Outcome")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
    num_games_half = num_games // 2

    # --- Games where agent1 starts ---
    print(
        f"Running {num_games_half} games with {agent1_name} starting (as Player 0)..."
    )
    for _ in tqdm(
        range(num_games_half), desc=f"{agent1_name} (P0) vs {agent2_name} (P1)"
    ):
        game_env = env.copy()  # Use a fresh copy for each game
        game_env.reset()
        winner = _play_one_game(game_env, agent1, agent2)  # agent1 is P0, agent2 is P1

        if winner == 0:
            results[agent1_name] += 1
        elif winner == 1:
            results[agent2_name] += 1
        else:
            results["draws"] += 1

    # --- Games where agent2 starts ---
    num_games_remaining = num_games - num_games_half
    print(
        f"Running {num_games_remaining} games with {agent2_name} starting (as Player 0)..."
    )
    for _ in tqdm(
        range(num_games_remaining), desc=f"{agent2_name} (P0) vs {agent1_name} (P1)"
    ):
        game_env = env.copy()  # Use a fresh copy for each game
        game_env.reset()
        winner = _play_one_game(game_env, agent2, agent1)  # agent2 is P0, agent1 is P1

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
    return results


# --- Elo Calculation Functions ---
def calculate_expected_score(rating1: float, rating2: float) -> float:
    """Calculate the expected score of player 1 against player 2."""
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))


def update_elo(
    old_rating: float, expected_score: float, actual_score: float, k_factor: int = 32
) -> float:
    """Update Elo rating based on performance."""
    return old_rating + k_factor * (actual_score - expected_score)


def calculate_elo_ratings(
    agent_names: List[str],
    all_results: Dict[tuple, Dict],
    baseline_agent: str = "Random",
    baseline_rating: float = 1000.0,
    k_factor: int = 32,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Calculate Elo ratings for agents based on pairwise game results.

    Args:
        agent_names: List of all agent names.
        all_results: Dictionary where keys are sorted tuples (agent1_name, agent2_name)
                     and values are results dictionaries from run_test_games.
        baseline_agent: Name of the agent whose Elo is fixed.
        baseline_rating: The fixed Elo rating for the baseline agent.
        k_factor: Elo K-factor (sensitivity to recent results).
        iterations: Number of iterations to run Elo updates for stabilization.

    Returns:
        Dictionary mapping agent names to their calculated Elo ratings.
    """
    print("\n--- Calculating Elo Ratings ---")
    elo_ratings = {name: baseline_rating for name in agent_names}

    for i in range(iterations):
        new_ratings = elo_ratings.copy()
        for (agent1_name, agent2_name), results in all_results.items():
            # Ensure results dict contains the correct keys
            if agent1_name not in results or agent2_name not in results:
                print(
                    f"Warning: Skipping Elo update for ({agent1_name}, {agent2_name}) due to missing keys in results: {results}"
                )
                continue

            wins1 = results[agent1_name]
            wins2 = results[agent2_name]
            draws = results.get("draws", 0)  # Handle potential missing 'draws' key
            total_games = wins1 + wins2 + draws

            if total_games == 0:
                continue

            # Calculate scores from agent1's perspective
            actual_score1 = (wins1 + 0.5 * draws) / total_games
            expected_score1 = calculate_expected_score(
                elo_ratings[agent1_name], elo_ratings[agent2_name]
            )

            # Calculate scores from agent2's perspective
            actual_score2 = (wins2 + 0.5 * draws) / total_games
            expected_score2 = calculate_expected_score(
                elo_ratings[agent2_name], elo_ratings[agent1_name]
            )

            # Update ratings (using ratings from the start of the iteration)
            # Don't update the baseline agent
            if agent1_name != baseline_agent:
                new_ratings[agent1_name] = update_elo(
                    elo_ratings[agent1_name], expected_score1, actual_score1, k_factor
                )
            if agent2_name != baseline_agent:
                new_ratings[agent2_name] = update_elo(
                    elo_ratings[agent2_name], expected_score2, actual_score2, k_factor
                )

        # Ensure baseline agent rating remains fixed
        new_ratings[baseline_agent] = baseline_rating
        elo_ratings = new_ratings  # Update ratings for the next iteration

        # Optional: Print progress or check for convergence
        # if i % 10 == 0:
        #     print(f"Elo Iteration {i+1}/{iterations}")

    # Print sorted Elo ratings
    print(
        f"\n--- Final Elo Ratings (Baseline: {baseline_agent} @ {baseline_rating:.0f}) ---"
    )
    sorted_elos = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
    for name, rating in sorted_elos:
        print(f"{name:<15}: {rating:.2f}")

    return elo_ratings


def run_evaluation(env: BaseEnvironment, agents: Dict[str, Agent], config: MainConfig):
    """
    Runs the full evaluation suite: round-robin games and Elo calculation.

    Args:
        env: An instance of the environment (copies will be made).
        agents: Dictionary mapping agent names to agent instances.
        config: The configuration object containing testing parameters.
    """
    print("\n--- Starting Round-Robin Agent Testing ---")
    agent_names = list(agents.keys())
    all_results = {}  # Store results for Elo calculation

    # Use combinations for unique pairwise tests
    for agent1_name, agent2_name in itertools.combinations(agent_names, 2):
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]

        # Run games for this pair (handles both starting orders internally)
        results = run_test_games(
            env,
            agent1_name,
            agent1,
            agent2_name,
            agent2,
            num_games=config.num_games_test,
        )

        # Store results using a consistent key (sorted tuple of names)
        result_key = tuple(sorted((agent1_name, agent2_name)))
        all_results[result_key] = results

    print("\n--- Agent Testing Complete ---")

    # --- Elo Calculation ---
    calculate_elo_ratings(
        agent_names=agent_names,
        all_results=all_results,
        baseline_agent=config.elo_baseline_agent,
        baseline_rating=config.elo_baseline_rating,
        k_factor=config.elo_k_factor,
        iterations=config.elo_iterations,
    )
