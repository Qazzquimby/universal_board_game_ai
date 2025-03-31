import itertools
import math # Added for Elo calculation
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mcts import MCTSAgent
from qlearning import QLearningAgent
from temp_env import BoardGameEnv  # Import your environment
from random_agent import RandomAgent


def train_agent(env, agent, num_episodes=1000, opponent=None):
    """Train agent with proper turn handling and sparse rewards"""
    opponent = opponent or RandomAgent(env)
    win_history = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_history = []

        while not done:
            current_player = env.get_current_player()

            if current_player == 0:  # Agent's turn
                state = obs
                action = agent.act(state)
                next_obs, reward, done = env.step(action)
                episode_history.append((state, action, reward, done))
                obs = next_obs
            else:  # Opponent's turn
                # Pass the current observation (state) to the opponent's act method
                action = opponent.act(obs)
                obs, _, done = env.step(action)

        # After episode ends, determine final reward
        if env.get_winning_player() == 0:
            final_reward = 1.0
            outcome = 1
        elif env.get_winning_player() is not None:
            final_reward = -1.0
            outcome = -1
        else:
            final_reward = 0.0
            outcome = 0

        # Update Q-values with final reward
        if episode_history:
            # Replace all rewards with final outcome
            episode_history = [
                (s, a, final_reward, d) for s, a, _, d in episode_history
            ]
            agent.learn(episode_history)

        # Track outcomes and decay exploration
        win_history.append(outcome)
        agent.exploration_rate = max(
            agent.exploration_rate * agent.exploration_decay, agent.min_exploration
        )

        # Print progress occasionally
        window_size = 200
        if (episode + 1) % window_size == 0:
            win_rate = win_history[-window_size:].count(1) / window_size
            draw_rate = win_history[-window_size:].count(0) / window_size
            loss_rate = win_history[-window_size:].count(-1) / window_size
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Win Rate: {win_rate:.2f} | "
                f"Draw Rate: {draw_rate:.2f} | "
                f"Loss Rate: {loss_rate:.2f} | "
                f"Exploration: {agent.exploration_rate:.4f}"
            )

    return win_history


def plot_results(win_history, window_size=100):
    """Plot the training results."""
    # Plot rewards
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(win_history)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Plot smoothed rewards
    if len(win_history) >= window_size:
        smoothed_rewards = [
            np.mean(win_history[i : i + window_size])
            for i in range(len(win_history) - window_size + 1)
        ]
        plt.plot(range(window_size - 1, len(win_history)), smoothed_rewards, "r-")
        plt.legend(["Rewards", f"Moving Average ({window_size})"])

    # Plot win rate
    plt.subplot(2, 1, 2)

    # Calculate win rate using a sliding window
    if len(win_history) >= window_size:
        win_rates = []
        draw_rates = []
        loss_rates = []

        for i in range(len(win_history) - window_size + 1):
            window = win_history[i : i + window_size]
            win_rates.append(window.count(1) / window_size)
            draw_rates.append(window.count(0) / window_size)
            loss_rates.append(window.count(-1) / window_size)

        plt.plot(range(window_size - 1, len(win_history)), win_rates, "g-")
        plt.plot(range(window_size - 1, len(win_history)), draw_rates, "y-")
        plt.plot(range(window_size - 1, len(win_history)), loss_rates, "r-")
        plt.legend(["Win Rate", "Draw Rate", "Loss Rate"])

    plt.title("Performance over Time")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.show()

# TODO: Consider moving evaluation logic (run_test_games, Elo) into a separate Evaluation class/module.
def _run_test_games(env, agent1_name, agent1, agent2_name, agent2, num_games=100):
    """
    Run test games between two agents, ensuring both play as player 1 and player 2.

    Args:
        env: The game environment instance.
        agent1_name (str): Name of the first agent.
        agent1: The first agent object.
        agent2_name (str): Name of the second agent.
        agent2: The second agent object.
        num_games (int): Total number of games to play (will be split between starting orders).

    Returns:
        dict: A dictionary containing the aggregated results:
              {agent1_name: total_wins, agent2_name: total_wins, "draws": total_draws}
    """
    print(f"\n--- Testing {agent1_name} vs {agent2_name} ---")
    results = {agent1_name: 0, agent2_name: 0, "draws": 0}
    num_games_half = num_games // 2

    # --- Games where agent1 starts ---
    print(f"Running {num_games_half} games with {agent1_name} starting...")
    for _ in tqdm(range(num_games_half)):
        state = env.reset()
        done = False
        while not done:
            current_player_idx = env.get_current_player()
            if current_player_idx == 0:
                action = agent1.act(state)
            else:
                action = agent2.act(state)
            state, _, done = env.step(action)

        winner = env.get_winning_player()
        if winner == 0:
            results[agent1_name] += 1
        elif winner == 1:
            results[agent2_name] += 1
        else:
            results["draws"] += 1

    # --- Games where agent2 starts ---
    num_games_remaining = num_games - num_games_half
    print(f"Running {num_games_remaining} games with {agent2_name} starting...")
    for _ in tqdm(range(num_games_remaining)):
        state = env.reset()
        done = False
        while not done:
            current_player_idx = env.get_current_player()
            # Note: Player indices are 0 and 1. agent2 is player 0 here.
            if current_player_idx == 0:
                action = agent2.act(state)
            else:
                action = agent1.act(state)
            state, _, done = env.step(action)

        winner = env.get_winning_player()
        # Adjust win recording based on starting player
        if winner == 0: # agent2 won
            results[agent2_name] += 1
        elif winner == 1: # agent1 won
            results[agent1_name] += 1
        else:
            results["draws"] += 1

    print(f"--- Results after {num_games} games ({agent1_name} vs {agent2_name}) ---")
    print(f"{agent1_name} total wins: {results[agent1_name]}")
    print(f"{agent2_name} total wins: {results[agent2_name]}")
    print(f"Total draws: {results['draws']}")
    print("-" * (len(f"--- Testing {agent1_name} vs {agent2_name} ---") + 5)) # Adjust separator length
    return results


# --- Elo Calculation Functions ---
def calculate_expected_score(rating1: float, rating2: float) -> float:
    """Calculate the expected score of player 1 against player 2."""
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

def update_elo(old_rating: float, expected_score: float, actual_score: float, k_factor: int = 32) -> float:
    """Update Elo rating based on performance."""
    return old_rating + k_factor * (actual_score - expected_score)


# Example usage
if __name__ == "__main__":
    # --- Configuration ---
    # TODO: Consider moving configuration to a separate file (e.g., YAML, JSON) or using argparse for flexibility.
    BOARD_SIZE = 4
    NUM_PLAYERS = 2
    NUM_EPISODES_TRAIN = 5000  # Set to 0 to skip training if loading
    NUM_GAMES_TEST = 100
    QL_SAVE_FILE = "q_agent_4x4.pkl"
    PLOT_WINDOW = 200

    # --- Environment Setup ---
    env = BoardGameEnv(board_size=BOARD_SIZE, num_players=NUM_PLAYERS)

    # --- Agent Initialization ---
    # TODO: Consider a more structured way to manage agent creation and loading, perhaps an AgentRegistry or factory pattern.
    ql_agent = QLearningAgent(env, exploration_rate=0.0) # Start with low exploration for loaded agent
    if not ql_agent.load(QL_SAVE_FILE):
        print(f"Training Q-learning agent for {NUM_EPISODES_TRAIN} episodes...")
        # TODO: Move training logic into a dedicated function or class (e.g., Trainer) for better separation.
        ql_agent.exploration_rate = 1.0 # Reset exploration for training
        wins = train_agent(env, ql_agent, num_episodes=NUM_EPISODES_TRAIN)
        plot_results(wins, window_size=PLOT_WINDOW)
        ql_agent.save(QL_SAVE_FILE)
        ql_agent.exploration_rate = ql_agent.min_exploration # Set low exploration after training
    else:
        print("Loaded pre-trained Q-learning agent.")
        # Optionally plot if training data were saved alongside agent
        # plot_results(loaded_wins, window_size=PLOT_WINDOW)

    # --- Define Agents for Testing ---
    # TODO: Define a formal Agent interface (e.g., using abc.ABC) that all agents must implement (act, learn, save, load, reset etc.)
    # Ensure agents used for testing have exploration turned off or minimized.
    ql_agent.exploration_rate = ql_agent.min_exploration # Ensure Q-agent is not exploring during tests

    agents = {
        "QLearning": ql_agent,
        "MCTS_50": MCTSAgent(env, num_simulations=50),
        "MCTS_200": MCTSAgent(env, num_simulations=200),
        "Random": RandomAgent(env),
    }

    # --- Round-Robin Testing ---
    print("\n--- Starting Round-Robin Agent Testing ---")
    agent_names = list(agents.keys())
    all_results = {} # Store results for Elo calculation, key: tuple(sorted(agent1_name, agent2_name))

    # Use combinations for unique pairwise tests (e.g., ('QLearning', 'Random'), not ('Random', 'QLearning'))
    for agent1_name, agent2_name in itertools.combinations(agent_names, 2):
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]

        # Need a fresh copy of the env for each test pair
        test_env = env.copy()

        # Run games for this pair (handles both starting orders internally)
        # The returned results dict contains aggregated wins for agent1_name and agent2_name
        results = _run_test_games(
            test_env, agent1_name, agent1, agent2_name, agent2, num_games=NUM_GAMES_TEST
        )

        # Store results using a consistent key (sorted tuple of names)
        result_key = tuple(sorted((agent1_name, agent2_name)))
        all_results[result_key] = results

    print("\n--- Agent Testing Complete ---")

    # --- Elo Calculation ---
    print("\n--- Calculating Elo Ratings ---")
    BASE_ELO = 1000
    K_FACTOR = 32
    ELO_ITERATIONS = 100 # Iterate multiple times for ratings to stabilize

    elo_ratings = {name: float(BASE_ELO) for name in agent_names}

    # TODO: Consider extracting Elo calculation into its own function or class.
    for _ in range(ELO_ITERATIONS):
        new_ratings = elo_ratings.copy()
        # Iterate through the results, keys are sorted tuples (name1, name2)
        for (agent1_name, agent2_name), results in all_results.items():
            # Ensure we correctly extract wins for agent1 and agent2 based on the names in the key
            wins1 = results[agent1_name]
            wins2 = results[agent2_name]
            draws = results["draws"]
            total_games = wins1 + wins2 + draws

            if total_games == 0:
                continue

            # Calculate scores from agent1's perspective
            actual_score1 = (wins1 + 0.5 * draws) / total_games
            expected_score1 = calculate_expected_score(elo_ratings[agent1_name], elo_ratings[agent2_name])

            # Calculate scores from agent2's perspective
            actual_score2 = (wins2 + 0.5 * draws) / total_games
            expected_score2 = calculate_expected_score(elo_ratings[agent2_name], elo_ratings[agent1_name])

            # Update ratings (using ratings from the start of the iteration)
            # Don't update the baseline "Random" agent
            if agent1_name != "Random":
                 new_ratings[agent1_name] = update_elo(elo_ratings[agent1_name], expected_score1, actual_score1, K_FACTOR)
            if agent2_name != "Random":
                 new_ratings[agent2_name] = update_elo(elo_ratings[agent2_name], expected_score2, actual_score2, K_FACTOR)

        elo_ratings = new_ratings # Update ratings for the next iteration

    # Print sorted Elo ratings
    print("\n--- Final Elo Ratings (Baseline: Random @ 1000) ---")
    sorted_elos = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
    for name, rating in sorted_elos:
        print(f"{name:<15}: {rating:.2f}")
