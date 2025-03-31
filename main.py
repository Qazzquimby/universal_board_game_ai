import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

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
                action = opponent.act()
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


def _run_test_games(env, agent1_name, agent1, agent2_name, agent2, num_games=100):
    """Run test games between two agents"""
    print(f"\n--- Testing {agent1_name} vs {agent2_name} ---")
    results = {agent1_name: 0, agent2_name: 0, "draws": 0}

    # Ensure agents have exploration turned off for testing if applicable
    original_exploration_rates = {}
    for agent, name in [(agent1, agent1_name), (agent2, agent2_name)]:
        if hasattr(agent, "exploration_rate"):
            original_exploration_rates[name] = agent.exploration_rate
            agent.exploration_rate = 0.0  # Turn off exploration

    for game_num in range(num_games):
        state = env.reset()
        done = False

        while not done:
            if env.get_current_player() == 0:
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

    # Restore original exploration rates
    for agent, name in [(agent1, agent1_name), (agent2, agent2_name)]:
        if name in original_exploration_rates:
            agent.exploration_rate = original_exploration_rates[name]

    print(f"Results after {num_games} games:")
    print(f"{agent1_name} wins: {results[agent1_name]}")
    print(f"{agent2_name} wins: {results[agent2_name]}")
    print(f"Draws: {results['draws']}")
    print("-" * (len(f"--- Testing {agent1_name} vs {agent2_name} ---")))


# Example usage
if __name__ == "__main__":
    # --- Configuration ---
    BOARD_SIZE = 4
    NUM_PLAYERS = 2
    NUM_EPISODES_TRAIN = 5000  # Set to 0 to skip training if loading
    NUM_GAMES_TEST = 100
    QL_SAVE_FILE = "q_agent_4x4.pkl"
    PLOT_WINDOW = 200

    # --- Environment Setup ---
    env = BoardGameEnv(board_size=BOARD_SIZE, num_players=NUM_PLAYERS)

    # --- Agent Initialization ---
    ql_agent = QLearningAgent(env)

    # --- Load or Train Q-Learning Agent ---
    if not ql_agent.load(QL_SAVE_FILE):
        if NUM_EPISODES_TRAIN > 0:
            print(f"Training Q-learning agent for {NUM_EPISODES_TRAIN} episodes...")
            wins = train_agent(env, ql_agent, num_episodes=NUM_EPISODES_TRAIN)
            plot_results(wins, window_size=PLOT_WINDOW)
            ql_agent.save(QL_SAVE_FILE)
        else:
            print("Skipping Q-learning training as NUM_EPISODES_TRAIN is 0.")
    else:
        print("Loaded pre-trained Q-learning agent.")
        # Optionally plot if training data were saved alongside agent
        # plot_results(loaded_wins, window_size=PLOT_WINDOW)

    # --- Define Agents for Testing ---
    # Ensure agents used for testing have exploration turned off or minimized
    # ql_agent.exploration_rate = ql_agent.min_exploration # Set low exploration for testing
    # Or handled within _run_test_games

    agents = {
        "QLearning": ql_agent,
        "MCTS_100": MCTSAgent(env, num_simulations=100),
        "MCTS_500": MCTSAgent(env, num_simulations=500), # Example of another MCTS variant
        "Random": RandomAgent(env),
    }

    # --- Round-Robin Testing ---
    print("\n--- Starting Round-Robin Agent Testing ---")
    agent_names = list(agents.keys())

    # Use combinations_with_replacement for pairwise tests including self-play
    for agent1_name, agent2_name in itertools.combinations_with_replacement(
        agent_names, 2
    ):
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]

        # Need a fresh copy of the env for each test pair
        test_env = env.copy()

        # If testing agent against itself, create a new instance for the opponent
        if agent1_name == agent2_name:
             # Re-create agent instance to avoid shared state issues (like MCTS tree)
            if isinstance(agent2, MCTSAgent):
                 agent2 = MCTSAgent(env, num_simulations=agent2.mcts.num_simulations)
            # QLearning and Random agents are generally okay state-wise for self-play,
            # but recreating MCTS is safer. If QL had internal state beyond Q-table,
            # it might need recreation too.

        _run_test_games(
            test_env, agent1_name, agent1, agent2_name, agent2, num_games=NUM_GAMES_TEST
        )

    print("\n--- Agent Testing Complete ---")
