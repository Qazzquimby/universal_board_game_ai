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


def _run_test_games(env, agent1, agent2, num_games=100):
    """Run test games between two agents"""
    results = {"agent1": 0, "agent2": 0, "draws": 0}

    for _ in range(num_games):
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
            results["agent1"] += 1
        elif winner == 1:
            results["agent2"] += 1
        else:
            results["draws"] += 1

    print(f"Results after {num_games} games:")
    print(f"Agent1 wins: {results['agent1']}")
    print(f"Agent2 wins: {results['agent2']}")
    print(f"Draws: {results['draws']}")


# Example usage
if __name__ == "__main__":
    # Create environment
    env = BoardGameEnv(board_size=4, num_players=2)

    # Create different agents
    random_agent = RandomAgent(env)
    ql_agent = QLearningAgent(env)
    mcts_agent = MCTSAgent(env)

    # Train Q-learning agent
    print("Training Q-learning agent...")
    wins = train_agent(env, ql_agent, num_episodes=5000)
    plot_results(wins, window_size=200)

    # Test agents against each other
    print("\nTesting MCTS vs Q-Learning...")
    _run_test_games(env.copy(), mcts_agent, ql_agent)

    print("\nTesting Q-Learning vs Random...")
    _run_test_games(env.copy(), ql_agent, random_agent)

    print("\nTesting MCTS vs MCTS...")
    _run_test_games(env.copy(), mcts_agent, MCTSAgent(env))
