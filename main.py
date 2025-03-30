import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from temp_env import BoardGameEnv, RandomAgent  # Import your environment


class QLearningAgent:
    """Q-learning agent for board games with sparse rewards"""

    def __init__(
        self,
        env: BoardGameEnv,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.999,
        min_exploration: float = 0.01,
    ):
        """
        Initialize the Q-learning agent.

        Args:
            env: The board game environment
            learning_rate: Learning rate for Q-learning updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate for epsilon-greedy policy
            exploration_decay: Rate at which exploration decreases
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Initialize Q-table
        # Using defaultdict to handle new state-action pairs
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _state_to_key(self, state):
        """Convert state (board) to a hashable key."""
        # Flatten the board and convert to tuple for hashing
        board = state["board"]
        return tuple(board.flatten())

    def act(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: The current state observation

        Returns:
            The chosen action
        """
        state_key = self._state_to_key(state)
        valid_actions = self.env.get_valid_actions()

        # If no valid actions, return an invalid action
        if not valid_actions:
            return -1, -1

        # Exploration: choose random action
        if np.random.random() < self.exploration_rate:
            index = np.random.choice(len(valid_actions))
            return valid_actions[index]

        # Exploitation: choose best action based on Q-values
        q_values = {action: self.q_table[state_key][action] for action in valid_actions}

        # If all Q-values are the same (e.g., all 0), choose randomly
        if len(set(q_values.values())) == 1:
            index = np.random.choice(len(valid_actions))
            return valid_actions[index]

        # Otherwise, choose the action with the highest Q-value
        return max(q_values.items(), key=lambda x: x[1])[0]

    def learn(self, episode_history):
        """Update Q-values for all steps in the episode"""
        final_reward = episode_history[-1][2]  # Reward from final step

        # Reverse update to propagate final reward back
        for t in reversed(range(len(episode_history))):
            state, action, _, done = episode_history[t]
            state_key = self._state_to_key(state)
            action = tuple(action)

            next_max = 0
            if not done and t < len(episode_history) - 1:
                next_state = episode_history[t + 1][0]
                next_valid = self.env.get_valid_actions()
                next_max = max(
                    (
                        self.q_table[self._state_to_key(next_state)][a]
                        for a in next_valid
                    ),
                    default=0,
                )

            # Q-learning update with discounted future rewards
            self.q_table[state_key][action] += self.learning_rate * (
                final_reward * (self.discount_factor ** (len(episode_history) - t - 1))
                + self.discount_factor * next_max
                - self.q_table[state_key][action]
            )

    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate *= self.exploration_decay


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
        if (episode + 1) % 100 == 0:
            win_rate = win_history[-100:].count(1) / 100
            draw_rate = win_history[-100:].count(0) / 100
            loss_rate = win_history[-100:].count(-1) / 100
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


# Example usage
if __name__ == "__main__":
    # Create environment
    env = BoardGameEnv(board_size=4, num_players=2)

    # Create agent
    agent = QLearningAgent(env)

    # Train agent
    print("Training agent...")
    wins = train_agent(env, agent, num_episodes=50000)

    # Plot results
    plot_results(wins)

    # Test agent against random opponent
    print("\nTesting agent against random opponent...")

    test_env = BoardGameEnv(board_size=4, num_players=2)
    test_opponent = RandomAgent(test_env)

    num_test_games = 100
    wins, draws, losses = 0, 0, 0

    for _ in range(num_test_games):
        obs = test_env.reset()
        done = False

        while not done:
            # Agent's turn
            if test_env.current_player == 0:
                action = agent.act(obs)
                obs, _, done = test_env.step(action)
            # Opponent's turn
            else:
                action = test_opponent.act()
                obs, _, done = test_env.step(action)

        # Record result
        if test_env.get_winning_player() == 0:
            wins += 1
        elif test_env.get_winning_player() is not None:
            losses += 1
        else:
            draws += 1

    print(f"Results against random opponent:")
    print(f"Win Rate: {wins/num_test_games:.2f}")
    print(f"Draw Rate: {draws/num_test_games:.2f}")
    print(f"Loss Rate: {losses/num_test_games:.2f}")
