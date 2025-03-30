import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from temp_env import BoardGameEnv, RandomAgent  # Import your environment


class QLearningAgent:
    """
    A Q-learning agent for the board game environment.
    Uses a tabular Q-table approach for simplicity.
    """

    def __init__(
        self,
        env: BoardGameEnv,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.99,
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

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values based on the Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Convert action to tuple if it's not already
        action = tuple(action) if not isinstance(action, tuple) else action

        # Get the best next action from next_state
        next_valid_actions = self.env.get_valid_actions()

        if done or not next_valid_actions:
            # If done or no valid actions, there's no future reward
            best_next_q = 0
        else:
            next_q_values = {
                a: self.q_table[next_state_key][a] for a in next_valid_actions
            }
            best_next_q = max(next_q_values.values()) if next_q_values else 0

        # Q-learning update rule
        self.q_table[state_key][action] += self.learning_rate * (
            reward
            + self.discount_factor * best_next_q
            - self.q_table[state_key][action]
        )

    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate *= self.exploration_decay


def train_agent(env, agent, num_episodes=1000, opponent=None):
    """
    Train the agent against an opponent (or random if none provided).

    Args:
        env: The environment
        agent: The agent to train
        num_episodes: Number of episodes to train for
        opponent: The opponent agent (uses RandomAgent if None)

    Returns:
        List of rewards per episode
    """
    if opponent is None:
        opponent = RandomAgent(env)

    rewards_history = []
    win_history = []

    for episode in range(num_episodes):
        total_reward = 0
        obs = env.reset()
        done = False

        while not done:
            # Agent's turn
            if env.current_player == 0:  # Assuming agent is player 0
                state = obs
                action = agent.act(state)
                next_obs, reward, done, info = env.step(action)

                # If the game ended with this move
                if done:
                    # Positive reward for winning, neutral for draw
                    if "winner" in info and info["winner"] == 0:
                        reward = 1.0
                    elif "draw" in info:
                        reward = 0.1

                agent.learn(state, action, reward, next_obs, done)
                total_reward += reward
                obs = next_obs

            # Opponent's turn
            else:
                action = opponent.act()
                obs, _, done, _ = env.step(action)

        # Decay exploration
        agent.decay_exploration()

        # Record results
        rewards_history.append(total_reward)

        # Record if agent won
        if "winner" in info and info["winner"] == 0:
            win_history.append(1)
        elif "winner" in info and info["winner"] != 0:
            win_history.append(-1)
        else:  # Draw
            win_history.append(0)

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

    return rewards_history, win_history


def plot_results(rewards_history, win_history, window_size=100):
    """Plot the training results."""
    # Plot rewards
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(rewards_history)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Plot smoothed rewards
    if len(rewards_history) >= window_size:
        smoothed_rewards = [
            np.mean(rewards_history[i : i + window_size])
            for i in range(len(rewards_history) - window_size + 1)
        ]
        plt.plot(range(window_size - 1, len(rewards_history)), smoothed_rewards, "r-")
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
    rewards, wins = train_agent(env, agent, num_episodes=5000)

    # Plot results
    plot_results(rewards, wins)

    # Test agent against random opponent
    print("\nTesting agent against random opponent...")

    test_env = BoardGameEnv(board_size=4, num_players=2)
    test_opponent = RandomAgent(test_env)

    num_test_games = 100
    wins, draws, losses = 0, 0, 0

    for _ in range(num_test_games):
        obs = test_env.reset()
        done = False
        info = {}

        while not done:
            # Agent's turn
            if test_env.current_player == 0:
                action = agent.act(obs)
                obs, _, done, info = test_env.step(action)
            # Opponent's turn
            else:
                action = test_opponent.act()
                obs, _, done, info = test_env.step(action)

        # Record result
        if "winner" in info:
            if info["winner"] == 0:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    print(f"Results against random opponent:")
    print(f"Win Rate: {wins/num_test_games:.2f}")
    print(f"Draw Rate: {draws/num_test_games:.2f}")
    print(f"Loss Rate: {losses/num_test_games:.2f}")
