import os
import pickle
from collections import defaultdict

import numpy as np

from temp_env import BoardGameEnv


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
        # Convert inner defaultdict to regular dict for pickling if needed,
        # but pickle should handle defaultdict directly.
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
        valid_actions = self.env.get_legal_actions()

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
                next_valid = self.env.get_legal_actions()
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
        self.exploration_rate = max(
            self.exploration_rate * self.exploration_decay, self.min_exploration
        )

    def save(self, filepath="q_agent.pkl"):
        """Save the Q-table and exploration rate to a file."""
        # Convert defaultdict to dict for potentially better compatibility
        q_table_dict = {k: dict(v) for k, v in self.q_table.items()}
        data = {
            "q_table": q_table_dict,
            "exploration_rate": self.exploration_rate,
        }
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            print(f"Agent saved to {filepath}")
        except Exception as e:
            print(f"Error saving agent: {e}")

    def load(self, filepath="q_agent.pkl"):
        """Load the Q-table and exploration rate from a file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                # Restore defaultdict structure
                self.q_table = defaultdict(
                    lambda: defaultdict(float),
                    {k: defaultdict(float, v) for k, v in data["q_table"].items()},
                )
                self.exploration_rate = data["exploration_rate"]
                # Ensure exploration doesn't go below minimum after loading
                self.exploration_rate = max(
                    self.exploration_rate, self.min_exploration
                )
                print(f"Agent loaded from {filepath}")
                return True
            else:
                print(f"Save file not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False
