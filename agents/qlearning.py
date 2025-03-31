import os
import pickle
import random
from collections import defaultdict
from typing import List, Tuple
from pathlib import Path # Import Path

import numpy as np

from environments.env_interface import BaseEnvironment, StateType, ActionType
from core.agent_interface import Agent
from core.config import QLearningConfig # Import QLearningConfig

# Define data directory relative to this file or project root
# Assuming project root is parent of 'agents' directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class QLearningAgent(Agent):
    """Q-learning agent adaptable to different environments."""

    def __init__(self, env: BaseEnvironment, config: QLearningConfig):
        """
        Initialize the Q-learning agent.

        Args:
            env: The environment instance.
            config: Configuration object with Q-learning parameters.
        """
        self.env = env
        self.config = config
        self.exploration_rate = config.exploration_rate # Track current exploration rate

        # Initialize Q-table
        # Using defaultdict to handle new state-action pairs
        # Convert inner defaultdict to regular dict for pickling if needed,
        # but pickle should handle defaultdict directly.
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _state_to_key(self, state: StateType) -> tuple:
        """
        Convert relevant parts of the state observation into a hashable key for the Q-table.
        Includes the primary game representation (e.g., 'board' or 'piles') and the current player.
        """
        key_parts = []
        # Add the primary game state representation (assuming it's hashable or convertible)
        if "board" in state:  # For grid-based games
            # Ensure board is hashable (convert numpy array to tuple)
            board_repr = tuple(state["board"].flatten())
            key_parts.append(("board", board_repr))
        elif "piles" in state:  # For Nim
            # Piles should already be a tuple in the observation
            key_parts.append(("piles", state["piles"]))
        else:
            # Fallback or error for unknown state structure
            # For now, let's try hashing items, excluding volatile ones
            print(
                "Warning: Unknown state structure in _state_to_key. Attempting generic hashing."
            )
            for k, v in sorted(state.items()):  # Sort for consistency
                if k not in [
                    "step_count",
                    "last_action",
                    "rewards",
                    "done",
                    "winner",
                ]:  # Exclude volatile keys
                    try:
                        hash(v)  # Check if value is hashable
                        key_parts.append((k, v))
                    except TypeError:
                        # If not hashable (like a list or dict), try converting common types
                        if isinstance(v, np.ndarray):
                            key_parts.append((k, tuple(v.flatten())))
                        elif isinstance(v, list):
                            key_parts.append((k, tuple(v)))
                        # Add more conversions if needed, or skip the key part

        # Always include the current player
        key_parts.append(("current_player", state["current_player"]))

        return tuple(sorted(key_parts))  # Sort outer tuple for consistency

    def act(self, state: StateType) -> ActionType:
        """
        Choose an action using epsilon-greedy policy based on the provided state.

        Args:
            state: The current state observation

        Returns:
            The chosen action
        """
        state_key = self._state_to_key(state)

        # Create a temporary environment copy and set its state
        temp_env = self.env.copy()
        temp_env.set_state(state)

        # Get legal actions from the temporary environment
        valid_actions = temp_env.get_legal_actions()

        # If no valid actions, return None
        if not valid_actions:
            return None

        # Exploration: choose random action
        if np.random.rand() < self.exploration_rate:
            # index = np.random.choice(len(valid_actions)) # np.random.choice is fine too
            return random.choice(valid_actions)  # Use random.choice for simplicity

        # Exploitation: choose best action based on Q-values
        # Ensure actions are tuples when used as keys
        q_values = {
            tuple(action): self.q_table[state_key][tuple(action)]
            for action in valid_actions
        }

        # If all Q-values for valid actions are the same (e.g., all 0 for an unseen state), choose randomly
        if len(set(q_values.values())) <= 1:
            return random.choice(valid_actions)

        # Otherwise, choose the action with the highest Q-value
        best_action = max(q_values.items(), key=lambda item: item[1])[0]
        return best_action

    # Modify learn to accept the final reward directly for MC update
    def learn(self, episode_history: List[Tuple[StateType, ActionType, float, bool]], final_reward_for_agent: float):
        """Update Q-values using Monte Carlo method based on final episode reward."""
        if not episode_history:
            return

        # Propagate the final reward back through the episode history
        for t in reversed(range(len(episode_history))):
            state, action, _, _ = episode_history[t] # Original reward in history is ignored for MC
            state_key = self._state_to_key(state)
            action_key = tuple(action)

            # Monte Carlo target is the discounted final reward from this state onwards
            target_value = final_reward_for_agent * (
                self.config.discount_factor ** (len(episode_history) - 1 - t)
            )

            # Original update combined MC and TD: update_target = target_value + self.discount_factor * next_state_max_q
            # Let's stick to the pure Monte Carlo update based on final reward, which is simpler for sparse rewards.
            # Q(s,a) <- Q(s,a) + alpha * (DiscountedFinalReward - Q(s,a))
            old_value = self.q_table[state_key][action_key]
            self.q_table[state_key][action_key] = old_value + self.config.learning_rate * (
                target_value - old_value
            )

    # Note: decay_exploration is now handled within the training loop itself

    # --- Agent Interface Methods ---

    def _get_save_path(self) -> Path:
        """Constructs the save file path based on environment type."""
        env_type_name = type(self.env).__name__
        filename = f"q_agent_{env_type_name}.pkl"
        return DATA_DIR / filename

    def save(self) -> None:
        """Save the Q-table and exploration rate to a file in the data directory."""
        filepath = self._get_save_path()
        q_table_dict = {k: dict(v) for k, v in self.q_table.items()}
        data = {
            "q_table": q_table_dict,
            "exploration_rate": self.exploration_rate,
            # Optionally save config used for training?
            # "config": self.config
        }
        try:
            # Ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            print(f"Agent saved to {filepath}")
        except Exception as e:
            print(f"Error saving agent to {filepath}: {e}")

    def load(self) -> bool:
        """Load the Q-table and exploration rate from the data directory."""
        filepath = self._get_save_path()
        try:
            if filepath.exists():
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                self.q_table = defaultdict(
                    lambda: defaultdict(float),
                    {k: defaultdict(float, v) for k, v in data["q_table"].items()},
                )
                # Load exploration rate, but ensure it respects the current config's min value
                loaded_exploration = data.get("exploration_rate", self.config.exploration_rate)
                self.exploration_rate = max(loaded_exploration, self.config.min_exploration)
                print(f"Agent loaded from {filepath}")
                return True
            else:
                print(f"Save file not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error loading agent from {filepath}: {e}")
            return False


# --- Training Function ---
# TODO: Consider moving training logic to a dedicated Trainer class for better separation, especially if adding more complex training regimes (e.g., for AlphaZero/MuZero).
from typing import Optional # Need Optional for opponent type hint
from agents.random_agent import RandomAgent # Need RandomAgent for default opponent

def train_agent(env: BaseEnvironment, agent: QLearningAgent, num_episodes: int, q_config: QLearningConfig, opponent: Optional[Agent] = None):
    """Train QLearning agent against an opponent (defaults to Random)."""
    opponent = opponent or RandomAgent(env)
    win_history = []
    print(f"Starting Q-Learning training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_history = []

        while not done:
            current_player = env.get_current_player()

            if current_player == 0:  # Agent being trained's turn
                state = obs
                action = agent.act(state)
                if action is None:
                    # This should ideally not happen if env has valid moves
                    print(f"Warning: Agent {type(agent).__name__} returned None action in state: {state}. Skipping turn?")
                    # Decide how to handle: break, raise error, skip? For now, let's break episode.
                    break
                next_obs, reward, done = env.step(action)
                episode_history.append((state, action, reward, done))
                obs = next_obs
            else:  # Opponent's turn
                state = obs
                action = opponent.act(state)
                if action is None:
                    print(f"Warning: Opponent {type(opponent).__name__} returned None action in state: {state}. Skipping turn?")
                    break # Break episode if opponent can't move
                obs, _, done = env.step(action) # Opponent reward/history not stored for agent

        # Determine final outcome for the agent (player 0)
        winner = env.get_winning_player()
        if winner == 0:
            final_reward_for_agent = 1.0
            outcome = 1
        elif winner is not None: # Opponent won
            final_reward_for_agent = -1.0
            outcome = -1
        else: # Draw
            final_reward_for_agent = 0.0
            outcome = 0

        # Learn from the episode (using Monte Carlo update with final reward)
        if episode_history:
            # Pass the history and the final reward directly to the learn method.
            agent.learn(episode_history, final_reward_for_agent)


        win_history.append(outcome)
        # Decay exploration rate after each episode
        agent.exploration_rate = max(
            agent.exploration_rate * agent.config.exploration_decay, agent.config.min_exploration
        )

        # Print progress occasionally
        # Ensure plot_window exists in q_config, fall back if necessary
        plot_window = getattr(q_config, 'plot_window', 200) # Use getattr for safety
        print_interval = num_episodes // 20 if num_episodes >= 20 else 1 # Print ~20 times
        if (episode + 1) % print_interval == 0:
             # Use plot_window from TrainingConfig if available, else default
             window_size = plot_window if plot_window <= episode else episode + 1
             if window_size > 0:
                 win_rate = win_history[-window_size:].count(1) / window_size
                 draw_rate = win_history[-window_size:].count(0) / window_size
                 loss_rate = win_history[-window_size:].count(-1) / window_size
                 print(
                     f"Episode {episode + 1}/{num_episodes} | "
                     f"Win Rate (last {window_size}): {win_rate:.2f} | "
                     f"Draw Rate: {draw_rate:.2f} | "
                     f"Loss Rate: {loss_rate:.2f} | "
                     f"Exploration: {agent.exploration_rate:.4f}"
                 )
             else:
                  print(f"Episode {episode + 1}/{num_episodes} | Exploration: {agent.exploration_rate:.4f}")


    print("Training complete.")
    return win_history
