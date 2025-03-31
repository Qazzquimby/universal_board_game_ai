import os
import pickle
import random
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np

# Use the generic EnvInterface and ActionType
from core.env_interface import EnvInterface, StateType, ActionType
from core.agent_interface import Agent


class QLearningAgent(Agent):
    """Q-learning agent adaptable to different environments."""

    def __init__(
        self,
        env: EnvInterface, # Use the interface type hint
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

    def _state_to_key(self, state: StateType) -> tuple:
        """
        Convert relevant parts of the state observation into a hashable key for the Q-table.
        Includes the primary game representation (e.g., 'board' or 'piles') and the current player.
        """
        key_parts = []
        # Add the primary game state representation (assuming it's hashable or convertible)
        if "board" in state: # For grid-based games
             # Ensure board is hashable (convert numpy array to tuple)
            board_repr = tuple(state["board"].flatten())
            key_parts.append(("board", board_repr))
        elif "piles" in state: # For Nim
            # Piles should already be a tuple in the observation
            key_parts.append(("piles", state["piles"]))
        else:
            # Fallback or error for unknown state structure
            # For now, let's try hashing items, excluding volatile ones
            print("Warning: Unknown state structure in _state_to_key. Attempting generic hashing.")
            for k, v in sorted(state.items()): # Sort for consistency
                 if k not in ["step_count", "last_action", "rewards", "done", "winner"]: # Exclude volatile keys
                    try:
                        hash(v) # Check if value is hashable
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

        return tuple(sorted(key_parts)) # Sort outer tuple for consistency

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
        if (
            np.random.rand() < self.exploration_rate
        ):  # Use np.random.rand() for standard uniform
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

    # Update action type hint in episode history
    def learn(self, episode_history: List[Tuple[StateType, ActionType, float, bool]]):
        """Update Q-values for all steps in the episode using Monte Carlo updates"""
        # This uses the final reward of the episode, suitable for sparse rewards.
        # For environments with intermediate rewards, a TD(0) update in a training loop might be better.
        if not episode_history:
            return

        final_reward = episode_history[-1][2] # Reward from final step for the agent being trained

        # Determine the player index of the agent being trained (assuming it's player 0 in the history)
        # This might need adjustment if the agent can be player 1.
        # Let's assume the history is always from player 0's perspective for now.
        agent_player_index = 0 # TODO: Make this more robust if agent can be other players

        # Determine the actual outcome for the agent being trained
        winner = self.env.get_winning_player() # Need env state at the end of episode
        # This requires the env state corresponding to the *end* of episode_history.
        # The current self.env might not be in that state.
        # Let's derive the outcome based on the final_reward relative to the player index.
        # Assuming reward is 1 for win, -1 for loss, 0 for draw for the player who made the last move recorded in history.
        # This is complex because episode_history only contains agent's moves.
        # Let's simplify: Assume final_reward passed IS the reward for the agent being trained.

        # Reverse update to propagate final reward back

        # Reverse update to propagate final reward back
        for t in reversed(range(len(episode_history))):
            state, action, _, done = episode_history[t]
            state_key = self._state_to_key(state)
            action_key = tuple(action)  # Ensure action is tuple for dict key

            # Calculate the expected value of the next state (max Q-value)
            next_state_max_q = 0.0
            if not done and t < len(episode_history) - 1:
                next_state = episode_history[t + 1][0]
                next_state_key = self._state_to_key(next_state)
                # Need valid actions for the *next* state. Use a temp env.
                temp_env = self.env.copy()
                temp_env.set_state(next_state)
                next_valid_actions = temp_env.get_legal_actions()

                if next_valid_actions:
                    next_state_max_q = max(
                        (
                            self.q_table[next_state_key][tuple(a)]
                            for a in next_valid_actions
                        ),
                        default=0.0,  # Default if no Q-values exist for next state actions
                    )

            target_value = final_reward * (
                self.discount_factor ** (len(episode_history) - 1 - t)
            )

            # Original update combined MC and TD: update_target = target_value + self.discount_factor * next_state_max_q
            # Let's stick to the pure Monte Carlo update based on final reward, which is simpler for sparse rewards.
            # Q(s,a) <- Q(s,a) + alpha * (DiscountedFinalReward - Q(s,a))
            old_value = self.q_table[state_key][action_key]
            self.q_table[state_key][action_key] = old_value + self.learning_rate * (target_value - old_value)

    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(
            self.exploration_rate * self.exploration_decay, self.min_exploration
        )

    # --- Agent Interface Methods ---

    def save(self, filepath: str) -> None:
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

    def load(self, filepath: str) -> bool:
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
                self.exploration_rate = max(self.exploration_rate, self.min_exploration)
                print(f"Agent loaded from {filepath}")
                return True
            else:
                print(f"Save file not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False
