import pickle
import random
from collections import defaultdict
from typing import List, Tuple
from pathlib import Path

import numpy as np

from environments.base import BaseEnvironment, StateType, ActionType
from core.agent_interface import Agent
from core.config import QLearningConfig, TrainingConfig

# Define data directory relative to project root
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
        self.exploration_rate = (
            config.exploration_rate
        )  # Track current exploration rate

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
        action_values = self.q_table[state_key]
        valid_actions = self.env.get_legal_actions()  # Get current legal actions

        # Filter Q-values for legal actions only
        valid_action_values = {}
        for action in valid_actions:
            # Action key is the action itself (int for connect4, tuple for Nim)
            action_key = action
            valid_action_values[action] = action_values.get(
                action_key, 0.0
            )  # Default to 0 if action not seen

        if not valid_action_values:
            print(
                "Warning: No legal actions found or no Q-values for legal actions in QLearningAgent.act"
            )
            # Fallback to random if no Q-values are known for legal actions
            return random.choice(valid_actions) if valid_actions else None

        # If all Q-values for valid actions are the same (e.g., all 0 for an unseen state), choose randomly
        if len(set(valid_action_values.values())) <= 1:
            return random.choice(valid_actions)

        # Otherwise, choose the action with the highest Q-value
        best_action = max(valid_action_values, key=valid_action_values.get)
        return best_action

    # This method is specific to QLearning's Monte Carlo update strategy
    def learn_from_episode(
        self,
        episode_history: List[Tuple[StateType, ActionType, float, bool]],
        final_reward_for_agent: float,
    ):
        """
        Update Q-values using Monte Carlo method based on the final episode reward.

        Args:
            episode_history: List of (state, action, reward, done) tuples.
            final_reward_for_agent: The final outcome (+1, -1, 0) for the agent being trained (player 0).
        """
        if not episode_history:
            return

        last_state, _, _, last_done = episode_history[-1]
        final_reward_for_agent = 0.0
        if last_done:
            winner = last_state.get(
                "winner"
            )  # Get winner from the state *before* the last action
            # This logic is likely flawed. The winner is determined *after* the move.
            # --> Using the explicit final_reward_for_agent passed by the training loop.

            # Propagate the final reward back through the episode history
            for t in reversed(range(len(episode_history))):
                state, action, _, _ = episode_history[
                    t
                ]  # reward in history tuple is ignored for MC
                state_key = self._state_to_key(state)
                # Action key is now just the action itself if it's simple (like int for connect4)
                # or tuple for complex actions (like Nim)
                action_key = action  # Assumes action is already hashable (int or tuple)

                # Monte Carlo target is the discounted final reward from this state onwards
                # Monte Carlo target is the discounted final reward from this state onwards
                target_value = final_reward_for_agent * (
                    self.config.discount_factor ** (len(episode_history) - 1 - t)
                )

                # Q(s,a) <- Q(s,a) + alpha * (Target - Q(s,a))
                old_value = self.q_table[state_key].get(
                    action_key, 0.0
                )  # Use .get for safety
                self.q_table[state_key][
                    action_key
                ] = old_value + self.config.learning_rate * (target_value - old_value)
        else:
            # Should not happen if called after a full episode
            print("Warning: QLearningAgent.learn called on incomplete episode history.")

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
                loaded_exploration = data.get(
                    "exploration_rate", self.config.exploration_rate
                )
                self.exploration_rate = max(
                    loaded_exploration, self.config.min_exploration
                )
                print(f"Agent loaded from {filepath}")
                return True
            else:
                print(f"Save file not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error loading agent from {filepath}: {e}")
            return False


# --- Training Function ---


def _play_one_qlearning_selfplay_episode(
    env: BaseEnvironment, agent: QLearningAgent
) -> Tuple[List[Tuple], float]:
    """Plays one episode of self-play, returning history and final reward for player 0."""
    obs = env.reset()
    done = False
    # Store history from player 0's perspective only
    episode_history_p0 = []
    current_trajectory = []  # Temp store for full (s, a, r, d) sequence

    while not done:
        current_player = env.get_current_player()
        state = obs  # Current observation before action

        # Agent acts using its policy (epsilon-greedy)
        action = agent.act(state)
        if action is None:
            print(
                f"Warning: Agent returned None action in state: {state}. Ending episode."
            )
            # Penalize the player who couldn't move? Assume draw for now.
            done = True
            reward = 0.0
            # Need to determine winner based on game rules if one player fails to move
            winner = 1 - current_player  # Assume opponent wins if player fails
            break  # Exit loop

        next_obs, reward, done = env.step(action)

        # Store step from the perspective of the player who acted
        current_trajectory.append(
            {
                "player": current_player,
                "state": state,
                "action": action,
                "reward": reward,  # Reward received *after* this action
                "done": done,
            }
        )

        # If the acting player was P0, store the state-action pair for learning later
        if current_player == 0:
            # Store state *before* action, action taken, reward *after* action, done *after* action
            episode_history_p0.append((state, action, reward, done))

        obs = next_obs

    # Determine final outcome for player 0
    winner = env.get_winning_player()
    if winner == 0:
        final_reward_for_agent0 = 1.0
    elif winner is not None:  # Player 1 won
        final_reward_for_agent0 = -1.0
    else:  # Draw
        final_reward_for_agent0 = 0.0

    # Ensure the last 'done' flag in the history is correct
    # If P1 made the last move ending the game, P0's history might not reflect done=True
    if episode_history_p0 and not episode_history_p0[-1][-1] and done:
        last_s, last_a, last_r, _ = episode_history_p0.pop()
        episode_history_p0.append((last_s, last_a, last_r, True))

    return episode_history_p0, final_reward_for_agent0


def run_qlearning_training(
    env: BaseEnvironment,
    agent: QLearningAgent,
    num_episodes: int,
    q_config: QLearningConfig,
    training_config: TrainingConfig,  # Pass general training config too
):
    """Train QLearning agent using self-play."""
    win_history_p0 = []  # Track outcomes for player 0
    print(f"Starting Q-Learning self-play training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Play one episode of self-play
        (
            episode_history_p0,
            final_reward_for_agent0,
        ) = _play_one_qlearning_selfplay_episode(env, agent)

        # Learn from the episode (only need history from P0's perspective)
        if episode_history_p0:
            agent.learn_from_episode(episode_history_p0, final_reward_for_agent0)

        # Record outcome for player 0
        if final_reward_for_agent0 > 0:
            outcome = 1
        elif final_reward_for_agent0 < 0:
            outcome = -1
        else:
            outcome = 0
        win_history_p0.append(outcome)

        # Decay exploration rate after each episode
        agent.exploration_rate = max(
            agent.exploration_rate * agent.config.exploration_decay,
            agent.config.min_exploration,
        )

        # Print progress occasionally
        plot_window = training_config.plot_window
        print_interval = (
            num_episodes // 20 if num_episodes >= 20 else 1
        )  # Print ~20 times
        if (episode + 1) % print_interval == 0:
            window_size = plot_window if plot_window <= episode else episode + 1
            if window_size > 0:
                win_rate = win_history_p0[-window_size:].count(1) / window_size
                draw_rate = win_history_p0[-window_size:].count(0) / window_size
                loss_rate = win_history_p0[-window_size:].count(-1) / window_size
                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"P0 Win Rate (last {window_size}): {win_rate:.2f} | "
                    f"Draw Rate: {draw_rate:.2f} | "
                    f"Loss Rate: {loss_rate:.2f} | "
                    f"Exploration: {agent.exploration_rate:.4f}"
                )
            else:
                print(
                    f"Episode {episode + 1}/{num_episodes} | Exploration: {agent.exploration_rate:.4f}"
                )

    print("Q-Learning Training complete.")
    return win_history_p0
