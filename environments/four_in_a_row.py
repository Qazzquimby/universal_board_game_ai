from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from core.env_interface import EnvInterface, StateType

BoardGameActionType = Tuple[int, int]


class FourInARow(EnvInterface):
    """
    A board game environment following PettingZoo-like conventions.
    Implements a Connect-4 style game on a square board.
    """

    metadata = {"render_modes": ["human"], "name": "four_in_a_row"}

    def __init__(self, board_size: int = 4, num_players: int = 2, max_steps: int = 100):
        """
        Initialize the board game environment.

        Args:
            board_size: The size of the board (board_size x board_size)
            num_players: Number of players in the game
            max_steps: Maximum number of steps before the game is considered a draw
        """
        self.board_size = board_size
        self.num_players = num_players
        self.max_steps = max_steps

        # Action space: all possible (row, col) combinations
        self.num_actions = board_size * board_size

        # State tracking
        self.board = None
        self.current_player = None
        self.done = None
        self.winner = None
        self.step_count = None
        self.last_action = None
        self.rewards = None

        self.reset()

    # Ensure method signatures match EnvInterface
    def reset(self) -> StateType:
        """
        Reset the environment to initial state.

        Returns:
            observation: The initial observation
        """
        # Initialize an empty board
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 0
        self.done = False
        self.winner = None
        self.step_count = 0
        self.last_action = None
        self.rewards = {i: 0.0 for i in range(self.num_players)}

        return self.get_observation()

    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool]:
        """
        Take a step in the environment.

        Args:
            action: A tuple (row, col) representing the position to place a piece

        Returns:
            observation: The current observation after taking the action
            reward: The reward received by the player who just acted.
            done: Boolean indicating if the game has ended.
        """
        if self.is_game_over():
            # Return current state, 0 reward for current player, and done=True
            return (
                self.get_observation(),
                self.rewards.get(self.current_player, 0.0),
                True,
            )

        row, col = action  # Assumes action is BoardGameActionType
        self.last_action = action

        # Reset rewards for all players
        self.rewards = {i: 0.0 for i in range(self.num_players)}

        # Check if action is valid
        if not self._is_valid_action(action):
            # Raise error for invalid actions, consistent with NimEnv
            raise ValueError(f"Invalid action {action} for board state.")

        # Place piece on the board
        self.board[row, col] = self.current_player + 1
        self.step_count += 1

        # Check for win
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            # Winner gets positive reward, loser gets negative reward
            self.rewards[self.current_player] = 1.0
            for other_player in range(self.num_players):
                if other_player != self.current_player:
                    self.rewards[other_player] = -1.0

        # Check for draw
        elif self.step_count >= self.max_steps or np.all(self.board != 0):
            self.done = True
            # All players get 0 reward for draw (consistent with win/loss being +/- 1)
            # Or keep 0.1 if desired, but 0 seems more standard for zero-sum games. Let's use 0.
            for player in range(self.num_players):
                self.rewards[player] = 0.0

        # Store the reward for the player who just moved *before* switching player
        reward_for_acting_player = self.rewards.get(self.current_player, 0.0)

        if not self.done:
            # Switch to next player only if game is not done
            self.current_player = (self.current_player + 1) % self.num_players

        return self.get_observation(), reward_for_acting_player, self.done

    def _is_valid_action(self, action: Tuple[int, int]) -> bool:
        """Check if an action is valid."""
        row, col = action

        # Check if position is within bounds
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False

        # Check if position is empty
        if self.board[row, col] != 0:
            return False

        return True

    # Make private methods conventionally private
    def _check_win(self, row: int, col: int) -> bool:
        """
        Check if the player who just moved to (row, col) has won.
        Only checks lines passing through the last move.
        Returns True if there's a winner, and sets self.winner.
        """
        player = self.current_player
        player_piece = player + 1
        board = self.board
        size = self.board_size
        win_condition = 4  # Connect-4 style

        # --- Check Horizontal ---
        count = 0
        for c in range(size):
            if board[row, c] == player_piece:
                count += 1
                if count >= win_condition:
                    self.winner = player
                    return True
            else:
                count = 0

        # --- Check Vertical ---
        count = 0
        for r in range(size):
            if board[r, col] == player_piece:
                count += 1
                if count >= win_condition:
                    self.winner = player
                    return True
            else:
                count = 0

        # --- Check Diagonal (top-left to bottom-right) ---
        count = 0
        # Iterate along the diagonal line passing through (row, col)
        for i in range(-(win_condition - 1), win_condition):
            r, c = row + i, col + i
            if 0 <= r < size and 0 <= c < size:
                if board[r, c] == player_piece:
                    count += 1
                    if count >= win_condition:
                        self.winner = player
                        return True
                else:
                    count = 0
            # Reset count if we hit the edge or a different piece *within the check range*
            # This handles cases where the winning line starts/ends near the check point
            if not (0 <= r < size and 0 <= c < size) or board[r, c] != player_piece:
                count = 0  # Reset if out of bounds or wrong piece

        # --- Check Anti-Diagonal (top-right to bottom-left) ---
        count = 0
        # Iterate along the anti-diagonal line passing through (row, col)
        for i in range(-(win_condition - 1), win_condition):
            r, c = row + i, col - i
            if 0 <= r < size and 0 <= c < size:
                if board[r, c] == player_piece:
                    count += 1
                    if count >= win_condition:
                        self.winner = player
                        return True
                else:
                    count = 0
            # Reset count if we hit the edge or a different piece *within the check range*
            if not (0 <= r < size and 0 <= c < size) or board[r, c] != player_piece:
                count = 0  # Reset if out of bounds or wrong piece

        return False  # No win found for this player on this move

    # Ensure method signatures match EnvInterface
    def get_observation(self) -> StateType:
        """Get the current observation of the environment."""
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "rewards": self.rewards.copy(),
            "winner": self.winner,
            "done": self.done,
            # Add legal actions to observation? Optional, but can be useful for some agents.
            # "legal_actions": self.get_legal_actions()
        }

    # Ensure method signatures match EnvInterface
    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.

        Args:
            mode: The mode to render with
        """
        if mode == "human":
            print(f"Step: {self.step_count}, Player: {self.current_player + 1}")
            for row in range(self.board_size):
                row_str = ""
                for col in range(self.board_size):
                    cell = self.board[row, col]
                    if cell == 0:
                        row_str += "· "
                    else:
                        row_str += f"{cell} "
                print(row_str)
            print()

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """
        Get a list of all valid actions.

        Returns:
            A list of valid actions as (row, col) tuples
        """
        valid_actions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:
                    valid_actions.append((row, col))
        return valid_actions

    # Ensure method signatures match EnvInterface
    def is_game_over(self) -> bool:
        """Return whether the game is over."""
        # Game is over if done flag is set. Check for no legal actions might be redundant if done is set correctly.
        return self.done

    # Ensure method signatures match EnvInterface
    def get_winning_player(self) -> Optional[int]:
        """Return the winning player number, or None if no winner."""
        return self.winner

    # This helper method isn't part of the interface, keep it if useful internally
    def is_draw(self) -> bool:
        """Return whether the game ended in a draw."""
        return self.is_game_over() and self.winner is None

    # Ensure method signatures match EnvInterface
    def get_current_player(self) -> int:
        """Return the current player's number."""
        return self.current_player

    # Ensure method signatures match EnvInterface
    def close(self) -> None:
        """Close the environment."""
        pass

    # Ensure method signatures match EnvInterface
    def copy(self) -> "FourInARow":  # Or EnvInterface if we want covariance
        """Create a copy of the environment"""
        new_env = FourInARow(
            board_size=self.board_size,
            num_players=self.num_players,
            max_steps=self.max_steps,
        )
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.step_count = self.step_count
        new_env.last_action = self.last_action
        new_env.rewards = self.rewards.copy()
        return new_env

    # Ensure method signatures match EnvInterface
    def set_state(self, state: StateType) -> None:
        """Set the environment state"""
        self.board = state["board"].copy()
        self.current_player = state["current_player"]
        self.step_count = state["step_count"]
        self.last_action = state["last_action"]
        self.rewards = state["rewards"].copy()
        self.winner = state["winner"]
        self.done = state["done"]
