import numpy as np
from typing import Dict, List, Tuple, Any


class BoardGameEnv:
    """
    A board game environment following PettingZoo-like conventions.
    Implements a Connect-4 style game on a square board.
    """

    metadata = {"render_modes": ["human"], "name": "connect4"}

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

    def reset(self) -> Dict[str, Any]:
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

        return self._get_observation()

    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool]:
        """
        Take a step in the environment.

        Args:
            action: A tuple (row, col) representing the position to place a piece

        Returns:
            observation: The current observation after taking the action
            reward: The reward for the current player
            done: Whether the episode is done
        """
        if self.is_game_over():
            return self._get_observation(), 0.0, True

        row, col = action
        self.last_action = action

        # Reset rewards for all players
        self.rewards = {i: 0.0 for i in range(self.num_players)}

        # Check if action is valid
        if not self._is_valid_action(action):
            self.rewards[self.current_player] = -10.0
            return self._get_observation(), self.rewards[self.current_player], False

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
            # All players get small positive reward for draw
            for player in range(self.num_players):
                self.rewards[player] = 0.1

        else:
            # Switch to next player
            self.current_player = (self.current_player + 1) % self.num_players

        return self._get_observation(), self.rewards[self.current_player], self.done

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

    def _check_win(self, row: int, col: int) -> bool:
        """
        Check if any player has won after placing a piece at (row, col).
        Returns True if there's a winner, and sets self.winner to the winning player.
        """
        # Check for all players
        for player in range(self.num_players):
            player_piece = player + 1

            # Check horizontal
            count = 0
            for c in range(self.board_size):
                if self.board[row, c] == player_piece:
                    count += 1
                    if count >= 4:  # Connect-4 style win condition
                        self.winner = player
                        return True
                else:
                    count = 0

            # Check vertical
            count = 0
            for r in range(self.board_size):
                if self.board[r, col] == player_piece:
                    count += 1
                    if count >= 4:
                        self.winner = player
                        return True
                else:
                    count = 0

            # Check both diagonal directions
            # Check diagonal (top-left to bottom-right)
            for r in range(self.board_size - 3):
                for c in range(self.board_size - 3):
                    if all(self.board[r + i][c + i] == player_piece for i in range(4)):
                        self.winner = player
                        return True

            # Check diagonal (top-right to bottom-left)
            for r in range(self.board_size - 3):
                for c in range(3, self.board_size):
                    if all(self.board[r + i][c - i] == player_piece for i in range(4)):
                        self.winner = player
                        return True

        return False

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation of the environment."""
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "rewards": self.rewards.copy(),
            "winner": self.winner,
            "done": self.done,
        }

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

    def is_game_over(self) -> bool:
        """Return whether the game is over."""
        return self.done or not self.get_legal_actions()

    def get_winning_player(self) -> int:
        """Return the winning player number, or None if no winner."""
        return self.winner

    def is_draw(self) -> bool:
        """Return whether the game ended in a draw."""
        return self.is_game_over() and self.winner is None

    def get_current_player(self) -> int:
        """Return the current player's number."""
        return self.current_player

    def close(self) -> None:
        """Close the environment."""
        pass

    def copy(self) -> "BoardGameEnv":
        """Create a copy of the environment"""
        new_env = BoardGameEnv(
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

    def set_state(self, state: dict) -> None:
        """Set the environment state"""
        self.board = state["board"].copy()
        self.current_player = state["current_player"]
        self.step_count = state["step_count"]
        self.last_action = state["last_action"]
        self.rewards = state["rewards"].copy()
        self.winner = state["winner"]
        self.done = state["done"]
