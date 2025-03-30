import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import random


class BoardGameEnv:
    """
    A lightweight board game environment for reinforcement learning.
    Designed to work on Windows without dependencies on gymnasium or petting zoo.
    """

    def __init__(self, board_size: int = 8, num_players: int = 2, max_steps: int = 100):
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

        # Return the initial observation
        return self._get_observation()

    def step(
        self, action: Tuple[int, int]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: A tuple (row, col) representing the position to place a piece

        Returns:
            observation: The current observation after taking the action
            reward: The reward for the current player
            done: Whether the episode is done
            info: Additional information
        """
        if self.done:
            return self._get_observation(), 0.0, True, {"info": "Game already over"}

        row, col = action

        # Check if action is valid
        if not self._is_valid_action(action):
            return self._get_observation(), -10.0, False, {"info": "Invalid action"}

        # Place piece on the board
        self.board[row, col] = self.current_player + 1
        self.step_count += 1

        # Check for win
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
            info = {"winner": self.current_player}
        # Check for draw
        elif self.step_count >= self.max_steps or np.all(self.board != 0):
            self.done = True
            reward = 0.0
            info = {"draw": True}
        else:
            reward = 0.0
            info = {}
            # Switch to next player
            self.current_player = (self.current_player + 1) % self.num_players

        return self._get_observation(), reward, self.done, info

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
        Check if the current player has won after placing a piece at (row, col).
        This is a simplified check for demonstration purposes.
        Modify this method based on your specific game rules.
        """
        player_piece = self.current_player + 1

        # Check horizontal
        count = 0
        for c in range(self.board_size):
            if self.board[row, c] == player_piece:
                count += 1
                if count >= 4:  # For a Connect-4 style win condition
                    return True
            else:
                count = 0

        # Check vertical
        count = 0
        for r in range(self.board_size):
            if self.board[r, col] == player_piece:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Check diagonal (you can expand this based on your game rules)
        # This is a simplified version

        return False

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation of the environment."""
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "step_count": self.step_count,
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

    def get_valid_actions(self) -> List[Tuple[int, int]]:
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

    def close(self) -> None:
        """Close the environment."""
        pass


# Example of a random agent
class RandomAgent:
    """A simple random agent for demonstration."""

    def __init__(self, env: BoardGameEnv):
        self.env = env

    def act(self) -> Tuple[int, int]:
        """Choose a random valid action."""
        valid_actions = self.env.get_valid_actions()
        if valid_actions:
            return random.choice(valid_actions)
        return (-1, -1)  # Invalid action if no valid actions available


# Example usage
if __name__ == "__main__":
    # Create the environment
    env = BoardGameEnv(board_size=6, num_players=2)

    # Create agents
    agents = [RandomAgent(env), RandomAgent(env)]

    # Run a game
    obs = env.reset()
    done = False

    while not done:
        env.render()

        # Get current player's action
        current_agent = agents[env.current_player]
        action = current_agent.act()

        # Take step in environment
        obs, reward, done, info = env.step(action)

        if done:
            env.render()
            if "winner" in info:
                print(f"Player {info['winner'] + 1} wins!")
            else:
                print("It's a draw!")
