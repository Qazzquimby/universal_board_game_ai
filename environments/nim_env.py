import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from core.env_interface import EnvInterface, StateType, ActionType

# Define Nim-specific action type for clarity
NimActionType = Tuple[int, int]  # (pile_index, num_to_remove)

class NimEnv(EnvInterface):
    """
    Environment for the game of Nim.

    State representation: A list/tuple of integers representing objects in each pile.
    Action representation: A tuple (pile_index, number_to_remove).
    Standard Nim rules: Player taking the last object wins.
    """

    metadata = {"render_modes": ["human"], "name": "nim"}

    def __init__(self, initial_piles: List[int] = [3, 5, 7], num_players: int = 2):
        """
        Args:
            initial_piles: List defining the number of objects in each pile at the start.
            num_players: Number of players (typically 2 for Nim).
        """
        if num_players != 2:
            raise ValueError("Nim is currently implemented for 2 players only.")

        self.initial_piles = tuple(initial_piles) # Use tuple for immutability
        self.num_players = num_players

        # State tracking
        self.piles: Optional[np.ndarray] = None
        self.current_player: Optional[int] = None
        self.done: bool = False
        self.winner: Optional[int] = None
        self.step_count: int = 0
        self.last_action: Optional[NimActionType] = None

        self.reset()

    def reset(self) -> StateType:
        """Reset the game to the initial pile configuration."""
        self.piles = np.array(self.initial_piles, dtype=np.int32)
        self.current_player = 0
        self.done = False
        self.winner = None
        self.step_count = 0
        self.last_action = None
        return self.get_observation()

    def step(self, action: NimActionType) -> Tuple[StateType, float, bool]:
        """Take a step in the Nim game."""
        if self.is_game_over():
            # Return current state, 0 reward, and done=True if game already finished
            return self.get_observation(), 0.0, True

        pile_index, num_to_remove = action
        self.last_action = action

        # --- Validate Action ---
        if not self._is_valid_action(action):
            # Invalid move: Penalize current player, game continues (but state doesn't change)
            # Note: In a real game, an invalid move might forfeit the game.
            # Here, we penalize and let the same player try again or handle differently.
            # For simplicity now, let's assume agents only choose legal moves.
            # If an invalid move *is* passed, we could raise an error or handle as below.
            # Let's raise an error for now to enforce agent correctness.
            raise ValueError(f"Invalid action {action} for piles {self.piles.tolist()}")
            # reward = -10.0
            # return self.get_observation(), reward, self.done

        # --- Apply Action ---
        self.piles[pile_index] -= num_to_remove
        self.step_count += 1

        # --- Check Game End ---
        # Game ends if all piles are empty
        if np.sum(self.piles) == 0:
            self.done = True
            # The player who just made the move (current_player) took the last object and wins.
            self.winner = self.current_player
            reward = 1.0
        else:
            # Game continues, switch player
            self.done = False
            self.winner = None
            reward = 0.0
            self.current_player = (self.current_player + 1) % self.num_players

        return self.get_observation(), reward, self.done

    def _is_valid_action(self, action: NimActionType) -> bool:
        """Check if an action is valid given the current piles."""
        pile_index, num_to_remove = action

        # Check pile index bounds
        if not (0 <= pile_index < len(self.piles)):
            return False
        # Check number to remove bounds (must remove at least 1)
        if num_to_remove <= 0:
            return False
        # Check if pile has enough objects
        if self.piles[pile_index] < num_to_remove:
            return False

        return True

    def get_legal_actions(self) -> List[NimActionType]:
        """Get all valid actions for the current state."""
        legal_actions = []
        for i, pile_count in enumerate(self.piles):
            for num_to_remove in range(1, pile_count + 1):
                legal_actions.append((i, num_to_remove))
        return legal_actions

    def get_current_player(self) -> int:
        """Return the index of the current player."""
        return self.current_player

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        # Game is over if done flag is set (win) or if no legal moves remain (shouldn't happen in Nim unless already won)
        return self.done or np.sum(self.piles) == 0

    def get_observation(self) -> StateType:
        """Return the current state observation."""
        # Use a tuple for the core state (piles) to make it hashable for Q-learning keys
        return {
            "piles": tuple(self.piles.tolist()), # Use tuple for hashability
            "current_player": self.current_player,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "winner": self.winner,
            "done": self.done,
        }

    def get_winning_player(self) -> Optional[int]:
        """Return the winner, or None if draw/not over."""
        return self.winner # Nim doesn't have draws

    def copy(self) -> 'NimEnv':
        """Create a deep copy of the Nim environment."""
        new_env = NimEnv(list(self.initial_piles), self.num_players)
        new_env.piles = self.piles.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.step_count = self.step_count
        new_env.last_action = self.last_action
        return new_env

    def set_state(self, state: StateType) -> None:
        """Set the environment state from an observation dictionary."""
        # Ensure piles are stored as numpy array internally
        self.piles = np.array(state["piles"], dtype=np.int32)
        self.current_player = state["current_player"]
        self.step_count = state["step_count"]
        self.last_action = state["last_action"]
        self.winner = state["winner"]
        self.done = state["done"]

    def render(self, mode: str = "human") -> None:
        """Print the current piles."""
        if mode == "human":
            print(f"Step: {self.step_count}, Player Turn: {self.current_player}, Piles: {self.piles.tolist()}")
