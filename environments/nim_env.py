from typing import List, Tuple, Optional

import numpy as np

from environments.base import (
    BaseEnvironment,
    StateType,
    SanityCheckState,
    ActionType,
)  # Import SanityCheckState, ActionType

# Define Nim-specific action type for clarity
NimActionType = Tuple[int, int]  # (pile_index, num_to_remove)


class NimEnv(BaseEnvironment):
    """
    Environment for the game of Nim.

    State representation: A list/tuple of integers representing objects in each pile.
    Action representation: A tuple (pile_index, number_to_remove).
    Standard Nim rules: Player taking the last object wins.
    """

    metadata = {"render_modes": ["human"], "name": "nim"}

    def __init__(self, initial_piles: List[int] = None, num_players: int = 2):
        """
        Args:
            initial_piles: List defining the number of objects in each pile at the start.
            num_players: Number of players (typically 2 for Nim).
        """
        if initial_piles is None:
            initial_piles = [3, 5, 7]

        if num_players != 2:
            raise ValueError("Nim is currently implemented for 2 players only.")

        self.initial_piles = tuple(initial_piles)
        self._num_players = num_players  # Use private attribute

        # State tracking
        self.piles: Optional[np.ndarray] = None
        self.current_player: Optional[int] = None
        self.done: bool = False
        self.winner: Optional[int] = None
        self.step_count: int = 0
        self.last_action: Optional[NimActionType] = None

        self.reset()

    @property
    def num_players(self) -> int:
        return self._num_players

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
            self.current_player = (self.current_player + 1) % self._num_players

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
        # Game is over if done flag is set (set correctly in step)
        return self.done

    def get_observation(self) -> StateType:
        """Return the current state observation."""
        # Use a tuple for the core state (piles) to make it hashable for Q-learning keys
        return {
            "piles": tuple(self.piles.tolist()),  # Use tuple for hashability
            "current_player": self.current_player,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "winner": self.winner,
            "done": self.done,
        }

    def get_winning_player(self) -> Optional[int]:
        """Return the winner, or None if draw/not over."""
        return self.winner  # Nim doesn't have draws

    def copy(self) -> "NimEnv":
        """Create a deep copy of the Nim environment."""
        new_env = NimEnv(list(self.initial_piles), self._num_players)
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

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        """Returns predefined states for sanity checking Nim."""
        # Expected value is from the perspective of the state's current player
        states = []

        # --- State 1: Initial State (e.g., [3, 5, 7]), Player 0 turn ---
        # Nim sum 3^5^7 = 011 ^ 101 ^ 111 = 001 != 0. Winning state for P0.
        env1 = NimEnv(list(self.initial_piles))
        # Optimal move depends on initial piles, e.g., for [3,5,7] -> (2, 1) makes Nim sum 0
        optimal_action1: Optional[ActionType] = (
            (2, 1) if self.initial_piles == (3, 5, 7) else None
        )  # Define for specific case
        states.append(
            SanityCheckState(
                description=f"Initial state {self.initial_piles}, Player 0 turn",
                state=env1.get_observation(),
                expected_value=1.0,
                expected_action=optimal_action1,
            )
        )

        # --- State 2: Simple winning state (Nim sum != 0), Player 0 turn ---
        # Piles [1, 2, 0] -> Nim sum = 1^2 = 3 != 0. Player 0 should win.
        # Optimal move: (1, 1) -> piles [1, 1, 0], Nim sum = 0
        # Nim sum 1^2^0 = 3 != 0. Winning state for P0.
        env2 = NimEnv([1, 2, 0])
        env2.current_player = 0
        states.append(
            SanityCheckState(
                description="Simple winning state [1, 2, 0], P0 turn (Optimal: (1,1))",
                state=env2.get_observation(),
                expected_value=1.0,
                expected_action=(1, 1),  # Take 1 from pile 1 (index 1)
            )
        )

        # --- State 3: Simple losing state (Nim sum == 0), Player 0 turn ---
        # Nim sum 1^1^0 = 0. Losing state for P0.
        env3 = NimEnv([1, 1, 0])
        env3.current_player = 0
        states.append(
            SanityCheckState(
                description="Simple losing state [1, 1, 0], P0 turn",
                state=env3.get_observation(),
                expected_value=-1.0,
                expected_action=None,  # Any move leads to a losing state, no single 'best' move to test
            )
        )

        # --- State 4: One pile left, Player 1 turn ---
        # Piles [0, 0, 5] -> Player 1 takes all 5 and wins.
        # Nim sum 0^0^5 = 5 != 0. Winning state for P1.
        env4 = NimEnv([0, 0, 5])
        env4.current_player = 1
        env4.step_count = 1  # Assume one move was made to get here
        states.append(
            SanityCheckState(
                description="One pile left [0, 0, 5], P1 turn (Optimal: (2,5))",
                state=env4.get_observation(),
                expected_value=1.0,
                expected_action=(2, 5),  # Take all 5 from pile 2 (index 2)
            )
        )

        return states

    def render(self, mode: str = "human") -> None:
        """Print the current piles."""
        if mode == "human":
            print(
                f"Step: {self.step_count}, Player Turn: {self.current_player}, Piles: {self.piles.tolist()}"
            )
