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
    Standard Nim rules: Player taking the last object loses.
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

    @property
    def observation_tensor_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the flattened observation tensor (piles + current_player)."""
        # Shape is 1D: (number_of_piles + 1 for current_player)
        return (len(self.initial_piles) + 1,)

    @property
    def policy_vector_size(self) -> int:
        """
        Calculates the fixed size of the policy vector.
        Maps all possible (pile_index, num_removed) actions to a flat vector.
        Size = num_piles * max_items_removable_from_any_pile.
        """
        if not self.initial_piles:
            return 0
        max_removable = max(self.initial_piles)
        num_piles = len(self.initial_piles)
        # Ensure max_removable is at least 1 if piles exist but are empty initially (edge case)
        if num_piles > 0 and max_removable == 0:
            max_removable = 1  # Allow removing 0? No, action must remove >= 1. Size should be 0 if max=0.
            # If max initial pile size is 0, no moves are possible. Policy size is 0.
            return 0
        return num_piles * max_removable

    def map_action_to_policy_index(self, action: ActionType) -> Optional[int]:
        """Maps a Nim action (pile_idx, num_removed) to a policy vector index."""
        if not isinstance(action, tuple) or len(action) != 2:
            return None  # Action must be a tuple (pile_idx, num_removed)

        pile_idx, num_removed = action
        num_piles = len(self.initial_piles)

        if not self.initial_piles:  # No piles, no actions
            return None

        max_removable = max(self.initial_piles)
        if max_removable == 0:  # No items initially, no valid actions
            return None

        # Validate the action components against the *potential* maximums
        if not (0 <= pile_idx < num_piles and 1 <= num_removed <= max_removable):
            # print(f"Warning: Action {action} is outside the bounds defined by initial_piles for index mapping.")
            return None  # Action is fundamentally invalid based on initial setup

        # Map (pile_idx, num_removed) to a flat index
        # index = pile_idx * max_removable + (num_removed - 1)
        return pile_idx * max_removable + (num_removed - 1)

    def map_policy_index_to_action(self, index: int) -> Optional[ActionType]:
        """Maps a policy vector index back to a Nim action (pile_idx, num_removed)."""
        policy_size = self.policy_vector_size
        if not (0 <= index < policy_size):
            # Index is out of the calculated bounds
            return None

        if not self.initial_piles:  # No piles, no actions
            return None
        max_removable = max(self.initial_piles)
        if max_removable == 0:  # No items initially, no valid actions
            return None

        pile_idx = index // max_removable
        num_removed = (index % max_removable) + 1

        # We don't need to check against current pile state here, just reconstruct the action
        # The caller (e.g., MCTS) should check if the reconstructed action is legal for the *current* state.
        return (pile_idx, num_removed)

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

        assert self._is_valid_action(action)

        # --- Apply Action ---
        self.piles[pile_index] -= num_to_remove
        self.step_count += 1

        # --- Check Game End ---
        # Game ends if all piles are empty
        if np.sum(self.piles) == 0:
            self.done = True
            self.winner = (self.current_player + 1) % self._num_players
        else:
            # Game continues
            self.done = False
            self.winner = None
        self.current_player = (self.current_player + 1) % self._num_players

        # reward is always 0. Terminal is determined by evaluation from winner.
        return self.get_observation(), 0, self.done

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
