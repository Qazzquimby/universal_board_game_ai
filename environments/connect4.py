from enum import Enum
from typing import List, Tuple, Optional, Generic, TypeVar, Any, Iterable

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

from environments.base import (
    BaseEnvironment,
    StateType,
    SanityCheckState,
    ActionType,
    ActionResult,
    StateWithKey,
)

ColumnActionType = int

# Could make state a separate data class.
# for connect4 is just a grid.
# Grid contains player identifier and can be empty
# In this case the grid is fixed size, but we want to be able to type if it's parameterized size.

PlayerId = int


class Player(BaseModel):
    id: PlayerId
    name: str


class Players:
    def __init__(self, num_players: int, player_labels: Optional[List[str]] = None):
        if player_labels is not None and len(player_labels) != num_players:
            raise ValueError("Number of player labels must match num_players.")
        if player_labels is None:
            player_labels = [f"Player {i}" for i in range(num_players)]
        self.players = [Player(id=i, name=player_labels[i]) for i in range(num_players)]


# type that players are a cycle of 2 and have names R, Y
# Cell is an Optional[Player]
# Board is a Grid with fixed dimensions w=7, h=6

Cell_T = TypeVar("Cell_T")


class Grid(BaseModel):
    width: int
    height: int

    cells: List[List[Cell_T]] = None

    @model_validator(mode="after")
    def init_cells(self):
        if self.cells is None:
            return [[None for _ in range(self.width)] for _ in range(self.height)]
        return self

    def __getitem__(self, item: Tuple[int, int]) -> Optional[int]:
        return self.cells[item[0]][item[1]]

    def __setitem__(self, key: Tuple[int, int], value: int) -> None:
        self.cells[key[0]][key[1]] = value

    def get_column(self, x: int) -> list[Optional[Cell_T]]:
        return [row[x] for row in self.cells]

    def get_row(self, y: int) -> list[Optional[Cell_T]]:
        return self.cells[y]

    def _is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def __str__(self) -> str:
        result = []
        for row in self.cells:
            result.append("".join(str(cell) if cell else "." for cell in row))
        return "\n".join(result)


class BaseState(BaseModel):
    current_player: Player
    rewards: dict[PlayerId, float] = {}
    done: bool = False

    class Config:
        arbitrary_types_allowed = True


class Connect4Board(Grid):



Connect4Cell = Optional[Player]

class Connect4State(BaseState):
    board: Connect4Board


class Connect4(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.state: Optional[Connect4State] = None

        self.board = None

    def map_action_to_policy_index(self, action: ActionType) -> Optional[int]:
        return action

    def map_policy_index_to_action(self, index: int) -> Optional[ActionType]:
        return index

    def _reset(self) -> StateWithKey:
        """
        Reset the environment to initial state.

        Returns:
            observation: The initial observation
        """
        # Initialize an empty board (height x width)
        self.board = np.zeros((self._height, self._width), dtype=np.int8)
        self.current_player = 0
        self.done = False
        self.winner = None
        self.step_count = 0
        self.last_action = None
        self.rewards = {i: 0.0 for i in range(self.num_players)}

        return self.get_state_with_key()

    def _step(self, action: ColumnActionType) -> ActionResult:
        """
        Take a step in the environment by dropping a piece in a column.

        Args:
            action: The column index where the piece is dropped.

        Returns:
            observation: The current observation after taking the action
            reward: The reward received by the player who just acted.
            done: Boolean indicating if the game has ended.
        """
        # TODO return a class. Default rewards to 0 and done to False.

        current_legal_actions = self.get_legal_actions()
        assert (
            action in current_legal_actions
        ), f"Connect4.step received illegal action {action}. Legal actions: {current_legal_actions}. Board:\n{self.board}"

        if self.done:
            return ActionResult(
                next_state_with_key=self.get_state_with_key(),
                reward=self.rewards.get(self.current_player, 0.0),
                done=True,
            )

        col = action  # Action is the column index
        self.last_action = col  # Store the column chosen

        # Reset rewards for all players
        self.rewards = {i: 0.0 for i in range(self.num_players)}

        # Check if action is valid
        if not self._is_valid_action(col):
            # Raise error for invalid actions (column full or out of bounds)
            raise ValueError(f"Invalid action (column {col}) for board state.")

        # Find the lowest available row in the chosen column (gravity)
        row = -1
        for r in range(
            self._height - 1, -1, -1
        ):  # Iterate from bottom row (height-1) up to 0
            if self.board[r, col] == 0:
                row = r
                break

        # Place piece on the board at the calculated row
        self.board[row, col] = self.current_player + 1
        self.step_count += 1

        # Check for win starting from the placed piece's location (row, col)
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            # Winner gets positive reward, loser gets negative reward
            self.rewards[self.current_player] = 1.0
            for other_player in range(self.num_players):
                if other_player != self.current_player:
                    self.rewards[other_player] = -1.0

        # Check for draw
        elif self.step_count >= self._max_steps or np.all(self.board != 0):
            self.done = True
            # All players get 0 reward for draw (consistent with win/loss being +/- 1)
            # Or keep 0.1 if desired, but 0 seems more standard for zero-sum games. Let's use 0.
            for player in range(self.num_players):
                self.rewards[player] = 0.0

        # Store the reward for the player who just moved *before* switching player
        reward_for_acting_player = self.rewards.get(self.current_player, 0.0)

        self.current_player = (self.current_player + 1) % self.num_players

        return ActionResult(
            next_state_with_key=self.get_state_with_key(),
            reward=reward_for_acting_player,
            done=self.done,
        )

    def _is_valid_action(self, col: ColumnActionType) -> bool:
        """Check if dropping a piece in the column is valid."""
        # Check column bounds
        if not (0 <= col < self._width):
            return False
        # Check if the top cell (row 0) of the column is empty
        if self.board[0, col] != 0:
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
        height = self._height
        width = self._width
        win_condition = 4

        # --- Check Horizontal ---
        count = 0
        for c in range(width):  # Iterate through columns
            if board[row, c] == player_piece:
                count += 1
                if count >= win_condition:
                    self.winner = player  # Use local variable
                    return True
            else:
                count = 0

        # --- Check Vertical ---
        count = 0
        for r in range(height):  # Iterate through rows
            if board[r, col] == player_piece:
                count += 1
                if count >= win_condition:
                    self.winner = player  # Use local variable
                    return True
            else:
                count = 0

        # --- Check Diagonal (top-left to bottom-right) ---
        count = 0
        # Iterate along the diagonal line passing through (row, col)
        for i in range(-(win_condition - 1), win_condition):
            r, c = row + i, col + i
            if 0 <= r < height and 0 <= c < width:  # Check against height and width
                if board[r, c] == player_piece:
                    count += 1
                    if count >= win_condition:
                        self.winner = player  # Use local variable
                        return True
                else:
                    count = 0  # Reset count if sequence breaks
            # No need for explicit reset on out-of-bounds, loop structure handles it.
            # Reset count only if the sequence of player pieces is broken.
            elif count > 0:  # If we were in a sequence and went out of bounds
                count = 0

        # --- Check Anti-Diagonal (top-right to bottom-left) ---
        count = 0
        # Iterate along the anti-diagonal line passing through (row, col)
        for i in range(-(win_condition - 1), win_condition):
            r, c = row + i, col - i
            if 0 <= r < height and 0 <= c < width:  # Check against height and width
                if board[r, c] == player_piece:
                    count += 1
                    if count >= win_condition:
                        self.winner = player  # Use local variable
                        return True
                else:
                    count = 0  # Reset count if sequence breaks
            # No need for explicit reset on out-of-bounds, loop structure handles it.
            # Reset count only if the sequence of player pieces is broken.
            elif count > 0:  # If we were in a sequence and went out of bounds
                count = 0

        return False  # No win found for this player on this move

    def _get_state(self) -> StateType:
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

    # Ensure method signatures match EnvInterface
    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.

        Args:
            mode: The mode to render with
        """
        if mode == "human":
            print(f"Step: {self.step_count}, Player: {self.current_player + 1}")
            # Print column numbers
            print("  " + " ".join(map(str, range(self._width))))
            print(" +" + "--" * self._width + "+")
            for r in range(self._height):
                row_str = f"{r}|"  # Print row numbers
                for c in range(self._width):
                    cell = self.board[r, c]
                    if cell == 0:
                        row_str += " ·"
                    else:
                        # Use different symbols or colors in a GUI
                        row_str += f" {int(cell)}"  # Player number (1 or 2)
                print(row_str + " |")
            print(" +" + "--" * self._width + "+")
            print()

    def get_legal_actions(self) -> List[ColumnActionType]:
        """
        Get a list of all valid actions (non-full column indices).

        Returns:
            A list of valid column indices.
        """
        valid_actions = []
        for col in range(self._width):  # Iterate through columns
            # A column is a legal action if its top cell (row 0) is empty
            if self.board[0, col] == 0:
                valid_actions.append(col)
        return valid_actions

    # Ensure method signatures match EnvInterface
    def get_winning_player(self) -> Optional[int]:
        """Return the winning player number, or None if no winner."""
        return self.winner

    # This helper method isn't part of the interface, keep it if useful internally
    def is_draw(self) -> bool:
        """Return whether the game ended in a draw."""
        return self.done and self.winner is None

    # Ensure method signatures match EnvInterface
    def get_current_player(self) -> int:
        """Return the current player's number."""
        return self.current_player

    # Ensure method signatures match EnvInterface
    def close(self) -> None:
        """Close the environment."""
        pass

    # Ensure method signatures match EnvInterface
    def copy(self) -> "Connect4":
        """Create a copy of the environment"""
        new_env = Connect4(
            width=self._width,
            height=self._height,
            num_players=self._num_players,
            max_steps=self._max_steps,
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

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        """Returns predefined states for sanity checking Connect4."""
        # Expected value is from the perspective of the state's current player
        states = []

        # --- State 1: Empty Board (Player 0's turn) ---
        # Outcome unclear, depends on perfect play. Use 0.0
        states.append(
            SanityCheckState(
                description="Empty board, Player 0 turn",
                state_with_key=Connect4(
                    width=self.width, height=self.height
                ).get_state_with_key(),
                expected_value=0.0,
                expected_action=None,
            )
        )

        # --- State 2: Player 0 can win horizontally in column 3 ---
        # Board:
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0
        # 1 1 1 0 2 2 0  <- Player 0 to move
        env2 = Connect4(width=self.width, height=self.height)
        env2.board[5, 0] = 1  # P0
        env2.board[5, 1] = 1  # P0
        env2.board[5, 2] = 1  # P0
        env2.board[5, 4] = 2  # P1
        env2.board[5, 5] = 2  # P1
        env2.current_player = 0
        env2.step_count = 5
        states.append(
            SanityCheckState(
                description="Player 0 can win horizontally (col 3)",
                state_with_key=env2.get_state_with_key(),
                expected_value=1.0,
                expected_action=3,
            )
        )

        # --- State 3: Player 1 can win vertically in column 0 ---
        # Board:
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0
        # 2 0 0 0 0 0 0
        # 2 0 1 0 0 0 0
        # 2 0 1 0 0 0 0
        # 1 0 1 0 0 0 0 <- Player 1 to move
        env3 = Connect4(width=self.width, height=self.height)
        env3.board[5, 0] = 1  # P0
        env3.board[4, 2] = 1  # P0
        env3.board[5, 2] = 1  # P0
        env3.board[3, 2] = 1  # P0
        env3.board[4, 0] = 2  # P1
        env3.board[3, 0] = 2  # P1
        env3.board[2, 0] = 2  # P1
        env3.current_player = 1
        env3.step_count = 7
        states.append(
            SanityCheckState(
                description="Player 1 can win vertically (col 0)",
                state_with_key=env3.get_state_with_key(),
                expected_value=1.0,
                expected_action=0,
            )
        )

        # --- State 4: Player 0 must block Player 1's win in column 6 ---
        # Board:
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 2
        # 1 1 0 0 0 0 2
        # 1 1 0 0 0 0 2 <- Player 0 to move
        env4 = Connect4(width=self.width, height=self.height)
        env4.board[5, 0] = 1  # P0
        env4.board[4, 0] = 1  # P0
        env4.board[5, 1] = 1  # P0
        env4.board[4, 1] = 1  # P0
        env4.board[3, 6] = 2  # P1
        env4.board[4, 6] = 2  # P1
        env4.board[5, 6] = 2  # P1
        env4.current_player = 0
        env4.step_count = 7
        states.append(
            SanityCheckState(
                description="Player 0 must block P1 win (col 6)",
                state_with_key=env4.get_state_with_key(),
                # Player 0 *must* block, but doesn't guarantee a win. Outcome unclear. Use 0.0
                # Alternatively, could argue it's slightly negative as P1 forced the block? Let's use 0.0 for simplicity.
                # expected_value=0.0,  # Blocking doesn't guarantee win/loss
                expected_action=6,
            )
        )

        return states
