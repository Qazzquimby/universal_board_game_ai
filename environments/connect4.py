from typing import List, Optional, Iterable

from environments.base import (
    BaseEnvironment,
    StateType,
    SanityCheckState,
    ActionType,
    ActionResult,
    StateWithKey,
    Grid,
    BaseState,
    Player,
    Players,
)

ColumnActionType = int

Connect4Cell = Optional[Player]


class Connect4Board(Grid[Player]):
    width: int = 7
    height: int = 6


class Connect4State(BaseState):
    players: Players = Players(player_labels=["Y", "R"])
    board: Connect4Board = Connect4Board()


class Connect4(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.state: Connect4State = Connect4State()

    @property
    def width(self) -> int:
        return self.state.board.width

    @property
    def num_action_types(self) -> int:
        return self.width

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
        self.state = Connect4State()
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
        ), f"Connect4.step received illegal action {action}. Legal actions: {current_legal_actions}. Board:\n{self.state.board}"

        if self.state.done:
            return ActionResult(
                next_state_with_key=self.get_state_with_key(),
                reward=self.state.rewards.get(self.state.current_player.id, 0.0),
                done=True,
            )

        col = action  # Action is the column index

        assert self._is_valid_action(col)

        # Find the lowest available row in the chosen column (gravity)
        row = -1
        for r in range(self.state.board.height - 1, -1, -1):
            if self.state.board[r, col] is None:
                row = r
                break

        self.state.board[row, col] = self.state.current_player

        if self._check_win(row, col):
            self.state.done = True
            self.state.rewards[self.state.current_player.id] = 1.0
            for other_player in self.state.players.players:
                if other_player.id != self.state.current_player.id:
                    self.state.rewards[other_player.id] = -1.0

        # Store the reward for the player who just moved *before* switching player
        reward_for_acting_player = self.state.rewards.get(
            self.state.current_player.id, 0.0
        )

        if not self.state.done:
            current_player_idx = self.state.players.players.index(
                self.state.current_player
            )
            next_player_idx = (current_player_idx + 1) % len(self.state.players)
            self.state.players.current_player = self.state.players.players[
                next_player_idx
            ]

        return ActionResult(
            next_state_with_key=self.get_state_with_key(),
            reward=reward_for_acting_player,
            done=self.state.done,
        )

    def _is_valid_action(self, col: ColumnActionType) -> bool:
        """Check if dropping a piece in the column is valid."""
        if not (0 <= col < self.width):
            return False
        if self.state.board[0, col] is not None:
            return False
        return True

    # Make private methods conventionally private
    def _check_win(self, row: int, col: int) -> bool:
        """
        Check if the player who just moved to (row, col) has won.
        Only checks lines passing through the last move.
        """
        player = self.state.current_player
        board = self.state.board
        win_condition = 4

        def check_line(line: Iterable[Optional[Player]]) -> bool:
            count = 0
            for cell in line:
                if cell is not None and cell.id == player.id:
                    count += 1
                    if count >= win_condition:
                        return True
                else:
                    count = 0
            return False

        # --- Check Horizontal ---
        if check_line(board.get_row(row)):
            return True

        # --- Check Vertical ---
        if check_line(board.get_column(col)):
            return True

        # --- Check Diagonal (top-left to bottom-right) ---
        diag = [
            board[row + i, col + i]
            for i in range(-(win_condition - 1), win_condition)
            if board._is_in_bounds(col + i, row + i)
        ]
        if check_line(diag):
            return True

        # --- Check Anti-Diagonal (top-right to bottom-left) ---
        anti_diag = [
            board[row + i, col - i]
            for i in range(-(win_condition - 1), win_condition)
            if board._is_in_bounds(col - i, row + i)
        ]
        if check_line(anti_diag):
            return True

        return False  # No win found for this player on this move

    def _get_state(self) -> StateType:
        """Get the current observation of the environment."""
        return self.state.model_dump()

    # Ensure method signatures match EnvInterface
    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.

        Args:
            mode: The mode to render with
        """
        if mode == "human":
            print(f"Player: {self.state.current_player.id}")
            # Print column numbers
            print("  " + " ".join(map(str, range(self.width))))
            print(" +" + "--" * self.width + "+")
            for r in range(self.state.board.height):
                row_str = f"{r}|"  # Print row numbers
                for c in range(self.width):
                    cell = self.state.board[r, c]
                    if cell is None:
                        row_str += " ·"
                    else:
                        # Use different symbols or colors in a GUI
                        row_str += f" {cell.id}"  # Player number (0 or 1)
                print(row_str + " |")
            print(" +" + "--" * self.width + "+")
            print()

    def get_legal_actions(self) -> List[ColumnActionType]:
        """
        Get a list of all valid actions (non-full column indices).

        Returns:
            A list of valid column indices.
        """
        valid_actions = []
        for col in range(self.width):  # Iterate through columns
            # A column is a legal action if its top cell (row 0) is empty
            if self.state.board[0, col] is None:
                valid_actions.append(col)
        return valid_actions

    # This helper method isn't part of the interface, keep it if useful internally
    def is_draw(self) -> bool:
        """Return whether the game ended in a draw."""
        return self.state.done and self.state.winner is None

    # Ensure method signatures match EnvInterface
    def get_current_player(self) -> int:
        """Return the current player's number."""
        return self.state.current_player.id

    # Ensure method signatures match EnvInterface
    def close(self) -> None:
        """Close the environment."""
        pass

    # Ensure method signatures match EnvInterface
    def copy(self) -> "Connect4":
        """Create a copy of the environment"""
        new_env = Connect4()
        new_env.state = self.state.model_copy(deep=True)
        return new_env

    # Ensure method signatures match EnvInterface
    def set_state(self, state: StateType) -> None:
        """Set the environment state"""
        self.state = Connect4State.model_validate(state)

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        """Returns predefined states for sanity checking Connect4."""
        # Expected value is from the perspective of the state's current player
        states = []

        # --- State 1: Empty Board (Player 0's turn) ---
        # Outcome unclear, depends on perfect play. Use 0.0
        states.append(
            SanityCheckState(
                description="Empty board, Player 0 turn",
                state_with_key=Connect4().get_state_with_key(),
                expected_value=0.0,
                expected_action=None,
            )
        )

        # --- State 2: Player 0 can win horizontally in column 3 ---
        # Board:
        # . . . . . . .
        # . . . . . . .
        # . . . . . . .
        # . . . . . . .
        # . . . . . . .
        # 0 0 0 . 1 1 .  <- Player 0 to move
        env2 = Connect4()
        p0 = env2.players.players[0]
        p1 = env2.players.players[1]
        env2.state.board[5, 0] = p0
        env2.state.board[5, 1] = p0
        env2.state.board[5, 2] = p0
        env2.state.board[5, 4] = p1
        env2.state.board[5, 5] = p1
        env2.state.current_player = p0
        env2.state.step_count = 5
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
        # . . . . . . .
        # . . . . . . .
        # 1 . . . . . .
        # 1 . 0 . . . .
        # 1 . 0 . . . .
        # 0 . 0 . . . . <- Player 1 to move
        env3 = Connect4(width=self.width, height=self.height)
        p0 = env3.players.players[0]
        p1 = env3.players.players[1]
        env3.state.board[5, 0] = p0
        env3.state.board[4, 2] = p0
        env3.state.board[5, 2] = p0
        env3.state.board[3, 2] = p0
        env3.state.board[4, 0] = p1
        env3.state.board[3, 0] = p1
        env3.state.board[2, 0] = p1
        env3.state.current_player = p1
        env3.state.step_count = 7
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
        # . . . . . . .
        # . . . . . . .
        # . . . . . . .
        # . . . . . . 1
        # 0 0 . . . . 1
        # 0 0 . . . . 1 <- Player 0 to move
        env4 = Connect4(width=self.width, height=self.height)
        p0 = env4.players.players[0]
        p1 = env4.players.players[1]
        env4.state.board[5, 0] = p0
        env4.state.board[4, 0] = p0
        env4.state.board[5, 1] = p0
        env4.state.board[4, 1] = p0
        env4.state.board[3, 6] = p1
        env4.state.board[4, 6] = p1
        env4.state.board[5, 6] = p1
        env4.state.current_player = p0
        env4.state.step_count = 7
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
