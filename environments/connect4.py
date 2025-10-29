from typing import List, Optional

from environments.base import (
    ActionResult,
    BaseEnvironment,
    DataFrame,
    SanityCheckState,
    StateType,
    StateWithKey,
)

ColumnActionType = int


class Connect4(BaseEnvironment):
    width: int = 7
    height: int = 6
    num_players: int = 2

    def __init__(self):
        super().__init__()
        self.reset()

    def _reset(self) -> StateWithKey:
        self.state = {
            "pieces": DataFrame(columns=["row", "col", "player_id"]),
            "game": DataFrame(
                data=[[0, False, None]],
                columns=["current_player", "done", "winner"],
            ),
        }
        return self.get_state_with_key()

    def _step(self, action: ColumnActionType) -> ActionResult:
        if self.is_done:
            winner = self.get_winning_player()
            current_player = self.get_current_player()
            if winner is None:
                reward = 0.0
            elif winner == current_player:
                # This should not happen in a typical game flow
                reward = 1.0
            else:
                reward = -1.0

            return ActionResult(
                next_state_with_key=self.get_state_with_key(),
                reward=reward,
                done=True,
            )

        col = action

        current_player = self.get_current_player()

        # Find the lowest available row in the chosen column
        pieces_in_col = self.state["pieces"].filter(("col", col))
        row = self.height - 1 - pieces_in_col.height
        assert row >= 0

        # Add piece
        new_piece = DataFrame(
            [{"row": row, "col": col, "player_id": current_player}],
            columns=self.state["pieces"].columns,
        )
        self.state["pieces"] = self.state["pieces"].concat(new_piece)

        done = False
        winner = None
        reward = 0.0

        player_id = self.get_current_player()
        player_coords = set(
            self.state["pieces"]
            .filter(("player_id", player_id))
            .select(["row", "col"])
            .rows()
        )

        if self._check_win(player_coords, row=row, col=col):
            done = True
            winner = current_player
            reward = 1.0
        elif self.state["pieces"].height == self.width * self.height:
            done = True  # Draw

        game_updates = {"done": done, "winner": winner}

        if not done:
            next_player = (current_player + 1) % self.num_players
            game_updates["current_player"] = next_player

        self.state["game"] = self.state["game"].with_columns(game_updates)

        return ActionResult(
            next_state_with_key=self.get_state_with_key(), reward=reward, done=done
        )

    def _is_valid_action(self, col: ColumnActionType) -> bool:
        """Check if dropping a piece in the column is valid."""
        if not (0 <= col < self.width):
            return False
        # A column is full if it has `height` pieces.
        if self.state["pieces"].is_empty():
            return True
        num_pieces_in_col = sum(v == col for v in self.state["pieces"]["col"])
        if num_pieces_in_col >= self.height:
            return False
        return True

    @staticmethod
    def check_for_winner_from_pieces(
        pieces_df: DataFrame, width: int, height: int
    ) -> Optional[int]:
        """Checks for a winner on the given board state. Returns winning player_id or None."""
        if pieces_df.is_empty():
            return None

        board = [[None] * width for _ in range(height)]
        for r, c, p in pieces_df.rows():
            board[r][c] = p

        # Check horizontal
        for r in range(height):
            for c in range(width - 3):
                if (
                    board[r][c] is not None
                    and board[r][c] == board[r][c + 1]
                    and board[r][c] == board[r][c + 2]
                    and board[r][c] == board[r][c + 3]
                ):
                    return board[r][c]

        # Check vertical
        for r in range(height - 3):
            for c in range(width):
                if (
                    board[r][c] is not None
                    and board[r][c] == board[r + 1][c]
                    and board[r][c] == board[r + 2][c]
                    and board[r][c] == board[r + 3][c]
                ):
                    return board[r][c]

        # Check diagonal (down-right)
        for r in range(height - 3):
            for c in range(width - 3):
                if (
                    board[r][c] is not None
                    and board[r][c] == board[r + 1][c + 1]
                    and board[r][c] == board[r + 2][c + 2]
                    and board[r][c] == board[r + 3][c + 3]
                ):
                    return board[r][c]

        # Check anti-diagonal (up-right)
        for r in range(3, height):
            for c in range(width - 3):
                if (
                    board[r][c] is not None
                    and board[r][c] == board[r - 1][c + 1]
                    and board[r][c] == board[r - 2][c + 2]
                    and board[r][c] == board[r - 3][c + 3]
                ):
                    return board[r][c]

        return None

    def _check_win(self, player_coords, row: int, col: int) -> bool:
        """Check if the player who just moved to (row, col) has won."""
        win_condition = 4

        def count_contiguous(dx: int, dy: int) -> int:
            count = 0
            for i in range(1, win_condition):
                r, c = row + i * dy, col + i * dx
                if (r, c) in player_coords:
                    count += 1
                else:
                    break
            return count

        # Horizontal
        if count_contiguous(1, 0) + count_contiguous(-1, 0) + 1 >= win_condition:
            return True
        # Vertical (only need to check down)
        if count_contiguous(0, 1) + 1 >= win_condition:
            return True
        # Diagonal
        if count_contiguous(1, 1) + count_contiguous(-1, -1) + 1 >= win_condition:
            return True
        # Anti-diagonal
        if count_contiguous(1, -1) + count_contiguous(-1, 1) + 1 >= win_condition:
            return True

        return False

    def _get_state(self) -> StateType:
        return self.state

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print(f"Player: {self.get_current_player()}")
            print("  " + " ".join(map(str, range(self.width))))
            print(" +" + "--" * self.width + "+")

            board = [["·"] * self.width for _ in range(self.height)]
            if self.state and self.state["pieces"].height > 0:
                for r, c, p in self.state["pieces"].rows():
                    board[r][c] = str(p)

            for r_idx, row_list in enumerate(board):
                row_str = f"{r_idx}|"
                row_str += " " + " ".join(row_list)
                print(row_str + " |")

            print(" +" + "--" * self.width + "+")
            print()

    def get_legal_actions(self) -> List[ColumnActionType]:
        if self.is_done:
            return []

        full_cols = set(self.state["pieces"].filter(("row", 0))["col"])
        return [c for c in range(self.width) if c not in full_cols]

    def get_current_player(self) -> int:
        return self.state["game"]["current_player"][0]

    def get_winning_player(self) -> Optional[int]:
        winner = self.state["game"]["winner"][0]
        return winner if winner is not None else None

    def get_network_spec(self) -> dict:
        """Returns the network specification for Connect4."""
        return {
            "action_space": {"components": ["col"]},
            "tables": {
                "pieces": {"columns": ["row", "col", "player_id"]},
                "game": {"columns": ["current_player", "done", "winner"]},
            },
            "cardinalities": {
                "row": self.height,
                "col": self.width,
                "player_id": self.num_players,
                "current_player": self.num_players,
                "done": 2,  # 0 for False, 1 for True
                "winner": self.num_players,  # 0, 1. None will be mapped to cardinality.
            },
            "transforms": {
                "player_id": lambda val, state: (
                    val - state["game"]["current_player"][0] + self.num_players
                )
                % self.num_players,
                "winner": lambda val, state: (
                    val - state["game"]["current_player"][0] + self.num_players
                )
                % self.num_players
                if val is not None
                else None,
                "current_player": lambda val, state: 0,
            },
        }

    def copy(self) -> "Connect4":
        new_env = Connect4()
        new_env.set_state(self.state)
        return new_env

    def set_state(self, state: StateType) -> None:
        self.state = {k: v.clone() for k, v in state.items()}
        self._dirty = True

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        states = []

        # # --- State 1: Empty Board (Player 0's turn) ---
        # env1 = Connect4()
        # states.append(
        #     SanityCheckState(
        #         description="Empty board, Player 0 turn",
        #         state_with_key=env1.get_state_with_key(),
        #         expected_value=0.0,
        #         expected_action=None,
        #     )
        # )

        # --- State 2: Player 0 can win horizontally in column 3 ---
        env2 = Connect4()
        env2.state["pieces"] = DataFrame(
            [
                (5, 0, 0),
                (5, 1, 0),
                (5, 2, 0),
                (5, 4, 1),
                (5, 5, 1),
            ],
            columns=["row", "col", "player_id"],
        )
        states.append(
            SanityCheckState(
                description="Player 0 can win horizontally (col 3)",
                state_with_key=env2.get_state_with_key(),
                expected_value=1.0,
                expected_action=3,
            )
        )

        # 2.5, player swapped state 2
        env2_5 = Connect4()
        env2_5.state["pieces"] = DataFrame(
            [
                (5, 0, 1),
                (5, 1, 1),
                (5, 2, 1),
                (5, 4, 0),
                (5, 5, 0),
            ],
            columns=["row", "col", "player_id"],
        )
        env2_5.state["game"] = env2_5.state["game"].with_columns({"current_player": 1})
        states.append(
            SanityCheckState(
                description="Player 1 can win horizontally (col 3)",
                state_with_key=env2_5.get_state_with_key(),
                expected_value=1.0,
                expected_action=3,
            )
        )

        # --- State 3: Player 0 can win vertically in column 0 ---
        env3 = Connect4()
        env3.state["pieces"] = DataFrame(
            [
                (5, 0, 1),
                (4, 2, 1),
                (5, 2, 1),
                (3, 2, 1),
                (4, 0, 0),
                (3, 0, 0),
                (2, 0, 0),
            ],
            columns=["row", "col", "player_id"],
        )
        states.append(
            SanityCheckState(
                description="Player 0 can win vertically (col 0)",
                state_with_key=env3.get_state_with_key(),
                expected_value=1.0,
                expected_action=0,
            )
        )

        # --- State 3.5: Player 1 can win vertically in column 0 ---
        env3_5 = Connect4()
        env3_5.state["pieces"] = DataFrame(
            [
                (5, 0, 0),
                (4, 2, 0),
                (5, 2, 0),
                (3, 2, 0),
                (4, 0, 1),
                (3, 0, 1),
                (2, 0, 1),
            ],
            columns=["row", "col", "player_id"],
        )
        env3_5.state["game"] = env3_5.state["game"].with_columns({"current_player": 1})
        states.append(
            SanityCheckState(
                description="Player 1 can win vertically (col 0)",
                state_with_key=env3_5.get_state_with_key(),
                expected_value=1.0,
                expected_action=0,
            )
        )

        # --- State 4: Player 0 must block Player 1's win in column 6 ---
        env4 = Connect4()
        env4.state["pieces"] = DataFrame(
            [
                (5, 0, 0),
                (4, 0, 0),
                (5, 1, 0),
                (4, 1, 0),
                (5, 6, 1),
                (4, 6, 1),
                (3, 6, 1),
            ],
            columns=["row", "col", "player_id"],
        )
        states.append(
            SanityCheckState(
                description="Player 0 must block P1 win (col 6)",
                state_with_key=env4.get_state_with_key(),
                expected_action=6,
            )
        )

        # --- State 4.5: Player 1 must block Player 0's win in column 6 ---
        env4_5 = Connect4()
        env4_5.state["pieces"] = DataFrame(
            [
                (5, 0, 1),
                (4, 0, 1),
                (5, 1, 1),
                (4, 1, 1),
                (5, 6, 0),
                (4, 6, 0),
                (3, 6, 0),
            ],
            columns=["row", "col", "player_id"],
        )
        env4_5.state["game"] = env4_5.state["game"].with_columns({"current_player": 1})
        states.append(
            SanityCheckState(
                description="Player 1 must block P1 win (col 6)",
                state_with_key=env4_5.get_state_with_key(),
                expected_action=6,
            )
        )

        return states
