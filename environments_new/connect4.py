from typing import List, Optional

import polars as pl

from environments_new.base import (
    ActionResult,
    BaseEnvironment,
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

    @property
    def num_action_types(self) -> int:
        return self.width

    def map_action_to_policy_index(self, action: ColumnActionType) -> Optional[int]:
        return action

    def map_policy_index_to_action(self, index: int) -> Optional[ColumnActionType]:
        return index

    def _reset(self) -> StateWithKey:
        self.state = {
            "pieces": pl.DataFrame(
                {"row": [], "col": [], "player_id": []},
                schema={"row": pl.Int8, "col": pl.Int8, "player_id": pl.Int8},
            ),
            "game": pl.DataFrame(
                {"current_player": [0], "done": [False], "winner": [None]},
                schema={
                    "current_player": pl.Int8,
                    "done": pl.Boolean,
                    "winner": pl.Int8,
                },
            ),
        }
        return self.get_state_with_key()

    def _is_done(self) -> bool:
        return self.state["game"]["done"][0]

    def _step(self, action: ColumnActionType) -> ActionResult:
        if self._is_done():
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
        pieces_in_col = self.state["pieces"].filter(pl.col("col") == col)
        row = self.height - 1 - pieces_in_col.height

        # Add piece
        new_piece = pl.DataFrame(
            [{"row": row, "col": col, "player_id": current_player}],
            schema=self.state["pieces"].schema,
        )
        self.state["pieces"] = pl.concat([self.state["pieces"], new_piece])

        done = False
        winner = None
        reward = 0.0

        player_id = self.get_current_player()
        player_coords = set(
            self.state["pieces"]
            .filter(pl.col("player_id") == player_id)
            .select(["row", "col"])
            .rows()
        )

        if self._check_win(player_coords, row=row, col=col):
            done = True
            winner = current_player
            reward = 1.0
        elif self.state["pieces"].height == self.width * self.height:
            done = True  # Draw

        game_updates = [
            pl.lit(done).alias("done"),
            pl.lit(winner, dtype=pl.Int8).alias("winner"),
        ]

        if not done:
            next_player = (current_player + 1) % self.num_players
            game_updates.append(pl.lit(next_player).alias("current_player"))

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
        num_pieces_in_col = (self.state["pieces"]["col"] == col).sum()
        if num_pieces_in_col >= self.height:
            return False
        return True

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
        if count_contiguous(0, -1) + 1 >= win_condition:
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
        if self._is_done():
            return []

        full_cols = set(
            self.state["pieces"].filter(pl.col("row") == 0)["col"].to_list()
        )
        return [c for c in range(self.width) if c not in full_cols]

    def get_current_player(self) -> int:
        return self.state["game"]["current_player"][0]

    def get_winning_player(self) -> Optional[int]:
        winner = self.state["game"]["winner"][0]
        return winner if winner is not None else None

    def copy(self) -> "Connect4":
        new_env = Connect4()
        new_env.set_state(self.state)
        return new_env

    def set_state(self, state: StateType) -> None:
        self.state = {
            k: v for k, v in state.items()
        }  # not v.clone() because all changes make new object

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        states = []

        # --- State 1: Empty Board (Player 0's turn) ---
        env1 = Connect4()
        states.append(
            SanityCheckState(
                description="Empty board, Player 0 turn",
                state_with_key=env1.get_state_with_key(),
                expected_value=0.0,
                expected_action=None,
            )
        )

        # --- State 2: Player 0 can win horizontally in column 3 ---
        env2 = Connect4()
        env2.state["pieces"] = pl.DataFrame(
            [
                (5, 0, 0),
                (5, 1, 0),
                (5, 2, 0),
                (5, 4, 1),
                (5, 5, 1),
            ],
            schema={"row": pl.Int8, "col": pl.Int8, "player_id": pl.Int8},
        )
        states.append(
            SanityCheckState(
                description="Player 0 can win horizontally (col 3)",
                state_with_key=env2.get_state_with_key(),
                expected_value=1.0,
                expected_action=3,
            )
        )

        # --- State 3: Player 1 can win vertically in column 0 ---
        env3 = Connect4()
        env3.state["pieces"] = pl.DataFrame(
            [
                (5, 0, 0),
                (4, 2, 0),
                (5, 2, 0),
                (3, 2, 0),
                (4, 0, 1),
                (3, 0, 1),
                (2, 0, 1),
            ],
            schema={"row": pl.Int8, "col": pl.Int8, "player_id": pl.Int8},
        )
        env3.state["game"] = env3.state["game"].with_columns(
            pl.lit(1).alias("current_player")
        )
        states.append(
            SanityCheckState(
                description="Player 1 can win vertically (col 0)",
                state_with_key=env3.get_state_with_key(),
                expected_value=1.0,
                expected_action=0,
            )
        )

        # --- State 4: Player 0 must block Player 1's win in column 6 ---
        env4 = Connect4()
        env4.state["pieces"] = pl.DataFrame(
            [
                (5, 0, 0),
                (4, 0, 0),
                (5, 1, 0),
                (4, 1, 0),
                (5, 6, 1),
                (4, 6, 1),
                (3, 6, 1),
            ],
            schema={"row": pl.Int8, "col": pl.Int8, "player_id": pl.Int8},
        )
        states.append(
            SanityCheckState(
                description="Player 0 must block P1 win (col 6)",
                state_with_key=env4.get_state_with_key(),
                expected_action=6,
            )
        )

        return states
