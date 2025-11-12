"""
Resembles 4x4 tic tac toe with different piece sizes and the ability to move
pieces already on the board, with bigger pieces covering smaller pieces.

Two colors, black and white.
Each has a *reserve* of 3 piles of 4 pieces with sizes 0-3, stacked.
Pieces always stack with larger on top of smaller. They don't need to be consecutive in size.
Pieces cannot stack on top of a piece the same size, so the maximum stack height is equal to the number of sizes.
Only the top piece of a stack can ever be moved, whether on the board or reserve. Lower pieces stay where they were.

A player can either
- move a piece from their *reserve* onto an empty space or onto one of your own pieces,
- move a piece on the board orthogonally one space

If a player ever has a 4-in-a-row at the end of their turn, they win.
Its possible for both players to have a 4-in-a-row simultaneously if a piece moves off another pieces,
in which case the currently acting player still wins.
"""
from typing import List, Optional, Union

from pydantic import BaseModel

from environments.base import (
    BaseEnvironment,
    DataFrame,
    SanityCheckState,
    StateType,
    StateWithKey,
    cached_method,
)


class MoveFromReserve(BaseModel):
    pile_index: int
    row: int
    col: int


class MoveFromBoard(BaseModel):
    from_row: int
    from_col: int
    to_row: int
    to_col: int


GobbletActionType = Union[MoveFromReserve, MoveFromBoard]


class Gobblet(BaseEnvironment):
    """
    Gobblet is a 4x4 board game where players try to get 4 of their pieces in a row.
    Pieces come in different sizes and can cover smaller pieces.
    """

    width: int = 4
    height: int = 4
    num_players: int = 2
    num_reserve_piles: int = 3

    def __init__(self):
        super().__init__()
        self.reset()

    def _reset(self) -> StateWithKey:
        reserves_data = []
        for player_id in range(self.num_players):
            for pile_index in range(self.num_reserve_piles):
                for size in range(4):  # sizes 0-3
                    reserves_data.append(
                        {
                            "player_id": player_id,
                            "pile_index": pile_index,
                            "size": size,
                        }
                    )

        self.state = {
            "pieces": DataFrame(
                columns=["row", "col", "player_id", "size"],
                indexed_columns=["row", "col", "player_id"],
            ),
            "reserves": DataFrame(
                data=reserves_data,
                columns=["player_id", "pile_index", "size"],
                indexed_columns=["player_id", "pile_index"],
            ),
            "game": DataFrame(
                data=[[0, False, None]],
                columns=["current_player", "done", "winner"],
            ),
        }
        return self.get_state_with_key()

    def _get_top_piece_at(self, row, col) -> Optional[dict]:
        return self._get_top_piece(self.state["pieces"], {"row": row, "col": col})

    def _get_top_reserve_piece(self, player_id: int, pile_index: int) -> Optional[dict]:
        return self._get_top_piece(
            self.state["reserves"],
            {"player_id": player_id, "pile_index": pile_index},
        )

    @cached_method
    def _get_top_piece(self, df: DataFrame, filters: dict) -> Optional[dict]:
        filtered_df = df.filter(filters)

        if filtered_df.is_empty():
            return None

        top_piece_row_data = None
        max_size = -1
        size_idx = filtered_df._col_to_idx["size"]

        for row_data in filtered_df._data:
            if row_data[size_idx] > max_size:
                max_size = row_data[size_idx]
                top_piece_row_data = row_data

        if top_piece_row_data is None:
            return None

        return {c: v for c, v in zip(df.columns, top_piece_row_data)}

    @cached_method
    def _get_legal_actions(self) -> List[GobbletActionType]:
        if self.is_done:
            return []

        legal_actions = []
        player = self.get_current_player()

        # Moves from reserve
        seen_sizes = set()
        for pile_index in range(self.num_reserve_piles):
            top_reserve_piece = self._get_top_reserve_piece(player, pile_index)
            if not top_reserve_piece:
                continue

            if top_reserve_piece["size"] in seen_sizes:
                continue
            seen_sizes.add(top_reserve_piece["size"])

            for row in range(self.height):
                for col in range(self.width):
                    top_board_piece = self._get_top_piece_at(row, col)
                    if top_board_piece is None:
                        legal_actions.append(
                            MoveFromReserve(pile_index=pile_index, row=row, col=col)
                        )
                    elif (
                        top_board_piece["player_id"] == player
                        and top_board_piece["size"] < top_reserve_piece["size"]
                    ):
                        legal_actions.append(
                            MoveFromReserve(pile_index=pile_index, row=row, col=col)
                        )

        # Moves from board
        player_pieces_on_board = self.state["pieces"].filter(("player_id", player))
        if not player_pieces_on_board.is_empty():
            top_pieces = {}  # (r,c) -> piece
            for r_tuple in player_pieces_on_board.rows():
                piece_dict = {
                    column: value
                    for column, value in zip(player_pieces_on_board.columns, r_tuple)
                }
                pos = (piece_dict["row"], piece_dict["col"])
                if (
                    pos not in top_pieces
                    or top_pieces[pos]["size"] < piece_dict["size"]
                ):
                    top_pieces[pos] = piece_dict

            for (from_row, from_col), piece in top_pieces.items():
                is_top_piece_overall = piece == self._get_top_piece_at(
                    from_row, from_col
                )
                if not is_top_piece_overall:
                    continue

                for (delta_row, delta_col) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    to_row, to_col = from_row + delta_row, from_col + delta_col
                    if not (0 <= to_row < self.height and 0 <= to_col < self.width):
                        continue

                    top_dest_piece = self._get_top_piece_at(to_row, to_col)
                    if top_dest_piece is None or top_dest_piece["size"] < piece["size"]:
                        legal_actions.append(
                            MoveFromBoard(
                                from_row=from_row,
                                from_col=from_col,
                                to_row=to_row,
                                to_col=to_col,
                            )
                        )
        return legal_actions

    def _step(self, action: GobbletActionType):
        player = self.get_current_player()

        if isinstance(action, MoveFromReserve):
            top_reserve_piece = self._get_top_reserve_piece(player, action.pile_index)

            # Remove from reserve
            self.state["reserves"] = self.state["reserves"].filter_out(
                {
                    "player_id": player,
                    "pile_index": action.pile_index,
                    "size": top_reserve_piece["size"],
                },
                expected_found=1,
            )

            # Add to board
            new_piece_data = [
                {
                    "row": action.row,
                    "col": action.col,
                    "player_id": player,
                    "size": top_reserve_piece["size"],
                }
            ]
            new_piece_df = DataFrame(
                data=new_piece_data, columns=self.state["pieces"].columns
            )
            self.state["pieces"] = self.state["pieces"].concat(new_piece_df)

        elif isinstance(action, MoveFromBoard):
            moving_piece = self._get_top_piece_at(action.from_row, action.from_col)

            # Remove from old position on board
            self.state["pieces"] = self.state["pieces"].filter_out(
                {
                    "row": action.from_row,
                    "col": action.from_col,
                    "player_id": moving_piece["player_id"],
                    "size": moving_piece["size"],
                },
                expected_found=1,
            )

            # Add to new position
            new_piece_data = [
                {
                    "row": action.to_row,
                    "col": action.to_col,
                    "player_id": player,
                    "size": moving_piece["size"],
                }
            ]
            new_piece_df = DataFrame(
                data=new_piece_data, columns=self.state["pieces"].columns
            )
            self.state["pieces"] = self.state["pieces"].concat(new_piece_df)
        else:
            raise TypeError(f"Unknown action type: {type(action)}")

        winner = self._check_for_winner()
        done = winner is not None
        reward = 0.0
        if done:
            if winner == player:
                reward = 1.0
            elif winner is not None:
                reward = -1.0

        game_updates = {"done": done, "winner": winner}
        if not done:
            next_player = (player + 1) % self.num_players
            game_updates["current_player"] = next_player

        self.state["game"] = self.state["game"].with_columns(game_updates)
        return reward, done

    def _check_for_winner(self) -> Optional[int]:
        board_top_pieces = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]
        for r in range(self.height):
            for c in range(self.width):
                top_piece = self._get_top_piece_at(r, c)
                if top_piece:
                    board_top_pieces[r][c] = top_piece["player_id"]

        current_player = self.get_current_player()
        other_player = (current_player + 1) % self.num_players

        def check(player):
            # Check rows
            for row_index in range(self.height):
                if all(
                    board_top_pieces[row_index][col_index] == player
                    for col_index in range(self.width)
                ):
                    return True
            # Check columns
            for col_index in range(self.width):
                if all(
                    board_top_pieces[row_index][col_index] == player
                    for row_index in range(self.height)
                ):
                    return True
            # Check diagonals
            if all(board_top_pieces[i][i] == player for i in range(self.width)):
                return True
            if all(
                board_top_pieces[i][self.width - 1 - i] == player
                for i in range(self.width)
            ):
                return True
            return False

        if check(current_player):
            return current_player  # even if both have 4, current player wins
        if check(other_player):
            return other_player

        return None

    def get_current_player(self) -> int:
        return self.state["game"]["current_player"][0]

    def _get_state(self) -> StateType:
        return self.state

    def get_winning_player(self) -> Optional[int]:
        winner = self.state["game"]["winner"][0]
        return winner if winner is not None else None

    def copy(self) -> "Gobblet":
        new_env = Gobblet()
        new_env.set_state(self.state)
        return new_env

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        from environments.gobblet.sanity import get_gobblet_sanity_states

        return get_gobblet_sanity_states()

    def _get_stack_at(self, row, col) -> List[dict]:
        pieces_at_loc = self.state["pieces"].filter({"row": row, "col": col})
        if pieces_at_loc.is_empty():
            return []

        stack_pieces = []
        for r_tuple in pieces_at_loc.rows():
            piece_dict = {
                column: value for column, value in zip(pieces_at_loc.columns, r_tuple)
            }
            stack_pieces.append(piece_dict)

        return sorted(stack_pieces, key=lambda p: p["size"])

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return

        player = self.get_current_player()
        print(f"Current player: {player}")

        # Render board
        print("Board:")

        # Get all stacks and prepare cell strings
        cell_strings = [["" for _ in range(self.width)] for _ in range(self.height)]
        max_len = 0
        for row in range(self.height):
            for col in range(self.width):
                stack = self._get_stack_at(row, col)
                if stack:
                    stack_string = ">".join(
                        [f"{p['player_id']},{p['size']}" for p in reversed(stack)]
                    )
                else:
                    stack_string = "."
                cell_strings[row][col] = stack_string
                if len(stack_string) > max_len:
                    max_len = len(stack_string)

        # Print board with padding
        for row in range(self.height):
            row_list = []
            for col in range(self.width):
                row_list.append(cell_strings[row][col].center(max_len))
            row_str = " | ".join(row_list)
            print(row_str)
            if row < self.height - 1:
                print("-" * len(row_str))

        # Render reserves
        print("\nReserves:")
        for p_id in range(self.num_players):
            p_reserves = []
            for pile_index in range(self.num_reserve_piles):
                top_piece = self._get_top_reserve_piece(p_id, pile_index)
                if top_piece:
                    p_reserves.append(f"Pile {pile_index}: size {top_piece['size']}")
                else:
                    p_reserves.append(f"Pile {pile_index}: empty")
            print(f"Player {p_id}: " + ", ".join(p_reserves))
        print()

    def get_network_spec(self) -> dict:
        """Returns the network specification for Gobblet."""
        return {
            "action_space_size": self.num_reserve_piles * self.width * self.height
            + self.width * self.height * 4,
            "action_space": {
                "types": {
                    "MoveFromReserve": ["pile_index", "row", "col"],
                    "MoveFromBoard": ["from_row", "from_col", "to_row", "to_col"],
                },
            },
            "tables": {
                "pieces": {"columns": ["row", "col", "player_id", "size"]},
                "reserves": {"columns": ["player_id", "pile_index", "size"]},
                "game": {"columns": ["current_player", "done", "winner"]},
            },
            "cardinalities": {
                "row": self.height,
                "col": self.width,
                "player_id": self.num_players,
                "size": 4,  # Sizes 0-3
                "pile_index": self.num_reserve_piles,
                "current_player": self.num_players,
                "done": 2,
                "winner": self.num_players + 1,  # including None
                # For actions
                "from_row": self.height,
                "from_col": self.width,
                "to_row": self.height,
                "to_col": self.width,
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
                else self.num_players,
                "current_player": lambda val, state: 0,
            },
        }
