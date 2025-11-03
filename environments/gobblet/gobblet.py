"""
Resembles 4x4 tic tac toe with different piece sizes and the ability to move
pieces already on the board, with bigger pieces covering smaller pieces.

Two colors, black and white.
Each has a *reserve* of 3 piles of 4 pieces with sizes 1-4, stacked.
Pieces always stack with larger on top of smaller. They don't need to be consecutive in size.
Only the top piece of a stack can ever be moved, whether on the board or reserve. Lower pieces stay where they were.

A player can either
- move a piece from their *reserve* onto an empty space or onto one of your own pieces,
- move a piece on the board orthogonally one space

If a player ever has a 4-in-a-row at the end of their turn, they win.
Its possible for both players to have a 4-in-a-row simultaneously if a piece moves off another pieces,
in which case the currently acting player still wins.
"""
from copy import deepcopy
from typing import Any, List, NamedTuple, Optional, Union

import numpy as np

from environments.base import (
    ActionResult,
    BaseEnvironment,
    DataFrame,
    StateType,
    StateWithKey,
)


class MoveFromReserve(NamedTuple):
    pile_idx: int
    row: int
    col: int


class MoveFromBoard(NamedTuple):
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
            for pile_idx in range(self.num_reserve_piles):
                for i, size in enumerate(range(1, 5)):  # sizes 1-4
                    reserves_data.append(
                        {
                            "player_id": player_id,
                            "pile_idx": pile_idx,
                            "size": size,
                            "stack_level": i,
                        }
                    )

        self.state = {
            "pieces": DataFrame(
                columns=["row", "col", "player_id", "size", "stack_level"]
            ),
            "reserves": DataFrame(
                data=reserves_data,
                columns=["player_id", "pile_idx", "size", "stack_level"],
            ),
            "game": DataFrame(
                data=[[0, False, None]],
                columns=["current_player", "done", "winner"],
            ),
        }
        return self.get_state_with_key()

    def _get_top_piece_at(self, row, col) -> Optional[dict]:
        pieces_at_loc = self.state["pieces"].filter(("row", row)).filter(("col", col))
        return self.__get_top_piece_of_stack(pieces_at_loc)

    def _get_top_reserve_piece(self, player_id: int, pile_idx: int) -> Optional[dict]:
        reserve_pile = (
            self.state["reserves"]
            .filter(("player_id", player_id))
            .filter(("pile_idx", pile_idx))
        )
        return self.__get_top_piece_of_stack(reserve_pile)

    def __get_top_piece_of_stack(self, pieces_at_loc: DataFrame):
        if pieces_at_loc.is_empty():
            return None

        max_level = -1
        top_piece_row = None
        for r_tuple in pieces_at_loc.rows():
            piece_dict = {c: v for c, v in zip(pieces_at_loc.columns, r_tuple)}
            if piece_dict["stack_level"] > max_level:
                max_level = piece_dict["stack_level"]
                top_piece_row = piece_dict
        return top_piece_row

    def get_legal_actions(self) -> List[GobbletActionType]:
        if self.is_done:
            return []

        legal_actions = []
        player = self.get_current_player()

        # Moves from reserve
        for pile_idx in range(self.num_reserve_piles):
            top_reserve_piece = self._get_top_reserve_piece(player, pile_idx)
            if not top_reserve_piece:
                continue

            for r in range(self.height):
                for c in range(self.width):
                    top_board_piece = self._get_top_piece_at(r, c)
                    if top_board_piece is None:
                        legal_actions.append(MoveFromReserve(pile_idx, r, c))
                    elif (
                        top_board_piece["player_id"] == player
                        and top_board_piece["size"] < top_reserve_piece["size"]
                    ):
                        legal_actions.append(MoveFromReserve(pile_idx, r, c))

        # Moves from board
        player_pieces_on_board = self.state["pieces"].filter(("player_id", player))
        if not player_pieces_on_board.is_empty():
            top_pieces = {}  # (r,c) -> piece
            for r_tuple in player_pieces_on_board.rows():
                piece_dict = {
                    c: v for c, v in zip(player_pieces_on_board.columns, r_tuple)
                }
                pos = (piece_dict["row"], piece_dict["col"])
                if (
                    pos not in top_pieces
                    or top_pieces[pos]["stack_level"] < piece_dict["stack_level"]
                ):
                    top_pieces[pos] = piece_dict

            for (from_r, from_c), piece in top_pieces.items():
                is_top_piece_overall = piece == self._get_top_piece_at(from_r, from_c)
                if not is_top_piece_overall:
                    continue

                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    to_r, to_c = from_r + dr, from_c + dc
                    if not (0 <= to_r < self.height and 0 <= to_c < self.width):
                        continue

                    top_dest_piece = self._get_top_piece_at(to_r, to_c)
                    if top_dest_piece is None or top_dest_piece["size"] < piece["size"]:
                        legal_actions.append(MoveFromBoard(from_r, from_c, to_r, to_c))
        return legal_actions

    def _step(self, action: GobbletActionType) -> ActionResult:
        player = self.get_current_player()

        if isinstance(action, MoveFromReserve):
            top_reserve_piece = self._get_top_reserve_piece(player, action.pile_idx)

            # Remove from reserve
            reserves_df = self.state["reserves"]
            new_reserves_data = []
            for r in reserves_df._data:
                rd = {c: v for c, v in zip(reserves_df.columns, r)}
                if not (
                    rd["player_id"] == player
                    and rd["pile_idx"] == action.pile_idx
                    and rd["size"] == top_reserve_piece["size"]
                ):
                    new_reserves_data.append(r)
            self.state["reserves"] = DataFrame(
                data=new_reserves_data, columns=reserves_df.columns
            )

            # Add to board
            dest_top_piece = self._get_top_piece_at(action.row, action.col)
            new_stack_level = dest_top_piece["stack_level"] + 1 if dest_top_piece else 0

            new_piece_data = [
                {
                    "row": action.row,
                    "col": action.col,
                    "player_id": player,
                    "size": top_reserve_piece["size"],
                    "stack_level": new_stack_level,
                }
            ]
            new_piece_df = DataFrame(
                data=new_piece_data, columns=self.state["pieces"].columns
            )
            self.state["pieces"] = self.state["pieces"].concat(new_piece_df)

        elif isinstance(action, MoveFromBoard):
            moving_piece = self._get_top_piece_at(action.from_row, action.from_col)

            # Remove from old position on board
            pieces_df = self.state["pieces"]
            new_pieces_data = []
            for r in pieces_df._data:
                rd = {c: v for c, v in zip(pieces_df.columns, r)}
                if not (
                    rd["row"] == action.from_row
                    and rd["col"] == action.from_col
                    and rd["player_id"] == moving_piece["player_id"]
                    and rd["size"] == moving_piece["size"]
                    and rd["stack_level"] == moving_piece["stack_level"]
                ):
                    new_pieces_data.append(r)
            self.state["pieces"] = DataFrame(
                data=new_pieces_data, columns=pieces_df.columns
            )

            # Add to new position
            dest_top_piece = self._get_top_piece_at(action.to_row, action.to_col)
            new_stack_level = dest_top_piece["stack_level"] + 1 if dest_top_piece else 0

            new_piece_data = [
                {
                    "row": action.to_row,
                    "col": action.to_col,
                    "player_id": player,
                    "size": moving_piece["size"],
                    "stack_level": new_stack_level,
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

        return ActionResult(
            next_state_with_key=self.get_state_with_key(), reward=reward, done=done
        )

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
            for r_idx in range(self.height):
                if all(
                    board_top_pieces[r_idx][c_idx] == player
                    for c_idx in range(self.width)
                ):
                    return True
            # Check columns
            for c_idx in range(self.width):
                if all(
                    board_top_pieces[r_idx][c_idx] == player
                    for r_idx in range(self.height)
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

    def map_action_to_policy_index(self, action: GobbletActionType) -> int:
        """Maps a Gobblet action to a unique integer index."""
        if isinstance(action, MoveFromReserve):
            # 3 piles, 16 squares -> 48 actions
            return (
                action.pile_idx * self.width * self.height
                + action.row * self.width
                + action.col
            )
        elif isinstance(action, MoveFromBoard):
            # 16 squares, 4 directions -> 64 actions
            dr = action.to_row - action.from_row
            dc = action.to_col - action.from_col
            if (dr, dc) == (-1, 0):  # up
                direction = 0
            elif (dr, dc) == (0, 1):  # right
                direction = 1
            elif (dr, dc) == (1, 0):  # down
                direction = 2
            elif (dr, dc) == (0, -1):  # left
                direction = 3
            else:
                raise ValueError(f"Invalid move from board action: {action}")

            offset = self.num_reserve_piles * self.width * self.height
            from_square_idx = action.from_row * self.width + action.from_col
            return offset + from_square_idx * 4 + direction
        else:
            raise TypeError(f"Unknown action type: {type(action)}")

    def map_policy_index_to_action(self, index: int) -> GobbletActionType:
        """Maps a policy index back to a Gobblet action."""
        reserve_moves_count = self.num_reserve_piles * self.width * self.height
        if index < reserve_moves_count:
            # MoveFromReserve
            col = index % self.width
            row = (index // self.width) % self.height
            pile_idx = index // (self.width * self.height)
            return MoveFromReserve(pile_idx, row, col)
        else:
            # MoveFromBoard
            index -= reserve_moves_count
            direction = index % 4
            from_col = (index // 4) % self.width
            from_row = index // (4 * self.width)

            if direction == 0:  # up
                dr, dc = -1, 0
            elif direction == 1:  # right
                dr, dc = 0, 1
            elif direction == 2:  # down
                dr, dc = 1, 0
            else:  # direction == 3, left
                dr, dc = 0, -1

            to_row, to_col = from_row + dr, from_col + dc
            return MoveFromBoard(from_row, from_col, to_row, to_col)

    def copy(self) -> "Gobblet":
        new_env = Gobblet()
        new_env.set_state(self.state)
        return new_env

    def set_state(self, state: StateType) -> None:
        self.state = {k: v.clone() for k, v in state.items()}
        self._dirty = True

    def augment_experiences(self, experiences: List[Any]) -> List[Any]:
        augmented_experiences = []
        for exp in experiences:
            # Original experience
            augmented_experiences.append(exp)

            # --- Horizontal flip augmentation ---
            sym_exp = deepcopy(exp)

            def h_flip_action(action: GobbletActionType) -> GobbletActionType:
                if isinstance(action, MoveFromReserve):
                    return MoveFromReserve(
                        action.pile_idx, action.row, self.width - 1 - action.col
                    )
                elif isinstance(action, MoveFromBoard):
                    return MoveFromBoard(
                        action.from_row,
                        self.width - 1 - action.from_col,
                        action.to_row,
                        self.width - 1 - action.to_col,
                    )
                raise TypeError(f"Unknown action type: {type(action)}")

            # 1. Augment state
            sym_state = sym_exp.state
            pieces_df = sym_state["pieces"]
            if not pieces_df.is_empty():
                col_idx = pieces_df._col_to_idx["col"]
                for row_data in pieces_df._data:
                    row_data[col_idx] = self.width - 1 - row_data[col_idx]

            if "legal_actions" in sym_state and not sym_state["legal_actions"].is_empty():
                la_df = sym_state["legal_actions"]
                action_id_idx = la_df._col_to_idx["action_id"]
                for row_data in la_df._data:
                    policy_index = row_data[action_id_idx]
                    action = self.map_policy_index_to_action(policy_index)
                    sym_action = h_flip_action(action)
                    row_data[action_id_idx] = self.map_action_to_policy_index(
                        sym_action
                    )

            # 2. Augment policy target and legal actions on experience
            action_prob_map = {
                a: p for a, p in zip(exp.legal_actions, exp.policy_target)
            }

            new_legal_actions = [h_flip_action(a) for a in exp.legal_actions]
            new_legal_actions.sort(key=self.map_action_to_policy_index)

            # h_flip_action is its own inverse
            new_policy_target = np.array(
                [action_prob_map[h_flip_action(a)] for a in new_legal_actions]
            )

            sym_exp.legal_actions = new_legal_actions
            sym_exp.policy_target = new_policy_target

            augmented_experiences.append(sym_exp)

        return augmented_experiences

    def _get_stack_at(self, row, col) -> List[dict]:
        pieces_at_loc = self.state["pieces"].filter(("row", row)).filter(("col", col))
        if pieces_at_loc.is_empty():
            return []

        stack_pieces = []
        for r_tuple in pieces_at_loc.rows():
            piece_dict = {c: v for c, v in zip(pieces_at_loc.columns, r_tuple)}
            stack_pieces.append(piece_dict)

        return sorted(stack_pieces, key=lambda p: p["stack_level"])

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
        for r in range(self.height):
            for c in range(self.width):
                stack = self._get_stack_at(r, c)
                if stack:
                    s = ">".join(
                        [f"{p['player_id']},{p['size']}" for p in reversed(stack)]
                    )
                else:
                    s = "."
                cell_strings[r][c] = s
                if len(s) > max_len:
                    max_len = len(s)

        # Print board with padding
        for r in range(self.height):
            row_list = []
            for c in range(self.width):
                row_list.append(cell_strings[r][c].center(max_len))
            row_str = " | ".join(row_list)
            print(row_str)
            if r < self.height - 1:
                print("-" * len(row_str))

        # Render reserves
        print("\nReserves:")
        for p_id in range(self.num_players):
            p_reserves = []
            for pile_idx in range(self.num_reserve_piles):
                top_piece = self._get_top_reserve_piece(p_id, pile_idx)
                if top_piece:
                    p_reserves.append(f"Pile {pile_idx}: size {top_piece['size']}")
                else:
                    p_reserves.append(f"Pile {pile_idx}: empty")
            print(f"Player {p_id}: " + ", ".join(p_reserves))
        print()

    def get_network_spec(self) -> dict:
        """Returns the network specification for Gobblet."""
        return {
            "action_space_size": self.num_reserve_piles * self.width * self.height
            + self.width * self.height * 4,
            "tables": {
                "pieces": {
                    "columns": ["row", "col", "player_id", "size", "stack_level"]
                },
                "reserves": {
                    "columns": ["player_id", "pile_idx", "size", "stack_level"]
                },
                "game": {"columns": ["current_player", "done", "winner"]},
            },
            "cardinalities": {
                "row": self.height,
                "col": self.width,
                "player_id": self.num_players,
                "size": 5,  # Sizes 1-4
                "stack_level": 24,  # Max possible stack
                "pile_idx": self.num_reserve_piles,
                "current_player": self.num_players,
                "done": 2,
                "winner": self.num_players + 1,  # including None
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
