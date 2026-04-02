from typing import List, Optional, Tuple
from copy import deepcopy

from environments.base import (
    BaseEnvironment,
    DataFrame,
    SanityCheckState,
    StateType,
    StateWithKey,
)

ActionType = Tuple[int, int]


class TicTacToe(BaseEnvironment):
    width: int = 3
    height: int = 3
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

    def _step(self, action: ActionType):
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

            return reward, True

        row, col = action
        current_player = self.get_current_player()

        # Add piece
        new_piece = DataFrame(
            [{"row": row, "col": col, "player_id": current_player}],
            columns=self.state["pieces"].columns,
        )
        self.state["pieces"] = self.state["pieces"].concat(new_piece)

        done = False
        winner = None
        reward = 0.0

        if self._check_win(current_player):
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

        return reward, done

    def _check_win(self, player_id: int) -> bool:
        """Check if the given player has won."""
        pieces = self.state["pieces"].filter(("player_id", player_id))
        if pieces.height < self.width:
            return False

        coords = set(pieces.select(["row", "col"]).rows())

        # Check rows
        for r in range(self.height):
            if all((r, c) in coords for c in range(self.width)):
                return True
        # Check columns
        for c in range(self.width):
            if all((r, c) in coords for r in range(self.height)):
                return True
        # Check diagonals
        if all((i, i) in coords for i in range(self.width)):
            return True
        if all((i, self.width - 1 - i) in coords for i in range(self.width)):
            return True

        return False

    def _get_state(self) -> StateType:
        return self.state

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print(f"Player: {self.get_current_player()}")
            board = [["·"] * self.width for _ in range(self.height)]
            if self.state and self.state["pieces"].height > 0:
                for r, c, p in self.state["pieces"].rows():
                    board[r][c] = str(p)

            for row_list in board:
                print(" " + " ".join(row_list))
            print()

    def _get_legal_actions(self) -> List[ActionType]:
        if self.is_done:
            return []

        occupied_coords = set(self.state["pieces"].select(["row", "col"]).rows())
        actions = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in occupied_coords:
                    actions.append((r, c))
        return actions

    def get_current_player(self) -> int:
        return self.state["game"]["current_player"][0]

    def get_winning_player(self) -> Optional[int]:
        winner = self.state["game"]["winner"][0]
        return winner if winner is not None else None

    def get_network_spec(self) -> dict:
        """Returns the network specification for TicTacToe."""
        return {
            "action_space": {"components": ["row", "col"]},
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
                "winner": self.num_players,
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

    def copy(self) -> "TicTacToe":
        new_env = TicTacToe()
        new_env.set_state(deepcopy(self.state))
        return new_env

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        from environments.tictactoe.sanity import get_tictactoe_sanity_states

        return get_tictactoe_sanity_states()
