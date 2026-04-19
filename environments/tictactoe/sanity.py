from environments.base import SanityCheckState, DataFrame
from environments.tictactoe.tictactoe import TicTacToe


def get_tictactoe_sanity_states():
    states = []

    # --- State 1: Player 0 can win horizontally ---
    env1 = TicTacToe()
    env1.state["pieces"] = DataFrame(
        [
            (0, 0, 0),
            (0, 1, 0),  # P0
            (1, 1, 1),
            (2, 1, 1),  # P1
        ],
        columns=["row", "col", "player_id"],
    )
    states.append(
        SanityCheckState(
            description="Player 0 can win horizontally (0, 2)",
            state_with_key=env1.get_state_with_key(),
            expected_value=1.0,
            expected_action=(0, 2),
        )
    )

    # --- State 2: Player 1 can win diagonally ---
    env2 = TicTacToe()
    env2.state["pieces"] = DataFrame(
        [
            (0, 0, 1),
            (1, 1, 1),  # P1
            (0, 1, 0),
            (0, 2, 0),
            (2, 0, 0),  # P0
        ],
        columns=["row", "col", "player_id"],
    )
    env2.state["game"] = env2.state["game"].with_columns({"current_player": 1})
    states.append(
        SanityCheckState(
            description="Player 1 can win diagonally (2, 2)",
            state_with_key=env2.get_state_with_key(),
            expected_value=1.0,
            expected_action=(2, 2),
        )
    )

    # --- State 3: Player 0 must block Player 1's win ---
    env3 = TicTacToe()
    env3.state["pieces"] = DataFrame(
        [
            (1, 0, 1),
            (1, 1, 1),  # P1
            (0, 0, 0),
            (2, 0, 0),  # P0
        ],
        columns=["row", "col", "player_id"],
    )
    states.append(
        SanityCheckState(
            description="Player 0 must block P1 win (1, 2)",
            state_with_key=env3.get_state_with_key(),
            expected_action=(1, 2),
        )
    )

    # --- State 4: Player 1 must block Player 0's win ---
    env4 = TicTacToe()
    env4.state["pieces"] = DataFrame(
        [
            (0, 0, 0),
            (2, 0, 0),  # P0
            (1, 1, 1),
            (2, 2, 1),  # P1
        ],
        columns=["row", "col", "player_id"],
    )
    env4.state["game"] = env4.state["game"].with_columns({"current_player": 1})
    states.append(
        SanityCheckState(
            description="Player 1 must block P0 win (1, 0)",
            state_with_key=env4.get_state_with_key(),
            expected_action=(1, 0),
        )
    )

    return states
