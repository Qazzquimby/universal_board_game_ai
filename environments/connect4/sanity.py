from environments.base import SanityCheckState, DataFrame
from environments.connect4.connect4 import Connect4


def get_connect4_sanity_states():
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
