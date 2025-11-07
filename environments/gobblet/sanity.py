from environments.base import SanityCheckState, DataFrame
from environments.gobblet.gobblet import Gobblet, MoveFromBoard, MoveFromReserve


def get_gobblet_sanity_states():
    """Returns a list of sanity check states for the Gobblet environment."""
    states = []

    # --- State 1: Player 0 can win horizontally ---
    env1 = Gobblet()
    env1.state["reserves"] = env1.state["reserves"]
    env1.state["pieces"] = DataFrame(
        [
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 2, 0, 0),  # P0 line
            (1, 0, 1, 0),
            (1, 1, 1, 0),
            (1, 2, 1, 0),
        ],
        columns=["row", "col", "player_id", "size"],
    )
    states.append(
        SanityCheckState(
            description="Player 0 can win horizontally (0,3)",
            state_with_key=env1.get_state_with_key(),
            expected_value=1.0,
            expected_action=MoveFromReserve(pile_index=0, row=0, col=3),
        )
    )

    # --- State 2: Player 1 can win horizontally ---
    env2 = Gobblet()
    env2.state["reserves"] = env2.state["reserves"]
    env2.state["game"] = env2.state["game"].with_columns({"current_player": 1})
    env2.state["pieces"] = DataFrame(
        [
            (0, 0, 1, 0),
            (0, 1, 1, 0),
            (0, 2, 1, 0),  # P1 line
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 2, 0, 0),
        ],
        columns=["row", "col", "player_id", "size"],
    )
    states.append(
        SanityCheckState(
            description="Player 1 can win horizontally (0,3)",
            state_with_key=env2.get_state_with_key(),
            expected_value=1.0,
            expected_action=MoveFromReserve(pile_index=0, row=0, col=3),
        )
    )

    # --- State 3: Player 0 must block Player 1's win ---
    env3 = Gobblet()
    env3.state["reserves"] = env3.state["reserves"]
    env3.state["pieces"] = DataFrame(
        [
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 1, 1, 0),
            (1, 2, 1, 0),  # P1 about to win
        ],
        columns=["row", "col", "player_id", "size"],
    )
    states.append(
        SanityCheckState(
            description="Player 0 must block P1 win (1,3)",
            state_with_key=env3.get_state_with_key(),
            expected_action=MoveFromReserve(pile_index=0, row=1, col=3),
        )
    )

    # --- State 4: Player 1 must block Player 0's win ---
    env4 = Gobblet()
    env4.state["reserves"] = env4.state["reserves"]
    env4.state["game"] = env4.state["game"].with_columns({"current_player": 1})
    env4.state["pieces"] = DataFrame(
        [
            (0, 0, 1, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 2, 0, 0),  # P0 about to win
        ],
        columns=["row", "col", "player_id", "size"],
    )
    states.append(
        SanityCheckState(
            description="Player 1 must block P0 win (1,3)",
            state_with_key=env4.get_state_with_key(),
            expected_action=MoveFromReserve(pile_index=0, row=1, col=3),
        )
    )

    # --- State 5: Player 0 can win by moving piece on board ---
    env5 = Gobblet()
    env5.state["reserves"] = env5.state["reserves"].filter(("pile_index", 0))
    env5.state["pieces"] = DataFrame(
        [
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 2, 0, 0),  # P0 line
            (1, 3, 0, 1),  # P0 piece to move for win
            (1, 0, 1, 0),
            (1, 1, 1, 0),
            (1, 2, 1, 0),
        ],
        columns=["row", "col", "player_id", "size"],
    )
    states.append(
        SanityCheckState(
            description="Player 0 can win with on-board move to (0,3)",
            state_with_key=env5.get_state_with_key(),
            expected_value=1.0,
            expected_action=MoveFromBoard(from_row=1, from_col=3, to_row=0, to_col=3),
        )
    )

    # --- State 6: Player 0 can win by moving on-board and gobbling P1 piece ---
    env6 = Gobblet()
    # Empty P0 reserves to force on-board move
    env6.state["reserves"] = env6.state["reserves"].filter_out({"player_id": 0})
    env6.state["pieces"] = DataFrame(
        [
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 2, 0, 0),  # P0 about to win at (0,3)
            (0, 3, 1, 0),  # P1 piece to be gobbled
            (1, 3, 0, 1),  # P0 piece to move for win
        ],
        columns=["row", "col", "player_id", "size"],
    )
    states.append(
        SanityCheckState(
            description="Player 0 can win with on-board move to (0,3) gobbling P1 piece",
            state_with_key=env6.get_state_with_key(),
            expected_value=1.0,
            expected_action=MoveFromBoard(from_row=1, from_col=3, to_row=0, to_col=3),
        )
    )

    # --- State 7: Player 0 must block by moving from board (no reserves) ---
    env7 = Gobblet()
    # Empty P0 reserves to force on-board move
    env7.state["reserves"] = env7.state["reserves"].filter_out({"player_id": 0})
    env7.state["pieces"] = DataFrame(
        [
            (1, 0, 1, 0),
            (1, 1, 1, 0),
            (1, 2, 1, 0),  # P1 about to win at (1,3)
            (2, 3, 0, 0),  # P0 piece to move for block
        ],
        columns=["row", "col", "player_id", "size"],
    )
    states.append(
        SanityCheckState(
            description="Player 0 must block P1 win with on-board move to (1,3)",
            state_with_key=env7.get_state_with_key(),
            expected_action=MoveFromBoard(from_row=2, from_col=3, to_row=1, to_col=3),
        )
    )

    return states
