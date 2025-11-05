import timeit
import numpy as np
from environments.gobblet.gobblet import Gobblet
from environments.base import DataFrame


def setup_initial_board():
    """Setup an empty Gobblet board."""
    return Gobblet()


def setup_mid_game_board():
    """Setup a mid-game Gobblet board."""
    env = Gobblet()
    pieces_data = [
        {"row": 0, "col": 0, "player_id": 0, "size": 1},
        {"row": 1, "col": 1, "player_id": 1, "size": 2},
        {"row": 0, "col": 1, "player_id": 0, "size": 0},
        {"row": 1, "col": 0, "player_id": 1, "size": 0},
    ]
    env.state["pieces"] = DataFrame(
        data=pieces_data, columns=["row", "col", "player_id", "size"]
    )

    # Remove used pieces from reserves. Assume they are all from pile 0.
    reserves_df = env.state["reserves"]
    pieces_to_remove = [
        (0, 0, 1),  # player 0, pile 0, size 1
        (1, 0, 2),  # player 1, pile 0, size 2
        (0, 0, 0),  # player 0, pile 0, size 0
        (1, 0, 0),  # player 1, pile 0, size 0
    ]

    mask = np.ones(reserves_df.height, dtype=bool)
    for p_id, p_idx, s in pieces_to_remove:
        piece_mask = (
            (reserves_df["player_id"] == p_id)
            & (reserves_df["pile_index"] == p_idx)
            & (reserves_df["size"] == s)
        )
        indices_to_remove = np.where(piece_mask)[0]
        if len(indices_to_remove) > 0:
            mask[indices_to_remove[0]] = False

    env.state["reserves"] = reserves_df[mask]
    env.state["game"] = DataFrame(
        data=[[0, False, None]], columns=["current_player", "done", "winner"]
    )
    return env


def setup_complex_board():
    """Setup a complex late-game Gobblet board with stacked pieces."""
    env = Gobblet()
    pieces_data = [
        {"row": 0, "col": 0, "player_id": 0, "size": 3},
        {"row": 0, "col": 0, "player_id": 1, "size": 2},
        {"row": 1, "col": 1, "player_id": 1, "size": 3},
        {"row": 1, "col": 1, "player_id": 0, "size": 1},
        {"row": 2, "col": 2, "player_id": 0, "size": 2},
        {"row": 3, "col": 3, "player_id": 1, "size": 1},
        {"row": 0, "col": 1, "player_id": 0, "size": 0},
    ]
    env.state["pieces"] = DataFrame(
        data=pieces_data, columns=["row", "col", "player_id", "size"]
    )

    # Pieces used by player 0: {0, 1, 2, 3} (one of each size)
    # Pieces used by player 1: {1, 2, 3}

    # Construct reserves with remaining pieces
    reserves_data = []
    # Player 0 has 2 full piles left
    for pile_index in range(2):
        for size in range(4):
            reserves_data.append(
                {"player_id": 0, "pile_index": pile_index, "size": size}
            )

    # Player 1 has 2 full piles and one pile with size 0
    for pile_index in range(2):
        for size in range(4):
            reserves_data.append(
                {"player_id": 1, "pile_index": pile_index, "size": size}
            )
    reserves_data.append({"player_id": 1, "pile_index": 2, "size": 0})

    env.state["reserves"] = DataFrame(
        data=reserves_data, columns=["player_id", "pile_index", "size"]
    )
    env.state["game"] = DataFrame(
        data=[[0, False, None]], columns=["current_player", "done", "winner"]
    )
    return env


if __name__ == "__main__":
    iterations = 1000

    scenarios = {
        "Initial Board": setup_initial_board,
        "Mid-Game Board": setup_mid_game_board,
        "Complex Board": setup_complex_board,
    }

    for name, setup_func in scenarios.items():
        env = setup_func()
        print(f"--- Profiling: {name} ---")

        # Verify correctness
        legal_actions = env.get_legal_actions()
        print(f"Found {len(legal_actions)} legal actions.")

        # Time it
        time_taken = timeit.timeit(lambda: env.get_legal_actions(), number=iterations)

        print(f"Time for {iterations} calls: {time_taken:.4f} seconds")
        print(f"Average time per call: {time_taken / iterations * 1e6:.2f} microseconds\n")
