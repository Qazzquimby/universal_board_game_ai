import random
import timeit
import numpy as np
from environments.gobblet.gobblet import Gobblet
from environments.base import DataFrame
from algorithms.mcts import MCTSNode, RandomRolloutEvaluation


def setup_initial_board():
    """Setup an empty Gobblet board."""
    return Gobblet()


def setup_mid_game_board_numpy():
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


def setup_mid_game_board_list():
    """Setup a mid-game Gobblet board for list-based DataFrame."""
    env = Gobblet()
    pieces_data = [
        {"row": 0, "col": 0, "player_id": 0, "size": 1},
        {"row": 1, "col": 1, "player_id": 1, "size": 2},
        {"row": 0, "col": 1, "player_id": 0, "size": 0},
        {"row": 1, "col": 0, "player_id": 1, "size": 0},
    ]
    env.state["pieces"] = DataFrame(
        data=pieces_data,
        columns=["row", "col", "player_id", "size"],
        indexed_columns=["row", "col", "player_id"],
    )

    # Manually construct reserves assuming used pieces are from pile 0.
    # Player 0 used sizes 0 and 1. Player 1 used sizes 0 and 2.
    reserves_data = []
    # Player 0 reserves
    reserves_data.extend(
        [
            {"player_id": 0, "pile_index": 0, "size": 2},
            {"player_id": 0, "pile_index": 0, "size": 3},
        ]
    )
    for pile_index in [1, 2]:
        for size in range(4):
            reserves_data.append(
                {"player_id": 0, "pile_index": pile_index, "size": size}
            )
    # Player 1 reserves
    reserves_data.extend(
        [
            {"player_id": 1, "pile_index": 0, "size": 1},
            {"player_id": 1, "pile_index": 0, "size": 3},
        ]
    )
    for pile_index in [1, 2]:
        for size in range(4):
            reserves_data.append(
                {"player_id": 1, "pile_index": pile_index, "size": size}
            )

    env.state["reserves"] = DataFrame(
        data=reserves_data,
        columns=["player_id", "pile_index", "size"],
        indexed_columns=["player_id", "pile_index"],
    )
    env.state["game"] = DataFrame(
        data=[[0, False, None]], columns=["current_player", "done", "winner"]
    )
    return env


def setup_complex_board_numpy():
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


def setup_complex_board_list():
    """Setup a complex late-game Gobblet board with stacked pieces for list-based DataFrame."""
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
        data=pieces_data,
        columns=["row", "col", "player_id", "size"],
        indexed_columns=["row", "col", "player_id"],
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
        data=reserves_data,
        columns=["player_id", "pile_index", "size"],
        indexed_columns=["player_id", "pile_index"],
    )
    env.state["game"] = DataFrame(
        data=[[0, False, None]], columns=["current_player", "done", "winner"]
    )
    return env


def profile_scenarios(df_type: str, scenarios: dict):
    """Run profiling for a given set of scenarios."""
    iterations = 1000
    print(f"--- Profiling with {df_type} DataFrame ---")

    for name, setup_func in scenarios.items():
        env = setup_func()
        print(f"\nScenario: {name}")

        # Verify correctness
        legal_actions = env.get_legal_actions()
        env._legal_actions = None
        print(f"Found {len(legal_actions)} legal actions.")

        # Time it
        time_taken = timeit.timeit(lambda: env.get_legal_actions(), number=iterations)

        print(f"Time for {iterations} calls: {time_taken:.4f} seconds")
        print(
            f"Average time per call: {time_taken / iterations * 1e6:.2f} microseconds"
        )


def profile_mcts_evaluation(df_type: str, scenarios: dict):
    """Run profiling for MCTS evaluation on a given set of scenarios."""
    random.seed(1)

    iterations = 100
    print(f"--- Profiling MCTS Evaluation with {df_type} DataFrame ---")

    evaluator = RandomRolloutEvaluation()

    for name, setup_func in scenarios.items():
        env = setup_func()
        node = MCTSNode(env.get_state_with_key())
        print(f"\nScenario: {name}")

        # Time it
        time_taken = timeit.timeit(
            lambda: evaluator.evaluate(node, env), number=iterations
        )

        print(f"Time for {iterations} calls to MCTS evaluate: {time_taken:.4f} seconds")
        print(
            f"Average time per call: {time_taken / iterations * 1e6:.2f} microseconds"
        )


if __name__ == "__main__":
    # scenarios = {
    #     "Initial Board": setup_initial_board,
    #     "Mid-Game Board": setup_mid_game_board_list,
    #     "Complex Board": setup_complex_board_list,
    # }
    # profile_scenarios("list-based", scenarios_list)

    # numpy
    scenarios = {
        "Initial Board": setup_initial_board,
        "Mid-Game Board": setup_mid_game_board_numpy,
        "Complex Board": setup_complex_board_numpy,
    }

    print("\n" + "-" * 20 + "\n")
    profile_mcts_evaluation("list-based", scenarios)

    print("\n" + "=" * 20 + "\n")

    # To run numpy-based comparison, you need to have the numpy version of DataFrame
    # in environments/base.py. The current setup functions will likely fail otherwise.
    # scenarios_numpy = {
    #     "Initial Board": setup_initial_board,
    #     "Mid-Game Board": setup_mid_game_board_numpy,
    #     "Complex Board": setup_complex_board_numpy,
    # }
    # profile_scenarios("numpy-based", scenarios_numpy)
