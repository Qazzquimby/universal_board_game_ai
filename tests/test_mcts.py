import math
from typing import Optional, List

import pytest

from algorithms.mcts import (
    MCTSNode,
    UCB1Selection,
    UniformExpansion,
    RandomRolloutEvaluation,
    StandardBackpropagation,
)
from environments.connect4 import Connect4, ActionResult
from environments.base import ActionType
import numpy as np


@pytest.fixture
def connect4_env_small() -> Connect4:
    """Fixture for a small Connect4 environment (5x4)."""
    return Connect4(width=5, height=4)  # Small state space for easier testing


@pytest.fixture
def root_node() -> MCTSNode:
    """Fixture for a basic root node."""
    return MCTSNode()


def test_ucb1_selection_init():
    """Test UCB1Selection initialization."""
    selector = UCB1Selection(exploration_constant=1.41)
    assert selector.exploration_constant == 1.41
    with pytest.raises(ValueError):
        UCB1Selection(exploration_constant=-1.0)


def test_ucb1_score_child():
    """Test the _score_child method of UCB1Selection."""
    selector = UCB1Selection(exploration_constant=math.sqrt(2))  # Common C value
    parent = MCTSNode()
    parent.visit_count = 10

    child_visited = MCTSNode(parent=parent, prior=0.5)
    child_visited.visit_count = 5
    child_visited.total_value = 2.0  # Value from child's perspective = 2/5 = 0.4
    # Score = -Q(child) + C * P(child) * sqrt(log(N(parent)) / N(child))
    # Score = -0.4 + sqrt(2) * 0.5 * sqrt(log(10) / 5)
    expected_score_visited = -0.4 + math.sqrt(2) * 0.5 * math.sqrt(math.log(10) / 5)
    assert selector._score_child(child_visited, parent.visit_count) == pytest.approx(
        expected_score_visited
    )

    child_unvisited = MCTSNode(parent=parent, prior=0.5)
    child_unvisited.visit_count = 0
    # Score for unvisited should be infinity
    assert selector._score_child(child_unvisited, parent.visit_count) == float("inf")


def test_ucb1_select_unvisited_child(root_node, connect4_env_small):
    """Test that UCB1 selects an unvisited child first."""
    selector = UCB1Selection(exploration_constant=1.41)
    env = connect4_env_small
    # state = env.get_observation() # Not strictly needed here as we manually build children

    env_copy_for_setup = env.copy()
    legal_actions = env_copy_for_setup.get_legal_actions()
    num_legal_actions = len(legal_actions)
    uniform_prior = 1.0 / num_legal_actions if num_legal_actions > 0 else 0.0

    action_visited_col = legal_actions[0]
    action_unvisited_col = (
        legal_actions[1] if num_legal_actions > 1 else legal_actions[0]
    )
    # Ensure they are distinct if possible, otherwise test logic might need adjustment
    if num_legal_actions == 1:
        action_visited_col = (
            action_unvisited_col  # Special case for single legal action
        )

    total_visits_for_root = 0
    for action in legal_actions:
        child = MCTSNode(parent=root_node, prior=uniform_prior)
        if (
            action == action_unvisited_col
            and action_visited_col != action_unvisited_col
        ):
            child.visit_count = 0
        else:  # Visited
            child.visit_count = 1
            child.total_value = -1.0 if action == action_visited_col else 0.0
        root_node.children[action] = child
        total_visits_for_root += child.visit_count
    root_node.visit_count = total_visits_for_root

    selection_result = selector.select(root_node, env.copy())

    # Should select the unvisited child (action_unvisited_col)
    assert selection_result.leaf_node == root_node.children[action_unvisited_col]
    assert len(selection_result.path) == 2  # Root -> Child for action_unvisited_col
    assert selection_result.path[0] == root_node
    assert selection_result.path[1] == root_node.children[action_unvisited_col]

    # Check environment state corresponds to action_unvisited_col
    # Initial board (5x4) is all zeros. Player 0 moves.
    # Action 1 means P0 places a piece in column 1.
    # The piece (1) will be at board[3, 1] (bottom row of col 1 for a 4-row high board).
    # Current player in leaf_env will be 1.
    leaf_env_state = selection_result.leaf_env.get_state_with_key()
    assert (
        leaf_env_state["board"][env.height - 1, action_unvisited_col] == 1
    )  # Player 0's piece
    assert leaf_env_state["current_player"] == 1


def test_ucb1_select_best_score(root_node, connect4_env_small):
    """Test that UCB1 selects the child with the highest score when all are visited."""
    selector = UCB1Selection(
        exploration_constant=1.0
    )  # Simpler constant for manual check
    env = connect4_env_small  # Player 0 to move on empty 5x4 board

    # Manually expand root with two visited children (actions for Player 0)
    action_col0 = 0  # P0 places in col 0
    action_col1 = 1  # P0 places in col 1

    # Child node after P0 plays action_col0. Player 1 is to move in this child state.
    # child_col0.total_value is accumulated for Player 1.
    # child_col0.value is Q_P1(child_state_after_col0, P1_action)
    env_copy_for_setup = env.copy()
    legal_actions = env_copy_for_setup.get_legal_actions()
    num_legal_actions = len(legal_actions)
    uniform_prior = 1.0 / num_legal_actions if num_legal_actions > 0 else 0.0

    action_col0 = legal_actions[
        0
    ]  # Assumes at least two legal actions for test variety
    action_col1 = legal_actions[1] if num_legal_actions > 1 else legal_actions[0]

    total_visits_for_root = 0
    for action in legal_actions:
        child = MCTSNode(parent=root_node, prior=uniform_prior)
        if action == action_col0:
            child.visit_count = 5
            child.total_value = 1.0  # For P1. From P0's view, Q_child is -0.2
        elif action == action_col1:
            child.visit_count = 3
            child.total_value = -1.5  # For P1. From P0's view, Q_child is 0.5
        else:
            child.visit_count = 1
            child.total_value = 0.0  # For P1. From P0's view, Q_child is 0.0
        root_node.children[action] = child
        total_visits_for_root += child.visit_count
    root_node.visit_count = total_visits_for_root

    # Scores will be calculated based on these new priors and visit counts.
    # The expectation is that action_col1 still has the highest score.
    # Score = -Q_child_perspective(child_state) + C * P(child) * sqrt(log(N(parent)) / N(child))
    # Score for action_col0: - (1.0/5) + 1.0 * 0.6 * math.sqrt(math.log(8) / 5)
    #                       = -0.2    + 0.6 * math.sqrt(2.07944 / 5)
    #                       = -0.2    + 0.6 * math.sqrt(0.41588)
    #                       = -0.2    + 0.6 * 0.64489 approx -0.2 + 0.38693 = 0.18693
    # Score for action_col1: - (-1.5/3) + 1.0 * 0.4 * math.sqrt(math.log(8) / 3)
    #                       = 0.5     + 0.4 * math.sqrt(2.07944 / 3)
    #                       = 0.5     + 0.4 * math.sqrt(0.69314)
    #                       = 0.5     + 0.4 * 0.83255 approx 0.5 + 0.33302 = 0.83302

    # action_col1 should have a higher score
    selection_result = selector.select(root_node, env.copy())

    assert selection_result.leaf_node == root_node.children[action_col1]
    assert len(selection_result.path) == 2
    assert selection_result.path[1] == root_node.children[action_col1]

    # Check environment state corresponds to action_col1
    # P0 places piece (1) in col 1 at board[3,1] (for H=4). Player 1 to move.
    leaf_env_state = selection_result.leaf_env.get_state_with_key()
    assert leaf_env_state["board"][env.height - 1, action_col1] == 1  # Player 0's piece
    assert leaf_env_state["current_player"] == 1


def test_ucb1_select_path(root_node, connect4_env_small):
    """Test selection down multiple levels."""
    selector = UCB1Selection(exploration_constant=1.0)
    env = connect4_env_small  # Empty 5x4 board, P0 turn

    action_r_c1 = 0
    action_c1_gc1 = 1

    # Setup root_node's children
    env_at_root = env.copy()
    legal_actions_root = env_at_root.get_legal_actions()
    num_legal_actions_root = len(legal_actions_root)
    uniform_prior_root = (
        1.0 / num_legal_actions_root if num_legal_actions_root > 0 else 0.0
    )

    child1_node_for_path = None
    root_total_visits = 0
    for action_at_root in legal_actions_root:
        node = MCTSNode(parent=root_node, prior=uniform_prior_root)
        if action_at_root == action_r_c1:
            node.visit_count = 5
            node.total_value = 0.0  # P1's value. P0 sees 0.0.
            child1_node_for_path = node
        else:
            node.visit_count = 1
            node.total_value = 1.0  # P1's value. P0 sees -1.0 (less attractive).
        root_node.children[action_at_root] = node
        root_total_visits += node.visit_count
    root_node.visit_count = root_total_visits
    assert (
        child1_node_for_path is not None
    ), f"Action {action_r_c1} not in {legal_actions_root}"

    # Setup child1_node_for_path's children
    env_at_child1 = env.copy()
    env_at_child1.step(action_r_c1)
    legal_actions_child1 = env_at_child1.get_legal_actions()
    num_legal_actions_child1 = len(legal_actions_child1)
    uniform_prior_child1 = (
        1.0 / num_legal_actions_child1 if num_legal_actions_child1 > 0 else 0.0
    )

    grandchild1_node_for_path = None
    for action_at_child1 in legal_actions_child1:
        node = MCTSNode(parent=child1_node_for_path, prior=uniform_prior_child1)
        if action_at_child1 == action_c1_gc1:
            node.visit_count = 0  # Unvisited, to be selected
            grandchild1_node_for_path = node
        else:
            node.visit_count = 1  # Visited
            node.total_value = 0.0
        child1_node_for_path.children[action_at_child1] = node
    assert (
        grandchild1_node_for_path is not None
    ), f"Action {action_c1_gc1} not in {legal_actions_child1}"
    # child1_node_for_path.visit_count is already 5.

    selection_result = selector.select(root_node, env.copy())
    # Update assertions to use the dynamically created child1 and grandchild1 nodes
    assert selection_result.leaf_node == grandchild1_node_for_path
    assert selection_result.path == [
        root_node,
        child1_node_for_path,
        grandchild1_node_for_path,
    ]

    # assert selection_result.leaf_node == grandchild1 # Replaced by assert above
    assert len(selection_result.path) == 3
    # assert selection_result.path == [root_node, child1, grandchild1] # Replaced by assert above

    # Expected leaf state: P0 played col 0 (board[H-1,0]=1), P1 played col 1 (board[H-1,1]=2)
    # Current player at leaf is P0.
    leaf_env_state = selection_result.leaf_env.get_state_with_key()
    expected_board = np.zeros((env.height, env.width), dtype=np.int8)
    expected_board[env.height - 1, action_r_c1] = 1  # P0's piece
    expected_board[env.height - 1, action_c1_gc1] = 2  # P1's piece
    assert np.array_equal(leaf_env_state["board"], expected_board)
    assert leaf_env_state["current_player"] == 0


# --- Test UniformExpansion ---


def test_uniform_expansion(root_node, connect4_env_small):
    """Test expanding a node with UniformExpansion."""
    expander = UniformExpansion()
    env = connect4_env_small  # Empty 5x4 board, P0 turn
    legal_actions = env.get_legal_actions()  # [0, 1, 2, 3, 4] for 5x4 board
    num_actions = len(legal_actions)
    assert num_actions == env.width  # For an empty board

    assert not root_node.is_expanded()
    expander.expand(root_node, env)

    assert root_node.is_expanded()
    assert len(root_node.children) == num_actions
    expected_prior = 1.0 / num_actions

    for action in legal_actions:
        # Connect4 actions are int, action_key will be the int itself
        action_key = action
        assert action_key in root_node.children
        child = root_node.children[action_key]
        assert child.parent == root_node
        assert child.prior == pytest.approx(expected_prior)
        assert child.visit_count == 0
        assert child.total_value == 0.0
        assert not child.is_expanded()


def test_uniform_expansion_terminal_node(root_node, connect4_env_small):
    """Test that expansion does nothing on a terminal node."""
    expander = UniformExpansion()
    env = connect4_env_small
    # Create a state where P0 has just won.
    # Board (5x4), P0 (piece 1) wins horizontally on bottom row.
    # . . . . .
    # . . . . .
    # . . . . .
    # 1 1 1 1 .
    # P0 played in col 3 (action=3) to win.
    # State should reflect: board, current_player=1 (next player), done=True, winner=0.
    board = np.zeros((env.height, env.width), dtype=np.int8)
    board[env.height - 1, 0] = 1
    board[env.height - 1, 1] = 1
    board[env.height - 1, 2] = 1
    board[env.height - 1, 3] = 1  # Winning piece
    state_terminal_p0_wins = {
        "board": board,
        "current_player": 1,  # Player whose turn it would be
        "step_count": 4,  # P0 made 4 moves (example)
        "last_action": 3,  # P0's last action (example)
        "rewards": {0: 1.0, 1: -1.0},  # Rewards after win
        "winner": 0,  # P0 is the winner
        "done": True,  # Game is over
    }
    env.set_state(state_terminal_p0_wins)
    assert env.is_game_over()
    assert env.get_winning_player() == 0

    expander.expand(root_node, env)
    assert not root_node.is_expanded()
    assert not root_node.children


def test_uniform_expansion_already_expanded(root_node, connect4_env_small):
    """Test that expansion does nothing on an already expanded node."""
    expander = UniformExpansion()
    env = connect4_env_small
    # Expand once
    expander.expand(root_node, env)
    children_before = root_node.children.copy()
    priors_before = {a: c.prior for a, c in children_before.items()}

    # Try expanding again
    expander.expand(root_node, env)
    children_after = root_node.children
    priors_after = {a: c.prior for a, c in children_after.items()}

    # Should be unchanged
    assert children_after == children_before
    assert priors_after == priors_before


# --- Test RandomRolloutEvaluation ---


@pytest.fixture
def rollout_evaluator() -> RandomRolloutEvaluation:
    """Fixture for the rollout evaluator."""
    return RandomRolloutEvaluation(max_rollout_depth=10)


def test_rollout_evaluation_terminal_win(
    root_node, connect4_env_small, rollout_evaluator
):
    """Test evaluation when the node itself is terminal (current player wins)."""
    env = connect4_env_small
    # We need a state where game is over, and env.current_player == env.winner.
    # This state might be "artificial" for Connect4's natural flow but tests evaluate() logic.
    # Player 0 (piece 1) has won. current_player is 0.
    board = np.zeros((env.height, env.width), dtype=np.int8)
    board[env.height - 1, 0] = 1  # P0
    board[env.height - 1, 1] = 1  # P0
    board[env.height - 1, 2] = 1  # P0
    board[env.height - 1, 3] = 1  # P0 wins

    state_terminal_p0_is_current_and_winner = {
        "board": board,
        "current_player": 0,  # P0 is current player
        "step_count": 7,  # Example steps
        "last_action": None,  # Not strictly relevant for this state's evaluation logic
        "rewards": {0: 1.0, 1: -1.0},  # Consistent with P0 win
        "winner": 0,  # P0 is winner
        "done": True,
    }
    env.set_state(state_terminal_p0_is_current_and_winner)
    assert env.is_game_over()
    assert env.get_winning_player() == 0
    assert env.get_current_player() == 0

    value = rollout_evaluator.evaluate(root_node, env)
    # Value is from perspective of player at leaf (P0). P0 won.
    assert value == 1.0


def test_rollout_evaluation_terminal_loss(
    root_node, connect4_env_small, rollout_evaluator
):
    """Test evaluation when the node itself is terminal (current player loses)."""
    env = connect4_env_small
    # We need a state where game is over, env.current_player != env.winner (and winner is not None).
    # Player 1 (piece 2) has won. current_player is 0.
    board = np.zeros((env.height, env.width), dtype=np.int8)
    board[env.height - 1, 0] = 2  # P1
    board[env.height - 1, 1] = 2  # P1
    board[env.height - 1, 2] = 2  # P1
    board[env.height - 1, 3] = 2  # P1 wins

    state_terminal_p0_is_current_p1_wins = {
        "board": board,
        "current_player": 0,  # P0 is current player
        "step_count": 8,  # Example steps
        "last_action": None,  # Not strictly relevant
        "rewards": {0: -1.0, 1: 1.0},  # Consistent with P1 win
        "winner": 1,  # P1 is winner
        "done": True,
    }
    env.set_state(state_terminal_p0_is_current_p1_wins)
    assert env.is_game_over()
    assert env.get_winning_player() == 1
    assert env.get_current_player() == 0

    value = rollout_evaluator.evaluate(root_node, env)
    # Value is from perspective of player at leaf (P0). P0 lost (P1 won).
    assert value == -1.0


# def test_rollout_evaluation_forced_loss(
#     root_node, connect4_env_small, rollout_evaluator
# ):
#     """Test evaluation from a state where the current player is likely to lose quickly in a random rollout."""
#     # Setup: P1 has a double threat. P0 is to move.
#     # P0 can block one threat, but P1 should win by taking the other.
#     # This test is somewhat probabilistic for a random rollout but aims to check -1.0 for P0.
#     env = connect4_env_small  # 5x4 board
#     board_double_threat = np.zeros((env.height, env.width), dtype=np.int8)
#     # P1 (piece 2) has three in a row: (H-1, 1), (H-1, 2), (H-1, 3)
#     # Threatening to win at (H-1, 0) or (H-1, 4)
#     # Board: (bottom row, H-1)
#     # _ 2 2 2 _
#     # To make this a double threat, P1 needs two such lines or a line that can be completed in two ways.
#     # Let's set P1 pieces at (H-1,1), (H-1,2), (H-1,3). P1 threatens (H-1,0) and (H-1,4) for a horizontal win.
#     # This requires P1 to have 3 pieces that form part of two potential winning lines.
#     # Example: P1 has (2,1), (2,2), (2,3). P0 to move. P1 threatens (2,0) and (2,4).
#     # If P0 plays (e.g. col 0, so piece at (H-1,0) or higher), P1 plays col 4 and wins.
#     # And vice-versa.
#     # (Using row index 2 for a 4-high board, which is env.height - 2)
#     # . . . . . (row 0)
#     # . . . . . (row 1)
#     # . 2 2 2 . (row 2: P1 at (2,1), (2,2), (2,3)) -> threatens (2,0) and (2,4)
#     # x . . . x (row 3: P0 pieces 'x' at (3,0) and (3,4) to ensure P1's threats are on row 2)
#     row_threat = env.height - 2  # Second row from bottom (index 2 for H=4)
#     if row_threat < 0:
#         row_threat = 0  # Ensure valid row for small height
#
#     board_double_threat[row_threat, 1] = 2  # P1
#     board_double_threat[row_threat, 2] = 2  # P1
#     board_double_threat[row_threat, 3] = 2  # P1
#     # Ensure P0 cannot simply place underneath these to change the row of threat
#     if env.height - 1 > row_threat:  # If there's a row below the threat row
#         board_double_threat[env.height - 1, 0] = 1  # P0 piece below left threat
#         board_double_threat[env.height - 1, 4] = 1  # P0 piece below right threat
#         # Potentially fill other spots below P1's pieces too if needed
#         board_double_threat[env.height - 1, 1] = 1
#         board_double_threat[env.height - 1, 2] = 1
#         board_double_threat[env.height - 1, 3] = 1
#
#     state_double_threat = {
#         "board": board_double_threat,
#         "current_player": 0,  # P0 to move
#         "step_count": 5,  # Example
#         "last_action": None,
#         "rewards": {0: 0.0, 1: 0.0},
#         "winner": None,
#         "done": False,
#     }
#     env.set_state(state_double_threat)
#     assert not env.is_game_over()
#     assert env.get_current_player() == 0
#
#     # P0 plays, e.g., blocks at (row_threat, 0) by playing in column 0.
#     # Then P1 plays. P1 should play at (row_threat, 4) by playing in column 4 and win.
#     # The random rollout should ideally find this sequence.
#     value = rollout_evaluator.evaluate(root_node, env)
#     # Value is from perspective of player at start of rollout (P0). P0 should lose.
#     assert value == -1.0  # This can be flaky with pure random rollout.
#
#
# def test_rollout_evaluation_terminal_draw(
#     root_node, connect4_env_small, rollout_evaluator
# ):
#     """Test evaluation when the node itself is terminal (draw)."""
#     env = connect4_env_small
#     # Create a full board state that is a draw for 5x4 Connect4
#     # A checkerboard pattern often results in a draw if no early win.
#     # 1 2 1 2 1
#     # 2 1 2 1 2
#     # 1 2 1 2 1
#     # 2 1 2 1 2
#     # This specific pattern is a draw on a 5x4 board.
#     board_draw = np.zeros((env.height, env.width), dtype=np.int8)
#     for r in range(env.height):
#         for c in range(env.width):
#             if (r + c) % 2 == 0:
#                 board_draw[r, c] = 1  # Player 0
#             else:
#                 board_draw[r, c] = 2  # Player 1
#
#     # Verify this board is indeed a draw (no winner)
#     # This requires a temporary env to check win condition on this board
#     temp_env = Connect4(width=env.width, height=env.height)
#     temp_env.board = board_draw.copy()
#     temp_env.current_player = 0  # Doesn't matter for win check
#     has_win = False
#     for r_check in range(env.height):
#         for c_check in range(env.width):
#             if board_draw[r_check, c_check] != 0:
#                 # Temporarily set current player to the one who owns the piece
#                 # to use _check_win correctly.
#                 temp_env.current_player = board_draw[r_check, c_check] - 1
#                 if temp_env._check_win(r_check, c_check):
#                     has_win = True
#                     break
#         if has_win:
#             break
#     assert not has_win, "Constructed draw board unexpectedly has a winner."
#
#     state_terminal_draw = {
#         "board": board_draw,
#         "current_player": 0,  # Next player, doesn't matter much for draw
#         "step_count": env.width * env.height,  # Board is full
#         "last_action": None,  # Not relevant
#         "rewards": {0: 0.0, 1: 0.0},  # Draw rewards
#         "winner": None,  # No winner
#         "done": True,
#     }
#     env.set_state(state_terminal_draw)
#     assert env.is_game_over()
#     assert env.get_winning_player() is None
#
#     value = rollout_evaluator.evaluate(root_node, env)
#     assert value == 0.0


def test_rollout_evaluation_max_depth(root_node, rollout_evaluator):
    """Test evaluation hitting max rollout depth."""

    class TrulyNeverEndingConnect4(Connect4):
        def __init__(self, width: int = 5, height: int = 4):
            # Initialize with large enough board that max_rollout_depth is hit first
            super().__init__(width=width, height=height)

        def is_game_over(self) -> bool:
            # Overridden to ensure game doesn't end by normal Connect4 rules
            return False

        def get_winning_player(self) -> Optional[int]:
            # No winner, so if is_game_over were true, it'd be a draw
            return None

        def get_legal_actions(self) -> List[ActionType]:
            # Always provide legal actions to prevent rollout from stopping due to no moves
            # For simplicity, always allow playing in the first column,
            # even if it means stacking infinitely (not physically possible, but tests depth).
            return [0]  # Always allow action in column 0

        def step(self, action: ActionType) -> "ActionResult":
            # Simplified step: increments step_count, switches player.
            # Does not modify board to prevent filling up or causing game end by full board.
            # This ensures the rollout can proceed for `max_rollout_depth` steps.
            assert action == 0  # Given get_legal_actions override

            self.last_action = action
            self.step_count += 1
            # self.board is not changed to allow infinite plays in column 0
            self.current_player = (self.current_player + 1) % self.num_players

            # Return an observation. Since board isn't changing, observation is mostly static.
            state_with_key = self.get_state_with_key()
            # Crucially, done must be false based on our is_game_over override
            state_with_key.state["done"] = self.is_game_over()

            return ActionResult(
                next_state_with_key=state_with_key,
                reward=0.0,  # No rewards during this type of rollout
                done=state_with_key.state["done"],
            )

    # Use a board size that wouldn't fill up within max_rollout_depth if step was normal
    env = TrulyNeverEndingConnect4(width=5, height=4)
    evaluator = RandomRolloutEvaluation(max_rollout_depth=5)  # Short depth

    # evaluate is called on root_node, but its state doesn't matter as env is not terminal.
    # The evaluator will use the passed 'env' for rollouts.
    value = evaluator.evaluate(root_node, env)
    assert value == 0.0  # Reaching max depth is treated as a draw


# --- Test StandardBackpropagation ---


def test_standard_backpropagation(root_node):
    """Test the StandardBackpropagation strategy."""
    backpropagator = StandardBackpropagation()

    # Create a path: root -> child -> grandchild
    child = MCTSNode(parent=root_node)
    grandchild = MCTSNode(parent=child)
    path = [root_node, child, grandchild]

    # Simulate value from grandchild's perspective (player at leaf)
    value_at_leaf = 1.0  # e.g., grandchild's player won the rollout
    player_at_leaf = 0  # Assume player 0 was at the grandchild node

    backpropagator.backpropagate(path, value_at_leaf, player_at_leaf)

    # Check grandchild (leaf)
    assert grandchild.visit_count == 1
    assert grandchild.total_value == pytest.approx(1.0)  # Value from P0 perspective

    # Check child (parent of leaf)
    assert child.visit_count == 1
    # Value for child is from perspective of player at child node.
    # If P0 was at grandchild, P1 was at child. Value for P1 is -value_at_leaf.
    assert child.total_value == pytest.approx(-1.0)

    # Check root (parent of child)
    assert root_node.visit_count == 1
    # Value for root is from perspective of player at root node.
    # If P1 was at child, P0 was at root. Value for P0 is -(-value_at_leaf).
    assert root_node.total_value == pytest.approx(1.0)

    # --- Second backpropagation on the same path ---
    # Simulate a loss from grandchild's perspective this time
    value_at_leaf_2 = -1.0
    player_at_leaf_2 = 0

    backpropagator.backpropagate(path, value_at_leaf_2, player_at_leaf_2)

    # Check grandchild
    assert grandchild.visit_count == 2
    assert grandchild.total_value == pytest.approx(1.0 + (-1.0)) == pytest.approx(0.0)

    # Check child
    assert child.visit_count == 2
    # Value added is -value_at_leaf_2 = -(-1.0) = 1.0
    assert child.total_value == pytest.approx(-1.0 + 1.0) == pytest.approx(0.0)

    # Check root
    assert root_node.visit_count == 2
    # Value added is -(-value_at_leaf_2) = value_at_leaf_2 = -1.0
    assert root_node.total_value == pytest.approx(1.0 + (-1.0)) == pytest.approx(0.0)


# --- Test MCTSNode ---


def test_mcts_node_init():
    """Test MCTSNode initialization."""
    node = MCTSNode()
    assert node.parent is None
    assert node.prior == 0.0
    assert not node.children
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.value == 0.0  # Test value property with zero visits
    assert not node.is_expanded()

    parent = MCTSNode()
    child = MCTSNode(parent=parent, prior=0.75)
    assert child.parent == parent
    assert child.prior == 0.75


def test_mcts_node_value_property():
    """Test the value property calculation."""
    node = MCTSNode()
    node.visit_count = 10
    node.total_value = 5.0
    assert node.value == 0.5

    node.visit_count = 0
    node.total_value = 0.0
    assert node.value == 0.0


def test_mcts_node_is_expanded():
    """Test the is_expanded method."""
    node = MCTSNode()
    assert not node.is_expanded()
    # Connect4 action is an int (column index)
    action_example_col = 0
    node.children[action_example_col] = MCTSNode(parent=node)
    assert node.is_expanded()
