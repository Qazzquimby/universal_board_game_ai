import math
from typing import Optional

import pytest

from algorithms.mcts import (
    MCTSNode,
    UCB1Selection,
    UniformExpansion,
    RandomRolloutEvaluation,
    StandardBackpropagation,
)
from environments.nim_env import NimEnv
from environments.base import StateType


@pytest.fixture
def nim_env_3_piles() -> NimEnv:
    """Fixture for a simple Nim environment."""
    return NimEnv(initial_piles=[1, 2, 1])  # Small state space


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


def test_ucb1_select_unvisited_child(root_node, nim_env_3_piles):
    """Test that UCB1 selects an unvisited child first."""
    selector = UCB1Selection(exploration_constant=1.41)
    env = nim_env_3_piles
    state = env.get_observation()

    # Manually expand root with two children, one visited, one not
    action1 = (0, 1)  # Take 1 from pile 0
    action2 = (1, 1)  # Take 1 from pile 1
    root_node.children[action1] = MCTSNode(parent=root_node, prior=0.5)
    root_node.children[action1].visit_count = 1
    root_node.children[action1].total_value = -1.0  # Visited, value -1
    root_node.children[action2] = MCTSNode(parent=root_node, prior=0.5)
    root_node.children[action2].visit_count = 0  # Unvisited
    root_node.visit_count = 1  # Parent visited once

    selection_result = selector.select(root_node, env.copy())

    # Should select the unvisited child (action2)
    assert selection_result.leaf_node == root_node.children[action2]
    assert len(selection_result.path) == 2  # Root -> Child2
    assert selection_result.path[0] == root_node
    assert selection_result.path[1] == root_node.children[action2]
    # Check environment state corresponds to action2
    expected_state_after_action2 = (1, 1, 1)  # Piles are tuples in observation
    assert (
        selection_result.leaf_env.get_observation()["piles"]
        == expected_state_after_action2
    )


def test_ucb1_select_best_score(root_node, nim_env_3_piles):
    """Test that UCB1 selects the child with the highest score when all are visited."""
    selector = UCB1Selection(
        exploration_constant=1.0
    )  # Simpler constant for manual check
    env = nim_env_3_piles
    state = env.get_observation()  # [1, 2, 1], player 0 turn

    # Manually expand root with two visited children
    action1 = (1, 1)  # Take 1 from pile 1 -> [1, 1, 1], player 1 turn
    action2 = (1, 2)  # Take 2 from pile 1 -> [1, 0, 1], player 1 turn

    root_node.children[action1] = MCTSNode(parent=root_node, prior=0.6)
    root_node.children[action1].visit_count = 5
    root_node.children[
        action1
    ].total_value = 1.0  # Q(s',a') = 1/5 = 0.2 (value for player 1)
    # Value for player 0 = -0.2

    root_node.children[action2] = MCTSNode(parent=root_node, prior=0.4)
    root_node.children[action2].visit_count = 3
    root_node.children[
        action2
    ].total_value = -1.5  # Q(s',a') = -1.5/3 = -0.5 (value for player 1)
    # Value for player 0 = 0.5

    root_node.visit_count = 8  # Total visits of parent

    # Calculate scores from player 0's perspective
    # Score = -Q(child) + C * P(child) * sqrt(log(N(parent)) / N(child))
    # Score1 = -0.2 + 1.0 * 0.6 * sqrt(log(8) / 5) approx -0.2 + 0.6 * sqrt(0.415) approx -0.2 + 0.386 = 0.186
    # Score2 = -(-0.5) + 1.0 * 0.4 * sqrt(log(8) / 3) approx 0.5 + 0.4 * sqrt(0.693) approx 0.5 + 0.333 = 0.833

    # Action 2 should have a higher score
    selection_result = selector.select(root_node, env.copy())

    assert selection_result.leaf_node == root_node.children[action2]
    assert len(selection_result.path) == 2
    assert selection_result.path[1] == root_node.children[action2]
    expected_state_after_action2 = (1, 0, 1)  # Piles are tuples in observation
    assert (
        selection_result.leaf_env.get_observation()["piles"]
        == expected_state_after_action2
    )


def test_ucb1_select_path(root_node, nim_env_3_piles):
    """Test selection down multiple levels."""
    selector = UCB1Selection(exploration_constant=1.0)
    env = nim_env_3_piles  # [1, 2, 1], P0 turn

    # Build a small tree manually
    # Root -> Child1 (Action (1,1)) -> Grandchild1 (Action (0,1))
    action_r_c1 = (1, 1)  # State [1, 1, 1], P1 turn
    action_c1_gc1 = (0, 1)  # State [0, 1, 1], P0 turn (leaf)

    child1 = MCTSNode(parent=root_node, prior=1.0)
    grandchild1 = MCTSNode(parent=child1, prior=1.0)

    root_node.children[action_r_c1] = child1
    child1.children[action_c1_gc1] = grandchild1

    # Set visits high enough so selection follows path
    root_node.visit_count = 10
    child1.visit_count = 5
    grandchild1.visit_count = 0  # Make grandchild the leaf

    selection_result = selector.select(root_node, env.copy())

    assert selection_result.leaf_node == grandchild1
    assert len(selection_result.path) == 3
    assert selection_result.path == [root_node, child1, grandchild1]
    expected_leaf_state = (0, 1, 1)  # Piles are tuples in observation
    assert selection_result.leaf_env.get_observation()["piles"] == expected_leaf_state
    assert selection_result.leaf_env.get_current_player() == 0


# --- Test UniformExpansion ---


def test_uniform_expansion(root_node, nim_env_3_piles):
    """Test expanding a node with UniformExpansion."""
    expander = UniformExpansion()
    env = nim_env_3_piles  # State [1, 2, 1], P0 turn
    legal_actions = env.get_legal_actions()  # [(0,1), (1,1), (1,2), (2,1)]
    num_actions = len(legal_actions)

    assert not root_node.is_expanded()
    expander.expand(root_node, env)

    assert root_node.is_expanded()
    assert len(root_node.children) == num_actions
    expected_prior = 1.0 / num_actions

    for action in legal_actions:
        action_key = tuple(action) if isinstance(action, list) else action
        assert action_key in root_node.children
        child = root_node.children[action_key]
        assert child.parent == root_node
        assert child.prior == pytest.approx(expected_prior)
        assert child.visit_count == 0
        assert child.total_value == 0.0
        assert not child.is_expanded()


def test_uniform_expansion_terminal_node(root_node, nim_env_3_piles):
    """Test that expansion does nothing on a terminal node."""
    expander = UniformExpansion()
    env = nim_env_3_piles
    # State [0, 0, 1], P1 turn. P1 has no moves. Game ends *before* P1's turn.
    # The player whose turn it *would* have been (P1) loses. Winner is P0.
    # NimEnv.step() sets done=True when sum(piles)==0.
    # NimEnv.is_game_over() checks self.done.
    # NimEnv.get_winning_player() returns self.winner.
    # We need to manually set the state *including* done=True and winner.
    state_terminal_p1_turn = {
        "piles": (0, 0, 1),  # Use tuple
        "current_player": 1,
        "step_count": 1,  # Assume some steps led here
        "last_action": None,  # Doesn't matter for this test
        "winner": 0,  # P0 wins because P1 has no moves
        "done": True,  # Game is over
    }
    env.set_state(state_terminal_p1_turn)
    assert env.is_game_over()
    assert env.get_winning_player() == 0

    expander.expand(root_node, env)
    assert not root_node.is_expanded()
    assert not root_node.children


def test_uniform_expansion_already_expanded(root_node, nim_env_3_piles):
    """Test that expansion does nothing on an already expanded node."""
    expander = UniformExpansion()
    env = nim_env_3_piles
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


def test_rollout_evaluation_terminal_win(root_node, nim_env_3_piles, rollout_evaluator):
    """Test evaluation when the node itself is terminal (current player wins)."""
    env = nim_env_3_piles
    # We need a state where the game is over, and the *current_player* at the leaf node has won.
    # State [0, 1, 0], P1 turn. P1 has no moves. Game over. Winner P0. Value for P1 = -1.0. (This is a loss for P1)
    # State [1, 0, 0], P0 turn. P0 has no moves. Game over. Winner P1. Value for P0 = -1.0. (This is a loss for P0)
    # Let's use the state *after* a winning move.
    # State [0, 0, 0], P1 turn. Game over. P0 made the last move and lost. Winner P1.
    # Value for P1 (current player at leaf) should be +1.0.
    state_terminal_p1_wins = {
        "piles": (0, 0, 0),  # Use tuple
        "current_player": 1,  # P1's turn (but game over)
        "step_count": 3,  # Example
        "last_action": (0, 1),  # Example: P0 took last from pile 0
        "winner": 1,  # P1 wins because P0 took last
        "done": True,
    }
    env.set_state(state_terminal_p1_wins)
    assert env.is_game_over()
    assert env.get_winning_player() == 1

    value = rollout_evaluator.evaluate(root_node, env)
    # Value is from perspective of player at leaf (P1). P1 won.
    assert value == 1.0


def test_rollout_evaluation_terminal_loss(
    root_node, nim_env_3_piles, rollout_evaluator
):
    """Test evaluation when the node itself is terminal (current player loses)."""
    env = nim_env_3_piles
    # We need a state where the game is over, and the *current_player* at the leaf node has lost.
    # State [0, 0, 1], P1 turn. P1 has no moves. Game over. Winner P0.
    # Value for P1 (current player at leaf) should be -1.0.
    state_terminal_p1_loses = {
        "piles": (0, 0, 1),  # Use tuple
        "current_player": 1,  # P1's turn (but game over)
        "step_count": 2,  # Example
        "last_action": (1, 2),  # Example: P0 took from pile 1
        "winner": 0,  # P0 wins because P1 has no moves
        "done": True,
    }
    env.set_state(state_terminal_p1_loses)
    assert env.is_game_over()
    assert env.get_winning_player() == 0

    value = rollout_evaluator.evaluate(root_node, env)
    # Value is from perspective of player at leaf (P1). P1 lost.
    assert value == -1.0


def test_rollout_evaluation_forced_loss(root_node, rollout_evaluator):
    """Test evaluation from a state where the current player is forced to lose."""
    # State [0, 1, 0], P0 turn. P0 must take (1,1). State becomes [0,0,0], P1 turn.
    # Game over. P0 took last object, P0 loses. Winner P1.
    # Rollout starting from P0 should result in a loss for P0. Value = -1.0.
    env_forced_loss = NimEnv(initial_piles=[0, 1, 0])
    env_forced_loss.set_state(
        {
            "piles": (0, 1, 0),
            "current_player": 0,
            "step_count": 1,
            "last_action": None,
            "winner": None,
            "done": False,
        }
    )
    assert not env_forced_loss.is_game_over()
    assert env_forced_loss.get_current_player() == 0

    # No need to mock random.choice, as there's only one legal move: (1, 1)
    value = rollout_evaluator.evaluate(root_node, env_forced_loss)

    # Value is from perspective of player at start of rollout (P0). P0 lost.
    assert value == -1.0


def test_rollout_evaluation_terminal_draw(root_node, rollout_evaluator):
    """Test evaluation when the node itself is terminal (draw). Nim doesn't have draws."""
    # This test requires a different environment or mocking BaseEnvironment heavily.
    # Skipping for now as Nim doesn't draw. If needed, mock BaseEnvironment.
    pass


def test_rollout_evaluation_max_depth(root_node, rollout_evaluator):
    """Test evaluation hitting max rollout depth."""
    # Mock an environment that never ends
    class NeverEndingEnv(NimEnv):
        def is_game_over(self) -> bool:
            return False  # Never ends

        def get_winning_player(self) -> Optional[int]:
            return None

        def get_observation(self) -> StateType:
            # Need to return a valid StateType dict
            return {
                "piles": tuple(self.piles.tolist()),
                "current_player": self.current_player,
                "step_count": self.step_count,
                "last_action": self.last_action,
                "winner": self.winner,
                "done": self.done,
            }

    env = NeverEndingEnv(initial_piles=[10] * 10)  # Large state
    evaluator = RandomRolloutEvaluation(max_rollout_depth=5)  # Short depth
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
    node.children[(0, 1)] = MCTSNode(parent=node)
    assert node.is_expanded()
