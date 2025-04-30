import pytest

from agents.mcts_agent import MCTSAgent
from environments.nim_env import NimEnv
from environments.base import SanityCheckState


@pytest.fixture
def nim_env_simple() -> NimEnv:
    """Fixture for a very simple Nim environment [1, 1]."""
    return NimEnv(initial_piles=[1, 1])


@pytest.fixture
def nim_env_3_piles() -> NimEnv:
    """Fixture for a standard Nim environment [1, 2, 1]."""
    return NimEnv(initial_piles=[1, 2, 1])


@pytest.fixture
def mcts_agent_nim_deterministic(nim_env_simple: NimEnv) -> MCTSAgent:
    """Fixture for a deterministic MCTS agent for Nim."""
    return MCTSAgent(
        env=nim_env_simple,
        num_simulations=10,  # Low sims for speed, but enough for basic tests
        exploration_constant=1.41,
        temperature=0.0,  # Deterministic selection
        tree_reuse=False,  # Easier testing without reuse
    )


@pytest.fixture
def mcts_agent_nim_stochastic(nim_env_simple: NimEnv) -> MCTSAgent:
    """Fixture for a stochastic MCTS agent for Nim."""
    return MCTSAgent(
        env=nim_env_simple,
        num_simulations=10,
        exploration_constant=1.41,
        temperature=1.0,  # Stochastic selection
        tree_reuse=False,
    )


def test_mcts_agent_act_deterministic_simple_win(nim_env_simple):
    """Test deterministic MCTS agent finds the winning move in [1, 1]."""
    # State [1, 1], P0 turn. Nim sum = 0 (Losing state). Any move leads to win for P1.
    # Agent should pick *a* legal move. Let's run it.
    env = nim_env_simple
    agent = MCTSAgent(env=env, num_simulations=20, temperature=0.0, tree_reuse=False)
    state = env.get_observation()
    action = agent.act(state)

    assert action is not None
    assert action in [(0, 1), (1, 1)]  # Should pick one of the two legal moves

    # Now test a winning state: [1, 0], P0 turn. Nim sum = 1 (Winning). Must take (0, 1).
    env.set_state(
        {
            "piles": (1, 0),
            "current_player": 0,
            "step_count": 1,
            "last_action": None,
            "winner": None,
            "done": False,
        }
    )
    state = env.get_observation()
    action = agent.act(state)
    assert action == (0, 1)  # Must choose the only winning move


def test_mcts_agent_act_returns_legal_action(mcts_agent_nim_stochastic, nim_env_simple):
    """Test that the agent always returns a legal action."""
    env = nim_env_simple
    state = env.get_observation()
    legal_actions = env.get_legal_actions()

    # Run act multiple times to check stochastic selection
    for _ in range(10):
        chosen_action = mcts_agent_nim_stochastic.act(state)
        assert chosen_action is not None
        # Ensure the action chosen is hashable (tuple) for the check
        action_key = (
            tuple(chosen_action) if isinstance(chosen_action, list) else chosen_action
        )
        assert action_key in legal_actions


def test_mcts_agent_reset(mcts_agent_nim_deterministic, nim_env_simple):
    """Test the agent's reset method."""
    agent = mcts_agent_nim_deterministic
    state = nim_env_simple.get_observation()
    # Run search to populate the tree
    agent.act(state)
    assert agent.mcts_orchestrator.root.visit_count > 0
    assert agent._last_action is not None

    # Reset the agent
    agent.reset()

    # Check if orchestrator root and last action are reset
    assert agent.mcts_orchestrator.root.visit_count == 0
    assert not agent.mcts_orchestrator.root.children  # Root should be fresh
    assert agent._last_action is None
