import unittest

import numpy as np

from core.config import AppConfig
from environments.base import BaseEnvironment
from agents.mcts_agent import MCTSAgent
from factories import get_environment


class TestSanityChecks(unittest.TestCase):
    """
    Tests agent behavior on predefined 'sanity check' game states.
    These tests verify basic logic (e.g., taking immediate wins, blocking losses)
    rather than optimal play or convergence.
    """

    def _get_config(self, env_name: str) -> AppConfig:
        """Helper to get a default config with the specified environment."""
        config = AppConfig()
        config.env.name = env_name
        # Use higher simulation counts for MCTS sanity checks to improve reliability
        config.mcts.num_simulations = 800
        config.alpha_zero.num_simulations = 50  # Keep AlphaZero low for now
        config.alpha_zero.debug_mode = False  # Keep tests quiet
        config.muzero.debug_mode = False
        return config

    def _run_single_mcts_check(
        self, agent: MCTSAgent, env: BaseEnvironment, check_case
    ):
        """Runs sanity check for a single case for MCTSAgent."""
        # This method now assumes check_case is passed in.
        # The loop and subTest are removed.
        print(f"\n--- Testing MCTS: {check_case.description} ---")
        # Set environment to the test state
        current_env = env.copy()
        current_env.set_state(check_case.state)
        agent.reset()  # Reset MCTS tree

        # Get the action chosen by the agent
        # MCTSAgent.act runs the search internally
        chosen_action = agent.act(check_case.state)

        # Get MCTS search results for analysis (optional but useful)
        root_node = agent.mcts.root  # Get root after search in act()
        if not root_node or not root_node.children:
            print("Warning: MCTS root has no children after search.")
            # Basic assertion: agent should return *some* legal action if available
            legal_actions = current_env.get_legal_actions()
            if legal_actions:
                self.assertIsNotNone(
                    chosen_action,
                    "Agent returned None action when legal moves exist.",
                )
                self.assertIn(
                    chosen_action,
                    legal_actions,
                    "Agent chose an illegal action.",
                )
            return  # Exit check if no children (changed from continue)

        # --- Basic Assertions for MCTS ---
        # 1. Did it choose a legal action?
        legal_actions = current_env.get_legal_actions()
        self.assertIn(
            chosen_action,
            legal_actions,
            f"Chosen action {chosen_action} not in legal actions {legal_actions}",
        )

        # 2. If an optimal/required action is defined, assert the agent chose it.
        if check_case.expected_action is not None:
            self.assertEqual(
                chosen_action,
                check_case.expected_action,
                f"Expected action {check_case.expected_action} but got {chosen_action}",
            )
        # Note: We could add more nuanced checks, e.g., if check_case.expected_value is 1.0,
        # ensure the chosen action *leads* to a win, even if multiple winning moves exist.
        # For now, checking against a single defined expected action is simpler.

        # 4. Print visit counts for manual inspection (optional)
        visit_counts = np.array(
            [child.visit_count for child in root_node.children.values()]
        )
        actions = list(root_node.children.keys())
        sorted_visits = sorted(
            zip(actions, visit_counts), key=lambda item: item[1], reverse=True
        )
        print("MCTS Visit Counts:")
        for act, visits in sorted_visits:
            highlight = " <<< CHOSEN" if act == chosen_action else ""
            print(f"  - {act}: {visits}{highlight}")

    # TODO: Add tests for AlphaZeroAgent, potentially checking value sign and policy peak

    # TODO: Add tests for AlphaZeroAgent, potentially checking value sign and policy peak


# --- Dynamic Test Generation for MCTS (Executed after class definition) ---


def _generate_mcts_test_method(env_name: str, check_case):  # Renamed factory slightly
    """Factory to create a test method for a specific MCTS sanity check case."""

    def test_func(self: TestSanityChecks):
        """Dynamically generated test for a specific MCTS sanity check case."""
        config = self._get_config(env_name)
        env = get_environment(config.env)
        agent = MCTSAgent(
            env,
            num_simulations=config.mcts.num_simulations,
            exploration_constant=config.mcts.exploration_constant,
        )
        # Call the actual check logic for this specific case
        self._run_single_mcts_check(agent, env, check_case)

    return test_func


# Generate tests for each environment and case
for _env_name_to_test in ["connect4", "nim"]:
    # Need a temporary env instance just to get the cases
    # Use default config settings for this temporary instance
    _temp_config = AppConfig()
    _temp_config.env.name = _env_name_to_test
    _temp_env_instance = None  # Initialize to prevent unbound local error
    _sanity_cases = []  # Initialize
    try:
        _temp_env_instance = get_environment(_temp_config.env)
        _sanity_cases = _temp_env_instance.get_sanity_check_states()
    except Exception as e:
        print(
            f"Warning: Could not instantiate or get sanity cases for {_env_name_to_test}: {e}"
        )
        # _sanity_cases remains []

    if not _sanity_cases:
        print(
            f"Warning: No sanity cases found for {_env_name_to_test}, skipping MCTS test generation."
        )
        # continue # No need to continue if loop variable isn't used after this

    for _i, _case in enumerate(_sanity_cases):
        # Sanitize description for method name (simple approach)
        _safe_desc = "".join(
            c if c.isalnum() or c == "_" else "_" for c in _case.description
        ).lower()
        # Ensure name starts with 'test_' and is unique
        _method_name = f"test_mcts_{_env_name_to_test}_case_{_i}_{_safe_desc}"

        # Create the test method using the factory
        _test_method = _generate_mcts_test_method(_env_name_to_test, _case)

        # Set descriptive name and docstring for better reporting
        _test_method.__name__ = _method_name
        _test_method.__doc__ = (
            f"MCTS Sanity Check ({_env_name_to_test}): {_case.description}"
        )

        # Add the method to the test class *after* it has been defined
        setattr(TestSanityChecks, _method_name, _test_method)

    # Clean up temporary instance and variables (optional, good practice)
    if _temp_env_instance:
        del _temp_env_instance
    del _sanity_cases
    # Delete loop variables to avoid potential leakage if this code were in a function
    # (though at module level it's less critical)
    try:
        del (
            _env_name_to_test,
            _temp_config,
            _i,
            _case,
            _safe_desc,
            _method_name,
            _test_method,
        )
    except NameError:
        pass  # In case the loop didn't run
