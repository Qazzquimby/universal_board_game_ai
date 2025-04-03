import unittest

import numpy as np

from core.config import AppConfig
from environments.base import BaseEnvironment
from agents.mcts_agent import MCTSAgent
from agents.alphazero_agent import AlphaZeroAgent
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
        config.mcts.num_simulations = 2400
        config.mcts.debug = True  # Enable MCTS debug prints for tests
        config.alpha_zero.num_simulations = 50  # Keep AlphaZero low for now
        config.alpha_zero.debug_mode = False  # Keep other agents quiet
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

    def _run_single_alphazero_check(
        self, agent: AlphaZeroAgent, env: BaseEnvironment, check_case
    ):
        """Runs sanity check for a single case for AlphaZeroAgent's network."""
        print(f"\n--- Testing AlphaZero Network: {check_case.description} ---")
        # Ensure network is in eval mode
        agent.network.eval()

        # Attempt to load weights - checks are more meaningful with trained weights
        if not agent.load():
            print(
                "Warning: No pre-trained AlphaZero weights found. Network predictions might be random."
            )

        # Get network predictions for the state
        try:
            policy_np, value_np = agent.network.predict(check_case.state)
        except Exception as e:
            self.fail(
                f"Network prediction failed for state {check_case.state} with error: {e}"
            )

        print(
            f"  State: {check_case.state.get('board', check_case.state.get('piles', 'N/A'))}"
        )
        print(f"  Player: {check_case.state['current_player']}")

        # --- 1. Value Prediction Check ---
        print(f"  Value Prediction: {value_np:.4f}")
        if check_case.expected_value is not None:
            print(f"  Expected Value: {check_case.expected_value:.1f}")
            # Check if signs match (more robust than exact value)
            if check_case.expected_value > 0.1:  # Expecting win
                self.assertGreater(
                    value_np,
                    0.0,
                    "Predicted value should be positive for expected win state.",
                )
            elif check_case.expected_value < -0.1:  # Expecting loss
                self.assertLess(
                    value_np,
                    0.0,
                    "Predicted value should be negative for expected loss state.",
                )
            else:  # Expecting draw (or near zero)
                self.assertAlmostEqual(
                    value_np,
                    0.0,
                    delta=0.2,
                    msg="Predicted value should be close to zero for expected draw state.",
                )
        else:
            print("  (No expected value defined for comparison)")

        # --- 2. Policy Prediction Check ---
        # Get legal actions for this state
        temp_env = env.copy()
        temp_env.set_state(check_case.state)
        legal_actions = temp_env.get_legal_actions()

        if not legal_actions:
            print("  Policy Prediction: (No legal actions in this state)")
            # If no legal actions, but an expected action was defined, that's a test setup error
            self.assertIsNone(
                check_case.expected_action,
                "Test case has expected_action but no legal actions exist.",
            )
            return  # Nothing more to check for policy

        action_probs = {}
        for action in legal_actions:
            idx = agent.network.get_action_index(action)
            if idx is not None and 0 <= idx < len(policy_np):
                action_probs[action] = policy_np[idx]
            else:
                action_probs[action] = -1  # Indicate mapping error

        # Sort actions by predicted probability
        sorted_probs = sorted(
            action_probs.items(), key=lambda item: item[1], reverse=True
        )

        print(f"  Predicted Probabilities (Top 5 Legal):")
        for i, (action, prob) in enumerate(sorted_probs[:5]):
            highlight = ""
            if prob < 0:
                print(f"    - {action}: (Error mapping action)")
                continue
            if i == 0:
                best_predicted_action = action
                highlight = " <<< BEST PREDICTED"
            print(f"    - {action}: {prob:.4f}{highlight}")

        # Check if the best predicted action matches the expected action
        if check_case.expected_action is not None:
            print(f"  Expected Action: {check_case.expected_action}")
            # Ensure best_predicted_action was assigned (i.e., there were legal actions)
            self.assertTrue(
                best_predicted_action is not None,
                "Could not determine best predicted action.",
            )
            self.assertEqual(
                best_predicted_action,
                check_case.expected_action,
                f"Action with highest predicted probability ({best_predicted_action} with p={action_probs.get(best_predicted_action, -1):.4f}) "
                f"does not match expected action ({check_case.expected_action}).",
            )
        else:
            print("  (No specific expected action defined for comparison)")
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
            debug=config.mcts.debug,  # Pass debug flag
        )
        # Call the actual check logic for this specific case
        self._run_single_mcts_check(agent, env, check_case)

    return test_func


# --- Generate MCTS Tests ---
print("\nGenerating MCTS Sanity Check Tests...")
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


# # --- Generate AlphaZero Tests ---
# # This loop correctly generates the AlphaZero tests using the factory defined earlier.
# print("\nGenerating AlphaZero Sanity Check Tests...")
# for _env_name_to_test in ["connect4", "nim"]:
#     # Need a temporary env instance just to get the cases
#     _temp_config = AppConfig()
#     _temp_config.env.name = _env_name_to_test
#     _temp_env_instance = None
#     _sanity_cases = []
#     try:
#         _temp_env_instance = get_environment(_temp_config.env)
#         _sanity_cases = _temp_env_instance.get_sanity_check_states()
#     except Exception as e:
#         print(
#             f"Warning: Could not instantiate or get sanity cases for {_env_name_to_test} (AlphaZero): {e}"
#         )
#
#     if not _sanity_cases:
#         print(
#             f"Warning: No sanity cases found for {_env_name_to_test}, skipping AlphaZero test generation."
#         )
#         continue  # Skip this environment if no cases
#
#     for _i, _case in enumerate(_sanity_cases):
#         _safe_desc = "".join(
#             c if c.isalnum() or c == "_" else "_" for c in _case.description
#         ).lower()
#         # Ensure name starts with 'test_' and is unique from MCTS tests
#         _method_name = f"test_alphazero_{_env_name_to_test}_case_{_i}_{_safe_desc}"
#
#         # Create the test method using the new factory
#         _test_method = _generate_alphazero_test_method(_env_name_to_test, _case)
#
#         _test_method.__name__ = _method_name
#         _test_method.__doc__ = (
#             f"AlphaZero Network Sanity Check ({_env_name_to_test}): {_case.description}"
#         )
#
#         setattr(TestSanityChecks, _method_name, _test_method)
#
#     # Clean up temporary instance and variables
#     if _temp_env_instance:
#         del _temp_env_instance
#     del _sanity_cases
#     # Delete loop variables
#     try:
#         del (
#             _env_name_to_test,
#             _temp_config,
#             _i,
#             _case,
#             _safe_desc,
#             _method_name,
#             _test_method,
#         )
#     except NameError:
#         pass
#
# print("Finished generating tests.")
