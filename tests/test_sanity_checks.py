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
        # Use low simulation counts for faster tests
        config.mcts.num_simulations = 25
        config.alpha_zero.num_simulations = 25
        config.alpha_zero.debug_mode = False  # Keep tests quiet
        config.muzero.debug_mode = False
        return config

    def _run_mcts_agent_check(self, agent: MCTSAgent, env: BaseEnvironment):
        """Runs sanity checks specifically for MCTSAgent."""
        sanity_states = env.get_sanity_check_states()
        self.assertTrue(
            sanity_states, f"No sanity states found for {type(env).__name__}"
        )

        for description, state, expected_value, expected_action in sanity_states: # Unpack expected_action
            with self.subTest(description=description):
                print(f"\n--- Testing MCTS: {description} ---")
                # Set environment to the test state
                current_env = env.copy()
                current_env.set_state(state)
                agent.reset()  # Reset MCTS tree

                # Get the action chosen by the agent
                # MCTSAgent.act runs the search internally
                chosen_action = agent.act(state)

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
                    continue  # Skip further checks if no children

                # --- Basic Assertions for MCTS ---
                # 1. Did it choose a legal action?
                legal_actions = current_env.get_legal_actions()
                self.assertIn(
                    chosen_action,
                    legal_actions,
                    f"Chosen action {chosen_action} not in legal actions {legal_actions}",
                )

                # 2. If an optimal/required action is defined, assert the agent chose it.
                if expected_action is not None:
                    self.assertEqual(
                        chosen_action,
                        expected_action,
                        f"Expected action {expected_action} but got {chosen_action}"
                    )
                # Note: We could add more nuanced checks, e.g., if expected_value is 1.0,
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

    # --- Test Methods ---

    def test_mcts_agent_sanity_connect4(self):
        """Run MCTSAgent sanity checks for Connect4."""
        config = self._get_config("connect4")
        env = get_environment(config.env)
        # Create agent directly (don't need full get_agents factory here)
        agent = MCTSAgent(
            env,
            num_simulations=config.mcts.num_simulations,
            exploration_constant=config.mcts.exploration_constant,
        )
        self._run_mcts_agent_check(agent, env)

    def test_mcts_agent_sanity_nim(self):
        """Run MCTSAgent sanity checks for Nim."""
        config = self._get_config("nim")
        env = get_environment(config.env)
        agent = MCTSAgent(
            env,
            num_simulations=config.mcts.num_simulations,
            exploration_constant=config.mcts.exploration_constant,
        )
        self._run_mcts_agent_check(agent, env)

    # TODO: Add tests for AlphaZeroAgent, potentially checking value sign and policy peak
