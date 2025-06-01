import pytest
import numpy as np

from core.config import AppConfig
from environments.base import BaseEnvironment, SanityCheckState
from agents.mcts_agent import MCTSAgent
from agents.alphazero_agent import AlphaZeroAgent
from factories import get_environment

# Helper to generate descriptive IDs
def _generate_id(env_name: str, case_desc: str, suffix: str = "") -> str:
    safe_desc = "".join(
        c if c.isalnum() or c == "_" else "_" for c in case_desc
    ).lower()
    return f"{env_name}-{safe_desc}{suffix}"


# List to hold parameters for each test type
MCTS_PARAMS = []
AZ_MCTS_PARAMS = []
AZ_NET_PARAMS = []

# Populate parameters by getting cases from environments
for env_name_param in ["connect4", "nim"]:
    try:
        # Need a temporary env instance just to get the cases
        _temp_config = AppConfig()
        _temp_config.env.name = env_name_param
        _temp_env = get_environment(_temp_config.env)
        _sanity_cases = _temp_env.get_sanity_check_states()

        for case in _sanity_cases:
            # MCTS Parameters
            mcts_id = _generate_id(env_name_param, case.description, suffix="-mcts")
            MCTS_PARAMS.append(pytest.param(env_name_param, case, id=mcts_id))

            # AlphaZero MCTS Parameters (No Load)
            az_mcts_noload_id = _generate_id(
                env_name_param, case.description, suffix="-az_mcts_noload"
            )
            AZ_MCTS_PARAMS.append(
                pytest.param(env_name_param, case, False, id=az_mcts_noload_id)
            )

            # AlphaZero MCTS Parameters (Load)
            az_mcts_load_id = _generate_id(
                env_name_param, case.description, suffix="-az_mcts_load"
            )
            AZ_MCTS_PARAMS.append(
                pytest.param(env_name_param, case, True, id=az_mcts_load_id)
            )

            # AlphaZero Network Eval Parameters (Load Only)
            az_net_load_id = _generate_id(
                env_name_param, case.description, suffix="-az_net_load"
            )
            AZ_NET_PARAMS.append(pytest.param(env_name_param, case, id=az_net_load_id))

    except Exception as e:
        print(f"Warning: Could not get/process sanity cases for {env_name_param}: {e}")
        # Optionally skip tests for this env or raise error depending on desired behavior

# --- Test Class ---


class TestSanityChecks:
    """
    Tests agent behavior on predefined 'sanity check' game states using pytest parameterization.
    """

    def _get_config(self, env_name: str) -> AppConfig:
        """Helper to get a default config with the specified environment."""
        config = AppConfig()
        config.env.name = env_name
        # Use higher simulation counts for MCTS sanity checks to improve reliability
        config.mcts.num_simulations = 1200  # Keep high for MCTS checks
        config.mcts.debug = False
        config.alpha_zero.num_simulations = 1200  # Keep high for AZ checks
        config.alpha_zero.debug_mode = False
        config.muzero.debug_mode = False
        return config

    # --- MCTS Agent Check Logic ---
    def _run_single_mcts_check(
        self, agent: MCTSAgent, env: BaseEnvironment, check_case: SanityCheckState
    ):
        """Runs sanity check for a single case for MCTSAgent."""
        print(f"\n--- Testing MCTS: {check_case.description} ---")
        current_env = env.copy()
        current_env.set_state(check_case.state_with_key)
        agent.reset()
        chosen_action = agent.act(check_case.state_with_key)
        root_node = agent.mcts.root

        if not root_node or not root_node.children:
            print("Warning: MCTS root has no children after search.")
            legal_actions = current_env.get_legal_actions()
            if legal_actions:
                assert (
                    chosen_action is not None
                ), "Agent returned None action when legal moves exist."
                assert chosen_action in legal_actions, "Agent chose an illegal action."
            return

        legal_actions = current_env.get_legal_actions()
        assert (
            chosen_action in legal_actions
        ), f"Chosen action {chosen_action} not in legal actions {legal_actions}"

        if check_case.expected_action is not None:
            assert (
                chosen_action == check_case.expected_action
            ), f"Expected action {check_case.expected_action} but got {chosen_action}"

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

    # --- AlphaZero Agent MCTS Action Check Logic ---
    def _run_single_alphazero_mcts_check(
        self, agent: AlphaZeroAgent, env: BaseEnvironment, check_case: SanityCheckState
    ):
        """Runs sanity check for a single case for AlphaZeroAgent's MCTS action selection."""
        load_attempted = agent.config.should_use_network
        load_status = (
            "(Load Attempted)"
            if load_attempted
            else "(No Load Attempted - Initial Weights)"
        )
        print(
            f"\n--- Testing AlphaZero Agent Action {load_status}: {check_case.description} ---"
        )

        current_env = env.copy()
        current_env.set_state(check_case.state_with_key)
        agent.reset()

        # Loading is handled by the agent based on its config (set in the test method)
        if load_attempted:
            if agent.load():
                print("Info: Using pre-trained weights for check.")
            else:
                print(
                    "Warning: Configured to load weights, but no weights file found or load failed. Using initial network weights."
                )

        chosen_action = agent.act(check_case.state_with_key, train=False)
        root_node = agent.mcts.root

        if not root_node or not root_node.children:
            print("Warning: AlphaZero MCTS root has no children after search.")
            legal_actions = current_env.get_legal_actions()
            if legal_actions:
                assert (
                    chosen_action is not None
                ), "Agent returned None action when legal moves exist."
                assert chosen_action in legal_actions, "Agent chose an illegal action."
            return

        legal_actions = current_env.get_legal_actions()
        assert (
            chosen_action in legal_actions
        ), f"Chosen action {chosen_action} not in legal actions {legal_actions}"

        if check_case.expected_action is not None:
            assert (
                chosen_action == check_case.expected_action
            ), f"Expected action {check_case.expected_action} but got {chosen_action}"

        visit_counts = np.array(
            [child.visit_count for child in root_node.children.values()]
        )
        actions = list(root_node.children.keys())
        sorted_visits = sorted(
            zip(actions, visit_counts), key=lambda item: item[1], reverse=True
        )
        print("AlphaZero MCTS Visit Counts:")
        for act, visits in sorted_visits:
            highlight = " <<< CHOSEN" if act == chosen_action else ""
            print(f"  - {act}: {visits}{highlight}")

    # --- AlphaZero Network Evaluation Check Logic ---
    def _run_single_alphazero_network_check(
        self, agent: AlphaZeroAgent, env: BaseEnvironment, check_case: SanityCheckState
    ):
        """Runs sanity check for a single case for AlphaZeroAgent's network's DIRECT OUTPUT."""
        print(
            f"\n--- Testing AlphaZero Network Direct Output: {check_case.description} ---"
        )

        # Ensure network exists and attempt load (agent handles this based on config)
        if agent.network is None:
            pytest.skip(
                f"Cannot run network check for {check_case.description}: Agent network is None."
            )
            return

        agent.network.eval()
        weights_loaded = False
        if agent.config.should_use_network:  # Check if load was intended
            if agent.load():
                weights_loaded = True
                print("Info: Using pre-trained weights for check.")
            else:
                print(
                    "Warning: Configured to load weights, but no weights file found or load failed. Network predictions might be random."
                )
        else:
            # This check shouldn't normally run if should_use_network is False, but handle defensively
            print(
                "Info: Running network check with un-trained (initial) network weights."
            )

        try:
            policy_np, value_np = agent.network.predict(check_case.state_with_key)
        except Exception as e:
            raise AssertionError(
                f"Network prediction failed for state {check_case.state_with_key} with error: {e}"
            )

        print(
            f"  State: {check_case.state_with_key.get('board', check_case.state_with_key.get('piles', 'N/A'))}"
        )
        print(f"  Player: {check_case.state_with_key['current_player']}")
        print(f"  Value Prediction: {value_np:.4f}")

        if check_case.expected_value is not None:
            print(f"  Expected Value: {check_case.expected_value:.1f}")
            if check_case.expected_value > 0.1:
                assert (
                    value_np > 0.0
                ), "Predicted value should be positive for expected win state."
            elif check_case.expected_value < -0.1:
                assert (
                    value_np < 0.0
                ), "Predicted value should be negative for expected loss state."
            else:
                assert (
                    abs(value_np - 0.0) <= 0.2
                ), "Predicted value should be close to zero for expected draw state."
        else:
            print("  (No expected value defined for comparison)")

        temp_env = env.copy()
        temp_env.set_state(check_case.state_with_key)
        legal_actions = temp_env.get_legal_actions()

        if not legal_actions:
            print("  Policy Prediction: (No legal actions in this state)")
            assert (
                check_case.expected_action is None
            ), "Test case has expected_action but no legal actions exist."
            return

        action_probs = {}
        best_predicted_action = None  # Initialize
        # Use env property for policy size if available, otherwise fallback (though network should know its size)
        policy_size = getattr(env, "policy_vector_size", len(policy_np))
        for action in legal_actions:
            # Use environment's mapping function
            idx = env.map_action_to_policy_index(action)
            if idx is not None and 0 <= idx < policy_size and idx < len(policy_np):
                action_probs[action] = policy_np[idx]
            else:
                # Print more info if mapping fails
                print(
                    f"  Warning: Failed to map action {action} to valid index (idx={idx}, policy_len={len(policy_np)})"
                )
                action_probs[action] = -1  # Indicate mapping error or out-of-bounds

        sorted_probs = sorted(
            action_probs.items(), key=lambda item: item[1], reverse=True
        )

        print(f"  Predicted Probabilities (Top 5 Legal):")
        for i, (action, prob) in enumerate(sorted_probs[:5]):
            highlight = ""
            if prob < 0:
                print(f"    - {action}: (Error mapping action)")
                continue
            if i == 0 and prob >= 0:  # Ensure best predicted action is valid
                best_predicted_action = action
                highlight = " <<< BEST PREDICTED"
            print(f"    - {action}: {prob:.4f}{highlight}")

        if check_case.expected_action is not None:
            print(f"  Expected Action: {check_case.expected_action}")
            assert (
                best_predicted_action is not None
            ), "Could not determine best predicted action (likely due to mapping errors or all probs negative)."
            assert (
                best_predicted_action == check_case.expected_action
            ), f"Action with highest predicted probability ({best_predicted_action} with p={action_probs.get(best_predicted_action, -1):.4f}) does not match expected action ({check_case.expected_action})."
        else:
            print("  (No specific expected action defined for comparison)")

    # --- Parameterized Test Methods ---

    @pytest.mark.parametrize("env_name, check_case", MCTS_PARAMS)
    def test_mcts_action_sanity(self, env_name: str, check_case: SanityCheckState):
        """Runs MCTS action selection sanity checks for various cases."""
        config = self._get_config(env_name)
        env = get_environment(config.env)
        agent = MCTSAgent(
            env,
            num_simulations=config.mcts.num_simulations,
            exploration_constant=config.mcts.exploration_constant,
        )
        self._run_single_mcts_check(agent, env, check_case)

    @pytest.mark.parametrize("env_name, check_case, load_weights", AZ_MCTS_PARAMS)
    def test_alphazero_mcts_action_sanity(
        self, env_name: str, check_case: SanityCheckState, load_weights: bool
    ):
        """Runs AlphaZero MCTS action selection sanity checks (with and without loading weights)."""
        config = self._get_config(env_name)
        config.alpha_zero.should_use_network = load_weights
        env = get_environment(config.env)
        agent = AlphaZeroAgent(env, config.alpha_zero, config.training)
        self._run_single_alphazero_mcts_check(agent, env, check_case)

    @pytest.mark.parametrize("env_name, check_case", AZ_NET_PARAMS)
    def test_alphazero_network_output_sanity(
        self, env_name: str, check_case: SanityCheckState
    ):
        """Runs AlphaZero direct network output sanity checks (assumes trying to load weights)."""
        config = self._get_config(env_name)
        config.alpha_zero.should_use_network = (
            True  # Always try to load for network eval check
        )
        env = get_environment(config.env)
        # Need training_config for agent init, even if not used in this check
        agent = AlphaZeroAgent(env, config.alpha_zero, config.training)
        self._run_single_alphazero_network_check(agent, env, check_case)


# --- End of File ---
