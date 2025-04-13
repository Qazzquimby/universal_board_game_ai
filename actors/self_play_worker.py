from typing import Tuple, List, Dict, Optional
import numpy as np
import ray
import torch  # Needed for network state dict handling

from core.config import AppConfig, EnvConfig, AlphaZeroConfig
from environments.base import BaseEnvironment, StateType, ActionType
from algorithms.mcts import AlphaZeroMCTS, get_state_key
from models.networks import AlphaZeroNet  # Import the network class
from factories import get_environment  # To create env inside actor


@ray.remote
class SelfPlayWorker:
    """
    A Ray actor responsible for running a single self-play game using AlphaZero MCTS.
    """

    def __init__(
        self,
        worker_id: int,
        env_config: EnvConfig,
        agent_config: AlphaZeroConfig,
        initial_network_state: Dict,  # Pass network state_dict
    ):
        self.worker_id = worker_id
        self.env_config = env_config
        self.agent_config = agent_config

        # Instantiate environment and network within the actor
        self.env = get_environment(self.env_config)
        # Create network instance and load state
        self.network = AlphaZeroNet(
            self.env,
            hidden_layer_size=agent_config.hidden_layer_size,
            num_hidden_layers=agent_config.num_hidden_layers,
        )
        self.network.load_state_dict(initial_network_state)
        self.network.eval()  # Set to evaluation mode

        self.mcts = AlphaZeroMCTS(
            env=self.env,
            config=self.agent_config,
            network=self.network,
            network_cache={},  # Each worker has its own cache for its game
        )
        self.network_cache = self.mcts.network_cache  # Alias for clarity

    def update_network_weights(self, network_state: Dict):
        """Updates the actor's local network weights."""
        try:
            self.network.load_state_dict(network_state)
            self.network.eval()
        except Exception as e:
            # Log error appropriately if logger is configured for Ray actors
            print(f"Worker {self.worker_id}: Error loading network state: {e}")

    def run_game(self) -> List[Tuple[StateType, np.ndarray, float]]:
        """
        Runs a complete self-play game and returns the collected experiences.
        """
        game_history: List[Tuple[StateType, ActionType, np.ndarray]] = []
        observation = self.env.reset()
        self.mcts.reset_root()  # Ensure MCTS starts fresh for the game
        self.network_cache.clear()  # Clear cache for the new game

        while not self.env.is_game_over():
            state_before_action = observation.copy()

            # Prepare MCTS for the search for this turn
            self.mcts.env = self.env.copy()  # Ensure MCTS internal env is current
            self.mcts.env.set_state(observation)
            self.mcts.prepare_for_next_search(train=True)

            # --- Run MCTS Simulations ---
            # The new run_simulations method handles the internal loop
            self.mcts.run_simulations()
            # --- End MCTS Simulations ---

            # Get the final action and policy target from MCTS results
            action, policy_target = self.mcts.get_result()

            # Validate and step environment
            current_legal_actions = self.env.get_legal_actions()
            if action not in current_legal_actions:
                # Handle illegal action - maybe log and end game?
                print(
                    f"Worker {self.worker_id}: Illegal action {action} chosen! Legal: {current_legal_actions}"
                )
                break  # End game prematurely

            if policy_target is None:
                print(f"Worker {self.worker_id}: MCTS returned None policy target.")
                break  # End game prematurely

            game_history.append((state_before_action, action, policy_target))
            observation, reward, done = self.env.step(action)

            if not done:
                self.mcts.advance_root(action)
            else:
                break  # Game finished

        # Process finished game to create experiences
        experiences = []
        if game_history:
            winner = self.env.get_winning_player()
            final_outcome = 1.0 if winner == 0 else -1.0 if winner == 1 else 0.0
            # Use the agent's processing logic (needs to be accessible or replicated)
            # For simplicity, let's replicate the core value assignment logic here:
            num_steps = len(game_history)
            for i, (state, _, policy) in enumerate(game_history):
                # Simple reward assignment: final outcome discounted back
                # TODO: Use proper N-step returns or GAE if needed later
                value_target = (
                    final_outcome  # Simplistic: assign final outcome to all steps
                )
                experiences.append((state, policy, value_target))

        return experiences
