import sys
import json
import datetime
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
)

import numpy as np
from tqdm import tqdm
from loguru import logger
import ray

from core.config import AppConfig, DATA_DIR, AlphaZeroConfig
from environments.base import BaseEnvironment, StateType, ActionType
from agents.alphazero_agent import AlphaZeroAgent
from factories import get_environment, get_agents
from utils.plotting import plot_losses
from algorithms.mcts import AlphaZeroMCTS, get_state_key, MCTSNode, PredictResult
from actors.inference_actor import InferenceActor

LOG_DIR = DATA_DIR / "game_logs"


def _default_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    # Add other custom types here if needed
    # Example: if isinstance(obj, SomeCustomClass): return obj.to_dict()
    try:
        return obj.__dict__  # Fallback for simple objects
    except AttributeError:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )


def save_game_log(
    logged_history: List[Tuple[StateType, ActionType, np.ndarray, float]],
    iteration: int,
    game_index: int,
    env_name: str,
):
    """Saves the processed game history to a JSON file."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{env_name}_game{game_index:04d}_{timestamp}.json"
        filepath = LOG_DIR / filename

        # Prepare data for JSON
        serializable_log = []
        for state, action, policy, value in logged_history:
            # Ensure state and action are serializable
            serializable_log.append(
                {
                    "state": state,
                    "action": action,
                    "policy_target": policy.tolist(),  # Convert numpy array
                    "value_target": value,
                }
            )

        with open(filepath, "w") as f:
            json.dump(serializable_log, f, indent=2, default=_default_serializer)

    except Exception as e:
        logger.error(
            f"Error saving game log for iter {iteration}, game {game_index}: {e}"
        )


def load_game_logs_into_buffer(agent: AlphaZeroAgent, env_name: str, buffer_limit: int):
    """Loads existing game logs from LOG_DIR into the agent's replay buffer."""
    loaded_games = 0
    loaded_steps = 0
    if not LOG_DIR.exists():
        logger.info("Log directory not found. Starting with an empty buffer.")
        return

    logger.info(f"Scanning {LOG_DIR} for existing '{env_name}' game logs...")
    log_files = sorted(LOG_DIR.glob(f"{env_name}_game*.json"), reverse=True)

    if not log_files:
        logger.info("No existing game logs found for this environment.")
        return

    for filepath in tqdm(log_files, desc="Loading Logs"):
        if len(agent.replay_buffer) >= buffer_limit:
            logger.info(
                f"Replay buffer reached limit ({buffer_limit}). Stopping log loading."
            )
            break
        try:
            with open(filepath, "r") as f:
                game_data = json.load(f)

            if not isinstance(game_data, list):
                logger.warning(
                    f"Skipping invalid log file (not a list): {filepath.name}"
                )
                continue

            steps_in_game = 0
            for step_data in game_data:
                if len(agent.replay_buffer) >= buffer_limit:
                    break  # Stop adding steps if buffer full mid-game

                # Extract required components for replay buffer
                state = step_data.get("state")
                policy_target_list = step_data.get("policy_target")
                value_target = step_data.get("value_target")

                if (
                    state is not None
                    and policy_target_list is not None
                    and value_target is not None
                ):
                    # Convert policy back to numpy array
                    policy_target = np.array(policy_target_list, dtype=np.float32)

                    # --- Standardize state loaded from JSON ---
                    # Convert board/piles list back to numpy array
                    if "board" in state and isinstance(state["board"], list):
                        state["board"] = np.array(
                            state["board"], dtype=np.int8
                        )  # Match Connect4 dtype
                    elif "piles" in state and isinstance(state["piles"], list):
                        state["piles"] = np.array(
                            state["piles"], dtype=np.int32
                        )  # Match NimEnv dtype

                    agent.replay_buffer.append((state, policy_target, value_target))
                    loaded_steps += 1
                    steps_in_game += 1
                else:
                    logger.warning(
                        f"Skipping step with missing data in {filepath.name}"
                    )

            if steps_in_game > 0:
                loaded_games += 1

        except json.JSONDecodeError:
            logger.warning(f"Skipping corrupted JSON file: {filepath.name}")
        except Exception as e:
            logger.warning(f"Error processing log file {filepath.name}: {e}")

    logger.info(
        f"Loaded {loaded_steps} steps from {loaded_games} game logs into replay buffer."
    )
    logger.info(f"Current buffer size: {len(agent.replay_buffer)}/{buffer_limit}")


def _initialize_parallel_games(num_parallel_games: int, env_factory: callable) -> Dict:
    """Initializes environments, observations, and tracking structures for parallel games."""
    envs = [env_factory() for _ in range(num_parallel_games)]
    observations = [env.reset() for env in envs]
    parallel_histories: Dict[int, List[Tuple[StateType, ActionType, np.ndarray]]] = {
        i: [] for i in range(num_parallel_games)
    }
    game_is_done = [False] * num_parallel_games
    # Track games needing an action search
    games_needing_action: List[int] = list(range(num_parallel_games))
    # Store MCTS intermediate results between phases
    # Updated mcts_pending_sims structure
    mcts_pending_sims: Dict[int, Dict[str, List[MCTSNode]]] = {}
    mcts_root_keys: Dict[int, str] = {}  # Store root key per game
    # Store pending network requests aggregated across all games, mapping key to list of game_idx
    pending_requests: Dict[str, Tuple[StateType, List[int]]] = {}
    # Store completed actions and policy targets after phase 2
    completed_actions: Dict[int, Tuple[Optional[ActionType], Optional[np.ndarray]]] = {}
    # Store state before MCTS search for replay buffer
    states_before_action: Dict[int, StateType] = {}

    return {
        "envs": envs,
        "observations": observations,
        "parallel_histories": parallel_histories,
        "game_is_done": game_is_done,
        "games_needing_action": games_needing_action,
        "mcts_pending_sims": mcts_pending_sims,
        "mcts_root_keys": mcts_root_keys,
        "pending_requests": pending_requests,
        "completed_actions": completed_actions,
        "states_before_action": states_before_action,
    }


def _process_network_batch(
    agent: AlphaZeroAgent,
    pending_requests: Dict[str, Tuple[StateType, List[int]]],
    inference_batch_size: int,
) -> Dict[str, PredictResult]:
    """
    Processes a batch of network prediction requests synchronously.

    Args:
        agent: The AlphaZeroAgent instance.
        pending_requests: List of (state_key, state_obs) tuples needing prediction.
        inference_batch_size: Maximum batch size for network inference.

    Returns:
        A dictionary mapping state_key to PredictResult (policy_np, value).
    """
    network_results: Dict[str, PredictResult] = {}
    if not agent.network or not pending_requests:
        return network_results

    # Process requests in batches based on the dictionary structure
    all_state_keys = list(pending_requests.keys())
    num_unique_requests = len(all_state_keys)
    logger.debug(
        f"Processing {num_unique_requests} unique network requests in batches..."
    )

    for i in range(0, num_unique_requests, inference_batch_size):
        batch_keys = all_state_keys[i : i + inference_batch_size]
        states_to_predict_batch = [pending_requests[key][0] for key in batch_keys]

        logger.debug(
            f"  Processing batch {i//inference_batch_size + 1}, size: {len(batch_keys)}"
        )

        policy_list, value_list = agent.network.predict_batch(states_to_predict_batch)

        if len(policy_list) != len(states_to_predict_batch):
            logger.error("Network batch result size mismatch!")
            # Continue processing other batches if possible, but log error
            continue

        for j, state_key in enumerate(batch_keys):  # Use batch_keys here
            network_results[state_key] = (policy_list[j], value_list[j])

    logger.debug(
        f"Finished processing network requests. Results count: {len(network_results)}"
    )
    return network_results


def _step_environments(
    completed_actions: Dict[int, Tuple[Optional[ActionType], Optional[np.ndarray]]],
    envs: List[BaseEnvironment],
    observations: List[StateType],
    game_is_done: List[bool],
    parallel_histories: Dict[int, List[Tuple[StateType, ActionType, np.ndarray]]],
    states_before_action: Dict[int, StateType],
) -> Tuple[Dict, Dict, Dict, int]:
    """
    Steps the environments for games where MCTS search has completed.

    Returns:
        A tuple containing:
        - next_observations: Dict mapping game_idx to the new observation.
        - rewards: Dict mapping game_idx to the reward received.
        - step_dones: Dict mapping game_idx to the done status after the step.
        - steps_taken_this_cycle: Count of successful steps taken.
    """
    next_observations = {}
    rewards = {}
    step_dones = {}
    steps_taken_this_cycle = 0

    games_to_step = list(completed_actions.keys())

    for game_idx in games_to_step:
        action, policy_target = completed_actions[game_idx]

        if game_is_done[game_idx]:
            step_dones[game_idx] = True
            next_observations[game_idx] = observations[game_idx]
            rewards[game_idx] = 0.0
            continue

        if action is None:
            logger.error(
                f"Game {game_idx} MCTS completed but action is None. Ending game."
            )
            step_dones[game_idx] = True
            next_observations[game_idx] = observations[game_idx]
            rewards[game_idx] = 0.0
            continue

        try:
            obs, reward, done = envs[game_idx].step(action)
            next_observations[game_idx] = obs
            rewards[game_idx] = reward
            step_dones[game_idx] = done
            steps_taken_this_cycle += 1

            state_before_action_val = states_before_action.get(game_idx)
            if policy_target is None:
                logger.warning(
                    f"Game {game_idx}, Step {len(parallel_histories[game_idx])}: Skipping history append - MCTS failed to produce policy target (policy_target is None)."
                )
            elif state_before_action_val is None:
                logger.error(
                    f"Game {game_idx}, Step {len(parallel_histories[game_idx])}: Skipping history append - Missing state_before_action."
                )
            else:
                # Only append if policy_target is valid
                parallel_histories[game_idx].append(
                    (state_before_action_val, action, policy_target)
                )

        except ValueError as e:
            logger.warning(
                f"Invalid action {action} in game {game_idx} step. Error: {e}. Ending game."
            )
            next_observations[game_idx] = observations[game_idx]
            rewards[game_idx] = 0.0
            step_dones[game_idx] = True

    return next_observations, rewards, step_dones, steps_taken_this_cycle


def _process_finished_games(
    games_stepped_indices: List[int],
    agent: AlphaZeroAgent,
    envs: List[BaseEnvironment],
    observations: List[StateType],
    game_is_done: List[bool],
    parallel_histories: Dict[int, List[Tuple[StateType, ActionType, np.ndarray]]],
    games_needing_action: List[int],  # Renamed from ready_for_action_search
    next_observations: Dict,
    step_dones: Dict,
    all_experiences: List[Tuple[StateType, np.ndarray, float]],
    iteration: int,
    env_name: str,
    finished_episodes_count_offset: int,  # To get correct game index for logging
    pbar: Optional[tqdm] = None,
):
    """
    Updates game states after stepping, processes finished games, resets environments,
    and updates tracking lists.
    """
    episodes_finished_this_cycle = 0
    for game_idx in games_stepped_indices:
        observations[game_idx] = next_observations[game_idx]
        game_is_done[game_idx] = step_dones[game_idx]

        if game_is_done[game_idx]:
            winner = envs[game_idx].get_winning_player()
            final_outcome = 1.0 if winner == 0 else -1.0 if winner == 1 else 0.0
            game_history = parallel_histories[game_idx]
            valid_history = [(s, a, p) for s, a, p in game_history if p is not None]

            if len(valid_history) != len(game_history):
                logger.warning(
                    f"Game {game_idx} history contained steps with None policy target."
                )

            if valid_history:
                episode_result = agent.process_finished_episode(
                    valid_history, final_outcome
                )
                all_experiences.extend(episode_result.buffer_experiences)
                save_game_log(
                    episode_result.logged_history,
                    iteration,
                    finished_episodes_count_offset
                    + episodes_finished_this_cycle
                    + 1,  # Calculate unique game index
                    env_name,
                )
            else:
                logger.warning(
                    f"Game {game_idx} finished with no valid history steps. No experiences added."
                )

            episodes_finished_this_cycle += 1
            if pbar:
                pbar.update(1)
            logger.debug(
                f"Game {game_idx} finished. Total finished this cycle: {episodes_finished_this_cycle}"
            )

            # Reset environment and history
            observations[game_idx] = envs[game_idx].reset()
            game_is_done[game_idx] = False
            parallel_histories[game_idx] = []
            if game_idx not in games_needing_action:
                games_needing_action.append(game_idx)
        else:
            # Game not done, mark ready for next action search
            if game_idx not in games_needing_action:
                games_needing_action.append(game_idx)

    return episodes_finished_this_cycle


# Experience = Tuple[StateType, np.ndarray, float]


class SelfPlayManager:
    def __init__(
        self,
        env_factory: callable,
        inference_actor_handle: ray.actor.ActorHandle,
        agent_config: AlphaZeroConfig,  # Still need agent config for MCTS
        num_games_to_collect: int,
        num_parallel_games: int,
        iteration: int,
        env_name: str,
        network_cache: Dict,
    ):
        self.num_games_to_collect = num_games_to_collect
        self.num_parallel_games = num_parallel_games
        self.network_cache = network_cache
        self.inference_actor = inference_actor_handle
        self.agent_config = agent_config
        self.inference_batch_size = agent_config.inference_batch_size  # Get from config
        self.iteration = iteration
        self.env_name = env_name

        self.all_experiences = []
        self.num_finished_games = 0
        self.num_steps_taken = 0

        self.envs = [env_factory() for _ in range(num_parallel_games)]
        self.observations = [env.reset() for env in self.envs]
        self.histories = [[] for _ in range(num_parallel_games)]
        self.states_before_action = {}
        self.mcts_instances: Dict[int, AlphaZeroMCTS] = {}
        self.pending_requests: Dict[int, StateType] = {}  # game_idx -> state_dict

    def run(self):
        pbar = tqdm(
            total=self.num_games_to_collect, desc="Self-Play Games", unit="game"
        )
        pbar.update(self.num_finished_games)

        while self.num_finished_games < self.num_games_to_collect:
            games_finished_before_iter = self.num_finished_games
            self._run_one_iter()
            games_finished_after_iter = self.num_finished_games
            # Update progress bar only by the number of games finished in this iteration
            pbar.update(games_finished_after_iter - games_finished_before_iter)

        pbar.close()

        return self.all_experiences

    def _run_one_iter(self):
        """Performs one iteration of the self-play loop."""
        self._start_needed_searches()
        self._repeatedly_handle_pending_requests()
        self._handle_completed_searches()

    def _start_needed_searches(self):
        """Starts MCTS search for games that need it (new games or games ready for next move)."""
        for game_idx in range(self.num_parallel_games):
            if self.envs[game_idx].is_game_over() or game_idx in self.pending_requests:
                continue

            if game_idx not in self.mcts_instances:
                # New game
                self.states_before_action[game_idx] = self.observations[game_idx].copy()
                mcts = AlphaZeroMCTS(
                    env=self.envs[game_idx],
                    config=self.agent_config,
                    network_cache=self.network_cache,
                )

                mcts.env = self.envs[game_idx].copy()
                mcts.env.set_state(self.observations[game_idx])
                mcts.prepare_for_next_search(train=True)
                self.mcts_instances[game_idx] = mcts
                self._get_mcts_network_request(game_idx=game_idx, response=None)
            else:
                # Continue game
                self.states_before_action[game_idx] = self.observations[game_idx].copy()
                mcts = self.mcts_instances[game_idx]
                mcts.prepare_for_next_search(train=True)
                self._get_mcts_network_request(game_idx=game_idx, response=None)

    def _get_mcts_network_request(
        self, game_idx: int, response: Optional[Tuple[np.ndarray, float]] = None
    ):
        mcts: AlphaZeroMCTS = self.mcts_instances[game_idx]
        request = mcts.get_network_request(previous_response=response)
        if request is not None:
            self.pending_requests[game_idx] = request

    def _repeatedly_handle_pending_requests(self):
        while len(self.pending_requests) >= self.inference_batch_size // 2:
            batch_size = min(self.inference_batch_size, len(self.pending_requests))
            game_indices = list(self.pending_requests.keys())[:batch_size]
            batch_states = [self.pending_requests[idx] for idx in game_indices]

            inference_task_ref = self.inference_actor.predict_batch.remote(batch_states)
            policy_list, value_list = ray.get(inference_task_ref)

            assert len(policy_list) == len(game_indices)

            for i, game_idx in enumerate(game_indices):
                assert game_idx in self.pending_requests
                evaluated_state = self.pending_requests[game_idx]
                state_key = get_state_key(evaluated_state)
                response = (policy_list[i], value_list[i])

                self.network_cache[state_key] = response
                del self.pending_requests[game_idx]
                self._get_mcts_network_request(game_idx=game_idx, response=response)

    def _handle_completed_searches(self):
        """Handles games where MCTS search has finished for the current step."""
        completed_game_indices = []
        for game_idx, mcts in list(self.mcts_instances.items()):
            if game_idx not in self.pending_requests:
                completed_game_indices.append(game_idx)

        for game_idx in completed_game_indices:
            mcts = self.mcts_instances[game_idx]
            action, action_visits = mcts.get_result()
            policy = self._calculate_policy(action_visits, self.envs[game_idx])

            # If this triggers, see if it can be moved into get_result()
            current_legal_actions = self.envs[game_idx].get_legal_actions()
            assert action is not None
            assert (
                action in current_legal_actions
            ), f"Illegal action chosen by MCTS! Action: {action}, Legal: {current_legal_actions}, State: {self.observations[game_idx]}"

            next_obs, reward, done = self.envs[game_idx].step(action)
            self.num_steps_taken += 1

            assert policy is not None, "MCTS returned valid action but None policy."
            self.histories[game_idx].append(
                (self.states_before_action[game_idx], action, policy)
            )
            self.observations[game_idx] = next_obs

            if game_idx in self.states_before_action:
                del self.states_before_action[game_idx]

            if done:
                self._handle_finished_game(game_idx=game_idx)
            else:
                mcts.advance_root(action)
                mcts.env = self.envs[game_idx].copy()

    def _handle_finished_game(self, game_idx):
        winner = self.envs[game_idx].get_winning_player()
        final_outcome = 1.0 if winner == 0 else -1.0 if winner == 1 else 0.0

        valid_history = [
            (s, a, p) for s, a, p in self.histories[game_idx] if p is not None
        ]

        assert valid_history
        logged_history_for_saving = []
        for i, (state, action, policy) in enumerate(valid_history):
            value_target = final_outcome
            self.all_experiences.append((state, policy, value_target))
            logged_history_for_saving.append((state, action, policy, value_target))

        save_game_log(
            logged_history=logged_history_for_saving,
            iteration=self.iteration,
            game_index=self.num_finished_games + 1,
            env_name=self.env_name,
        )

        self.observations[game_idx] = self.envs[game_idx].reset()
        self.histories[game_idx] = []

        if game_idx in self.mcts_instances:
            del self.mcts_instances[game_idx]
        self.num_finished_games += 1

    def _calculate_policy(
        self, action_visits: Dict[ActionType, int], env: BaseEnvironment
    ) -> np.ndarray:
        """Calculates the policy target vector from visit counts."""
        policy_size = env.policy_vector_size

        policy_target = np.zeros(policy_size, dtype=np.float32)
        total_visits = sum(action_visits.values())
        assert total_visits

        for action, visits in action_visits.items():
            action_key = tuple(action) if isinstance(action, list) else action
            action_idx = env.map_action_to_policy_index(action_key)
            assert action_idx is not None and 0 <= action_idx < policy_size
            policy_target[action_idx] = visits / total_visits

        # Normalize policy target (important if some actions couldn't be mapped)
        current_sum = policy_target.sum()
        if abs(current_sum - 1.0) > 1e-5:
            if current_sum > 1e-6:
                policy_target /= current_sum
            elif policy_target.size > 0:
                logger.warning(
                    f"Policy target sum is near zero ({current_sum}). Setting uniform."
                )
                policy_target.fill(1.0 / policy_target.size)
            else:
                logger.error("Cannot normalize zero-size policy target.")
                assert False

        return policy_target


def run_training(config: AppConfig, env_name_override: str = None):
    """Runs the AlphaZero training process."""

    # --- Environment Selection ---
    if env_name_override:
        config.env.name = env_name_override

    # --- Instantiation ---
    env = get_environment(config.env)
    agents = get_agents(env, config)
    agent = agents.get("AlphaZero")

    if not isinstance(agent, AlphaZeroAgent):
        logger.error("Failed to retrieve AlphaZeroAgent from factory.")
        return

    load_game_logs_into_buffer(
        agent, config.env.name, config.alpha_zero.replay_buffer_size
    )

    logger.info(
        f"Starting AlphaZero training for {config.training.num_iterations} iterations..."
    )
    logger.info(
        f"({config.training.num_games_per_iteration} self-play games per iteration)"
    )

    # Lists to store losses for plotting
    total_losses = []
    value_losses = []
    policy_losses = []

    outer_loop_iterator = range(config.training.num_iterations)

    for iteration in outer_loop_iterator:
        logger.info(
            f"\n--- Iteration {iteration + 1}/{config.training.num_iterations} ---"
        )

        # 1. Self-Play Phase
        if agent.network:
            agent.network.eval()
        if agent.network:
            agent.network.eval()  # Ensure main agent network is in eval mode

        logger.info("Collecting self-play data using remote inference actor...")

        # --- Create Inference Actor ---
        network_state_ref = None
        if agent.network:
            # Send weights to CPU before putting in object store / sending to actor
            network_state_dict = {
                k: v.cpu() for k, v in agent.network.state_dict().items()
            }
            network_state_ref = network_state_dict  # Pass the dict directly
        else:
            network_state_ref = {}  # Empty dict if no network

        # Create the single inference actor
        inference_actor = InferenceActor.remote(
            initial_network_state=network_state_ref,
            env_config=config.env,
            agent_config=config.alpha_zero,
        )
        # Verify actor started and device (optional)
        try:
            actor_device = ray.get(inference_actor.get_device.remote())
            logger.info(
                f"InferenceActor created successfully on device: {actor_device}"
            )
        except Exception as e:
            logger.error(f"Failed to create or communicate with InferenceActor: {e}")
            # Cannot proceed without inference actor
            if ray.is_initialized():
                ray.shutdown()
            return

        # --- Run SelfPlayManager (locally, using remote inference) ---
        env_factory = lambda: get_environment(config.env)
        # The network cache is still managed locally by SelfPlayManager
        network_cache = {}

        # Pass inference actor handle to SelfPlayManager
        manager = SelfPlayManager(
            env_factory=env_factory,
            inference_actor_handle=inference_actor,  # Pass handle
            agent_config=config.alpha_zero,  # Pass agent config
            num_games_to_collect=config.training.num_games_per_iteration,
            num_parallel_games=config.alpha_zero.num_parallel_games,
            iteration=iteration + 1,
            env_name=config.env.name,
            network_cache=network_cache,
        )
        self_play_experiences = manager.run()  # Run the local manager loop

        # --- Add experiences to buffer ---
        agent.add_experiences_to_buffer(self_play_experiences)
        logger.info(
            f"Collected and added {len(self_play_experiences)} total experiences to replay buffer."
        )

        # --- Optional: Update Inference Actor Weights ---
        # If the network was updated during the learning phase (which happens next),
        # you might want to update the inference actor's weights before the *next*
        # self-play iteration. This can be done here or at the start of the next iteration.
        # Example:
        # if agent.network and iteration > 0: # Check if network exists and not first iteration
        #     new_weights = {k: v.cpu() for k, v in agent.network.state_dict().items()}
        #     inference_actor.update_weights.remote(new_weights)

        # 2. Learning Phase
        logger.info("Running learning step...")
        loss_results = agent.learn()
        if loss_results:
            total_loss, value_loss, policy_loss = loss_results
            total_losses.append(total_loss)
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)
        else:
            logger.info("Learning Time: Skipped (buffer too small)")

        # 3. Save Checkpoint Periodically
        # TODO: Make save frequency configurable
        if (iteration + 1) % 10 == 0:  # Save every 10 iterations
            logger.info("Saving agent checkpoint...")
            agent.save()

        buffer_size = len(agent.replay_buffer)
        logger.info(
            f"Iteration {iteration + 1} complete. Buffer size: {buffer_size}/{config.alpha_zero.replay_buffer_size}"
        )
        if total_losses:
            logger.info(
                f"  Latest Losses: Total={total_losses[-1]:.4f}, Value={value_losses[-1]:.4f}, Policy={policy_losses[-1]:.4f}"
            )
        else:
            logger.info("  Latest Losses: (No learning step occurred)")

        # 4. Run Sanity Checks Periodically (and not on first iteration if frequency > 1)
        if (
            config.training.sanity_check_frequency > 0
            and (iteration + 1) % config.training.sanity_check_frequency == 0
        ):

            run_sanity_checks(env, agent)

    logger.info("\nTraining complete. Saving final agent state.")
    agent.save()

    logger.info("Plotting training losses...")
    plot_losses(total_losses, value_losses, policy_losses)

    # Run sanity checks one last time on the final trained agent
    # This ensures checks run even if num_iterations isn't a multiple of frequency
    logger.info("\n--- Running Final Sanity Checks ---")
    run_sanity_checks(env, agent)

    logger.info("\n--- AlphaZero Training Finished ---")


# --- Sanity Check Function ---
def run_sanity_checks(env: BaseEnvironment, agent: AlphaZeroAgent):
    """Runs network predictions on predefined sanity check states."""
    logger.info("\n--- Running Periodic Sanity Checks ---")
    sanity_states = env.get_sanity_check_states()

    if not agent.network:
        logger.warning("Cannot run sanity checks: Agent network not initialized.")
        return

    agent.network.eval()

    if not sanity_states:
        logger.info("No sanity check states defined for this environment.")
        return

    for check_case in sanity_states:
        logger.info(f"\nChecking State: {check_case.description}")
        if "board" in check_case.state:
            logger.info("Board:")
            logger.info(f"\n{check_case.state['board']}")
        elif "piles" in check_case.state:
            logger.info(f"Piles: {check_case.state['piles']}")
        logger.info(f"Current Player: {check_case.state['current_player']}")

        try:
            policy_np, value_np = agent.network.predict(check_case.state)
            logger.info(
                f"  Value: Expected={check_case.expected_value:.1f}, Predicted={value_np:.4f}"
            )

            temp_env = env.copy()
            temp_env.set_state(check_case.state)
            legal_actions = temp_env.get_legal_actions()

            action_probs = {}
            for action in legal_actions:
                idx = agent.network.get_action_index(action)
                if idx is not None and 0 <= idx < len(policy_np):
                    action_probs[action] = policy_np[idx]
                else:
                    action_probs[action] = -1  # Indicate mapping error

            sorted_probs = sorted(
                action_probs.items(), key=lambda item: item[1], reverse=True
            )

            logger.info(f"  Predicted Probabilities for Legal Actions:")
            if not legal_actions:
                logger.info("    - (No legal actions)")
            else:
                for action, prob in sorted_probs:
                    if prob >= 0:
                        highlight = " <<< BEST" if action == sorted_probs[0][0] else ""
                        logger.info(f"    - {action}: {prob:.4f}{highlight}")
                    else:
                        logger.info(f"    - {action}: (Error mapping action)")

        except Exception as e:
            logger.error(f"  Error during prediction for this state: {e}")


if __name__ == "__main__":
    config = AppConfig()
    env_override = None

    if len(sys.argv) > 1:
        env_override = sys.argv[1]  # e.g., python train_alphazero.py Nim

    # Remove the default handler to prevent duplicate messages if re-adding stderr
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    run_training(config, env_name_override=env_override)
