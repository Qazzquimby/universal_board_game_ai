import sys
import json
import datetime
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
)
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from loguru import logger

from core.config import AppConfig, DATA_DIR
from environments.base import BaseEnvironment, StateType, ActionType
from agents.alphazero_agent import AlphaZeroAgent
from factories import get_environment, get_agents
from utils.plotting import plot_losses
from algorithms.mcts import Timer, PredictResult, MCTSNode


LOG_DIR = DATA_DIR / "game_logs"


# Removed PredictRequest dataclass
# Removed _handle_predict_request function


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
    log_files = sorted(
        LOG_DIR.glob(f"{env_name}_game*.json"), reverse=True
    )  # Sort newest first (now correctly based on timestamp in filename)

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


@dataclass
class SelfPlayResult:
    """Holds the results of the self-play data collection phase."""

    experiences: List[Tuple[StateType, np.ndarray, float]]
    collection_time_ms: float
    network_time_ms: float


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


# Removed _start_mcts_searches function


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


# Removed _advance_and_process_generators function


def collect_parallel_self_play_data(
    env_factory: callable,  # Function to create a new env instance
    agent: AlphaZeroAgent,
    num_episodes_to_collect: int,
    num_parallel_games: int,
    iteration: int,
    env_name: str,
    inference_batch_size: int,  # Add batch size from config
    use_tqdm: bool = True,
) -> SelfPlayResult:
    """
    Runs multiple self-play games concurrently, managing MCTS search generators
    and performing batched network inference across games.

    Args:
        env_factory: A function that returns a new instance of the environment.
        agent: AlphaZeroAgent instance (contains the network and MCTS).
        num_episodes_to_collect: Target number of episodes to complete.
        num_parallel_games: Number of games to run simultaneously.
        iteration: Current training iteration number (for logging).
        env_name: Name of the environment (for logging).
        inference_batch_size: Max number of requests to batch for the network.
        use_tqdm: Whether to display a progress bar.

    Returns:
        SelfPlayResult: An object containing the collected experiences and timing information.
    """
    all_experiences = []
    finished_episodes_count = 0
    total_steps_taken = 0

    # --- Initialize Parallel Game States ---
    game_states = _initialize_parallel_games(num_parallel_games, env_factory)
    envs = game_states["envs"]
    observations = game_states["observations"]
    parallel_histories = game_states["parallel_histories"]
    game_is_done = game_states["game_is_done"]
    games_needing_action = game_states["games_needing_action"]
    mcts_pending_sims = game_states["mcts_pending_sims"]
    mcts_root_keys = game_states["mcts_root_keys"]
    pending_requests = game_states["pending_requests"]
    completed_actions = game_states["completed_actions"]
    states_before_action = game_states["states_before_action"]

    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=num_episodes_to_collect, desc="Parallel Self-Play", leave=False
        )

    collection_timer = Timer()
    with collection_timer:
        while finished_episodes_count < num_episodes_to_collect:

            # --- Phase 1: Prepare MCTS Simulations & Aggregate Requests ---
            games_to_search = games_needing_action[:]  # Copy list
            games_needing_action.clear()
            # pending_requests is now the global dict, cleared later
            games_prepared_this_cycle: List[int] = []  # Track games prepared

            if not games_to_search:
                # If no games need action and no actions are pending, check if done
                if (
                    not completed_actions
                    and finished_episodes_count >= num_episodes_to_collect
                ):
                    break
                elif not completed_actions:
                    logger.warning(
                        "No games need action, but collection not complete. Waiting..."
                    )
                    # Add a small sleep or check condition? For now, just continue loop.
                    # This might indicate a logic error if it persists.
                    continue

            logger.debug(f"Preparing MCTS simulations for games: {games_to_search}")
            for game_idx in games_to_search:
                if game_is_done[game_idx]:
                    continue

                state = observations[game_idx]
                states_before_action[game_idx] = state.copy()

                try:
                    # Updated return signature for prepare_simulations
                    (
                        requests_dict,
                        pending_sim_dict, # Now Dict[str, List[MCTSNode]]
                        root_key,
                    ) = agent.mcts.prepare_simulations(
                        envs[game_idx], state, train=True
                    )

                    mcts_pending_sims[game_idx] = pending_sim_dict
                    mcts_root_keys[game_idx] = root_key
                    games_prepared_this_cycle.append(game_idx)

                    # Aggregate requests into the global pending_requests dict
                    for state_key, state_obs in requests_dict.items():
                        if state_key not in pending_requests:
                            pending_requests[state_key] = (state_obs, [])
                        if game_idx not in pending_requests[state_key][1]:
                            pending_requests[state_key][1].append(game_idx)

                # I made it IOError to disable it. There is no IOError happening
                except IOError as e:
                    logger.error(
                        f"Error during MCTS Phase 1 for game {game_idx}: {e}",
                        exc_info=True,
                    )
                    game_is_done[game_idx] = True  # Treat as error/end game
                    if game_idx not in completed_actions:
                        completed_actions[game_idx] = (None, None)
                    # Clean up any partial state after error during prepare_simulations
                    if game_idx in mcts_pending_sims:
                        del mcts_pending_sims[game_idx]
                    if game_idx in mcts_root_keys:
                        del mcts_root_keys[game_idx]

            # --- Phase 2: Batch Network Inference ---
            network_results: Dict[str, PredictResult] = {}
            # Track which games are waiting for which keys
            games_awaiting_results: Dict[int, List[str]] = {}

            if pending_requests:
                logger.debug(
                    f"Processing {len(pending_requests)} unique network requests..."
                )
                # Store which games are waiting for which keys *before* processing
                for key, (_, game_indices) in pending_requests.items():
                    for game_idx in game_indices:
                        if game_idx not in games_awaiting_results:
                            games_awaiting_results[game_idx] = []
                        games_awaiting_results[game_idx].append(key)

                network_results = _process_network_batch(
                    agent=agent,
                    pending_requests=pending_requests,  # Pass the dict
                    inference_batch_size=inference_batch_size,
                )
                # Clear pending requests *after* processing
                pending_requests.clear()
            else:
                logger.debug("No network requests needed this cycle.")

            # --- Phase 3: Process Results and Select Actions ---
            games_to_process = list(games_awaiting_results.keys())
            logger.debug(f"Processing MCTS results for games: {games_to_process}")

            for game_idx in games_to_process:
                # Check for pending sims and root key
                if (
                    game_idx not in mcts_pending_sims
                    or game_idx not in mcts_root_keys  # Check root key exists
                ):
                    logger.warning(
                        f"Missing MCTS intermediate state for game {game_idx}. Skipping."
                    )
                    # Ensure it doesn't get stuck waiting
                    if game_idx in games_awaiting_results:
                        del games_awaiting_results[game_idx]
                    continue

                # Check if all required network results are available for this game
                required_keys = games_awaiting_results.get(game_idx, [])
                results_for_game = {
                    k: network_results[k] for k in required_keys if k in network_results
                }

                # Only proceed if all required results were obtained
                if len(results_for_game) == len(required_keys):
                    # try:
                    (
                        action,
                        policy_target,
                    ) = agent.mcts.process_results_and_select_action(
                        network_results=results_for_game,
                        pending_sims=mcts_pending_sims[game_idx],
                        root_state_key=mcts_root_keys[game_idx],
                        train=True,
                        current_step=observations[game_idx].get("step_count", 0),
                        env=envs[game_idx],
                    )
                    completed_actions[game_idx] = (action, policy_target)
                    logger.debug(f"Game {game_idx} completed search. Action: {action}")
                else:
                    logger.warning(
                        f"Game {game_idx}: Missing some network results ({len(results_for_game)}/{len(required_keys)}). Will retry next cycle."
                    )
                    # Add game back to needing action if it didn't complete processing
                    if game_idx not in games_needing_action:
                        games_needing_action.append(game_idx)

                # Clean up stored MCTS intermediate data for this game regardless of success/failure
                if game_idx in mcts_pending_sims:
                    del mcts_pending_sims[game_idx]
                # completed_sims cleanup removed
                if game_idx in mcts_root_keys:
                    del mcts_root_keys[game_idx]
                if game_idx in games_awaiting_results:
                    del games_awaiting_results[game_idx]

            # --- Step Environments ---
            games_ready_to_step = list(completed_actions.keys())
            if games_ready_to_step:
                logger.debug(f"Stepping environments for games: {games_ready_to_step}")
                (
                    next_observations,
                    rewards,
                    step_dones,
                    steps_this_cycle,
                ) = _step_environments(
                    completed_actions=completed_actions,
                    envs=envs,
                    observations=observations,
                    game_is_done=game_is_done,
                    parallel_histories=parallel_histories,
                    states_before_action=states_before_action,
                )
                total_steps_taken += steps_this_cycle

                # --- Update Game States and Handle Finished Games ---
                episodes_finished_this_cycle = _process_finished_games(
                    games_stepped_indices=games_ready_to_step,
                    agent=agent,
                    envs=envs,
                    observations=observations,
                    game_is_done=game_is_done,
                    parallel_histories=parallel_histories,
                    games_needing_action=games_needing_action,  # Pass renamed list
                    next_observations=next_observations,
                    step_dones=step_dones,
                    all_experiences=all_experiences,
                    iteration=iteration,
                    env_name=env_name,
                    finished_episodes_count_offset=finished_episodes_count,
                    pbar=pbar,
                )
                finished_episodes_count += episodes_finished_this_cycle

                # Clean up completed actions and stored states
                for game_idx in games_ready_to_step:
                    del completed_actions[game_idx]
                    if game_idx in states_before_action:
                        del states_before_action[game_idx]

            else:
                # If no actions were completed, check termination condition
                if finished_episodes_count >= num_episodes_to_collect:
                    break
                else:
                    # This might indicate a problem if no games are progressing
                    logger.warning(
                        "No actions completed and collection not finished. Checking status..."
                    )
                    if not games_needing_action:
                        logger.error(
                            "Stuck: No games need action, no actions completed. Breaking."
                        )
                        break

    # --- End of Collection Loop ---
    if pbar:
        pbar.close()

    # Calculate network time spent during this self-play phase
    network_time_ms = 0.0

    logger.info(
        f"Collected {len(all_experiences)} experiences from {finished_episodes_count} games ({total_steps_taken} steps)."
    )
    # Clear agent's internal episode history just in case act() left something
    agent.reset()

    return SelfPlayResult(
        experiences=all_experiences,
        collection_time_ms=collection_timer.elapsed_ms,
        network_time_ms=network_time_ms,
    )


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

    # --- Load Existing Game Logs into Buffer ---
    load_game_logs_into_buffer(
        agent, config.env.name, config.alpha_zero.replay_buffer_size
    )

    # --- Training Loop ---
    num_training_iterations = config.training.num_iterations
    num_episodes_per_iteration = config.training.num_episodes_per_iteration

    logger.info(
        f"Starting AlphaZero training for {num_training_iterations} iterations..."
    )
    logger.info(f"({num_episodes_per_iteration} self-play games per iteration)")

    # Lists to store losses for plotting
    total_losses = []
    value_losses = []
    policy_losses = []

    # Disable tqdm if running smoke test to potentially avoid encoding issues
    use_tqdm = not config.smoke_test
    outer_loop_iterator = range(num_training_iterations)

    for iteration in outer_loop_iterator:
        # Start timer for the whole iteration
        iteration_timer = Timer()
        with iteration_timer:
            logger.info(
                f"\n--- Iteration {iteration + 1}/{num_training_iterations} ---"
            )

            # 1. Self-Play Phase
            if agent.network:  # Check if network exists before setting mode
                agent.network.eval()  # Ensure network is in eval mode for self-play actions
            logger.info("Collecting self-play data (parallel)...")

            # Need a way to create new env instances for parallel collection
            env_factory = lambda: get_environment(config.env)

            # Call using keyword arguments and receive the dataclass object
            self_play_result = collect_parallel_self_play_data(
                env_factory=env_factory,
                agent=agent,
                num_episodes_to_collect=num_episodes_per_iteration,
                num_parallel_games=config.alpha_zero.num_parallel_games,
                iteration=iteration + 1,
                env_name=config.env.name,
                inference_batch_size=config.alpha_zero.inference_batch_size,  # Pass batch size
                use_tqdm=use_tqdm,
            )

            # Log self-play duration and network time from the result object
            logger.info(
                f"Self-Play Total Time: {self_play_result.collection_time_ms:.2f} ms"
            )

            # Add collected data to the agent's replay buffer from the result object
            if self_play_result.experiences:
                agent.add_experiences_to_buffer(self_play_result.experiences)
                logger.info(
                    f"Added {len(self_play_result.experiences)} experiences to replay buffer."
                )
            else:
                logger.warning("No experiences collected in this iteration.")

            # 2. Learning Phase
            logger.info("Running learning step...")
            learn_timer = Timer()  # Use Timer directly
            with learn_timer:
                loss_results = agent.learn()
            # Log learning time only if learning actually happened
            if loss_results:
                logger.info(f"Learning Time: {learn_timer.elapsed_ms:.2f} ms")
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
                with Timer() as save_timer:
                    agent.save()
                logger.info(f"Save Time: {save_timer.elapsed_ms:.2f} ms")

            # Print buffer size and latest losses if available
            buffer_size = len(agent.replay_buffer)
            logger.info(
                f"Iteration {iteration + 1} complete. Buffer size: {buffer_size}/{config.alpha_zero.replay_buffer_size}"
            )
            if total_losses:  # Check if any learning steps have occurred
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
                with Timer() as sanity_timer:
                    run_sanity_checks(
                        env, agent
                    )  # Run checks on the current agent state
                logger.info(f"Sanity Check Time: {sanity_timer.elapsed_ms:.2f} ms")

        # Log total iteration time *after* the 'with iteration_timer' block finishes
        logger.info(f"Total Iteration Time: {iteration_timer.elapsed_ms:.2f} ms")

    logger.info("\nTraining complete. Saving final agent state.")
    agent.save()

    logger.info("Plotting training losses...")
    plot_losses(
        total_losses, value_losses, policy_losses
    )  # Call the new plotting function

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

    agent.network.eval()  # Ensure network is in eval mode

    if not sanity_states:
        logger.info("No sanity check states defined for this environment.")
        return

    for check_case in sanity_states:  # Iterate over SanityCheckState objects
        logger.info(f"\nChecking State: {check_case.description}")
        # Print board/piles for context
        if "board" in check_case.state:
            logger.info("Board:")
            logger.info(
                f"\n{check_case.state['board']}"
            )  # Add newline for better formatting
        elif "piles" in check_case.state:
            logger.info(f"Piles: {check_case.state['piles']}")
        logger.info(f"Current Player: {check_case.state['current_player']}")

        try:
            # Get network predictions
            policy_np, value_np = agent.network.predict(check_case.state)
            # Print expected vs predicted value
            logger.info(
                f"  Value: Expected={check_case.expected_value:.1f}, Predicted={value_np:.4f}"
            )

            # Get legal actions for this state to interpret policy
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

            # Sort actions by predicted probability for display
            sorted_probs = sorted(
                action_probs.items(), key=lambda item: item[1], reverse=True
            )

            logger.info(f"  Predicted Probabilities for Legal Actions:")
            if not legal_actions:
                logger.info("    - (No legal actions)")
            else:
                for action, prob in sorted_probs:
                    if prob >= 0:
                        # Highlight the best predicted action
                        highlight = " <<< BEST" if action == sorted_probs[0][0] else ""
                        logger.info(f"    - {action}: {prob:.4f}{highlight}")
                    else:
                        logger.info(f"    - {action}: (Error mapping action)")

        except Exception as e:
            logger.error(f"  Error during prediction for this state: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    config = AppConfig()
    env_override = None

    # --- Environment Selection (Optional: Add CLI arg parsing) ---
    if len(sys.argv) > 1:
        env_override = sys.argv[1]  # e.g., python train_alphazero.py Nim

    # --- Loguru Configuration ---
    # Remove the default handler to prevent duplicate messages if re-adding stderr
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # You could also add file logging here if needed:
    # logger.add("file_{time}.log", level="INFO")
    # --- End Loguru Configuration ---

    run_training(config, env_name_override=env_override)
