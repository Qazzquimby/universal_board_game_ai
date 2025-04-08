import sys
import json
import datetime
from typing import (
    Tuple,
    List,
    Dict,
    Generator,
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
from algorithms.mcts import (
    Timer,
    PredictResult,
    GeneratorYield,
    PredictRequestYield,
)

LOG_DIR = DATA_DIR / "game_logs"


@dataclass
class PredictRequest:
    """Bundles information needed for a network prediction request."""

    game_idx: int
    state_key: str
    state_obs: StateType


def _handle_predict_request(
    game_idx: int,
    state_key: str,
    state_obs: StateType,
    pending_reqs: List[PredictRequest],
    waiting_gens: Dict[str, List[int]],
    gen_states: Dict[int, str],
):
    """Adds a new prediction request and updates tracking state."""
    logger.debug(
        f"Game {game_idx} yielded request for state {state_key}"
    )  # Use consistent log format
    pending_reqs.append(
        PredictRequest(game_idx=game_idx, state_key=state_key, state_obs=state_obs)
    )
    if state_key not in waiting_gens:
        waiting_gens[state_key] = []
    if game_idx not in waiting_gens[state_key]:
        waiting_gens[state_key].append(game_idx)
    gen_states[game_idx] = "waiting_predict"


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
    ready_for_action_search: List[int] = list(range(num_parallel_games))
    mcts_generators: Dict[int, Generator] = {}
    generator_states: Dict[int, str] = {}
    pending_requests: List[PredictRequest] = []
    waiting_generators: Dict[str, List[int]] = {}
    completed_searches: Dict[
        int, Tuple[Optional[ActionType], Optional[np.ndarray]]
    ] = {}
    states_before_action: Dict[int, StateType] = {}

    return {
        "envs": envs,
        "observations": observations,
        "parallel_histories": parallel_histories,
        "game_is_done": game_is_done,
        "ready_for_action_search": ready_for_action_search,
        "mcts_generators": mcts_generators,
        "generator_states": generator_states,
        "pending_requests": pending_requests,
        "waiting_generators": waiting_generators,
        "completed_searches": completed_searches,
        "states_before_action": states_before_action,
    }


def _start_mcts_searches(
    ready_list: List[int],
    game_is_done: List[bool],
    observations: List[StateType],
    envs: List[BaseEnvironment],
    agent: AlphaZeroAgent,
    mcts_generators: Dict[int, Generator],
    generator_states: Dict[int, str],
    states_before_action: Dict[int, StateType],
):
    """Starts MCTS search generators for games ready for an action."""
    games_to_start = ready_list[:]  # Copy the list
    ready_list.clear()  # Clear the original list

    for game_idx in games_to_start:
        if game_is_done[game_idx]:
            continue

        state = observations[game_idx]
        states_before_action[game_idx] = state.copy()  # Store state before search
        mcts_generators[game_idx] = agent.mcts.search_generator(envs[game_idx], state)
        generator_states[game_idx] = "running"
        logger.debug(f"Started MCTS search for game {game_idx}")


def _process_network_batch(
    agent: AlphaZeroAgent,
    pending_requests: List[PredictRequest],
    waiting_generators: Dict[str, List[int]],
    mcts_generators: Dict[int, Generator],
    generator_states: Dict[int, str],
    # completed_searches: Dict[int, Tuple[Optional[ActionType], Optional[np.ndarray]]], # Removed
    # envs: List[BaseEnvironment], # Removed
    inference_batch_size: int,
) -> List[PredictRequest]:
    """
    Processes a batch of network prediction requests.

    Args:
        agent: The AlphaZeroAgent instance.
        pending_requests: List of pending PredictRequest objects.
        waiting_generators: Maps state_key -> List[game_idx] waiting for that key.
        mcts_generators: Maps game_idx -> active MCTS generator instance.
        generator_states: Maps game_idx -> current state ('running', 'waiting_predict').
        inference_batch_size: Maximum batch size for network inference.

    Returns:
        The updated list of pending_requests (requests not processed in this batch).

    Updates the following dictionaries/lists in place:
    - waiting_generators
    - mcts_generators
    - generator_states
    - completed_searches
    """
    if not agent.network or not pending_requests:
        return pending_requests  # Return unchanged list if no network or requests

    batch_size_to_process = min(len(pending_requests), inference_batch_size)
    logger.debug(f"Processing network batch. Size: {batch_size_to_process}")

    # --- Prepare Batch ---
    batch_dict: Dict[str, Tuple[StateType, List[int]]] = {}
    requests_in_batch = pending_requests[:batch_size_to_process]
    remaining_requests = pending_requests[
        batch_size_to_process:
    ]  # Keep track of unprocessed requests

    for request in requests_in_batch:
        if request.state_key not in batch_dict:
            batch_dict[request.state_key] = (request.state_obs, [])
        if request.game_idx not in batch_dict[request.state_key][1]:
            batch_dict[request.state_key][1].append(request.game_idx)

    state_keys_in_batch = list(batch_dict.keys())
    states_to_predict_batch = [batch_dict[key][0] for key in state_keys_in_batch]

    # --- Call Network ---
    if agent.profiler:
        with Timer() as net_timer:
            policy_list, value_list = agent.network.predict_batch(
                states_to_predict_batch
            )
        agent.profiler.record_network_time(net_timer.elapsed_ms)
    else:
        policy_list, value_list = agent.network.predict_batch(states_to_predict_batch)

    # --- Distribute Results ---
    if len(policy_list) != len(state_keys_in_batch):
        logger.error("Network batch result size mismatch!")
        return remaining_requests  # Return unprocessed requests

    for i, state_key in enumerate(state_keys_in_batch):
        policy_result, value_result = policy_list[i], value_list[i]
        result_tuple: PredictResult = (policy_result, value_result)
        logger.debug(f"Distributing result for state {state_key}")

        waiting_game_indices = batch_dict[state_key][1]
        if state_key in waiting_generators:
            del waiting_generators[state_key]  # Clear from global waiting dict

        for waiting_game_idx in waiting_game_indices:
            if waiting_game_idx in mcts_generators:
                generator_to_resume = mcts_generators[waiting_game_idx]
                try:
                    assert (
                        result_tuple is not None
                    ), f"Attempting to send None result for state {state_key}"
                    assert (
                        isinstance(result_tuple, tuple) and len(result_tuple) == 2
                    ), f"Attempting to send invalid result type {type(result_tuple)} for state {state_key}"
                    assert isinstance(
                        generator_to_resume, Generator
                    ), f"Attempting to send to non-generator object for game {waiting_game_idx}"

                    generator_to_resume.send(result_tuple)
                    generator_states[waiting_game_idx] = "running"
                    logger.debug(
                        f"Sent result to generator {waiting_game_idx}. Marked as 'running'."
                    )

                except StopIteration:
                    # Generator finished immediately upon receiving the result.
                    # This might happen if the result leads directly to the end of the search.
                    logger.debug(
                        f"Generator {waiting_game_idx} finished (StopIteration) immediately after receiving result."
                    )
                    # Clean up state and generator tracking
                    if waiting_game_idx in mcts_generators:
                        del mcts_generators[waiting_game_idx]
                    if waiting_game_idx in generator_states:
                        del generator_states[waiting_game_idx]

                except Exception as e:
                    logger.error(
                        f"Error sending result to generator for game {waiting_game_idx}: {e}",
                        exc_info=True,
                    )
                    if waiting_game_idx in mcts_generators:
                        del mcts_generators[waiting_game_idx]
                    if waiting_game_idx in generator_states:
                        del generator_states[waiting_game_idx]
            else:
                logger.warning(
                    f"Generator for game {waiting_game_idx} no longer active while distributing results for state {state_key}."
                )
                if waiting_game_idx in generator_states:
                    del generator_states[waiting_game_idx]

    return remaining_requests  # Return the list of requests not processed in this batch


def _step_environments(
    completed_searches: Dict[int, Tuple[Optional[ActionType], Optional[np.ndarray]]],
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
    games_ready_to_step = list(completed_searches.keys())
    next_observations = {}
    rewards = {}
    step_dones = {}
    steps_taken_this_cycle = 0

    for game_idx in games_ready_to_step:
        action, policy_target = completed_searches[game_idx]

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
                    f"Game {game_idx}, Step {len(parallel_histories[game_idx])}: Skipping history append - MCTS failed to produce policy target."
                )
            elif state_before_action_val is None:
                logger.error(
                    f"Game {game_idx}: Skipping history append - Missing state_before_action"
                )
            else:
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
    ready_for_action_search: List[int],
    next_observations: Dict,
    step_dones: Dict,
    all_experiences: List[Tuple[StateType, np.ndarray, float]],
    iteration: int,
    env_name: str,
    finished_episodes_count_offset: int,  # To get correct game index for logging
    pbar: Optional[tqdm] = None,
) -> int:
    """
    Updates game states after stepping, processes finished games, resets environments,
    and updates tracking lists.

    Returns:
        The number of episodes finished in this cycle.
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
            if game_idx not in ready_for_action_search:
                ready_for_action_search.append(game_idx)
        else:
            # Game not done, mark ready for next action search
            if game_idx not in ready_for_action_search:
                ready_for_action_search.append(game_idx)

    return episodes_finished_this_cycle


def _advance_and_process_generators(
    agent: AlphaZeroAgent,
    mcts_generators: Dict[int, Generator],
    generator_states: Dict[int, str],
    pending_requests: List[PredictRequest],
    waiting_generators: Dict[str, List[int]],
    completed_searches: Dict[int, Tuple[Optional[ActionType], Optional[np.ndarray]]],
    # envs: List[BaseEnvironment], # Removed
):
    """
    Advances running MCTS generators and processes their yielded results.

    Updates the following dictionaries/lists in place:
    - mcts_generators
    - generator_states
    - pending_requests
    - waiting_generators
    - completed_searches
    """
    active_generator_indices = list(mcts_generators.keys())
    generators_finished_this_cycle = []

    for game_idx in active_generator_indices:
        current_state = generator_states.get(game_idx)
        logger.trace(f"Advancing Gen {game_idx}: Checking state '{current_state}'")

        # Skip if generator finished or is waiting for prediction
        if game_idx not in mcts_generators or current_state == "waiting_predict":
            logger.trace(f"Advancing Gen {game_idx}: Skipping (Not found or waiting)")
            continue

        # If we reach here, state should be 'running'. Add extra check for safety.
        if current_state != "running":
            logger.error(
                f"Advancing Gen {game_idx}: State is '{current_state}', expected 'running'! Skipping send(None)."
            )
            continue  # Avoid sending None if state is unexpected

        generator = mcts_generators[game_idx]
        try:
            logger.trace(
                f"Advancing Gen {game_idx}: State is '{current_state}', sending None..."
            )
            # Advance the main search_generator by sending None
            yielded_value: GeneratorYield = generator.send(None)
            yield_type = yielded_value[0]

            if yield_type == "predict_request":
                _, state_key, state_obs = yielded_value
                # Use the helper function
                _handle_predict_request(
                    game_idx=game_idx,
                    state_key=state_key,
                    state_obs=state_obs,
                    pending_reqs=pending_requests,
                    waiting_gens=waiting_generators,
                    gen_states=generator_states,
                )
            elif yield_type == "search_complete":
                _, root_node = yielded_value
                # Add logging for received root node state
                logger.debug(
                    f"MCTS search complete for game {game_idx}. Received Root ID: {id(root_node)}, Has Children: {bool(root_node.children)}"
                )
                generators_finished_this_cycle.append(game_idx)
                # Process completed search result
                if not root_node.children:
                    # --- Error Handling: Root has no children after search ---
                    # This indicates a problem, possibly the root state was terminal
                    # or an error occurred during the very first simulation/expansion.
                    logger.error(
                        f"MCTS root for game {game_idx} has no children after search complete. Setting action=None."
                    )
                    action = None  # Signal error state
                    policy_target = None
                else:
                    # --- Standard processing ---
                    visit_counts = np.array(
                        [child.visit_count for child in root_node.children.values()]
                    )
                    actions = list(root_node.children.keys())
                    chosen_action_index = np.argmax(visit_counts)
                    action = actions[chosen_action_index]
                    policy_target = agent._calculate_policy_target(
                        root_node, actions, visit_counts
                    )
                completed_searches[game_idx] = (action, policy_target)

            elif yield_type == "resumed":
                logger.debug(f"Generator {game_idx} resumed, remains running.")
                generator_states[game_idx] = "running"  # Ensure state is correct

            else:
                logger.warning(
                    f"Unexpected yield value from running generator {game_idx}: {yielded_value}"
                )

        except StopIteration:
            logger.debug(f"MCTS generator for game {game_idx} stopped.")
            generators_finished_this_cycle.append(game_idx)
            if game_idx not in completed_searches:
                completed_searches[game_idx] = (None, None)  # Mark error

        except Exception as e:
            logger.error(
                f"Error processing MCTS generator for game {game_idx}: {e}",
                exc_info=True,
            )
            generators_finished_this_cycle.append(game_idx)
            if game_idx not in completed_searches:
                completed_searches[game_idx] = (None, None)  # Mark error

    # Clean up finished generators *after* the loop
    for game_idx in generators_finished_this_cycle:
        if game_idx in mcts_generators:
            del mcts_generators[game_idx]
        if game_idx in generator_states:
            del generator_states[game_idx]


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
    ready_for_action_search = game_states["ready_for_action_search"]
    mcts_generators = game_states["mcts_generators"]
    generator_states = game_states["generator_states"]
    pending_requests = game_states["pending_requests"]
    waiting_generators = game_states["waiting_generators"]
    completed_searches = game_states["completed_searches"]
    states_before_action = game_states["states_before_action"]

    # Get network time *before* self-play if profiling
    net_time_before_self_play = 0.0
    if agent.profiler:
        agent.profiler.reset()  # Reset profiler for this collection phase
        net_time_before_self_play = (
            agent.profiler.get_total_network_time()
        )  # Should be 0

    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=num_episodes_to_collect, desc="Parallel Self-Play", leave=False
        )

    collection_timer = Timer()
    with collection_timer:
        while finished_episodes_count < num_episodes_to_collect:

            # --- Start MCTS searches for games needing an action ---
            _start_mcts_searches(
                ready_list=ready_for_action_search,
                game_is_done=game_is_done,
                observations=observations,
                envs=envs,
                agent=agent,
                mcts_generators=mcts_generators,
                generator_states=generator_states,
                states_before_action=states_before_action,
            )

            # --- Advance MCTS Generators and Collect Network Requests ---
            # Check if any generators are active before proceeding
            if not mcts_generators and not pending_requests:
                # Logic to check if enough episodes collected or if stuck
                if finished_episodes_count >= num_episodes_to_collect:
                    break
                if ready_for_action_search:
                    continue
                logger.warning(
                    "No active MCTS generators or pending requests, but not enough episodes collected. Checking game states."
                )
                all_done = all(game_is_done)
                if all_done:
                    break
                stuck = True
                for i in range(num_parallel_games):
                    if (
                        not game_is_done[i]
                        and i not in mcts_generators
                        and i not in ready_for_action_search
                    ):
                        logger.warning(
                            f"Game {i} is not done but has no MCTS generator. Marking ready."
                        )
                        ready_for_action_search.append(i)
                        stuck = False
                if stuck and not ready_for_action_search:
                    logger.error(
                        "Stuck state: No active generators, no pending requests, but games not done. Breaking."
                    )
                    break
                else:
                    continue

            # Advance running generators
            _advance_and_process_generators(
                agent=agent,
                mcts_generators=mcts_generators,
                generator_states=generator_states,
                pending_requests=pending_requests,
                waiting_generators=waiting_generators,
                completed_searches=completed_searches,
                # envs=envs, # Removed
            )

            # --- Process Network Batch if Ready ---
            # Check if all active generators are waiting for prediction
            all_waiting = False
            if mcts_generators:  # Only check if there are active generators
                all_waiting = all(
                    generator_states.get(idx) == "waiting_predict"
                    for idx in mcts_generators
                )

            # Process batch if enough pending requests OR if requests exist and all active generators are waiting
            if agent.network and (
                len(pending_requests) >= inference_batch_size
                or (pending_requests and all_waiting)  # Modified condition
            ):
                pending_requests = _process_network_batch(
                    agent=agent,
                    pending_requests=pending_requests,
                    waiting_generators=waiting_generators,
                    mcts_generators=mcts_generators,
                    generator_states=generator_states,
                    # completed_searches=completed_searches, # Removed
                    # envs=envs, # Removed
                    inference_batch_size=inference_batch_size,
                )

            # --- Step Environments for games where MCTS search completed ---
            games_ready_to_step = list(completed_searches.keys())
            if games_ready_to_step:
                (
                    next_observations,
                    rewards,
                    step_dones,
                    steps_this_cycle,
                ) = _step_environments(
                    completed_searches=completed_searches,
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
                    ready_for_action_search=ready_for_action_search,
                    next_observations=next_observations,
                    step_dones=step_dones,
                    all_experiences=all_experiences,
                    iteration=iteration,
                    env_name=env_name,
                    finished_episodes_count_offset=finished_episodes_count,
                    pbar=pbar,
                )
                finished_episodes_count += episodes_finished_this_cycle

                # Clean up completed search data after processing
                for game_idx in games_ready_to_step:
                    del completed_searches[game_idx]
                    if game_idx in states_before_action:
                        del states_before_action[game_idx]
                    # Generator state should already be cleaned up when generator finishes

            elif not mcts_generators and not pending_requests:
                # If no games ready to step, no generators running, and no requests pending, check termination
                if finished_episodes_count >= num_episodes_to_collect:
                    break
                else:
                    logger.warning(
                        "Stuck state: No games ready, no generators, no requests. Checking status."
                    )
                    if not ready_for_action_search:
                        logger.error("Stuck: Breaking loop.")
                        break
                    else:
                        continue  # Go back to start searches for games marked ready

    # --- End of Collection Loop ---
    if pbar:
        pbar.close()

    # Calculate network time spent during this self-play phase
    network_time_ms = 0.0
    if agent.profiler:
        net_time_after_self_play = agent.profiler.get_total_network_time()
        network_time_ms = (
            net_time_after_self_play - net_time_before_self_play
        )  # Total time recorded by profiler

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
            if agent.profiler:
                logger.info(
                    f"Self-Play Network Time: {self_play_result.network_time_ms:.2f} ms"
                )
            else:
                logger.info("Self-Play Network Time: (Profiling Disabled)")

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

            # --- Log MCTS Profiler Report ---
            # Check if profiling is enabled and report periodically
            report_freq = config.training.mcts_profiling_report_frequency
            if (
                agent.profiler
                and report_freq > 0
                and (iteration + 1) % report_freq == 0
            ):
                logger.info(
                    "\n--- MCTS Profiling Stats (Last {} Iterations) ---".format(
                        report_freq
                    )
                )
                logger.info(agent.profiler.report())  # Log the aggregated report
                agent.profiler.reset()  # Reset for the next reporting period
            # --- End MCTS Profiler Report ---

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

    # --- Optional: Final MCTS Profiler Report ---
    # Log any remaining stats if the loop didn't end on a reporting interval
    if agent.profiler and agent.profiler.get_num_searches() > 0:
        logger.info("\n--- Final MCTS Profiler Stats (Since Last Report) ---")
        logger.info(agent.profiler.report())
    # --- End Final Report ---

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
    # logger.remove()
    # logger.add(sys.stderr, level="INFO")
    # You could also add file logging here if needed:
    # logger.add("file_{time}.log", level="INFO")
    # --- End Loguru Configuration ---

    run_training(config, env_name_override=env_override)
