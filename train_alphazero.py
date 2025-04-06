import sys
import json
import datetime
import random  # Import random
from typing import (
    Tuple,
    List,
    Dict,
    Generator,
    Any,
    Optional,
)  # Import Generator, Any, Optional
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from loguru import logger

from core.config import AppConfig, DATA_DIR
from environments.base import BaseEnvironment, StateType, ActionType
from agents.alphazero_agent import AlphaZeroAgent, EpisodeResult
from factories import get_environment, get_agents
from utils.plotting import plot_losses
from algorithms.mcts import Timer, MCTSNode  # Import MCTSNode

LOG_DIR = DATA_DIR / "game_logs"


def run_self_play_game(
    env: BaseEnvironment, agent: AlphaZeroAgent
) -> Tuple[float, int, EpisodeResult]:
    """
    Plays one game of self-play using the AlphaZero agent.
    The agent collects training data internally via agent.act(train=True) and
    processes it in finish_episode.

    Args:
        env: The environment instance.
        agent: The AlphaZero agent instance.

    Returns:
        A tuple containing:
        - final_outcome: The outcome for player 0 (+1 win, -1 loss, 0 draw).
        - num_steps: The number of steps taken in the game.
        - episode_result: An EpisodeResult object containing buffer experiences and logged history.
    """
    obs = env.reset()
    agent.reset()
    done = False
    game_steps = 0

    while not done:
        current_player = env.get_current_player()
        state = obs

        # agent.act now uses the profiler internally if it exists
        action = agent.act(state, train=True)

        if action is None:
            logger.warning(f"Agent returned None action in self-play. Ending game.")
            # Assign outcome based on rules (e.g., loss for player who can't move)
            # For simplicity, let's call it a draw if this happens unexpectedly.
            final_outcome = 0.0
            break

        try:
            obs, _, done = env.step(action)
            game_steps += 1
        except ValueError as e:
            logger.warning(
                f"Invalid action {action} during self-play. Error: {e}. Ending game."
            )
            # Penalize the player who made the invalid move
            final_outcome = -1.0 if current_player == 0 else 1.0
            break

    # If loop finished normally
    if "final_outcome" not in locals():
        winner = env.get_winning_player()
        if winner == 0:
            final_outcome = 1.0
        elif winner == 1:
            final_outcome = -1.0
        else:  # Draw
            final_outcome = 0.0


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

    # --- Initialize Parallel Environments and States ---
    envs = [env_factory() for _ in range(num_parallel_games)]
    observations = [env.reset() for env in envs]
    # Track temporary history for each parallel game
    parallel_histories: Dict[int, List[Tuple[StateType, ActionType, np.ndarray]]] = {
        i: [] for i in range(num_parallel_games)
    }
    dones = [False] * num_parallel_games
    # Track MCTS search generators for games waiting for an action
    # Maps game_idx -> generator instance
    mcts_generators: Dict[int, Generator] = {}
    # Track which games are ready for the next environment step (need an action)
    ready_for_action_search: List[int] = list(range(num_parallel_games))

    # --- State for External Batching ---
    # Stores requests yielded by generators: List[(game_idx, state_key, state_obs, node)]
    pending_requests: List[Tuple[int, str, StateType, MCTSNode]] = []
    # Maps state_key -> List[Tuple[int, MCTSNode]] for generators waiting on that state
    waiting_generators: Dict[str, List[Tuple[int, MCTSNode]]] = {}
    # Store chosen actions/policies when search completes, before stepping env
    # Maps game_idx -> (action, policy_target)
    completed_searches: Dict[
        int, Tuple[Optional[ActionType], Optional[np.ndarray]]
    ] = {}
    # Store state before action for history
    states_before_action: Dict[int, StateType] = {}

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
            games_to_start_search = ready_for_action_search[:]
            ready_for_action_search.clear()  # Clear the list as we are starting searches
            for game_idx in games_to_start_search:
                if dones[game_idx]:
                    continue  # Skip finished games

                state = observations[game_idx]
                states_before_action[
                    game_idx
                ] = state.copy()  # Store state before action search
                # Create and store the generator for this game's MCTS search
                mcts_generators[game_idx] = agent.mcts.search_generator(
                    envs[game_idx], state
                )
                logger.debug(f"Started MCTS search for game {game_idx}")

            # --- Advance MCTS Generators and Collect Network Requests ---
            active_generator_indices = list(mcts_generators.keys())
            if not active_generator_indices and not pending_requests:
                # Check if we have collected enough episodes
                if finished_episodes_count >= num_episodes_to_collect:
                    break
                # Otherwise, check if there are games waiting for action
                if ready_for_action_search:
                    continue  # Go back to start searches
                # If no generators, no requests, and not enough episodes, something is wrong
                logger.warning(
                    "No active MCTS generators or pending requests, but not enough episodes collected. Checking game states."
                )
                all_done = True
                for i in range(num_parallel_games):
                    if not dones[i]:
                        all_done = False
                        if (
                            i not in mcts_generators
                            and i not in ready_for_action_search
                        ):
                            logger.warning(
                                f"Game {i} is not done but has no MCTS generator. Marking ready."
                            )
                            ready_for_action_search.append(i)
                if all_done:
                    break  # All games finished unexpectedly early?
                if (
                    not ready_for_action_search
                ):  # If still nothing, break to avoid infinite loop
                    logger.error(
                        "Stuck state: No active generators, no pending requests, but games not done. Breaking."
                    )
                    break
                else:
                    continue  # Go back to start searches

            generators_finished_this_cycle = []
            # Use while loop to handle generators finishing mid-iteration
            current_generator_indices = active_generator_indices[:]
            idx_pointer = 0
            while idx_pointer < len(current_generator_indices):
                game_idx = current_generator_indices[idx_pointer]
                idx_pointer += 1  # Increment pointer for next iteration

                if game_idx not in mcts_generators:
                    continue  # Generator might have finished already

                generator = mcts_generators[game_idx]
                try:
                    # Send None initially or after receiving a result
                    # Use next() for the very first call to prime the generator
                    yielded_value = (
                        next(generator)
                        if generator.gi_frame.f_lasti == -1
                        else generator.send(None)
                    )

                    if yielded_value[0] == "predict_request":
                        _, state_key, state_obs, node = yielded_value
                        logger.debug(
                            f"Game {game_idx} yielded request for state {state_key}"
                        )
                        # Store request
                        pending_requests.append((game_idx, state_key, state_obs, node))
                        # Register generator as waiting for this state_key
                        if state_key not in waiting_generators:
                            waiting_generators[state_key] = []
                        # Avoid adding duplicates if generator yields same state multiple times before batch processes
                        if (game_idx, node) not in waiting_generators[state_key]:
                            waiting_generators[state_key].append((game_idx, node))
                        # Generator is now paused until result is sent back

                    elif yielded_value[0] == "search_complete":
                        _, root_node = yielded_value
                        logger.debug(f"MCTS search complete for game {game_idx}")
                        generators_finished_this_cycle.append(
                            game_idx
                        )  # Mark for removal later

                        # --- Process completed search (similar to agent.act logic) ---
                        if not root_node.children:
                            logger.warning(
                                f"MCTS root for game {game_idx} has no children. Choosing random."
                            )
                            legal_actions = envs[game_idx].get_legal_actions()
                            action = (
                                random.choice(legal_actions) if legal_actions else None
                            )
                            policy_target = None  # Cannot calculate policy target
                        else:
                            visit_counts = np.array(
                                [
                                    child.visit_count
                                    for child in root_node.children.values()
                                ]
                            )
                            actions = list(root_node.children.keys())
                            # Use temperature=0 for action selection (greedy) for now
                            # TODO: Re-introduce temperature if needed for exploration during collection
                            chosen_action_index = np.argmax(visit_counts)
                            action = actions[chosen_action_index]
                            # Calculate policy target
                            policy_target = agent._calculate_policy_target(
                                root_node, actions, visit_counts
                            )

                        # Store the result, ready for env step
                        completed_searches[game_idx] = (action, policy_target)

                except StopIteration:
                    # Generator finished without yielding 'search_complete' (should not happen)
                    logger.warning(
                        f"MCTS generator for game {game_idx} stopped unexpectedly."
                    )
                    generators_finished_this_cycle.append(game_idx)
                    if game_idx not in completed_searches:
                        completed_searches[game_idx] = (
                            None,
                            None,
                        )  # Mark as completed with error

                except Exception as e:
                    logger.error(
                        f"Error processing MCTS generator for game {game_idx}: {e}",
                        exc_info=True,
                    )
                    generators_finished_this_cycle.append(game_idx)
                    if game_idx not in completed_searches:
                        completed_searches[game_idx] = (
                            None,
                            None,
                        )  # Mark as completed with error

            # Remove finished generators *after* iterating
            for game_idx in generators_finished_this_cycle:
                if game_idx in mcts_generators:
                    del mcts_generators[game_idx]

            # --- Process Network Batch if Ready ---
            # Process batch if enough pending requests OR if no generators are running (to flush remaining requests)
            if agent.network and (
                len(pending_requests) >= inference_batch_size
                or (not mcts_generators and pending_requests)
            ):
                batch_size_to_process = len(pending_requests)
                logger.debug(f"Processing network batch. Size: {batch_size_to_process}")

                # --- Prepare Batch ---
                # Deduplicate requests for the same state_key within this batch
                batch_dict: Dict[str, Tuple[StateType, List[Tuple[int, MCTSNode]]]] = {}
                requests_in_batch = pending_requests[
                    :batch_size_to_process
                ]  # Take the requests for this batch
                pending_requests = pending_requests[
                    batch_size_to_process:
                ]  # Remove them from pending list

                for game_idx, state_key, state_obs, node in requests_in_batch:
                    if state_key not in batch_dict:
                        # Store the state observation only once per unique state key
                        batch_dict[state_key] = (state_obs, [])
                    # Add the waiting game/node to the list for this state key
                    # Check if this specific game/node pair is already waiting (shouldn't happen with current logic, but safety)
                    if (game_idx, node) not in batch_dict[state_key][1]:
                        batch_dict[state_key][1].append((game_idx, node))

                state_keys_in_batch = list(batch_dict.keys())
                states_to_predict_batch = [
                    batch_dict[key][0] for key in state_keys_in_batch
                ]

                # --- Call Network ---
                policy_list, value_list = [], []
                if agent.profiler:
                    with Timer() as net_timer:
                        policy_list, value_list = agent.network.predict_batch(
                            states_to_predict_batch
                        )
                    agent.profiler.record_network_time(net_timer.elapsed_ms)
                else:
                    policy_list, value_list = agent.network.predict_batch(
                        states_to_predict_batch
                    )

                # --- Distribute Results ---
                if len(policy_list) != len(state_keys_in_batch):
                    logger.error("Network batch result size mismatch!")
                else:
                    for i, state_key in enumerate(state_keys_in_batch):
                        policy_result, value_result = policy_list[i], value_list[i]
                        logger.debug(f"Distributing result for state {state_key}")

                        # Find all generators/nodes waiting for this state_key from the batch_dict
                        waiting_list = batch_dict[state_key][1]
                        # Also clear from the global waiting dict (important!)
                        if state_key in waiting_generators:
                            del waiting_generators[state_key]

                        for waiting_game_idx, waiting_node in waiting_list:
                            if waiting_game_idx in mcts_generators:
                                generator_to_resume = mcts_generators[waiting_game_idx]
                                try:
                                    # Send result back to the specific waiting generator
                                    generator_to_resume.send(
                                        (policy_result, value_result)
                                    )
                                    # Generator will resume execution in the next cycle's generator loop
                                except StopIteration:
                                    logger.warning(
                                        f"Generator for game {waiting_game_idx} finished while sending result."
                                    )
                                    if waiting_game_idx in mcts_generators:
                                        del mcts_generators[
                                            waiting_game_idx
                                        ]  # Clean up
                                except Exception as e:
                                    logger.error(
                                        f"Error sending result to generator for game {waiting_game_idx}: {e}",
                                        exc_info=True,
                                    )
                                    if waiting_game_idx in mcts_generators:
                                        del mcts_generators[
                                            waiting_game_idx
                                        ]  # Clean up
                            else:
                                # This can happen if the generator finished between yielding and batch processing
                                logger.warning(
                                    f"Generator for game {waiting_game_idx} no longer active while distributing results for state {state_key}."
                                )

            # --- Step Environments for games where MCTS search completed ---
            games_ready_to_step = list(completed_searches.keys())
            if not games_ready_to_step:
                # If no games ready to step, and no generators running, and requests pending, continue to process requests
                if not mcts_generators and pending_requests:
                    continue
                # If no games ready, no generators, no requests, check termination condition
                elif not mcts_generators and not pending_requests:
                    if finished_episodes_count >= num_episodes_to_collect:
                        break
                    else:
                        # This state should ideally be caught earlier
                        logger.warning(
                            "No games ready to step, no generators running, no pending requests. Checking status."
                        )
                        if (
                            not ready_for_action_search
                        ):  # If no games need action search either, break
                            logger.error(
                                "Stuck: No games ready to step and none need action search. Breaking."
                            )
                            break
                        else:
                            continue  # Go back to start searches

            next_observations = {}
            rewards = {}
            step_dones = {}

            for game_idx in games_ready_to_step:
                action, policy_target = completed_searches[game_idx]

                if dones[game_idx]:  # If game ended previously (e.g., MCTS error)
                    step_dones[game_idx] = True
                    next_observations[game_idx] = observations[game_idx]
                    rewards[game_idx] = 0.0
                    continue

                if action is None:
                    logger.error(
                        f"Game {game_idx} MCTS completed but action is None. Ending game."
                    )
                    dones[game_idx] = True
                    step_dones[game_idx] = True
                    next_observations[game_idx] = observations[game_idx]
                    rewards[game_idx] = 0.0
                    continue

                try:
                    obs, reward, done = envs[game_idx].step(action)
                    next_observations[game_idx] = obs
                    rewards[game_idx] = reward
                    step_dones[game_idx] = done
                    total_steps_taken += 1

                    # Store history step using state saved *before* MCTS search
                    state_before_action = states_before_action.get(game_idx)
                    if state_before_action and policy_target is not None:
                        parallel_histories[game_idx].append(
                            (state_before_action, action, policy_target)
                        )
                    elif not state_before_action:
                        logger.error(f"Missing state_before_action for game {game_idx}")
                    elif policy_target is None:
                        logger.warning(
                            f"Missing policy_target for game {game_idx} step."
                        )

                except ValueError as e:
                    logger.warning(
                        f"Invalid action {action} in game {game_idx} step. Error: {e}. Ending game."
                    )
                    next_observations[game_idx] = observations[game_idx]
                    rewards[game_idx] = 0.0
                    step_dones[game_idx] = True  # Mark done

            # --- Update Game States and Handle Finished Games ---
            for game_idx in games_ready_to_step:  # Only update games we tried to step
                observations[game_idx] = next_observations[game_idx]
                dones[game_idx] = step_dones[game_idx]

                if dones[game_idx]:
                    # Game finished, process history
                    winner = envs[game_idx].get_winning_player()
                    final_outcome = 1.0 if winner == 0 else -1.0 if winner == 1 else 0.0

                    game_history = parallel_histories[game_idx]
                    # Ensure policy target is valid if last step had None
                    valid_history = [
                        (s, a, p) for s, a, p in game_history if p is not None
                    ]
                    if len(valid_history) != len(game_history):
                        logger.warning(
                            f"Game {game_idx} history contained steps with None policy target."
                        )

                    if valid_history:  # Only process if there's valid history
                        episode_result = agent.process_finished_episode(
                            valid_history, final_outcome
                        )
                        all_experiences.extend(episode_result.buffer_experiences)
                        save_game_log(
                            episode_result.logged_history,
                            iteration,
                            finished_episodes_count + 1,
                            env_name,
                        )
                    else:
                        logger.warning(
                            f"Game {game_idx} finished with no valid history steps. No experiences added."
                        )

                    finished_episodes_count += 1
                    if pbar:
                        pbar.update(1)
                    logger.debug(
                        f"Game {game_idx} finished. Total finished: {finished_episodes_count}/{num_episodes_to_collect}"
                    )

                    # Reset environment and history for this index
                    observations[game_idx] = envs[game_idx].reset()
                    dones[game_idx] = False
                    parallel_histories[game_idx] = []
                    # Mark ready for next action search
                    if game_idx not in ready_for_action_search:
                        ready_for_action_search.append(game_idx)
                else:
                    # Game not done, mark ready for next action search
                    if game_idx not in ready_for_action_search:
                        ready_for_action_search.append(game_idx)

                # Clean up completed search data
                del completed_searches[game_idx]
                if game_idx in states_before_action:
                    del states_before_action[game_idx]

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
    logger.remove()
    # Add a handler for standard error (console) with level INFO or higher
    # This will automatically filter out DEBUG messages.
    logger.add(sys.stderr, level="INFO")
    # You could also add file logging here if needed:
    # logger.add("file_{time}.log", level="INFO")
    # --- End Loguru Configuration ---

    run_training(config, env_name_override=env_override)
