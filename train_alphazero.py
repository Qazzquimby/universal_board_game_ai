import sys
import json
import datetime
from typing import Tuple, List, Dict
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from loguru import logger

from core.config import AppConfig, DATA_DIR
from environments.base import BaseEnvironment, StateType, ActionType
from agents.alphazero_agent import AlphaZeroAgent, EpisodeResult
from factories import get_environment, get_agents
from utils.plotting import plot_losses
from algorithms.mcts import Timer

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

    episode_result = agent.finish_episode(final_outcome)

    return final_outcome, game_steps, episode_result


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
    use_tqdm: bool = True,
) -> SelfPlayResult:
    """
    Runs multiple self-play games in parallel using batched inference and collects experiences.

    Args:
        env_factory: A function that returns a new instance of the environment.
        agent: AlphaZeroAgent instance (contains the network and MCTS).
        num_episodes_to_collect: Target number of episodes to complete.
        num_parallel_games: Number of games to run simultaneously.
        iteration: Current training iteration number (for logging).
        env_name: Name of the environment (for logging).
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
    # Track temporary history for each parallel game before finish_episode is called
    # Maps game_idx -> List[Tuple[StateType, ActionType, np.ndarray]]
    parallel_histories: Dict[int, List[Tuple[StateType, ActionType, np.ndarray]]] = {
        i: [] for i in range(num_parallel_games)
    }
    dones = [False] * num_parallel_games
    active_game_indices = list(range(num_parallel_games))

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
            if not active_game_indices:
                logger.warning(
                    "No active games left, but haven't collected enough episodes. Breaking."
                )
                break

            current_batch_indices = active_game_indices[
                :
            ]  # Process all active games in this step
            logger.debug(
                f"Processing step for {len(current_batch_indices)} active games."
            )

            # --- Get Actions (using batched MCTS internally) ---
            actions = {}  # game_idx -> action
            policy_targets = {}  # game_idx -> policy_target (from MCTS)
            states_for_step = {}  # game_idx -> state_dict (before action)

            for game_idx in current_batch_indices:
                state = observations[game_idx]
                states_for_step[game_idx] = state.copy()  # Store state before action

                # agent.act performs MCTS search (which now uses batching internally)
                # It also stores the state/action/policy_target internally if train=True
                # We need to retrieve the policy target *after* act is called for logging/buffer
                # Let's modify agent.act slightly or add a way to get the last policy target.
                # --- Modification needed in Agent.act ---
                # For now, assume agent.act stores the necessary info and we retrieve it later.
                # agent.act now returns (action, policy_target) when train=True
                act_result = agent.act(state, train=True)

                if isinstance(act_result, tuple):
                    action, policy_target = act_result
                else:  # Should happen only if train=False, but handle defensively
                    action = act_result
                    policy_target = None  # No policy target if not training

                if action is None:
                    logger.warning(
                        f"Agent returned None action for game {game_idx}. Ending game."
                    )
                    dones[game_idx] = True
                    actions[game_idx] = None
                    policy_targets[game_idx] = None
                else:
                    actions[game_idx] = action
                    if (
                        policy_target is None
                    ):  # Fallback if train=True but target is None
                        logger.warning(
                            f"Policy target missing for game {game_idx}. Using zeros."
                        )
                        policy_size = agent.network._calculate_policy_size(
                            envs[game_idx]
                        )
                        policy_targets[game_idx] = np.zeros(
                            policy_size, dtype=np.float32
                        )
                    else:
                        policy_targets[game_idx] = policy_target

            # --- Step Environments ---
            next_observations = {}
            rewards = {}  # Not used by AlphaZero, but store anyway
            step_dones = {}  # Track dones from this step

            for game_idx in current_batch_indices:
                action = actions.get(game_idx)
                if (
                    dones[game_idx] or action is None
                ):  # Skip if already marked done or no action
                    step_dones[game_idx] = dones[game_idx]
                    next_observations[game_idx] = observations[game_idx]  # Keep old obs
                    rewards[game_idx] = 0.0
                    continue

                try:
                    obs, reward, done = envs[game_idx].step(action)
                    next_observations[game_idx] = obs
                    rewards[game_idx] = reward
                    step_dones[game_idx] = done
                    total_steps_taken += 1

                    # Store the step history for this game (using state *before* action)
                    state_before_action = states_for_step[game_idx]
                    policy_target = policy_targets[game_idx]
                    parallel_histories[game_idx].append(
                        (state_before_action, action, policy_target)
                    )

                except ValueError as e:
                    logger.warning(
                        f"Invalid action {action} in game {game_idx}. Error: {e}. Ending game."
                    )
                    # Penalize the player who made the invalid move? AZ uses outcome. Mark done.
                    next_observations[game_idx] = observations[game_idx]  # Keep old obs
                    rewards[game_idx] = 0.0  # No reward signal needed here
                    step_dones[game_idx] = True  # Mark as done due to error
                    # How to set outcome? Let finish_episode handle it based on winner.

            # --- Update Game States and Handle Finished Games ---
            new_active_game_indices = []
            for game_idx in current_batch_indices:
                observations[game_idx] = next_observations[game_idx]
                dones[game_idx] = step_dones[game_idx]

                if dones[game_idx]:
                    # Game finished, process history
                    winner = envs[game_idx].get_winning_player()
                    if winner == 0:
                        final_outcome = 1.0
                    elif winner == 1:
                        final_outcome = -1.0
                    else:
                        final_outcome = 0.0  # Draw

                    # Use the refactored agent method to process the history
                    game_history = parallel_histories[game_idx]
                    episode_result = agent.process_finished_episode(
                        game_history, final_outcome
                    )

                    # Add experiences to the main list
                    all_experiences.extend(episode_result.buffer_experiences)

                    # Save log for the finished game using the processed logged_history
                    save_game_log(
                        episode_result.logged_history,
                        iteration,
                        finished_episodes_count + 1,
                        env_name,
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
                    # Keep the game index active for the next iteration
                    new_active_game_indices.append(game_idx)

                else:
                    # Game not done, keep it active
                    new_active_game_indices.append(game_idx)

            active_game_indices = new_active_game_indices  # Update list of active games

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
