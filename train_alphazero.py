import sys
import json
import datetime
from typing import Tuple, List

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


def collect_self_play_data(
    env: BaseEnvironment,
    agent: AlphaZeroAgent,
    num_episodes: int,
    iteration: int,
    env_name: str,
    use_tqdm: bool = True,
) -> Tuple[List[Tuple[StateType, np.ndarray, float]], float, float]:
    """
    Runs multiple self-play games sequentially and collects experiences.

    Args:
        env: Environment instance.
        agent: AlphaZeroAgent instance.
        num_episodes: Number of games to play.
        iteration: Current training iteration number (for logging).
        env_name: Name of the environment (for logging).
        use_tqdm: Whether to display a progress bar.

    Returns:
        A tuple containing:
        - all_experiences: A flat list of (state, policy_target, value_target) tuples from all games.
        - total_self_play_time_ms: Total time spent in self-play games (excluding logging).
        - total_network_time_ms: Total network inference time during self-play.
    """
    all_experiences = []
    total_games = 0
    total_steps = 0

    # Get network time *before* self-play if profiling
    net_time_before_self_play = 0.0
    if agent.profiler:
        net_time_before_self_play = agent.profiler.get_total_network_time()

    inner_loop_iterator = (
        tqdm(range(num_episodes), desc="Self-Play", leave=False)
        if use_tqdm
        else range(num_episodes)
    )

    with Timer() as self_play_timer:
        for game_idx in inner_loop_iterator:
            outcome, steps, episode_result = run_self_play_game(env, agent)
            all_experiences.extend(episode_result.buffer_experiences)
            total_games += 1
            total_steps += steps
            # Save the game log after each game
            save_game_log(
                episode_result.logged_history, iteration, game_idx + 1, env_name
            )

    # Calculate network time spent during this self-play phase
    network_time_ms = 0.0
    if agent.profiler:
        net_time_after_self_play = agent.profiler.get_total_network_time()
        network_time_ms = net_time_after_self_play - net_time_before_self_play

    logger.info(
        f"Collected {len(all_experiences)} experiences from {total_games} games ({total_steps} steps)."
    )
    return all_experiences, self_play_timer.elapsed_ms, network_time_ms


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
            logger.info("Collecting self-play data...")

            (
                collected_experiences,
                self_play_time,
                network_time,
            ) = collect_self_play_data(
                env,
                agent,
                num_episodes_per_iteration,
                iteration + 1,
                config.env.name,
                use_tqdm,
            )

            # Log self-play duration and network time
            logger.info(f"Self-Play Total Time: {self_play_time:.2f} ms")
            if agent.profiler:
                logger.info(f"Self-Play Network Time: {network_time:.2f} ms")
            else:
                logger.info("Self-Play Network Time: (Profiling Disabled)")

            # Add collected data to the agent's replay buffer
            if collected_experiences:
                agent.add_experiences_to_buffer(collected_experiences)
                logger.info(
                    f"Added {len(collected_experiences)} experiences to replay buffer."
                )
            else:
                logger.warning("No experiences collected in this iteration.")

            # 2. Learning Phase
            logger.info("Running learning step...")
            loss_results = None
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
