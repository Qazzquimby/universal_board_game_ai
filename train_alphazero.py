import math
import sys
import json

import numpy as np
from tqdm import tqdm
from loguru import logger
import ray

from core.config import AppConfig
from core.serialization import LOG_DIR, save_game_log
from environments.base import BaseEnvironment
from agents.alphazero_agent import AlphaZeroAgent
from factories import get_environment, get_agents
from utils.plotting import plot_losses
from actors.inference_actor import InferenceActor
from actors.self_play_manager_actor import SelfPlayWorkerActor


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


# Experience = Tuple[StateType, np.ndarray, float]


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

    # --- Ray Initialization ---
    # Ensure Ray is initialized (or initialize it)
    if not ray.is_initialized():
        context = ray.init(
            ignore_reinit_error=True,
            log_to_driver=config.alpha_zero.debug_mode,
            include_dashboard=True,
            # local_mode=True,
        )
        print(f"Dashboard at {context.dashboard_url}")

    if agent.network:
        network_state_dict = {k: v.cpu() for k, v in agent.network.state_dict().items()}
        try:
            inference_actor = InferenceActor.remote(
                initial_network_state=network_state_dict,
                env_config=config.env,
                agent_config=config.alpha_zero,
            )
            # Verify actor started and device (optional, blocks until ready)
            actor_device = ray.get(inference_actor.get_device.remote())
            logger.info(
                f"InferenceActor created successfully on device: {actor_device}"
            )
        except Exception as e:
            logger.error(f"Failed to create or communicate with InferenceActor: {e}")
            if ray.is_initialized():
                ray.shutdown()
            return
    else:
        logger.warning("No agent network found. Self-play will use DummyNet locally.")

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

        logger.info("Collecting self-play data using parallel workers...")

        # --- Parallel Self-Play Setup ---
        num_games_total = config.training.num_games_per_iteration
        total_parallel_games_across_workers = config.training.num_games_per_iteration

        # Determine number of workers
        if config.alpha_zero.num_self_play_workers > 0:
            num_workers = config.alpha_zero.num_self_play_workers
        else:
            num_workers = 2
            # temp  os.cpu_count() or 1  # Default to CPU count, fallback to 1
        # Ensure we don't use more workers than total parallel games needed
        num_workers = min(num_workers, total_parallel_games_across_workers)
        logger.info(f"Using {num_workers} self-play workers.")

        if num_workers <= 0:
            logger.error("Number of self-play workers must be > 0.")
            if ray.is_initialized():
                ray.shutdown()
            return

        # Distribute work among workers
        games_per_worker = math.ceil(num_games_total / num_workers)
        # Distribute the *concurrent* games each worker manages internally
        internal_parallel_games_per_worker = math.ceil(
            total_parallel_games_across_workers / num_workers
        )
        logger.info(f"  Total concurrent games: {total_parallel_games_across_workers}")
        logger.info(
            f"  Internal concurrent games per worker: ~{internal_parallel_games_per_worker}"
        )
        logger.info(f"  Total games to collect per worker: ~{games_per_worker}")

        # Create worker actors
        workers = [
            SelfPlayWorkerActor.remote(
                actor_id=i,
                env_config=config.env,
                agent_config=config.alpha_zero,
                inference_actor_handle=inference_actor,  # Pass handle to central inference
                num_internal_parallel_games=internal_parallel_games_per_worker,
                iteration=iteration + 1,
            )
            for i in range(num_workers)
        ]

        # Launch tasks
        tasks = [w.collect_n_games.remote(games_per_worker) for w in workers]

        # Collect results
        all_experiences_iteration = []
        results = []
        while tasks:
            ready, tasks = ray.wait(tasks, num_returns=1)
            if not ready:
                break
            try:
                result_experiences = ray.get(ready[0])
                results.append(result_experiences)
            except ray.exceptions.RayTaskError as e:
                logger.error(f"Self-play worker task failed: {e}")
                # TODO: Consider adding retry logic or handling worker failure more robustly.

        # Aggregate experiences
        for exp_list in results:
            all_experiences_iteration.extend(
                exp_list
            )  # exp_list is now List[Tuple[raw_history, final_outcome]]

        # --- Process collected game results and add to buffer ---
        total_experiences_added = 0
        total_games_processed = 0
        game_log_index_offset = (
            iteration * config.training.num_games_per_iteration
        )  # Base index for this iteration

        logger.info(
            f"Processing {len(all_experiences_iteration)} collected game results..."
        )
        for i, (raw_history, final_outcome) in enumerate(all_experiences_iteration):
            if not raw_history:
                logger.warning(f"Skipping game {i} with empty raw history.")
                continue

            # Process the raw history using the agent to get buffer experiences and loggable history
            episode_result = agent.process_finished_episode(raw_history, final_outcome)

            # Add processed experiences to the central replay buffer
            agent.add_experiences_to_buffer(episode_result.buffer_experiences)
            total_experiences_added += len(episode_result.buffer_experiences)

            # Save the processed game log (now includes value targets)
            # Use a unique game index across the entire training run
            current_game_log_index = game_log_index_offset + total_games_processed + 1
            save_game_log(
                logged_history=episode_result.logged_history,
                iteration=iteration + 1,
                game_index=current_game_log_index,
                env_name=config.env.name,
            )
            total_games_processed += 1

        logger.info(
            f"Processed {total_games_processed} games, adding {total_experiences_added} experiences to replay buffer."
        )

        # --- Update Inference Actor Weights ---
        # If the network was updated during the learning phase (which happens next),
        # you might want to update the inference actor's weights before the *next*
        # self-play iteration. This can be done here or at the start of the next iteration.
        # Example: Update inference actor weights *before* next self-play
        if inference_actor and agent.network:
            new_weights = {k: v.cpu() for k, v in agent.network.state_dict().items()}
            # Update asynchronously, don't need to wait for it usually
            inference_actor.update_weights.remote(new_weights)
            logger.debug("Sent updated weights to InferenceActor.")

        # 2. Learning Phase
        logger.info("Running learning step...")
        loss_results = agent.learn()  # Agent learns using its local network
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

    # --- Ray Shutdown (after loop) ---
    if ray.is_initialized():
        if inference_actor:
            ray.kill(inference_actor)
            logger.info("InferenceActor terminated.")
        ray.shutdown()
        print("Ray shut down.")


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
            if check_case.expected_value is None:
                logger.info(f"  Value: Predicted={value_np:.4f}")
            else:
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
