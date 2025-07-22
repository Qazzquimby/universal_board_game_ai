import sys
import json
import time

import numpy as np
from tqdm import tqdm
from loguru import logger

from agents.mcts_agent import make_pure_mcts
from core.config import AppConfig
from core.serialization import LOG_DIR, save_game_log
from environments.base import BaseEnvironment
from agents.alphazero_agent import AlphaZeroAgent
from factories import (
    get_environment,
    get_agents,
)
from utils.plotting import plot_losses
from utils.training_reporter import TrainingReporter
import evaluation


def load_game_logs_into_buffer(agent: AlphaZeroAgent, env_name: str, buffer_limit: int):
    """
    Loads existing game logs from LOG_DIR into the agent's train and validation
    replay buffers, splitting them to maintain persistence across runs.
    """
    loaded_games = 0
    if not LOG_DIR.exists():
        logger.info("Log directory not found. Starting with empty buffers.")
        return

    logger.info(f"Scanning {LOG_DIR} for existing '{env_name}' game logs...")
    log_files = sorted(LOG_DIR.glob(f"{env_name}_game*.json"), reverse=True)

    if not log_files:
        logger.info("No existing game logs found for this environment.")
        return

    all_experiences = []
    for filepath in tqdm(log_files, desc="Scanning Logs"):
        if len(all_experiences) >= buffer_limit:
            break

        with open(filepath, "r") as f:
            game_data = json.load(f)

        loaded_games += 1
        for step_data in game_data:
            state = step_data.get("state")
            policy_target_list = step_data.get("policy_target")
            value_target = step_data.get("value_target")

            if (
                state is not None
                and policy_target_list is not None
                and value_target is not None
            ):
                policy_target = np.array(policy_target_list, dtype=np.float32)
                all_experiences.append((state, policy_target, value_target))

    # Add to agent's buffers, which will handle shuffling, splitting, and capacity
    agent.add_experiences_to_buffer(all_experiences)
    loaded_steps = len(agent.train_replay_buffer) + len(agent.val_replay_buffer)

    logger.info(
        f"Loaded {loaded_steps} steps from {loaded_games} games into replay buffers. "
        f"Train: {len(agent.train_replay_buffer)}, Val: {len(agent.val_replay_buffer)}"
    )


def run_training(config: AppConfig, env_name_override: str = None):
    """Runs the AlphaZero training process."""

    if env_name_override:
        config.env.name = env_name_override

    env = get_environment(config.env)
    agents = get_agents(env, config)
    agent = agents["AZ_400"]
    assert isinstance(agent, AlphaZeroAgent)

    load_game_logs_into_buffer(
        agent, config.env.name, config.alpha_zero.replay_buffer_size
    )

    logger.info(
        f"Starting AlphaZero training for {config.training.num_iterations} iterations...\n"
        f"({config.training.num_games_per_iteration} self-play games per iteration)"
    )

    if not agent.network:
        logger.warning("No agent network found. Self-play will use DummyNet.")

    # Lists to store losses for plotting
    total_losses = []
    value_losses = []
    policy_losses = []

    outer_loop_iterator = range(config.training.num_iterations)
    start_time = time.time()
    reporter = TrainingReporter(config, agent, start_time)

    logger.info(f"\n--- Running Initial Evaluation ---")
    run_eval_against_benchmark(
        iteration=-1,
        reporter=reporter,
        agent=agent,
        config=config,
        env=env,
    )

    for iteration in outer_loop_iterator:
        reporter.log_iteration_start(iteration)

        all_experiences_iteration = run_self_play(agent=agent, env=env, config=config)
        add_results_to_buffer(
            iteration=iteration,
            all_experiences_iteration=all_experiences_iteration,
            agent=agent,
            config=config,
        )

        logger.info("Running learning step...")
        metrics = agent.learn()  # Agent learns using its local network
        if metrics:
            total_losses.append(metrics.train.loss)
            value_losses.append(metrics.train.value_loss)
            policy_losses.append(metrics.train.policy_loss)
            reporter.log_iteration_end(iteration=iteration, metrics=metrics)

        if (
            config.training.save_checkpoint_frequency > 0
            and (iteration + 1) % config.training.save_checkpoint_frequency == 0
        ):
            logger.info(f"Saving agent checkpoint (Iteration {iteration + 1})...")
            agent.save()

        if (
            config.evaluation.run_periodic_evaluation
            and (iteration + 1) % config.evaluation.periodic_eval_frequency == 0
        ):
            logger.info(
                f"\n--- Running Periodic Evaluation (Iteration {iteration + 1}) ---"
            )
            run_eval_against_benchmark(
                iteration=iteration,
                reporter=reporter,
                agent=agent,
                config=config,
                env=env,
            )

    logger.info("\nTraining complete. Saving final agent state.")
    agent.save()

    plot_losses(total_losses, value_losses, policy_losses)

    logger.info("\n--- AlphaZero Training Finished ---")

    reporter.finish()


def run_self_play(agent: AlphaZeroAgent, env: BaseEnvironment, config: AppConfig):
    logger.info("Running self play")
    if agent.network:
        agent.network.eval()

    num_games_total = config.training.num_games_per_iteration
    all_experiences_iteration = []

    for _ in tqdm(range(num_games_total), desc="Self-Play Games"):
        # for _ in tqdm(range(2), desc="Self-Play Games"):
        game_env = env.copy()
        state_with_key = game_env.reset()
        game_history = []

        while not game_env.state.done:
            state = state_with_key.state
            action = agent.act(game_env, train=True)
            policy_target = agent.get_policy_target()
            game_history.append((state, action, policy_target))
            action_result = game_env.step(action)
            state_with_key = action_result.next_state_with_key

        final_outcome = game_env.state.get_reward_for_player(0)
        all_experiences_iteration.append((game_history, final_outcome))
    return all_experiences_iteration


def add_results_to_buffer(
    iteration: int,
    all_experiences_iteration: list,
    agent: AlphaZeroAgent,
    config: AppConfig,
):
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


def run_eval_against_benchmark(
    iteration: int,
    reporter: TrainingReporter,
    agent: AlphaZeroAgent,
    config: AppConfig,
    env: BaseEnvironment,
):
    logger.info(f"\n--- Running Periodic Evaluation (Iteration {iteration + 1}) ---")
    agent.network.eval()
    benchmark_agent = make_pure_mcts(num_simulations=config.mcts.num_simulations)
    benchmark_agent_name = f"MCTS_{config.evaluation.benchmark_mcts_simulations}"

    eval_results = evaluation.run_test_games(
        env=env,
        agent0_name="AlphaZero",
        agent0=agent,
        agent1_name=benchmark_agent_name,
        agent1=benchmark_agent,
        config=config,
        num_games=config.evaluation.periodic_eval_num_games,
    )
    reporter.log_evaluation_results(
        eval_results=eval_results,
        benchmark_agent_name=benchmark_agent_name,
        iteration=iteration,
    )
    # Note: agent.learn() will put the network back in train mode if needed


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
        if "board" in check_case.state_with_key:
            logger.info("Board:")
            logger.info(f"\n{check_case.state_with_key['board']}")
        elif "piles" in check_case.state_with_key:
            logger.info(f"Piles: {check_case.state_with_key['piles']}")
        logger.info(f"Current Player: {check_case.state_with_key['current_player']}")

        try:
            policy_np, value_np = agent.network.predict(check_case.state_with_key)
            if check_case.expected_value is None:
                logger.info(f"  Value: Predicted={value_np:.4f}")
            else:
                logger.info(
                    f"  Value: Expected={check_case.expected_value:.1f}, Predicted={value_np:.4f}"
                )

            temp_env = env.copy()
            temp_env.set_state(check_case.state_with_key)
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
