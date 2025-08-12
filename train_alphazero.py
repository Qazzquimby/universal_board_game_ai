import sys
import json
import time
from typing import Union, List, Tuple

import numpy as np
from tqdm import tqdm
from loguru import logger

from agents.mcts_agent import MCTSAgent, make_pure_mcts
from core.config import AppConfig, DATA_DIR
from core.serialization import LOG_DIR, save_game_log
from environments.base import BaseEnvironment, StateType
from agents.alphazero_agent import AlphaZeroAgent, make_pure_az
from factories import (
    get_environment,
    get_agents,
)
from utils.plotting import plot_losses
from utils.training_reporter import TrainingReporter


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
    current_agent = next(
        agent for agent in agents.values() if isinstance(agent, AlphaZeroAgent)
    )
    assert isinstance(current_agent, AlphaZeroAgent)

    logger.info("Initializing with pure MCTS as the starting 'best' agent.")
    best_agent = make_pure_mcts(num_simulations=config.mcts.num_simulations)
    best_agent.temperature = 1.0  # Use temperature for exploration during self-play
    self_play_agent = best_agent
    best_agent_name = f"MCTS_{config.mcts.num_simulations}"

    load_game_logs_into_buffer(
        current_agent, config.env.name, config.alpha_zero.replay_buffer_size
    )

    logger.info(
        f"Starting AlphaZero training for {config.training.num_iterations} iterations...\n"
        f"({config.training.num_games_per_iteration} self-play games per iteration)"
    )

    total_losses, value_losses, policy_losses = [], [], []

    outer_loop_iterator = range(config.training.num_iterations)
    start_time = time.time()
    reporter = TrainingReporter(config, current_agent, start_time)

    for iteration in outer_loop_iterator:
        reporter.log_iteration_start(iteration)

        logger.info(f"Running self-play with '{best_agent_name}'...")
        all_experiences_iteration = run_self_play(
            agent=self_play_agent, env=env, config=config
        )
        add_results_to_buffer(
            iteration=iteration,
            all_experiences_iteration=all_experiences_iteration,
            agent=current_agent,
            config=config,
        )

        logger.info("Running learning step...")
        metrics = current_agent.learn()
        if metrics:
            total_losses.append(metrics.train.loss)
            value_losses.append(metrics.train.value_loss)
            policy_losses.append(metrics.train.policy_loss)
            reporter.log_iteration_end(iteration=iteration, metrics=metrics)

        if (
            config.evaluation.run_periodic_evaluation
            and (iteration + 1) % config.evaluation.periodic_eval_frequency == 0
        ):
            eval_results, tournament_experiences = run_eval_against_benchmark(
                iteration=iteration,
                reporter=reporter,
                current_agent=current_agent,
                best_agent=best_agent,
                best_agent_name=best_agent_name,
                config=config,
                env=env,
            )
            # Add tournament games to buffer
            if tournament_experiences:
                logger.info(
                    f"Adding {len(tournament_experiences)} experiences from tournament games to replay buffer."
                )
                current_agent.add_experiences_to_buffer(tournament_experiences)

            # Check for new best agent
            win_rate = (
                eval_results["wins"]["AlphaZero"] / eval_results["total_games"]
                if eval_results["total_games"] > 0
                else 0
            )
            win_rate_threshold = 0.55  # TODO: Move to config

            if win_rate > win_rate_threshold:
                logger.info(
                    f"New best agent found! Win rate: {win_rate:.2f} > {win_rate_threshold:.2f}"
                )
                best_agent_name = f"AlphaZero_iter{iteration + 1}"

                if not isinstance(best_agent, AlphaZeroAgent):
                    best_agent = make_pure_az(
                        env,
                        config.alpha_zero,
                        config.training,
                        should_use_network=True,
                    )

                best_agent.network.load_state_dict(current_agent.network.state_dict())
                if best_agent.optimizer and current_agent.optimizer:
                    best_agent.optimizer.load_state_dict(
                        current_agent.optimizer.state_dict()
                    )
                self_play_agent = best_agent

                checkpoint_path = (
                    DATA_DIR
                    / f"alphazero_net_{config.env.name}_best_iter_{iteration + 1}.pth"
                )
                current_agent.save(checkpoint_path)
                logger.info(f"Saved new best model to {checkpoint_path}")
            else:
                logger.info(
                    f"Current agent failed to beat best. Win rate: {win_rate:.2f} <= {win_rate_threshold:.2f}"
                )
                if isinstance(best_agent, AlphaZeroAgent):
                    logger.info(
                        "Resetting current agent's weights to best agent's weights."
                    )
                    current_agent.network.load_state_dict(
                        best_agent.network.state_dict()
                    )

    logger.info("\nTraining complete. Saving final agent state.")
    current_agent.save()

    plot_losses(total_losses, value_losses, policy_losses)

    logger.info("\n--- AlphaZero Training Finished ---")

    reporter.finish()


def run_self_play(
    agent: Union[AlphaZeroAgent, MCTSAgent], env: BaseEnvironment, config: AppConfig
):
    logger.info("Running self play")
    if hasattr(agent, "network") and agent.network:
        agent.network.eval()

    num_games_total = config.training.num_games_per_iteration
    all_experiences_iteration = []

    for _ in tqdm(range(num_games_total), desc="Self-Play Games"):
        game_env = env.copy()
        state_with_key = game_env.reset()
        game_history = []

        while not game_env.state.done:
            state = state_with_key.state
            action = agent.act(game_env, train=True)

            policy_target = np.zeros(game_env.num_action_types, dtype=np.float32)
            if isinstance(agent, AlphaZeroAgent):
                policy_target = agent.get_policy_target()
            elif isinstance(agent, MCTSAgent):
                if not agent.root:
                    raise RuntimeError("MCTSAgent has no root after act()")
                action_visits = {
                    edge_action: edge.num_visits
                    for edge_action, edge in agent.root.edges.items()
                }
                total_visits = sum(action_visits.values())
                if total_visits > 0:
                    for act_key, visits in action_visits.items():
                        action_idx = game_env.map_action_to_policy_index(act_key)
                        if action_idx is not None:
                            policy_target[action_idx] = visits / total_visits
            else:
                raise TypeError(f"Unsupported agent type for self-play: {type(agent)}")

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
    current_agent: AlphaZeroAgent,
    best_agent: Union[AlphaZeroAgent, MCTSAgent],
    best_agent_name: str,
    config: AppConfig,
    env: BaseEnvironment,
) -> Tuple[dict, List[Tuple[StateType, np.ndarray, float]]]:
    logger.info(
        f"\n--- Running Evaluation vs '{best_agent_name}' (Iteration {iteration + 1}) ---"
    )
    current_agent.network.eval()
    if hasattr(best_agent, "network") and best_agent.network:
        best_agent.network.eval()

    num_games = config.evaluation.periodic_eval_num_games
    wins = {"AlphaZero": 0, best_agent_name: 0, "draw": 0}
    all_tournament_experiences = []

    for game_num in tqdm(range(num_games), desc=f"Eval vs {best_agent_name}"):
        game_env = env.copy()
        state_with_key = game_env.reset()
        game_history = []

        # Alternate who goes first
        agents = {0: current_agent, 1: best_agent}
        if game_num % 2 == 1:
            agents = {0: best_agent, 1: current_agent}

        while not game_env.state.done:
            player = game_env.get_current_player()
            agent_for_turn = agents[player]

            state = state_with_key.state
            action = agent_for_turn.act(game_env, train=False)

            policy_target = np.zeros(game_env.num_action_types, dtype=np.float32)
            if isinstance(agent_for_turn, AlphaZeroAgent):
                policy_target = agent_for_turn.get_policy_target()
            elif isinstance(agent_for_turn, MCTSAgent):
                if agent_for_turn.root:
                    action_visits = {
                        k: v.num_visits for k, v in agent_for_turn.root.edges.items()
                    }
                    total_visits = sum(action_visits.values())
                    if total_visits > 0:
                        for act_key, visits in action_visits.items():
                            idx = game_env.map_action_to_policy_index(act_key)
                            if idx is not None:
                                policy_target[idx] = visits / total_visits

            game_history.append((state, action, policy_target))
            action_result = game_env.step(action)
            state_with_key = action_result.next_state_with_key

        outcome = game_env.state.get_reward_for_player(0)
        episode_result = current_agent.process_finished_episode(game_history, outcome)
        all_tournament_experiences.extend(episode_result.buffer_experiences)

        winner = game_env.get_winning_player()
        if winner is None:
            wins["draw"] += 1
        else:
            winner_agent = agents[winner]
            if winner_agent == current_agent:
                wins["AlphaZero"] += 1
            else:
                wins[best_agent_name] += 1

    eval_results = {
        "wins": wins,
        "total_games": num_games,
        "win_rate": wins["AlphaZero"] / num_games if num_games > 0 else 0,
    }

    if iteration > -1:  # Don't log for initial evaluation
        reporter.log_evaluation_results(
            eval_results=eval_results,
            benchmark_agent_name=best_agent_name,
            iteration=iteration,
        )
    return eval_results, all_tournament_experiences


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
        temp_env = env.copy()
        temp_env.set_state(check_case.state_with_key.state)
        temp_env.render()

        try:
            policy_np, value_np = agent.network.predict(check_case.state_with_key)
            if check_case.expected_value is None:
                logger.info(f"  Value: Predicted={value_np:.4f}")
            else:
                logger.info(
                    f"  Value: Expected={check_case.expected_value:.1f}, Predicted={value_np:.4f}"
                )

            legal_actions = temp_env.get_legal_actions()

            action_probs = {}
            for action in legal_actions:
                idx = temp_env.map_action_to_policy_index(action)
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
