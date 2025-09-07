import json
import time
from typing import Union, Tuple

import numpy as np
from tqdm import tqdm
from loguru import logger

from agents.base_learning_agent import BaseLearningAgent
from agents.mcts_agent import MCTSAgent, make_pure_mcts
from core.config import AppConfig, DATA_DIR
from core.serialization import LOG_DIR, save_game_log
from environments.base import BaseEnvironment, DataFrame
from agents.alphazero.alphazero_agent import AlphaZeroAgent, make_pure_az
from agents.muzero.muzero_agent import make_pure_muzero
from factories import get_environment
from utils.plotting import plot_losses
from utils.training_reporter import TrainingReporter




def run_training_loop(
    config: AppConfig, model_type: str, env_name_override: str = None
):
    """Runs the training process for a given model type (AlphaZero or MuZero)."""

    if env_name_override:
        config.env.name = env_name_override

    env = get_environment(config.env)

    if model_type == "alphazero":
        current_agent = make_pure_az(
            env=env,
            config=config.alphazero,
            training_config=config.training,
        )
    elif model_type == "muzero":
        current_agent = make_pure_muzero(
            env=env,
            config=config.muzero,
            training_config=config.training,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    mcts_agent = make_pure_mcts(num_simulations=config.mcts.num_simulations)
    has_checkpoint = current_agent.load()

    if has_checkpoint:
        logger.info(
            f"Checkpoint found, starting with {current_agent.name} for self-play."
        )
        self_play_agent = current_agent
    else:
        logger.info("No checkpoint, starting with pure MCTS for self-play.")
        self_play_agent = mcts_agent

    self_play_agent.temperature = 0.15

    current_agent.load_game_logs(
        config.env.name, current_agent.config.replay_buffer_size
    )

    logger.info(
        f"Starting {current_agent.name} training for {config.training.num_iterations} iterations...\n"
        f"({config.training.num_games_per_iteration} self-play games per iteration)"
    )

    total_losses, value_losses, policy_losses = [], [], []

    outer_loop_iterator = range(config.training.num_iterations)
    start_time = time.time()
    reporter = TrainingReporter(config, current_agent, start_time)

    for iteration in outer_loop_iterator:
        reporter.log_iteration_start(iteration)

        logger.info(f"Running self-play with '{type(self_play_agent).__name__}'...")
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
        metrics = current_agent.train_network()
        if metrics:
            total_losses.append(metrics.train.loss)
            value_losses.append(metrics.train.value_loss)
            policy_losses.append(metrics.train.policy_loss)
            reporter.log_iteration_end(iteration=iteration, metrics=metrics)

        if isinstance(self_play_agent, MCTSAgent):
            eval_results = check_if_agent_outperforms_mcts(
                iteration=iteration,
                reporter=reporter,
                current_agent=current_agent,
                mcts_agent=mcts_agent,
                env=env,
                config=config,
            )

            if eval_results["win_rate"] > 0.6:
                logger.info(
                    f"{current_agent.name} outperformed MCTS with win rate: {eval_results['win_rate']:.2f}. "
                    f"Promoting to use {current_agent.name} for self-play."
                )
                self_play_agent = current_agent
            else:
                logger.info(
                    f"{current_agent.name} did not outperform MCTS (win rate: {eval_results['win_rate']:.2f}). "
                    "Continuing with MCTS for self-play."
                )
        else:
            self_play_agent = current_agent

        checkpoint_path = (
            DATA_DIR
            / f"{current_agent.model_name}_net_{config.env.name}_iter_{iteration + 1}.pth"
        )
        current_agent.save(checkpoint_path)
        current_agent.save()
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    plot_losses(total_losses, value_losses, policy_losses)

    logger.info(f"\n--- {current_agent.name} Training Finished ---")

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
        agent.reset_game()

        while not state_with_key.done:
            state = state_with_key.state
            legal_actions = game_env.get_legal_actions()
            action = agent.act(game_env, train=True)

            if isinstance(agent, AlphaZeroAgent):
                policy_target = agent.get_policy_target(legal_actions)
            elif isinstance(agent, MCTSAgent):
                policy_target = np.zeros(len(legal_actions), dtype=np.float32)
                if not agent.root:
                    raise RuntimeError("MCTSAgent has no root after act()")
                action_visits = {
                    edge_action: edge.num_visits
                    for edge_action, edge in agent.root.edges.items()
                }
                total_visits = sum(action_visits.values())
                if total_visits > 0:
                    visit_probs = {
                        act: visits / total_visits
                        for act, visits in action_visits.items()
                    }
                    for i, act in enumerate(legal_actions):
                        act_key = tuple(act) if isinstance(act, list) else act
                        policy_target[i] = visit_probs.get(act_key, 0.0)
            else:
                raise TypeError(f"Unsupported agent type for self-play: {type(agent)}")

            state_with_actions = state.copy()
            if legal_actions:
                action_data = [[a] for a in legal_actions]
                state_with_actions["legal_actions"] = DataFrame(
                    data=action_data, columns=["action_id"]
                )
            else:
                state_with_actions["legal_actions"] = DataFrame(
                    data=[], columns=["action_id"]
                )

            game_history.append((state_with_actions, action, policy_target))
            action_result = game_env.step(action)
            state_with_key = action_result.next_state_with_key

        final_outcome = game_env.get_reward_for_player(player=0)
        all_experiences_iteration.append((game_history, final_outcome))
    return all_experiences_iteration


def check_if_agent_outperforms_mcts(
    iteration, current_agent, mcts_agent, env, config, reporter=None
):
    logger.info(f"Evaluating {current_agent.name} against MCTS for promotion...")
    original_num_games = config.evaluation.periodic_eval_num_games
    config.evaluation.periodic_eval_num_games = 20
    eval_results, tournament_experiences = run_eval_against_benchmark(
        iteration=iteration,
        reporter=reporter,
        agent_in_training=current_agent,
        benchmark_agent=mcts_agent,
        benchmark_agent_name="MCTS",
        config=config,
        env=env,
    )
    config.evaluation.periodic_eval_num_games = original_num_games
    add_results_to_buffer(
        iteration=iteration,
        all_experiences_iteration=tournament_experiences,
        agent=current_agent,
        config=config,
    )
    return eval_results


def add_results_to_buffer(
    iteration: int,
    all_experiences_iteration: list,
    agent: BaseLearningAgent,
    config: AppConfig,
):
    total_experiences_added = 0
    total_games_processed = 0
    game_log_index_offset = iteration * config.training.num_games_per_iteration

    logger.info(
        f"Processing {len(all_experiences_iteration)} collected game results..."
    )
    for i, (raw_history, final_outcome) in enumerate(all_experiences_iteration):
        if not raw_history:
            logger.warning(f"Skipping game {i} with empty raw history.")
            continue

        episode_result = agent.process_finished_episode(raw_history, final_outcome)

        agent.add_experiences_to_buffer(episode_result.buffer_experiences)
        total_experiences_added += len(episode_result.buffer_experiences)

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
    agent_in_training: AlphaZeroAgent,
    benchmark_agent: Union[AlphaZeroAgent, MCTSAgent],
    benchmark_agent_name: str,
    config: AppConfig,
    env: BaseEnvironment,
    reporter: TrainingReporter = None,
) -> Tuple[dict, list]:
    logger.info(
        f"\n--- Running Evaluation vs '{benchmark_agent_name}' (Iteration {iteration + 1}) ---"
    )
    benchmark_agent.name = benchmark_agent_name
    agent_in_training.network.eval()
    benchmark_agent.temperature = 0.0
    if hasattr(benchmark_agent, "network") and benchmark_agent.network:
        benchmark_agent.network.eval()

    num_games = config.evaluation.periodic_eval_num_games
    wins = {agent_in_training.name: 0, benchmark_agent.name: 0, "draw": 0}
    all_experiences = []

    for game_num in tqdm(range(num_games), desc=f"Eval vs {benchmark_agent.name}"):
        game_env = env.copy()
        state_with_key = game_env.reset()
        game_history = []

        agents = {0: agent_in_training, 1: benchmark_agent}
        if game_num % 2 == 1:
            agents = {0: benchmark_agent, 1: agent_in_training}
        agents[0].reset_game()
        agents[1].reset_game()

        while not state_with_key.done:
            player = game_env.get_current_player()
            agent_for_turn = agents[player]

            state = state_with_key.state
            legal_actions = game_env.get_legal_actions()
            action = agent_for_turn.act(game_env, train=False)

            if isinstance(agent_for_turn, AlphaZeroAgent):
                policy_target = agent_for_turn.get_policy_target(legal_actions)
            elif isinstance(agent_for_turn, MCTSAgent):
                policy_target = np.zeros(len(legal_actions), dtype=np.float32)
                if agent_for_turn.root:
                    action_visits = {
                        k: v.num_visits for k, v in agent_for_turn.root.edges.items()
                    }
                    total_visits = sum(action_visits.values())
                    if total_visits > 0:
                        visit_probs = {
                            k: v / total_visits for k, v in action_visits.items()
                        }
                        for i, act in enumerate(legal_actions):
                            act_key = tuple(act) if isinstance(act, list) else act
                            policy_target[i] = visit_probs.get(act_key, 0.0)

            state_with_actions = state.copy()
            if legal_actions:
                action_data = [[a] for a in legal_actions]
                state_with_actions["legal_actions"] = DataFrame(
                    data=action_data, columns=["action_id"]
                )
            else:
                state_with_actions["legal_actions"] = DataFrame(
                    data=[], columns=["action_id"]
                )
            game_history.append((state_with_actions, action, policy_target))
            action_result = game_env.step(action)
            state_with_key = action_result.next_state_with_key

        outcome = game_env.get_reward_for_player(player=0)
        all_experiences.append((game_history, outcome))

        winner = game_env.get_winning_player()
        if winner is None:
            wins["draw"] += 1
        else:
            winner_agent = agents[winner]
            wins[winner_agent.name] += 1

    eval_results = {
        "wins": wins,
        "total_games": num_games,
        "win_rate": wins[agent_in_training.name] / num_games if num_games > 0 else 0,
    }

    if iteration > -1 and reporter:
        reporter.log_evaluation_results(
            eval_results=eval_results,
            benchmark_agent_name=benchmark_agent_name,
            iteration=iteration,
        )
    benchmark_agent.temperature = 1.0
    return eval_results, all_experiences


def run_sanity_checks(env: BaseEnvironment, agent: AlphaZeroAgent):
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

        legal_actions = temp_env.get_legal_actions()
        try:
            policy_dict, value_np = agent.network.predict(
                check_case.state_with_key, legal_actions
            )
            if check_case.expected_value is None:
                logger.info(f"  Value: Predicted={value_np:.4f}")
            else:
                logger.info(
                    f"  Value: Expected={check_case.expected_value:.1f}, Predicted={value_np:.4f}"
                )

            action_probs = policy_dict

            sorted_probs = sorted(
                action_probs.items(), key=lambda item: item[1], reverse=True
            )

            logger.info(f"  Predicted Probabilities for Legal Actions:")
            if not sorted_probs:
                logger.info("    - (No legal actions)")
            else:
                for action, prob in sorted_probs:
                    highlight = " <<< BEST" if action == sorted_probs[0][0] else ""
                    logger.info(f"    - {action}: {prob:.4f}{highlight}")

        except Exception as e:
            logger.error(f"  Error during prediction for this state: {e}")
