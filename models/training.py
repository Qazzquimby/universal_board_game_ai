from typing import List, Union, Tuple
import time
import os
import asyncio

import numpy as np
from tqdm import tqdm
from loguru import logger

from agents.base_learning_agent import GameHistoryStep, BaseLearningAgent
from agents.mcts_agent import MCTSAgent, make_pure_mcts
from core.config import AppConfig
from core.serialization import save_game_log
from environments.base import BaseEnvironment, DataFrame
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from agents.muzero.muzero_agent import MuZeroAgent
from factories import get_environment, create_learning_agent
from remote_play.client import RemotePlayClient
from utils.training_reporter import TrainingReporter, BenchmarkResults

SELF_PLAY_ON_FIRST_ITER = True

USE_REMOTE_SELF_PLAY = True


def run_training_loop(
    config: AppConfig, model_type: str, env_name_override: str = None
):
    """Runs the training process for a given model type (AlphaZero or MuZero)."""

    if env_name_override:
        config.env.name = env_name_override

    env = get_environment(config.env)

    current_agent = create_learning_agent(model_type, env, config)
    mcts_agent = make_pure_mcts(num_simulations=config.mcts.num_simulations)
    mcts_agent.name = "mcts"
    mcts_agent.model_name = "mcts"
    self_play_agent, start_iteration = get_self_play_agent_and_start_iteration(
        env=env,
        config=config,
        model_type=model_type,
        current_agent=current_agent,
        base_agent=mcts_agent,
    )

    self_play_agent.temperature = 0.15

    current_agent.load_game_logs(
        config.env.name, current_agent.config.replay_buffer_size
    )

    logger.info(
        f"Starting {current_agent.name} training for {config.training.num_iterations} iterations...\n"
        f"({config.training.num_games_per_iteration} self-play games per iteration)"
    )
    outer_loop_iterator = range(start_iteration, config.training.num_iterations)
    start_time = time.time()
    reporter = TrainingReporter(config, current_agent, start_time)

    for iteration in outer_loop_iterator:
        reporter.log_iteration_start(iteration)

        current_agent.model_name = f"{model_type}_iter_{iteration:03d}"
        current_agent.name = current_agent.model_name.capitalize()

        if iteration > start_iteration or SELF_PLAY_ON_FIRST_ITER:
            logger.info(f"Running self-play with '{self_play_agent.name}'...")
            if USE_REMOTE_SELF_PLAY:
                if not os.path.exists("servers.json"):
                    raise ValueError("Run start.py first to start the remote server")
                run_remote_self_play(
                    learning_agent=current_agent,
                    self_play_agent=self_play_agent,
                    env=env,
                    config=config,
                    iteration=iteration,
                )
            else:
                run_self_play(
                    learning_agent=current_agent,
                    self_play_agent=self_play_agent,
                    env=env,
                    config=config,
                    iteration=iteration,
                )

        logger.info("Running learning step...")
        metrics = current_agent.train_network()
        if metrics:
            reporter.log_iteration_end(iteration=iteration, metrics=metrics)

        current_agent.save(iteration=iteration)
        logger.info(f"Saved checkpoint for iteration {iteration}")

        if isinstance(self_play_agent, MCTSAgent):
            eval_results = check_if_agent_outperforms_mcts(
                iteration=iteration,
                reporter=reporter,
                current_agent=current_agent,
                mcts_agent=mcts_agent,
                env=env,
                config=config,
            )

            if eval_results.win_rate >= 0.6:
                logger.info(
                    f"{current_agent.name} outperformed MCTS with win rate: {eval_results.win_rate:.2f}. "
                    f"Promoting to use {current_agent.name} for self-play."
                )
                current_agent.promote_to_self_play(iteration)
                self_play_agent = create_learning_agent(model_type, env, config)
                model_path = current_agent.get_model_iter_path(iteration)
                self_play_agent.load(model_path)
                self_play_agent.model_name = f"{model_type}_iter_{iteration:03d}"
                self_play_agent.name = self_play_agent.model_name.capitalize()
            else:
                logger.info(
                    f"{current_agent.name} did not outperform MCTS (win rate: {eval_results.win_rate:.2f}). "
                    "Continuing with MCTS for self-play."
                )
        else:
            # Once promoted, the self-play agent is a learning agent.
            # We should have logic here to see if the new agent is better than the current self-play agent.
            # For now, we just update to the latest agent.
            self_play_agent = create_learning_agent(model_type, env, config)
            model_path = current_agent.get_model_iter_path(iteration)
            self_play_agent.load(model_path)
            self_play_agent.model_name = f"{model_type}_iter_{iteration:03d}"
            self_play_agent.name = self_play_agent.model_name.capitalize()

    # plot_losses(total_losses, value_losses, policy_losses)

    logger.info(f"\n--- {current_agent.name} Training Finished ---")

    reporter.finish()


def get_self_play_agent_and_start_iteration(
    env, config, model_type, current_agent, base_agent
):
    current_agent.load_latest_version()
    start_iteration = current_agent.iteration_to_start_training_at

    self_play_iter = current_agent.get_self_play_agent_iter()
    if self_play_iter is not None:
        logger.info(f"Loading agent from iter {self_play_iter} for self-play.")
        self_play_agent = create_learning_agent(model_type, env, config)
        model_path = self_play_agent.get_model_iter_path(self_play_iter)
        self_play_agent.load(model_path)
        self_play_agent.model_name = f"{model_type}_iter_{self_play_iter:03d}"
        self_play_agent.name = self_play_agent.model_name.capitalize()
    else:
        logger.info("No promoted self-play agent found. Using pure MCTS.")
        self_play_agent = base_agent

    return self_play_agent, start_iteration


def _run_one_self_play_game(
    env: BaseEnvironment, self_play_agent: Union[AlphaZeroAgent, MuZeroAgent, MCTSAgent]
) -> Tuple[List[GameHistoryStep], float]:
    """Runs a single game of self-play and returns the history."""
    game_env = env.copy()
    state_with_key = game_env.reset()
    game_history = []
    self_play_agent.reset_game()

    turn = 0
    while not state_with_key.done:
        state = state_with_key.state
        legal_actions = game_env.get_legal_actions()
        action_index = self_play_agent.act(game_env, train=True)

        if isinstance(self_play_agent, AlphaZeroAgent) or isinstance(
            self_play_agent, MuZeroAgent
        ):
            policy_target = self_play_agent.get_policy_target(legal_actions)
        elif isinstance(self_play_agent, MCTSAgent):
            policy_target = np.zeros(len(legal_actions), dtype=np.float32)
            if not self_play_agent.root:
                raise RuntimeError("MCTSAgent has no root after act()")
            action_visits = {
                edge_action: edge.num_visits
                for edge_action, edge in self_play_agent.root.edges.items()
            }
            total_visits = sum(action_visits.values())
            if total_visits > 0:
                visit_probs = {
                    act: visits / total_visits for act, visits in action_visits.items()
                }
                for i, act in enumerate(legal_actions):
                    policy_target[i] = visit_probs.get(i, 0.0)
        else:
            raise TypeError(
                f"Unsupported agent type for self-play: {type(self_play_agent)}"
            )

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

        turn += 1
        if turn >= 100:
            print("WARN: game timed out, setting draw")
            state_with_key.state["game"]["done"][0] = True

        game_history.append(
            GameHistoryStep(
                state=state_with_actions,
                action_index=action_index,
                policy=policy_target,
                legal_actions=legal_actions,
            )
        )
        action = legal_actions[action_index]
        action_result = game_env.step(action)
        state_with_key = action_result.next_state_with_key

    final_outcome = game_env.get_reward_for_player(player=0)
    return game_history, final_outcome


def _process_and_save_game_results(
    game_history: List[GameHistoryStep],
    final_outcome: float,
    learning_agent: BaseLearningAgent,
    game_log_index: int,
    iteration: int,
    config: AppConfig,
    model_name: str,
    env: BaseEnvironment,
) -> int:
    """Processes a single game's results and saves them."""
    episode_result = learning_agent.process_finished_episode(
        game_history, final_outcome
    )

    buffer_experiences = episode_result.buffer_experiences
    if hasattr(env, "augment_experiences"):
        buffer_experiences = env.augment_experiences(buffer_experiences)

    learning_agent.add_experiences_to_buffer(buffer_experiences)

    save_game_log(
        logged_history=episode_result.logged_history,
        iteration=iteration + 1,
        game_index=game_log_index,
        env_name=config.env.name,
        model_name=model_name,
    )
    return len(buffer_experiences)


def run_remote_self_play(
    learning_agent: BaseLearningAgent,
    self_play_agent: Union[AlphaZeroAgent, MuZeroAgent, MCTSAgent],
    env: BaseEnvironment,
    config: AppConfig,
    iteration: int,
):
    logger.info("Running remote self play")
    client = RemotePlayClient()
    if not client.ips:
        run_self_play(
            learning_agent=learning_agent,
            self_play_agent=self_play_agent,
            env=env,
            config=config,
            iteration=iteration,
        )
        return

    model_path = learning_agent.get_model_iter_path(iteration - 1)
    num_games = config.training.num_games_per_iteration

    async def _run_remote_and_process():
        game_log_index_offset = iteration * config.training.num_games_per_iteration
        total_experiences_added = 0
        total_games_processed = 0
        pbar = tqdm(total=num_games, desc="Remote Self-Play Games")

        async for game_history, final_outcome in client.run_self_play_games(
            model_path=str(model_path),
            num_games=num_games,
            config=config,
            model_type=self_play_agent.model_type,
        ):
            if not game_history:
                continue

            current_game_log_index = game_log_index_offset + total_games_processed + 1
            experiences_added = _process_and_save_game_results(
                game_history=game_history,
                final_outcome=final_outcome,
                learning_agent=learning_agent,
                game_log_index=current_game_log_index,
                iteration=iteration,
                config=config,
                model_name=self_play_agent.model_name,
                env=env,
            )
            total_experiences_added += experiences_added
            total_games_processed += 1
            pbar.update(1)

        pbar.close()
        logger.info(
            f"Processed {total_games_processed} games, adding {total_experiences_added} experiences to replay buffer."
        )

    asyncio.run(_run_remote_and_process())


def run_self_play(
    learning_agent: BaseLearningAgent,
    self_play_agent: Union[AlphaZeroAgent, MuZeroAgent, MCTSAgent],
    env: BaseEnvironment,
    config: AppConfig,
    iteration: int,
):
    logger.info("Running self play")
    if hasattr(self_play_agent, "network") and self_play_agent.network:
        self_play_agent.network.eval()

    num_games_total = config.training.num_games_per_iteration
    game_log_index_offset = iteration * config.training.num_games_per_iteration
    total_experiences_added = 0
    total_games_processed = 0

    for game_num in tqdm(range(num_games_total), desc="Self-Play Games"):
        game_history, final_outcome = _run_one_self_play_game(env, self_play_agent)

        if not game_history:
            logger.warning(f"Skipping game {game_num} with empty raw history.")
            continue

        current_game_log_index = game_log_index_offset + total_games_processed + 1
        experiences_added = _process_and_save_game_results(
            game_history=game_history,
            final_outcome=final_outcome,
            learning_agent=learning_agent,
            game_log_index=current_game_log_index,
            iteration=iteration,
            config=config,
            model_name=self_play_agent.model_name,
            env=env,
        )
        total_experiences_added += experiences_added
        total_games_processed += 1

    logger.info(
        f"Processed {total_games_processed} games, adding {total_experiences_added} experiences to replay buffer."
    )


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
    return eval_results


def add_results_to_buffer(
    iteration: int,
    all_experiences_iteration: list,
    agent: BaseLearningAgent,
    config: AppConfig,
    model_name: str,
    env: BaseEnvironment,
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

        buffer_experiences = episode_result.buffer_experiences
        if hasattr(env, "augment_experiences"):
            buffer_experiences = env.augment_experiences(buffer_experiences)

        agent.add_experiences_to_buffer(buffer_experiences)
        total_experiences_added += len(buffer_experiences)

        current_game_log_index = game_log_index_offset + total_games_processed + 1
        save_game_log(
            logged_history=episode_result.logged_history,
            iteration=iteration + 1,
            game_index=current_game_log_index,
            env_name=config.env.name,
            model_name=model_name,
        )
        total_games_processed += 1

    logger.info(
        f"Processed {total_games_processed} games, adding {total_experiences_added} experiences to replay buffer."
    )


def run_eval_against_benchmark(
    iteration: int,
    agent_in_training: AlphaZeroAgent,
    benchmark_agent: Union[AlphaZeroAgent, MuZeroAgent, MCTSAgent],
    benchmark_agent_name: str,
    config: AppConfig,
    env: BaseEnvironment,
    reporter: TrainingReporter = None,
) -> Tuple[BenchmarkResults, List[Tuple[List[GameHistoryStep], float]]]:
    logger.info(
        f"\n--- Running Evaluation vs '{benchmark_agent_name}' (Iteration {iteration}) ---"
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
            action_index = agent_for_turn.act(game_env, train=False)

            if isinstance(agent_for_turn, AlphaZeroAgent) or isinstance(
                agent_for_turn, MuZeroAgent
            ):
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
                            policy_target[i] = visit_probs.get(i, 0.0)
            else:
                raise ValueError(f"Unsupported agent type {agent_for_turn}")

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
            game_history_step = GameHistoryStep(
                state=state_with_actions,
                action_index=action_index,
                policy=policy_target,
                legal_actions=legal_actions,
            )
            game_history.append(game_history_step)
            action = legal_actions[action_index]
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

    eval_results = BenchmarkResults(
        wins=wins,
        total_games=num_games,
        win_rate=wins[agent_in_training.name] / num_games if num_games > 0 else 0,
    )

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
