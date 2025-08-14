import sys
import time
from typing import Union, List, Tuple

import numpy as np
from tqdm import tqdm
from loguru import logger

from agents.mcts_agent import MCTSAgent, make_pure_mcts  # Keep for benchmark
from core.config import AppConfig
from environments.base import BaseEnvironment, StateType
from agents.muzero_agent import MuZeroAgent  # TODO: need factory
from agents.alphazero_agent import AlphaZeroAgent  # For type hinting until refactor
from factories import (
    get_environment,
    get_agents,  # This will need to be adapted for MuZero
)
from utils.plotting import plot_losses
from utils.training_reporter import TrainingReporter


# TODO: This is nearly identical to the one in train_alphazero.py
# Refactoring opportunity: Move this to a shared module.
def load_game_logs_into_buffer(agent: MuZeroAgent, env_name: str, buffer_limit: int):
    # ... This function will need to be adapted for MuZero's buffer format
    logger.warning("load_game_logs_into_buffer for MuZero not fully implemented.")
    pass


def run_training(config: AppConfig, env_name_override: str = None):
    """Runs the MuZero training process."""

    # TODO: This setup is nearly identical to train_alphazero.py
    # Refactoring opportunity: abstract the main training loop.
    if env_name_override:
        config.env.name = env_name_override

    env = get_environment(config.env)
    # agents = get_agents(env, config) # This needs to be adapted for MuZero
    # For now, let's assume we have a way to create a MuZero agent.
    # current_agent = ... a MuZeroAgent instance
    current_agent: MuZeroAgent = None  # Placeholder
    if not current_agent:
        logger.error("MuZero agent creation not implemented yet. Exiting.")
        sys.exit(1)

    logger.info("Initializing with pure MCTS as the starting 'best' agent.")
    best_agent = make_pure_mcts(num_simulations=config.mcts.num_simulations)
    best_agent.temperature = 1.0
    self_play_agent = best_agent
    best_agent_name = f"MCTS_{config.mcts.num_simulations}"

    # load_game_logs_into_buffer(
    #     current_agent, config.env.name, config.muzero.replay_buffer_size # TODO: Muzero config
    # )

    logger.info(
        f"Starting MuZero training for {config.training.num_iterations} iterations...\n"
        f"({config.training.num_games_per_iteration} self-play games per iteration)"
    )

    total_losses, value_losses, policy_losses = [], [], []

    outer_loop_iterator = range(config.training.num_iterations)
    start_time = time.time()
    # reporter = TrainingReporter(config, current_agent, start_time) # Reporter might need adaptation

    for iteration in outer_loop_iterator:
        # reporter.log_iteration_start(iteration)

        logger.info(f"Running self-play with '{best_agent_name}'...")
        # TODO: Self-play for MuZero needs to store rewards at each step.
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
            # ... update losses, log metrics
            pass

        # TODO: Evaluation logic is identical to AlphaZero's.
        # Refactoring opportunity: move to shared module.
        if (
            config.evaluation.run_periodic_evaluation
            and (iteration + 1) % config.evaluation.periodic_eval_frequency == 0
        ):
            # eval_results, tournament_experiences = run_eval_against_benchmark(...)
            # ... logic to check for new best agent, save checkpoint, etc.
            pass

    logger.info("\nTraining complete. Saving final agent state.")
    current_agent.save()

    plot_losses(total_losses, value_losses, policy_losses)
    logger.info("\n--- MuZero Training Finished ---")

    # reporter.finish()


# TODO: This is very similar to train_alphazero.py.
# The main difference for MuZero is that `game_history` must also include the reward at each step.
def run_self_play(
    agent: Union[MuZeroAgent, MCTSAgent], env: BaseEnvironment, config: AppConfig
):
    logger.info("Running self play for MuZero")
    num_games_total = config.training.num_games_per_iteration
    all_experiences_iteration = []

    for _ in tqdm(range(num_games_total), desc="Self-Play Games"):
        game_env = env.copy()
        game_env.reset()
        # For MuZero, history needs to store (observation, action, reward, policy_target)
        game_history = []

        while not game_env.state.done:
            # ... similar to AlphaZero's self-play loop ...
            # action = agent.act(game_env, train=True)
            # policy_target = agent.get_policy_target()
            # observation = game_env.get_state_with_key().state
            # action_result = game_env.step(action)
            # reward = action_result.reward # Assuming reward per step is available
            # game_history.append((observation, action, reward, policy_target))
            pass  # placeholder

        # final_outcome doesn't mean much for step-by-step reward storage
        all_experiences_iteration.append(game_history)
    return all_experiences_iteration


# TODO: This is similar to train_alphazero.py
# For MuZero, it will take the game history (with rewards) and just add it to the buffer.
def add_results_to_buffer(
    iteration: int,
    all_experiences_iteration: list,
    agent: MuZeroAgent,
    config: AppConfig,
):
    logger.warning("add_results_to_buffer for MuZero not implemented.")
    # for raw_history in all_experiences_iteration:
    #   agent.add_experiences_to_buffer([raw_history]) # Add whole game trajectory
    pass


# TODO: This function is IDENTICAL to the one in train_alphazero.py.
# It can be moved to a shared module without any changes.
def run_eval_against_benchmark(
    iteration: int,
    reporter: TrainingReporter,
    current_agent: AlphaZeroAgent,  # Should be generic Agent
    best_agent: Union[AlphaZeroAgent, MCTSAgent],  # Should be generic Agent
    best_agent_name: str,
    config: AppConfig,
    env: BaseEnvironment,
) -> Tuple[dict, List[Tuple[StateType, np.ndarray, float]]]:
    pass  # Not implementing here, just pointing out it's a refactor candidate.


if __name__ == "__main__":
    config = AppConfig()
    env_override = None

    if len(sys.argv) > 1:
        env_override = sys.argv[1]

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    run_training(config, env_name_override=env_override)
