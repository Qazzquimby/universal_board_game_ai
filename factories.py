from typing import Dict

from core.agent_interface import Agent
from core.config import AppConfig, EnvConfig
from environments.base import BaseEnvironment
from environments.four_in_a_row import FourInARow
from environments.nim_env import NimEnv
from agents.mcts_agent import MCTSAgent
from agents.qlearning import (
    QLearningAgent,
    train_agent as train_q_agent,
)
from agents.random_agent import RandomAgent
from utils.plotting import plot_results


def get_environment(env_config: EnvConfig) -> BaseEnvironment:
    """Factory function to create environment instances."""
    if env_config.name.lower() == "fourinarow":
        print(
            f"Using FourInARow environment ({env_config.board_size}x{env_config.board_size})"
        )
        return FourInARow(
            board_size=env_config.board_size,
            num_players=env_config.num_players,
            max_steps=env_config.max_steps,
        )
    elif env_config.name.lower() == "nim":
        print(f"Using Nim environment with piles: {env_config.nim_piles}")
        return NimEnv(
            initial_piles=env_config.nim_piles, num_players=env_config.num_players
        )
    else:
        raise ValueError(f"Unknown environment name: {env_config.name}")


def get_agents(env: BaseEnvironment, config: AppConfig) -> Dict[str, Agent]:
    """Factory function to create agent instances for the given environment."""

    # --- QLearning Agent Initialization & Training ---
    ql_agent = QLearningAgent(env, config.q_learning)
    if not ql_agent.load():  # Load uses internal path logic now
        print(
            f"Training Q-learning agent for {config.training.num_episodes} episodes..."
        )
        ql_agent.exploration_rate = (
            config.q_learning.exploration_rate
        )  # Reset exploration for training
        wins = train_q_agent(
            env,
            ql_agent,
            num_episodes=config.training.num_episodes,
            q_config=config.q_learning,
        )
        plot_results(wins, window_size=config.training.plot_window)
        ql_agent.save()  # Save uses internal path logic now
        ql_agent.exploration_rate = (
            config.q_learning.min_exploration
        )  # Set low exploration after training/saving
    else:
        print(f"Loaded pre-trained Q-learning agent.")

    # Ensure Q-agent used for testing has exploration turned off or minimized.
    ql_agent.exploration_rate = config.q_learning.min_exploration

    # --- Other Agents ---
    agents = {
        "QLearning": ql_agent,
        "MCTS_50": MCTSAgent(
            env,
            num_simulations=config.mcts.num_simulations_short,
            exploration_constant=config.mcts.exploration_constant,
        ),
        "MCTS_200": MCTSAgent(
            env,
            num_simulations=config.mcts.num_simulations_long,
            exploration_constant=config.mcts.exploration_constant,
        ),
        "Random": RandomAgent(env),
    }
    return agents
