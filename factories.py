from typing import Dict
from loguru import logger

from core.agent_interface import Agent
from core.config import (
    AppConfig,
    EnvConfig,
)
from environments.base import BaseEnvironment
from environments.connect4 import Connect4
from environments.nim_env import NimEnv
from agents.mcts_agent import MCTSAgent
from agents.alphazero_agent import AlphaZeroAgent


def get_environment(env_config: EnvConfig) -> BaseEnvironment:
    """Factory function to create environment instances."""
    if env_config.name.lower() == "connect4":
        # Use connect4 specific dimensions from config
        return Connect4(
            width=env_config.width,
            height=env_config.height,
            num_players=env_config.num_players,
            max_steps=env_config.max_steps,
        )
    elif env_config.name.lower() == "nim":
        return NimEnv(
            initial_piles=env_config.nim_piles, num_players=env_config.num_players
        )
    else:
        raise ValueError(f"Unknown environment name: {env_config.name}")


def get_agents(env: BaseEnvironment, config: AppConfig) -> Dict[str, Agent]:
    """Factory function to create agent instances for the given environment."""

    az_agent = AlphaZeroAgent(
        env=env,
        config=config.alpha_zero,
        training_config=config.training,
    )
    if not az_agent.load():
        logger.warning(
            "Could not load pre-trained AlphaZero weights. Agent will play randomly/poorly."
        )
    else:
        logger.info("Loaded pre-trained AlphaZero agent.")
    if az_agent.network:
        az_agent.network.eval()

    mcts_agent_name = f"MCTS_{config.mcts.num_simulations}"
    mcts_agent = MCTSAgent(
        env,
        num_simulations=config.mcts.num_simulations,
        exploration_constant=config.mcts.exploration_constant,
    )

    agents = {
        "AlphaZero": az_agent,
        mcts_agent_name: mcts_agent,
    }

    return agents


def get_benchmark_mcts_agent(env: BaseEnvironment, config: AppConfig) -> MCTSAgent:
    """Creates the benchmark MCTS agent for evaluation."""
    benchmark_sims = config.evaluation.benchmark_mcts_simulations
    logger.info(f"Creating benchmark MCTSAgent with {benchmark_sims} simulations.")
    return MCTSAgent(
        env,
        num_simulations=benchmark_sims,
        exploration_constant=config.mcts.exploration_constant, # Use same exploration constant for now
    )


# TODO: Add factory for MuZero agent when implemented
