from typing import Dict
from loguru import logger

from agents.mcts_agent_old import MCTSAgent_Old

import torch.optim as optim
from agents.alphazero_agent import (
    AlphaZeroAgent,
    AlphaZeroExpansion,
    AlphaZeroEvaluation,
)
from core.agent_interface import Agent
from core.config import (
    AppConfig,
    EnvConfig,
)
from environments.base import BaseEnvironment
from environments.connect4 import Connect4
from agents.mcts_agent import make_pure_mcts
from models.networks import AlphaZeroNet
from algorithms.mcts import UCB1Selection, StandardBackpropagation


def get_environment(env_config: EnvConfig) -> BaseEnvironment:
    """Factory function to create environment instances."""
    if env_config.name.lower() == "connect4":
        # Use connect4 specific dimensions from config
        return Connect4()
    else:
        raise ValueError(f"Unknown environment name: {env_config.name}")


def get_agents(env: BaseEnvironment, config: AppConfig) -> Dict[str, Agent]:
    """Factory function to create agent instances for the given environment."""
    az_config = config.alpha_zero
    training_config = config.training

    network = AlphaZeroNet(env=env)
    network.init_zero()
    optimizer = optim.AdamW(network.parameters(), lr=training_config.learning_rate)

    az_agent_name = f"AZ_{config.mcts.num_simulations}"
    az_agent = AlphaZeroAgent(
        selection_strategy=UCB1Selection(exploration_constant=az_config.cpuct),
        expansion_strategy=AlphaZeroExpansion(network=network),
        evaluation_strategy=AlphaZeroEvaluation(network=network),
        backpropagation_strategy=StandardBackpropagation(),
        network=network,
        optimizer=optimizer,
        env=env,
        config=az_config,
        training_config=training_config,
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
    mcts_agent = make_pure_mcts(
        num_simulations=config.mcts.num_simulations,
    )

    # mcts_agent_old_name = f"MCTS_old_{config.mcts.num_simulations}"
    # mcts_agent_old = MCTSAgent_Old(
    #     env=env,
    #     num_simulations=config.mcts.num_simulations,
    #     exploration_constant=config.mcts.exploration_constant,
    # )

    agents = {
        az_agent_name: az_agent,
        mcts_agent_name: mcts_agent,
        # mcts_agent_old_name: mcts_agent_old,
    }

    return agents


# TODO: Add factory for MuZero agent when implemented
