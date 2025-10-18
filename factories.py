from typing import Dict, Union

import torch.optim as optim

from agents.alphazero.alphazero_agent import (
    AlphaZeroAgent,
    AlphaZeroExpansion,
    AlphaZeroEvaluation,
    make_pure_az,
)
from agents.alphazero.alphazero_net import AlphaZeroNet
from agents.muzero.muzero_agent import MuZeroAgent, make_pure_muzero
from core.agent_interface import Agent
from core.config import (
    AppConfig,
    EnvConfig,
)
from environments.base import BaseEnvironment
from environments.connect4 import Connect4
from agents.mcts_agent import make_pure_mcts
from algorithms.mcts import UCB1Selection, StandardBackpropagation


def _create_az_agent(env: BaseEnvironment, config: AppConfig) -> AlphaZeroAgent:
    az_config = config.alphazero
    training_config = config.training
    network = AlphaZeroNet(env=env)
    network.init_zero()
    optimizer = optim.AdamW(network.parameters(), lr=training_config.learning_rate)

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
    return az_agent


def _create_mz_agent(env: BaseEnvironment, config: AppConfig) -> MuZeroAgent:
    mz_config = config.muzero
    training_config = config.training
    mz_agent = make_pure_muzero(
        env=env,
        config=mz_config,
        training_config=training_config,
    )
    return mz_agent


def get_environment(env_config: EnvConfig) -> BaseEnvironment:
    """Factory function to create environment instances."""
    if env_config.name.lower() == "connect4":
        # Use connect4 specific dimensions from config
        return Connect4()
    else:
        raise ValueError(f"Unknown environment name: {env_config.name}")


def get_agents(env: BaseEnvironment, config: AppConfig) -> Dict[str, Agent]:
    """Factory function to create agent instances for the given environment."""
    agents = {}

    # AlphaZero agent
    az_agent_name = f"AZ_{config.mcts.num_simulations}"
    # az_agent = _create_az_agent(env, config)
    # _load_and_prepare_agent(az_agent, "AlphaZero")
    az_agent = _create_learning_agent(
        model_type="alphazero",
        env=env,
        config=config,
    )
    agents[az_agent_name] = az_agent

    # # MuZero agent
    # mz_agent_name = f"MZ_{config.mcts.num_simulations}"
    # mz_agent = _create_mz_agent(env, config)
    # _load_and_prepare_agent(mz_agent, "MuZero")
    # agents[mz_agent_name] = mz_agent

    # MCTS agent
    mcts_agent_name = f"MCTS_{config.mcts.num_simulations}"
    mcts_agent = make_pure_mcts(
        num_simulations=config.mcts.num_simulations,
    )
    agents[mcts_agent_name] = mcts_agent

    return agents


def _create_learning_agent(
    model_type: str,
    env: BaseEnvironment,
    config: AppConfig,
) -> Union[AlphaZeroAgent, MuZeroAgent]:
    if model_type == "alphazero":
        agent = make_pure_az(
            env=env,
            config=config.alphazero,
            training_config=config.training,
        )
    elif model_type == "muzero":
        agent = make_pure_muzero(
            env=env,
            config=config.muzero,
            training_config=config.training,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return agent
