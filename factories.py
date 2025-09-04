import re
from pathlib import Path
from typing import Dict
from loguru import logger

import torch.optim as optim
from agents.alphazero.alphazero_agent import (
    AlphaZeroAgent,
    AlphaZeroExpansion,
    AlphaZeroEvaluation,
)
from agents.alphazero.alphazero_net import AlphaZeroNet
from core.agent_interface import Agent
from core.config import (
    AppConfig,
    EnvConfig,
)
from environments.base import BaseEnvironment
from environments.connect4 import Connect4
from agents.mcts_agent import make_pure_mcts
from algorithms.mcts import UCB1Selection, StandardBackpropagation


def get_environment(env_config: EnvConfig) -> BaseEnvironment:
    """Factory function to create environment instances."""
    if env_config.name.lower() == "connect4":
        # Use connect4 specific dimensions from config
        return Connect4()
    else:
        raise ValueError(f"Unknown environment name: {env_config.name}")


def get_agents(
    env: BaseEnvironment, config: AppConfig, load_all_az_iterations: bool = False
) -> Dict[str, Agent]:
    """Factory function to create agent instances for the given environment."""
    az_config = config.alphazero
    training_config = config.training

    agents = {}
    if load_all_az_iterations:
        data_dir = Path("data")

        def get_iter_from_path(p):
            m = re.search(r"iter_(\d+)\.pth", str(p))
            return int(m.group(1)) if m else -1

        files = sorted(
            list(data_dir.glob("alphazero_net_connect4_iter_*.pth")),
            key=get_iter_from_path,
        )

        for pth_file in files:
            iteration = get_iter_from_path(pth_file)
            network = AlphaZeroNet(env=env)
            network.init_zero()
            optimizer = optim.AdamW(
                network.parameters(), lr=training_config.learning_rate
            )
            az_agent_name = f"AZ_{config.mcts.num_simulations}_iter_{iteration}"
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
            if not az_agent.load(filepath=pth_file):
                logger.warning(
                    f"Could not load pre-trained AlphaZero weights from {pth_file}."
                )
            else:
                logger.info(f"Loaded pre-trained AlphaZero agent from {pth_file}.")
            if az_agent.network:
                az_agent.network.eval()
            agents[az_agent_name] = az_agent
    else:
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
        agents[az_agent_name] = az_agent

    mcts_agent_name = f"MCTS_{config.mcts.num_simulations}"
    mcts_agent = make_pure_mcts(
        num_simulations=config.mcts.num_simulations,
    )
    agents[mcts_agent_name] = mcts_agent

    # agents["Random"] = RandomAgent(env=env)

    # mcts_agent_old_name = f"MCTS_old_{config.mcts.num_simulations}"
    # mcts_agent_old = MCTSAgent_Old(
    #     env=env,
    #     num_simulations=config.mcts.num_simulations,
    #     exploration_constant=config.mcts.exploration_constant,
    # )

    return agents


# TODO: Add factory for MuZero agent when implemented
