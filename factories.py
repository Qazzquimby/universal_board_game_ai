from typing import Dict, Optional
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
from algorithms.mcts import MCTSProfiler

# Import MuZeroAgent when ready
# from agents.muzero_agent import MuZeroAgent


def get_environment(env_config: EnvConfig) -> BaseEnvironment:
    """Factory function to create environment instances."""
    if env_config.name.lower() == "connect4":
        # Use connect4 specific dimensions from config
        logger.info(
            f"Using connect4 environment ({env_config.width}x{env_config.height})"
        )
        return Connect4(
            width=env_config.width,
            height=env_config.height,
            num_players=env_config.num_players,
            max_steps=env_config.max_steps,
        )
    elif env_config.name.lower() == "nim":
        logger.info(f"Using Nim environment with piles: {env_config.nim_piles}")
        return NimEnv(
            initial_piles=env_config.nim_piles, num_players=env_config.num_players
        )
    else:
        raise ValueError(f"Unknown environment name: {env_config.name}")


def get_agents(env: BaseEnvironment, config: AppConfig) -> Dict[str, Agent]:
    """Factory function to create agent instances for the given environment."""

    profiler: Optional[MCTSProfiler] = None  # Initialize profiler variable

    # Decide whether to enable profiling based on the config flag
    if config.training.enable_mcts_profiling:
        logger.info("MCTS Profiling enabled.")
        profiler = MCTSProfiler()  # Create the profiler instance

    # --- AlphaZero Agent Initialization ---
    # Instantiate AlphaZero agent and attempt to load weights.
    # Training should happen separately via train_alphazero.py
    # Pass the profiler instance to the agent's constructor
    az_agent = AlphaZeroAgent(
        env,
        config.alpha_zero,
        config.training,
        profiler=profiler,  # Pass the created profiler here (can be None)
    )
    if not az_agent.load():
        logger.warning(
            "Could not load pre-trained AlphaZero weights. Agent will play randomly/poorly."
        )
    else:
        logger.info("Loaded pre-trained AlphaZero agent.")
    if az_agent.network:
        az_agent.network.eval()

    # --- Benchmark MCTS Agent ---
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

    # --- MuZero Agent (Example) ---
    # if config.training.agent_type == "muzero" or ... :
    #     mz_agent = MuZeroAgent(
    #         env,
    #         config.muzero,
    #         profiler=profiler # Pass the same profiler if needed
    #     )
    #     agents["muzero"] = mz_agent

    return agents
