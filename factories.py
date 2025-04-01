from typing import Dict

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

# Import MuZeroAgent when ready
# from agents.muzero_agent import MuZeroAgent


def get_environment(env_config: EnvConfig) -> BaseEnvironment:
    """Factory function to create environment instances."""
    if env_config.name.lower() == "connect4":
        # Use connect4 specific dimensions from config
        print(f"Using connect4 environment ({env_config.width}x{env_config.height})")
        return Connect4(
            width=env_config.width,
            height=env_config.height,
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

    # QLearning and Random agents removed from evaluation setup

    # --- AlphaZero Agent Initialization ---
    # Instantiate AlphaZero agent and attempt to load weights.
    # Training should happen separately via train_alphazero.py
    az_agent = AlphaZeroAgent(env, config.alpha_zero)
    if not az_agent.load():
        print(
            "WARNING: Could not load pre-trained AlphaZero weights. Agent will play randomly/poorly."
        )
    else:
        print("Loaded pre-trained AlphaZero agent.")
    az_agent.network.eval()  # Ensure network is in eval mode for evaluation

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

    return agents
