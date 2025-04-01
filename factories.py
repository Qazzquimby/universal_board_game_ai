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
from agents.qlearning import (
    QLearningAgent,
    train_agent as train_q_agent,  # Keep Q-learning training here for now
)
from agents.random_agent import RandomAgent
from agents.alphazero_agent import AlphaZeroAgent  # Import AlphaZeroAgent
from utils.plotting import plot_results


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

    # --- QLearning Agent Initialization & Training ---
    ql_agent = QLearningAgent(env, config.q_learning)
    if not ql_agent.load():  # Load uses internal path logic now
        print(
            f"Training Q-learning agent for {config.training.num_episodes} episodes..."
        )

        # Reset exploration for training
        ql_agent.exploration_rate = config.q_learning.exploration_rate
        wins = train_q_agent(
            env,
            ql_agent,
            num_episodes=config.training.num_episodes,
            q_config=config.q_learning,
        )
        plot_results(wins, window_size=config.training.plot_window)
        ql_agent.save()  # Save uses internal path logic now

        # Set low exploration after training/saving
        ql_agent.exploration_rate = config.q_learning.min_exploration
    else:
        print(f"Loaded pre-trained Q-learning agent.")

    # Ensure Q-agent used for testing has exploration turned off or minimized
    ql_agent.exploration_rate = config.q_learning.min_exploration

    # --- AlphaZero Agent Initialization ---
    # Instantiate AlphaZero agent and attempt to load weights.
    # Training should happen separately via train_alphazero.py
    az_agent = AlphaZeroAgent(env, config.alpha_zero)
    if not az_agent.load():
        print(
            "WARNING: Could not load pre-trained AlphaZero weights. Agent will play randomly/poorly."
        )
        # Do NOT train here. Evaluation assumes agent is already trained.
    else:
        print("Loaded pre-trained AlphaZero agent.")
    az_agent.network.eval()  # Ensure network is in eval mode for evaluation

    # --- Other Agents ---
    agents = {
        "QLearning": ql_agent,
        "AlphaZero": az_agent,  # Add the AlphaZero agent
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

    # If running smoke test, remove the slower MCTS agent
    if config.smoke_test and "MCTS_200" in agents:
        print("Smoke test mode: Removing MCTS_200 agent.")
        del agents["MCTS_200"]
    return agents
