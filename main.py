import sys # For basic CLI args or environment selection
import numpy as np
from typing import Optional, Dict, Type

# Core imports
from core.config import MainConfig
from core.env_interface import EnvInterface
from core.agent_interface import Agent

# Environment imports - Choose the environment to use
from environments.four_in_a_row import FourInARow
from environments.nim_env import NimEnv

# Agent imports
from mcts import MCTSAgent
from qlearning import QLearningAgent
from random_agent import RandomAgent

# Evaluation import
from evaluation import plot_results, run_evaluation


def train_q_agent(env: EnvInterface, agent: QLearningAgent, num_episodes=1000, opponent: Optional[Agent]=None):
    """Train agent with proper turn handling and sparse rewards"""
    opponent = opponent or RandomAgent(env)
    win_history = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_history = []

        while not done:
            current_player = env.get_current_player()

            if current_player == 0:  # Agent being trained's turn
                state = obs
                action = agent.act(state)
                if action is None:
                    raise RuntimeError(f"Agent {type(agent).__name__} returned None action in state: {state}")
                next_obs, reward, done = env.step(action)
                # Store the state *before* the action, the action, the reward *received for that action*, and done status
                episode_history.append((state, action, reward, done))
                obs = next_obs
            else:  # Opponent's turn
                state = obs # Use current observation for opponent
                action = opponent.act(state)
                if action is None:
                     raise RuntimeError(f"Opponent {type(opponent).__name__} returned None action in state: {state}")
                # We don't store opponent moves in the agent's history
                obs, _, done = env.step(action)

        # After episode ends, determine final reward
        if env.get_winning_player() == 0:
            final_reward = 1.0
            outcome = 1
        elif env.get_winning_player() is not None:
            final_reward = -1.0
            outcome = -1
        else:
            final_reward = 0.0
            outcome = 0

        # Update Q-values with final reward
        if episode_history:
            # Replace all rewards with final outcome
            episode_history = [
                (s, a, final_reward, d) for s, a, _, d in episode_history
            ]
            agent.learn(episode_history)

        # Track outcomes and decay exploration
        win_history.append(outcome)
        agent.exploration_rate = max(
            agent.exploration_rate * agent.exploration_decay, agent.min_exploration
        )

        # Print progress occasionally
        window_size = 200
        if (episode + 1) % window_size == 0:
            win_rate = win_history[-window_size:].count(1) / window_size
            draw_rate = win_history[-window_size:].count(0) / window_size
            loss_rate = win_history[-window_size:].count(-1) / window_size
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Win Rate: {win_rate:.2f} | "
                f"Draw Rate: {draw_rate:.2f} | "
                f"Loss Rate: {loss_rate:.2f} | "
                f"Exploration: {agent.exploration_rate:.4f}"
            )



def get_environment(env_name: str, config: MainConfig) -> EnvInterface:
    """Factory function to create environment instances."""
    if env_name.lower() == "fourinarow":
        print(f"Using FourInARow environment ({config.board_size}x{config.board_size})")
        return FourInARow(board_size=config.board_size, num_players=config.num_players, max_steps=config.env_max_steps)
    elif env_name.lower() == "nim":
        # Example Nim config - could be added to MainConfig or passed differently
        nim_piles = [3, 5, 7]
        print(f"Using Nim environment with piles: {nim_piles}")
        return NimEnv(initial_piles=nim_piles, num_players=config.num_players)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

def get_agents(env: EnvInterface, config: MainConfig) -> Dict[str, Agent]:
    """Factory function to create agent instances for the given environment."""
    # Adjust Q-learning save file based on environment type
    env_type_name = type(env).__name__
    ql_save_file = f"q_agent_{env_type_name}.pkl"
    print(f"Q-Learning save file: {ql_save_file}")

    # --- Agent Initialization ---
    ql_agent = QLearningAgent(env, exploration_rate=0.0) # Start with low exploration for loaded agent
    if not ql_agent.load(ql_save_file):
        print(f"Training Q-learning agent for {config.num_episodes_train} episodes...")
        ql_agent.exploration_rate = 1.0 # Reset exploration for training
        wins = train_q_agent(env, ql_agent, num_episodes=config.num_episodes_train)
        plot_results(wins, window_size=config.plot_window)
        ql_agent.save(ql_save_file)
        ql_agent.exploration_rate = ql_agent.min_exploration # Set low exploration after training/saving
    else:
        print(f"Loaded pre-trained Q-learning agent from {ql_save_file}.")

    # Ensure agents used for testing have exploration turned off or minimized.
    ql_agent.exploration_rate = ql_agent.min_exploration

    agents = {
        "QLearning": ql_agent,
        "MCTS_50": MCTSAgent(env, num_simulations=config.mcts_simulations_short),
        "MCTS_200": MCTSAgent(env, num_simulations=config.mcts_simulations_long),
        "Random": RandomAgent(env),
    }
    return agents


if __name__ == "__main__":
    config = MainConfig()

    # env_name = "FourInARow"
    env_name = "Nim"
    if len(sys.argv) > 1:
        env_name = sys.argv[1] # e.g., python main.py Nim
    env = get_environment(env_name, config)

    agents = get_agents(env, config)

    run_evaluation(env, agents, config)
