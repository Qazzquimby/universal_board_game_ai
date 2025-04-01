# Standard library imports
import sys
import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Third-party imports
# import torch.optim as optim # Optimizer managed internally by agent
import numpy as np
from tqdm import tqdm

# Local application imports
from core.config import AppConfig
from environments.base import BaseEnvironment, StateType, ActionType
from agents.muzero_agent import MuZeroAgent  # Import MuZeroAgent
from factories import get_environment


# LOG_DIR = DATA_DIR / "muzero_game_logs" # Define if trajectory logging is added later


# --- Helper Function for Self-Play ---


def run_self_play_game(env: BaseEnvironment, agent: MuZeroAgent) -> Tuple[float, int]:
    """
    Plays one game of self-play using the MuZero agent.
    The agent collects trajectory data internally via agent.observe().

    Args:
        env: The environment instance.
        agent: The MuZero agent instance.

    Returns:
        A tuple containing:
        - final_outcome: The outcome for player 0 (+1 win, -1 loss, 0 draw).
        - num_steps: The number of steps taken in the game.
    """
    obs = env.reset()
    agent.reset()  # Reset agent state (e.g., MCTS tree, trajectory buffer)
    done = False
    game_steps = 0
    cumulative_reward = 0.0  # Track reward if needed for logging

    while not done:
        current_player = env.get_current_player()
        state = obs  # MuZero uses the observation dict

        # Use agent.act with train=True to get action and policy target
        action, policy_target = agent.act(state, train=True)

        if (
            action is None
        ):  # Policy target might also be None or invalid if action is None
            print(f"Warning: Agent returned None action in self-play. Ending game.")
            # Assign outcome based on rules (e.g., loss for player who can't move)
            final_outcome = 0.0  # Default to draw for unexpected None
            # Observe the final state before breaking? MuZero might need this.
            # Let's assume the game ends here, no final observation step needed yet.
            break

        try:
            next_obs, reward, done = env.step(action)
            game_steps += 1
            cumulative_reward += reward  # Optional tracking

            # --- MuZero Specific: Observe the transition ---
            # Store the observation *before* the action, the action taken,
            # the reward received *after* the action, the done flag,
            # and the policy target corresponding to the observation.
            agent.observe(obs, action, reward, done, policy_target)
            # ---------------------------------------------

            obs = next_obs  # Update observation for the next loop

        except ValueError as e:
            print(
                f"Warning: Invalid action {action} during self-play. Error: {e}. Ending game."
            )
            # Penalize the player who made the invalid move
            final_outcome = -1.0 if current_player == 0 else 1.0
            # Should we observe this terminal state? Maybe not if it's an error state.
            done = True  # Ensure loop terminates

    # Determine final outcome after the loop finishes
    if "final_outcome" not in locals():  # If loop finished normally
        winner = env.get_winning_player()
        if winner == 0:
            final_outcome = 1.0
        elif winner == 1:
            final_outcome = -1.0
        else:  # Draw
            final_outcome = 0.0

    # Tell the agent the episode is finished so it can store the trajectory
    agent.finish_episode()

    # Return outcome for win rate tracking, and steps
    return final_outcome, game_steps


# --- Game Log Saving/Loading (Removed for MuZero initial setup) ---
# MuZero typically saves trajectories (obs, action, reward), not (state, policy, value)
# We can add trajectory saving later if needed.


# --- Main Training Function ---


def run_training(config: AppConfig, env_name_override: str = None):
    """Runs the MuZero training process (Self-Play only for now)."""

    # --- Environment Selection ---
    if env_name_override:
        config.env.name = env_name_override

    # --- Instantiation ---
    env = get_environment(config.env)
    # Use MuZeroAgent and MuZeroConfig
    agent = MuZeroAgent(env, config.muzero)

    # --- Training Setup ---
    # Try loading existing weights for the MuZero network
    if not agent.load():
        print("No pre-trained MuZero weights found. Starting training from scratch.")
    else:
        print("Loaded existing MuZero weights. Continuing training.")

    # Optimizer is managed internally by the agent

    # --- Load Existing Game Logs (Removed - Needs MuZero format) ---
    # load_game_logs_into_buffer(...) # Needs adaptation for trajectories

    # --- Training Loop ---
    # Use values from config (consider adding MuZero specific iteration counts)
    num_training_iterations = config.training.num_iterations
    num_episodes_per_iteration = config.training.num_episodes_per_iteration

    print(
        f"Starting MuZero training (Self-Play Phase) for {num_training_iterations} iterations..."
    )
    print(f"({num_episodes_per_iteration} self-play games per iteration)")

    game_outcomes = []  # Track outcomes for win rate stats

    # Disable tqdm if running smoke test
    use_tqdm = not config.smoke_test
    outer_loop_iterator = range(num_training_iterations)
    inner_loop_iterator = (
        tqdm(range(num_episodes_per_iteration), desc="Self-Play")
        if use_tqdm
        else range(num_episodes_per_iteration)
    )

    for iteration in outer_loop_iterator:
        print(f"\n--- Iteration {iteration + 1}/{num_training_iterations} ---")

        # 1. Self-Play Phase
        agent.network.eval()  # Ensure network is in eval mode for self-play actions
        print("Running self-play games...")
        current_iteration_games = 0
        for game_idx in inner_loop_iterator:
            # Run self-play game, collect trajectory via agent.observe internally
            outcome, steps = run_self_play_game(env, agent)
            game_outcomes.append(outcome)
            current_iteration_games += 1

            # --- Game Log Saving (Removed - Needs MuZero format) ---
            # save_game_log(...)

        print(
            f"Completed {current_iteration_games} self-play games for iteration {iteration + 1}."
        )

        # 2. Learning Phase
        print("Running learning step...")
        agent.learn()  # Call the agent's learning method

        # 3. Save Checkpoint Periodically
        # TODO: Make save frequency configurable in MuZeroConfig?
        if (iteration + 1) % 10 == 0:  # Save every 10 iterations
            print("Saving agent checkpoint...")
            agent.save()  # Uses MuZero agent's save method

        # Print progress (e.g., buffer size, recent win rate)
        # MuZero buffer stores trajectories, len() gives number of games/trajectories
        buffer_size = len(agent.replay_buffer)
        window_size = min(len(game_outcomes), 100)  # Look at last 100 games
        if window_size > 0:
            win_rate = game_outcomes[-window_size:].count(1) / window_size
            loss_rate = game_outcomes[-window_size:].count(-1) / window_size
            draw_rate = game_outcomes[-window_size:].count(0) / window_size
            print(
                f"Iteration {iteration + 1} complete. Buffer size: {buffer_size} trajectories (Max: {config.muzero.replay_buffer_size})"
            )
            print(
                f"  Recent Performance (last {window_size} games): Wins={win_rate:.2f}, Losses={loss_rate:.2f}, Draws={draw_rate:.2f}"
            )

    # --- Final Save & Plot ---
    print("\nTraining complete. Saving final agent state.")
    agent.save()

    print("Plotting training results (win rate)...")
    plot_results(game_outcomes, window_size=config.training.plot_window)

    print("\n--- MuZero Training (Self-Play Phase) Finished ---")


# --- Script Entry Point ---

if __name__ == "__main__":
    # --- Configuration ---
    # Load the main AppConfig, which includes MuZeroConfig
    config = AppConfig()
    env_override = None

    # --- Environment Selection (Optional: Add CLI arg parsing) ---
    if len(sys.argv) > 1:
        env_override = sys.argv[1]  # e.g., python train_muzero.py Nim

    run_training(config, env_name_override=env_override)
