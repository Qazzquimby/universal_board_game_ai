# Standard library imports
import sys
from typing import Tuple

# Third-party imports
import torch.optim as optim
from tqdm import tqdm

# Local application imports
from core.config import AppConfig
from environments.base import BaseEnvironment
from agents.alphazero_agent import AlphaZeroAgent
from factories import get_environment
from utils.plotting import plot_results


# --- Helper Function for Self-Play ---


def run_self_play_game(
    env: BaseEnvironment, agent: AlphaZeroAgent
) -> Tuple[float, int]:
    """
    Plays one game of self-play using the AlphaZero agent.
    The agent collects training data internally via agent.act(train=True).

    Args:
        env: The environment instance.
        agent: The AlphaZero agent instance.

    Returns:
        A tuple containing:
        - final_outcome: The outcome for player 0 (+1 win, -1 loss, 0 draw).
        - num_steps: The number of steps taken in the game.
    """
    obs = env.reset()
    agent.reset()  # Reset agent state (e.g., MCTS tree) for the new game
    done = False
    game_steps = 0

    while not done:
        current_player = env.get_current_player()
        state = obs  # Use the full observation dict

        # Use agent.act with train=True to enable data collection and exploration
        action = agent.act(state, train=True)

        if action is None:
            print(f"Warning: Agent returned None action in self-play. Ending game.")
            # Assign outcome based on rules (e.g., loss for player who can't move)
            # For simplicity, let's call it a draw if this happens unexpectedly.
            final_outcome = 0.0
            break

        try:
            obs, _, done = env.step(action)
            game_steps += 1
        except ValueError as e:
            print(
                f"Warning: Invalid action {action} during self-play. Error: {e}. Ending game."
            )
            # Penalize the player who made the invalid move
            final_outcome = -1.0 if current_player == 0 else 1.0
            break

    # Determine final outcome after the loop finishes
    if "final_outcome" not in locals():  # If loop finished normally
        winner = env.get_winning_player()
        if winner == 0:
            final_outcome = 1.0
        elif winner == 1:
            final_outcome = -1.0
        else:  # Draw
            final_outcome = 0.0

    # Tell the agent the final outcome to store experiences in the buffer
    agent.finish_episode(final_outcome)

    return final_outcome, game_steps


# --- Main Training Function ---

def run_training(config: AppConfig, env_name_override: str = None):
    """Runs the AlphaZero training process."""

    # --- Environment Selection ---
    if env_name_override:
        config.env.name = env_name_override

    # --- Instantiation ---
    env = get_environment(config.env)
    agent = AlphaZeroAgent(env, config.alpha_zero)

    # --- Training Setup ---
    # Try loading existing weights
    if not agent.load():
        print("No pre-trained weights found. Starting training from scratch.")
    else:
        print("Loaded existing weights. Continuing training.")

    # Optimizer is managed internally by the agent

    # --- Training Loop ---
    # Use values from config
    num_training_iterations = config.training.num_iterations
    num_episodes_per_iteration = config.training.num_episodes_per_iteration

    print(f"Starting AlphaZero training for {num_training_iterations} iterations...")
    print(f"({num_episodes_per_iteration} self-play games per iteration)")

    game_outcomes = []

    # Disable tqdm if running smoke test to potentially avoid encoding issues
    use_tqdm = not config.smoke_test
    outer_loop_iterator = range(num_training_iterations)
    inner_loop_iterator = tqdm(range(num_episodes_per_iteration), desc="Self-Play") if use_tqdm else range(num_episodes_per_iteration)


    for iteration in outer_loop_iterator:
        print(f"\n--- Iteration {iteration + 1}/{num_training_iterations} ---")

        # 1. Self-Play Phase
        agent.network.eval()
        print("Running self-play games...")
        for _ in inner_loop_iterator:
            outcome, steps = run_self_play_game(env, agent)
            game_outcomes.append(outcome)

        # 2. Learning Phase
        print("Running learning step...")
        # agent.network.train() # Network mode is handled within agent.learn()
        # TODO: Add epochs per learning step if desired (call learn multiple times)
        agent.learn() # Call the agent's learning method (optimizer is internal)

        # 3. Save Checkpoint Periodically
        # TODO: Make save frequency configurable
        if (iteration + 1) % 10 == 0:  # Save every 10 iterations
            print("Saving agent checkpoint...")
            agent.save()

        # Print progress (e.g., buffer size, recent win rate)
        buffer_size = len(agent.replay_buffer)
        window_size = min(len(game_outcomes), 100)  # Look at last 100 games
        if window_size > 0:
            win_rate = game_outcomes[-window_size:].count(1) / window_size
            loss_rate = game_outcomes[-window_size:].count(-1) / window_size
            draw_rate = game_outcomes[-window_size:].count(0) / window_size
            print(
                f"Iteration {iteration + 1} complete. Buffer size: {buffer_size}/{config.alpha_zero.replay_buffer_size}"
            )
            print(
                f"  Recent Performance (last {window_size} games): Wins={win_rate:.2f}, Losses={loss_rate:.2f}, Draws={draw_rate:.2f}"
            )

    # --- Final Save & Plot ---
    print("\nTraining complete. Saving final agent state.")
    agent.save()

    print("Plotting training results...")
    plot_results(game_outcomes, window_size=config.training.plot_window)

    print("\n--- AlphaZero Training Finished ---")


# --- Script Entry Point ---

if __name__ == "__main__":
    # --- Configuration ---
    config = AppConfig()
    env_override = None

    # --- Environment Selection (Optional: Add CLI arg parsing) ---
    if len(sys.argv) > 1:
        env_override = sys.argv[1] # e.g., python train_alphazero.py Nim

    run_training(config, env_name_override=env_override)
