import sys
import json
import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch.optim as optim
import numpy as np
from tqdm import tqdm


from core.config import AppConfig, DATA_DIR
from environments.base import BaseEnvironment, StateType, ActionType
from agents.alphazero_agent import AlphaZeroAgent
from factories import get_environment
from utils.plotting import plot_losses  # Import the new plotting function

LOG_DIR = DATA_DIR / "game_logs"


def run_self_play_game(
    env: BaseEnvironment, agent: AlphaZeroAgent
) -> Tuple[float, int, List]:
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
        - game_history: The processed game history list from the agent.
    """
    obs = env.reset()
    agent.reset()
    done = False
    game_steps = 0

    while not done:
        current_player = env.get_current_player()
        state = obs

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

    # If loop finished normally
    if "final_outcome" not in locals():
        winner = env.get_winning_player()
        if winner == 0:
            final_outcome = 1.0
        elif winner == 1:
            final_outcome = -1.0
        else:  # Draw
            final_outcome = 0.0

    game_history = agent.finish_episode(final_outcome)

    return final_outcome, game_steps, game_history


def _default_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    # Add other custom types here if needed
    # Example: if isinstance(obj, SomeCustomClass): return obj.to_dict()
    try:
        return obj.__dict__  # Fallback for simple objects
    except AttributeError:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )


def save_game_log(
    game_history: List[Tuple[StateType, ActionType, np.ndarray, float]],
    iteration: int,
    game_index: int,
    env_name: str,
):
    """Saves the processed game history to a JSON file."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            f"{env_name}_iter{iteration:04d}_game{game_index:04d}_{timestamp}.json"
        )
        filepath = LOG_DIR / filename

        # Prepare data for JSON (convert numpy arrays)
        serializable_history = []
        for state, action, policy, value in game_history:
            # Ensure state is serializable (should be if it's dicts/lists/primitives)
            # Ensure action is serializable (should be if int/tuple)
            serializable_history.append(
                {
                    "state": state,
                    "action": action,
                    "policy_target": policy.tolist(),  # Convert numpy array
                    "value_target": value,
                }
            )

        with open(filepath, "w") as f:
            # Pass the custom serializer to handle numpy arrays within state dicts etc.
            json.dump(serializable_history, f, indent=2, default=_default_serializer)

    except Exception as e:
        print(f"Error saving game log for iter {iteration}, game {game_index}: {e}")


def load_game_logs_into_buffer(agent: AlphaZeroAgent, env_name: str, buffer_limit: int):
    """Loads existing game logs from LOG_DIR into the agent's replay buffer."""
    loaded_games = 0
    loaded_steps = 0
    if not LOG_DIR.exists():
        print("Log directory not found. Starting with an empty buffer.")
        return

    print(f"Scanning {LOG_DIR} for existing '{env_name}' game logs...")
    log_files = sorted(
        LOG_DIR.glob(f"{env_name}_iter*.json")
    )  # Sort for potential consistency

    if not log_files:
        print("No existing game logs found for this environment.")
        return

    for filepath in tqdm(log_files, desc="Loading Logs"):
        if len(agent.replay_buffer) >= buffer_limit:
            print(
                f"Replay buffer reached limit ({buffer_limit}). Stopping log loading."
            )
            break
        try:
            with open(filepath, "r") as f:
                game_data = json.load(f)

            if not isinstance(game_data, list):
                print(
                    f"Warning: Skipping invalid log file (not a list): {filepath.name}"
                )
                continue

            steps_in_game = 0
            for step_data in game_data:
                if len(agent.replay_buffer) >= buffer_limit:
                    break  # Stop adding steps if buffer full mid-game

                # Extract required components for replay buffer
                state = step_data.get("state")
                policy_target_list = step_data.get("policy_target")
                value_target = step_data.get("value_target")

                if (
                    state is not None
                    and policy_target_list is not None
                    and value_target is not None
                ):
                    # Convert policy back to numpy array
                    policy_target = np.array(policy_target_list, dtype=np.float32)

                    # --- Standardize state loaded from JSON ---
                    # Convert board/piles list back to numpy array
                    if 'board' in state and isinstance(state['board'], list):
                        state['board'] = np.array(state['board'], dtype=np.int8) # Match Connect4 dtype
                    elif 'piles' in state and isinstance(state['piles'], list):
                        state['piles'] = np.array(state['piles'], dtype=np.int32) # Match NimEnv dtype

                    agent.replay_buffer.append((state, policy_target, value_target))
                    loaded_steps += 1
                    steps_in_game += 1
                else:
                    print(
                        f"Warning: Skipping step with missing data in {filepath.name}"
                    )

            if steps_in_game > 0:
                loaded_games += 1

        except json.JSONDecodeError:
            print(f"Warning: Skipping corrupted JSON file: {filepath.name}")
        except Exception as e:
            print(f"Warning: Error processing log file {filepath.name}: {e}")

    print(
        f"Loaded {loaded_steps} steps from {loaded_games} game logs into replay buffer."
    )
    print(f"Current buffer size: {len(agent.replay_buffer)}/{buffer_limit}")


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

    # --- Load Existing Game Logs into Buffer ---
    load_game_logs_into_buffer(
        agent, config.env.name, config.alpha_zero.replay_buffer_size
    )

    # --- Training Loop ---
    num_training_iterations = config.training.num_iterations
    num_episodes_per_iteration = config.training.num_episodes_per_iteration

    print(f"Starting AlphaZero training for {num_training_iterations} iterations...")
    print(f"({num_episodes_per_iteration} self-play games per iteration)")

    # Lists to store losses for plotting
    total_losses = []
    value_losses = []
    policy_losses = []

    # Disable tqdm if running smoke test to potentially avoid encoding issues
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
        # Use enumerate to get game index for logging
        current_iteration_games = 0
        # No need to track game outcomes for plotting anymore
        for game_idx in inner_loop_iterator:
            _, _, history = run_self_play_game(env, agent)  # Discard outcome and steps
            # game_outcomes.append(outcome) # Removed
            current_iteration_games += 1

            # Save the game log after each game
            save_game_log(history, iteration + 1, game_idx + 1, config.env.name)

        print(
            f"Completed {current_iteration_games} self-play games for iteration {iteration + 1}."
        )

        # 2. Learning Phase
        print("Running learning step...")
        # agent.network.train() # Network mode is handled within agent.learn()
        # TODO: Add epochs per learning step if desired (call learn multiple times)
        losses = agent.learn()
        if losses:
            total_loss, value_loss, policy_loss = losses
            total_losses.append(total_loss)
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)

        # 3. Save Checkpoint Periodically
        # TODO: Make save frequency configurable
        if (iteration + 1) % 10 == 0:  # Save every 10 iterations
            print("Saving agent checkpoint...")
            agent.save()

        # Print buffer size and latest losses if available
        buffer_size = len(agent.replay_buffer)
        print(
            f"Iteration {iteration + 1} complete. Buffer size: {buffer_size}/{config.alpha_zero.replay_buffer_size}"
        )
        if total_losses:  # Check if any learning steps have occurred
            print(
                f"  Latest Losses: Total={total_losses[-1]:.4f}, Value={value_losses[-1]:.4f}, Policy={policy_losses[-1]:.4f}"
            )

        # 4. Run Sanity Checks Periodically
        if config.training.sanity_check_frequency > 0 and (iteration + 1) % config.training.sanity_check_frequency == 0:
            run_sanity_checks(env, agent) # Run checks on the current agent state

    print("\nTraining complete. Saving final agent state.")
    agent.save()

    print("Plotting training losses...")
    plot_losses(
        total_losses, value_losses, policy_losses
    )  # Call the new plotting function

    # Run sanity checks one last time on the final trained agent
    # This ensures checks run even if num_iterations isn't a multiple of frequency
    print("\n--- Running Final Sanity Checks ---")
    run_sanity_checks(env, agent)

    print("\n--- AlphaZero Training Finished ---")


def run_sanity_checks(env: BaseEnvironment, agent: AlphaZeroAgent):
    """Runs network predictions on predefined sanity check states."""
    print("\n--- Running Sanity Checks on Final Network ---")
    sanity_states = env.get_sanity_check_states()
    agent.network.eval() # Ensure network is in eval mode

    if not sanity_states:
        print("No sanity check states defined for this environment.")
        return

    for description, state in sanity_states:
        print(f"\nChecking State: {description}")
        # Print board/piles for context
        if 'board' in state:
            print("Board:")
            print(state['board'])
        elif 'piles' in state:
            print(f"Piles: {state['piles']}")
        print(f"Current Player: {state['current_player']}")

        try:
            # Get network predictions
            policy_np, value_np = agent.network.predict(state)
            print(f"  Predicted Value: {value_np:.4f}")

            # Get legal actions for this state to interpret policy
            temp_env = env.copy()
            temp_env.set_state(state)
            legal_actions = temp_env.get_legal_actions()

            action_probs = {}
            for action in legal_actions:
                idx = agent.network.get_action_index(action)
                if idx is not None and 0 <= idx < len(policy_np):
                    action_probs[action] = policy_np[idx]
                else:
                    action_probs[action] = -1 # Indicate mapping error

            # Sort actions by predicted probability
            sorted_probs = sorted(
                action_probs.items(), key=lambda item: item[1], reverse=True
            )

            print(f"  Top Predicted Legal Actions:")
            top_k = 5
            for action, prob in sorted_probs[:top_k]:
                if prob >= 0:
                    print(f"    - {action}: {prob:.4f}")
                else:
                    print(f"    - {action}: (Error mapping action)")
            if not legal_actions:
                print("    - (No legal actions)")

        except Exception as e:
            print(f"  Error during prediction for this state: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    config = AppConfig()
    env_override = None

    # --- Environment Selection (Optional: Add CLI arg parsing) ---
    if len(sys.argv) > 1:
        env_override = sys.argv[1]  # e.g., python train_alphazero.py Nim

    run_training(config, env_name_override=env_override)
