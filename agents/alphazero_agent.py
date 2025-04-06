import random
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from loguru import logger

from core.agent_interface import Agent
from environments.base import BaseEnvironment, StateType, ActionType
from algorithms.mcts import AlphaZeroMCTS, MCTSProfiler
from models.networks import AlphaZeroNet
from core.config import AlphaZeroConfig, DATA_DIR, TrainingConfig


# --- Define a simple Dataset wrapper for the replay buffer ---
class ReplayBufferDataset(Dataset):
    def __init__(self, buffer: deque, network: AlphaZeroNet):
        # Convert deque to list for easier indexing by DataLoader
        self.buffer_list = list(buffer)
        self.network = network  # Need network for flattening

    def __len__(self):
        return len(self.buffer_list)

    def __getitem__(self, idx):
        state_dict, policy_target, value_target = self.buffer_list[idx]
        # Flatten state here before returning
        flat_state = self.network._flatten_state(state_dict)
        # Return flattened state, policy target, value target as tensors
        return (
            flat_state,
            torch.tensor(policy_target, dtype=torch.float32),
            torch.tensor(
                [value_target], dtype=torch.float32
            ),  # Ensure value is shape [1]
        )


@dataclass
class EpisodeResult:
    """Holds the results of a finished self-play episode."""

    buffer_experiences: List[Tuple[StateType, np.ndarray, float]]
    logged_history: List[Tuple[StateType, ActionType, np.ndarray, float]]


class AlphaZeroAgent(Agent):
    """Agent implementing the AlphaZero algorithm."""

    def __init__(
        self,
        env: BaseEnvironment,
        config: AlphaZeroConfig,
        training_config: TrainingConfig,
        profiler: Optional[MCTSProfiler] = None,  # Add profiler argument
    ):
        """
        Initialize the AlphaZero agent.

        Args:
            env: The environment instance.
            config: Configuration object with AlphaZero parameters.
            training_config: Configuration object with general training parameters.
            profiler: An optional MCTSProfiler instance for collecting timing data.
        """
        self.env = env
        self.config = config
        self.training_config = training_config  # Store training config
        self.profiler = profiler  # Store the profiler instance

        if self.config.should_use_network:
            self.network = AlphaZeroNet(
                env,
                hidden_layer_size=config.hidden_layer_size,
                num_hidden_layers=config.num_hidden_layers,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.network = None
            self.device = torch.device("cpu")

        # Pass the stored profiler instance to the MCTS constructor
        self.mcts = AlphaZeroMCTS(
            exploration_constant=config.cpuct,
            num_simulations=config.num_simulations,
            network=self.network,
            profiler=self.profiler,
            # inference_batch_size is not needed by MCTS itself anymore
            # TODO: Add dirichlet noise parameters from config if/when implemented
        )

        # Learning:

        # Experience buffer for training (stores (state, policy_target, value))
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        self._current_episode_history = []

        # Optimizer (managed internally by the agent)
        if self.config.should_use_network:
            self.optimizer = optim.AdamW(
                self.network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = None
        # # Learning Rate Scheduler
        # self.scheduler = StepLR(
        #     self.optimizer,
        #     step_size=config.lr_scheduler_step_size,
        #     gamma=config.lr_scheduler_gamma,
        # )

    def act(self, state: StateType, train: bool = False) -> ActionType:
        """
        Choose an action using MCTS guided by the neural network.
        If train=True, also stores the state and MCTS policy target for learning.

        Args:
            state: The current environment state observation.
            train: If True, use temperature sampling for exploration during training.
                   If False, choose the most visited node (greedy).

        Returns:
            The chosen action.
        """
        # Ensure network is in evaluation mode for inference
        if self.network:
            self.network.eval()

        # --- Optional Debug Print: Network Prediction ---
        if self.config.debug_mode and self.network:
            try:
                policy_np, value_np = self.network.predict(state)
                logger.debug(
                    f"[DEBUG Act] State: {state.get('board', state.get('piles', 'N/A'))}"
                )
                logger.debug(f"[DEBUG Act] To Play: {state['current_player']}")
                logger.debug(f"[DEBUG Act] Network Value Prediction: {value_np:.4f}")
                legal_actions = self.env.get_legal_actions()
                action_probs = {}
                for action in legal_actions:
                    idx = self.network.get_action_index(action)
                    if idx is not None and 0 <= idx < len(policy_np):
                        action_probs[action] = policy_np[idx]
                sorted_probs = sorted(
                    action_probs.items(), key=lambda item: item[1], reverse=True
                )
                top_k = 5
                logger.debug(f"[DEBUG Act] Network Policy Priors (Top {top_k} Legal):")
                for action, prob in sorted_probs[:top_k]:
                    logger.debug(f"  - {action}: {prob:.4f}")
            except Exception as e:
                logger.error(f"[DEBUG Act] Error during network predict debug: {e}")

        # MCTS search now uses the profiler internally if it exists
        root_node = self.mcts.search(self.env, state)

        if not root_node.children:
            logger.warning(
                "MCTS root has no children after search. Choosing random action."
            )
            temp_env = self.env.copy()
            temp_env.set_state(state)
            legal_actions = temp_env.get_legal_actions()
            return random.choice(legal_actions) if legal_actions else None

        # Select action based on visit counts
        visit_counts = np.array(
            [child.visit_count for child in root_node.children.values()]
        )
        actions = list(root_node.children.keys())

        if train:
            current_step = state.get("step_count", 0)
            temperature = (
                1.0 if current_step < self.config.temperature_decay_steps else 0.1
            )  # Example decay

            if temperature > 1e-6:  # Avoid division by zero / issues with temp=0
                visit_counts_temp = visit_counts ** (1.0 / temperature)
                sum_visits_temp = np.sum(visit_counts_temp)
                if sum_visits_temp > 0:
                    action_probs = visit_counts_temp / sum_visits_temp
                    # Ensure probabilities sum to 1 due to potential floating point issues
                    action_probs /= action_probs.sum()
                    try:
                        chosen_action_index = np.random.choice(
                            len(actions), p=action_probs
                        )
                    except ValueError as e:
                        logger.warning(
                            f"Error during np.random.choice (probs might not sum to 1?): {e}"
                        )
                        logger.warning(
                            f"Action probs: {action_probs}, Sum: {np.sum(action_probs)}"
                        )
                        chosen_action_index = np.argmax(
                            visit_counts
                        )  # Fallback to greedy
                else:
                    # Fallback if sum is zero (e.g., only one action, one visit?) - choose greedily
                    logger.warning(
                        "Sum of visit counts with temperature was zero. Choosing greedily."
                    )
                    chosen_action_index = np.argmax(visit_counts)
            else:
                # If temperature is effectively zero, choose greedily
                chosen_action_index = np.argmax(visit_counts)
        else:
            # Greedy selection for evaluation/play
            chosen_action_index = np.argmax(visit_counts)

        chosen_action = actions[chosen_action_index]

        # --- Optional Debug Print: MCTS Results ---
        if self.config.debug_mode:
            logger.debug(f"[DEBUG Act] MCTS Visit Counts:")
            sorted_visits = sorted(
                zip(actions, visit_counts), key=lambda item: item[1], reverse=True
            )
            for action, visits in sorted_visits:
                logger.debug(f"  - {action}: {visits}")
            logger.debug(f"[DEBUG Act] Chosen Action (Train={train}): {chosen_action}")

        policy_target = None
        # --- Store data for training if in training mode ---
        if train:
            # Calculate policy target based on visit counts
            policy_target = self._calculate_policy_target(
                root_node, actions, visit_counts
            )
            # NOTE: History is no longer stored internally in the agent for parallel execution.
            # The caller (training loop) is responsible for storing state, action, policy_target.

        # Return action, and policy_target if training
        if train:
            # Ensure policy_target is not None before returning
            if policy_target is None:
                raise ValueError(
                    "Policy target is None during training act(). This should not happen."
                )
                # Handle error case: maybe return a default policy or raise exception?
                # For now, let's create a zero policy as a fallback, though this indicates a deeper issue.
                # policy_size = self.network._calculate_policy_size(self.env) if self.network else self.env.policy_vector_size
                # policy_target = np.zeros(policy_size, dtype=np.float32)
            return chosen_action, policy_target
        else:
            return chosen_action

    def _calculate_policy_target(self, root_node, actions, visit_counts) -> np.ndarray:
        """Calculates the policy target vector based on MCTS visit counts."""
        if self.network:
            policy_size = self.network._calculate_policy_size(self.env)
        else:
            # Estimate policy size based on max possible actions if no network
            # This is less ideal, but needed for the fallback case
            # A better approach might be to require the network or pass policy size explicitly
            try:
                policy_size = self.env.policy_vector_size
            except AttributeError:
                logger.error(
                    "Cannot determine policy size without network or env.policy_vector_size"
                )
                return np.array([])  # Return empty array on error

        policy_target = np.zeros(policy_size, dtype=np.float32)
        total_visits = np.sum(visit_counts)

        # Log visit counts for debugging, especially if total_visits is 0
        if self.config.debug_mode or total_visits == 0:
            visit_dict = {str(a): v for a, v in zip(actions, visit_counts)}
            logger.debug(f"[DEBUG PolicyTarget] Visit Counts: {visit_dict}")
            logger.debug(f"[DEBUG PolicyTarget] Total Visits: {total_visits}")

        if total_visits > 0:
            # --- Calculate policy based on visit counts ---
            logger.debug(
                "[DEBUG PolicyTarget] Calculating policy based on VISIT COUNTS."
            )
            # Use the network's action mapping if available
            if self.network:
                for i, action in enumerate(actions):
                    action_key = tuple(action) if isinstance(action, list) else action
                    # Get index from the network
                    action_idx = self.network.get_action_index(action_key)

                    if action_idx is not None and 0 <= action_idx < policy_size:
                        policy_target[action_idx] = visit_counts[i] / total_visits
                    else:
                        logger.warning(
                            f"Action {action_key} could not be mapped to index during policy target calculation."
                        )
            else:
                # Fallback if no network: Assume actions are indices directly? Risky.
                # Or try mapping via env if available?
                if hasattr(self.env, "map_action_to_policy_index"):
                    for i, action in enumerate(actions):
                        action_key = (
                            tuple(action) if isinstance(action, list) else action
                        )
                        action_idx = self.env.map_action_to_policy_index(action_key)
                        if action_idx is not None and 0 <= action_idx < policy_size:
                            policy_target[action_idx] = visit_counts[i] / total_visits
                        else:
                            logger.warning(
                                f"Action {action_key} could not be mapped via env during policy target calculation."
                            )
                else:
                    logger.error(
                        "Cannot calculate policy target without network or env mapping."
                    )

        else:
            # Handle case with no visits (should be rare if search runs)
            logger.warning(
                "No visits recorded in MCTS root. Policy target will be zeros."
            )
        # Ensure policy target sums to 1 (handle potential float issues)
        current_sum = policy_target.sum()
        if current_sum > 1e-6:  # Avoid division by zero
            policy_target /= current_sum
        elif policy_target.size > 0:
            # If sum is zero but size > 0, distribute uniformly as a last resort?
            logger.warning(
                f"Policy target sum is near zero ({current_sum}). Setting uniform distribution."
            )
            policy_target.fill(1.0 / policy_target.size)

        return policy_target

    def process_finished_episode(
        self,
        game_history: List[Tuple[StateType, ActionType, np.ndarray]],
        final_outcome: float,
    ) -> EpisodeResult:
        """
        Processes the history of a completed episode to generate training data.
        Assigns the final outcome to all steps and prepares data for buffer and logging.

        Args:
            game_history: A list of tuples (state, action, policy_target) for the episode.
            final_outcome: The outcome for player 0 (+1 win, -1 loss, 0 draw).

        Returns:
            EpisodeResult: An object containing lists of experiences for the
                           replay buffer and the raw history for logging.
        """
        buffer_experiences = []
        logged_history = []
        num_steps = len(game_history)

        if num_steps == 0:
            logger.warning("process_finished_episode called with empty history.")
            return EpisodeResult(buffer_experiences=[], logged_history=[])

        for i, (state_at_step, action_taken, policy_target) in enumerate(game_history):
            # Determine the value target from the perspective of the player at that state
            player_at_step = state_at_step.get("current_player", -1)

            if player_at_step == 0:
                value_target = final_outcome
            elif player_at_step == 1:
                value_target = -final_outcome  # Flip outcome for opponent
            else:
                logger.warning(
                    f"Unknown player {player_at_step} at step {i}. Assigning value target 0.0."
                )
                value_target = 0.0

            if i < 5:  # Print for first few steps
                logger.debug(
                    f"[DEBUG finish_episode] Step {i}: Player={player_at_step}, FinalOutcome(P0)={final_outcome}, AssignedValue={value_target}"
                )

            # --- Standardize state before adding to buffer ---
            # Ensure board/piles are numpy arrays for consistent network input processing
            buffer_state = (
                state_at_step.copy()
            )  # Avoid modifying original history state
            if "board" in buffer_state and not isinstance(
                buffer_state["board"], np.ndarray
            ):
                buffer_state["board"] = np.array(
                    buffer_state["board"], dtype=np.int8
                )  # Or appropriate dtype
            elif "piles" in buffer_state and not isinstance(
                buffer_state["piles"], np.ndarray
            ):
                # NimEnv currently returns tuple, convert it
                buffer_state["piles"] = np.array(buffer_state["piles"], dtype=np.int32)

            # Prepare the experience tuple for the buffer
            buffer_experiences.append((buffer_state, policy_target, value_target))

            # Store the original step info (without modification) for the returned history log
            logged_history.append(
                (state_at_step, action_taken, policy_target, value_target)
            )

        # Note: Internal history (_current_episode_history) is no longer used by this method.

        return EpisodeResult(
            buffer_experiences=buffer_experiences, logged_history=logged_history
        )

    def add_experiences_to_buffer(
        self, experiences: List[Tuple[StateType, np.ndarray, float]]
    ):
        """Adds a list of experiences to the replay buffer."""
        self.replay_buffer.extend(experiences)

    def _calculate_loss(
        self, policy_logits, value_preds, policy_targets, value_targets
    ):
        """Calculates the combined loss for AlphaZero."""
        # Value loss: Mean Squared Error
        value_loss = F.mse_loss(value_preds, value_targets)

        # Policy loss: Cross-Entropy between predicted policy logits and MCTS policy target
        # Ensure targets are probabilities (sum to 1) and logits are raw scores
        # CrossEntropyLoss expects logits as input and class indices or probabilities as target.
        # Here, policy_targets are probabilities derived from MCTS visits.
        policy_loss = F.cross_entropy(policy_logits, policy_targets)

        # Combine losses using configured weight for value loss
        total_loss = (self.config.value_loss_weight * value_loss) + policy_loss
        return total_loss, value_loss, policy_loss

    # Conforms to Agent interface (no arguments)
    def learn(self) -> Optional[Tuple]:
        """
        Update the neural network by training for multiple epochs over the replay buffer.
        Returns average losses for the learning step.
        """
        if not self.network or not self.optimizer:
            logger.warning("Cannot learn: Network or optimizer not initialized.")
            return None

        if len(self.replay_buffer) < self.config.batch_size:
            logger.info(
                f"Skipping learn step: Buffer size {len(self.replay_buffer)} < Batch size {self.config.batch_size}"
            )
            return None  # Indicate no learning happened

        # --- Create DataLoader for efficient batching and shuffling ---
        dataset = ReplayBufferDataset(self.replay_buffer, self.network)
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # --- Training Loop over Epochs ---
        self.network.train()

        total_loss_epoch_avg = 0.0
        value_loss_epoch_avg = 0.0
        policy_loss_epoch_avg = 0.0
        epochs_done = 0

        num_epochs = self.training_config.num_epochs_per_iteration

        for epoch in range(num_epochs):
            total_loss_batches = 0.0
            value_loss_batches = 0.0
            policy_loss_batches = 0.0
            batches_in_epoch = 0

            batch_iterator = (
                tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
                if not self.config.debug_mode
                else data_loader
            )

            for batch_data in batch_iterator:
                states_batch, policy_targets_batch, value_targets_batch = batch_data

                # Move batch data to the correct device
                states_batch = states_batch.to(self.device)
                policy_targets_batch = policy_targets_batch.to(self.device)
                value_targets_batch = value_targets_batch.to(self.device)

                self.optimizer.zero_grad()
                # Network input (states_batch) is already on the device from DataLoader/Dataset
                policy_logits, value_preds = self.network(states_batch)
                total_loss, value_loss, policy_loss = self._calculate_loss(
                    policy_logits,
                    value_preds,
                    policy_targets_batch,
                    value_targets_batch,
                )
                total_loss.backward()
                self.optimizer.step()

                total_loss_batches += total_loss.item()
                value_loss_batches += value_loss.item()
                policy_loss_batches += policy_loss.item()
                batches_in_epoch += 1

                if not self.config.debug_mode and isinstance(batch_iterator, tqdm):
                    batch_iterator.set_postfix(
                        {
                            "Loss": f"{total_loss.item():.3f}",
                            "V_Loss": f"{value_loss.item():.3f}",
                            "P_Loss": f"{policy_loss.item():.3f}",
                        }
                    )

            if batches_in_epoch > 0:
                total_loss_epoch_avg += total_loss_batches / batches_in_epoch
                value_loss_epoch_avg += value_loss_batches / batches_in_epoch
                policy_loss_epoch_avg += policy_loss_batches / batches_in_epoch
                epochs_done += 1

            if self.config.debug_mode:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(
                    f"[DEBUG Learn] Epoch {epoch+1}/{num_epochs} Avg Losses: "
                    f"Total={total_loss_batches / batches_in_epoch:.4f}, "
                    f"Value={value_loss_batches / batches_in_epoch:.4f}, "
                    f"Policy={policy_loss_batches / batches_in_epoch:.4f} | LR={current_lr:.6f}"
                )

        self.network.eval()
        # self.scheduler.step()

        if epochs_done > 0:
            final_total_loss = total_loss_epoch_avg / epochs_done
            final_value_loss = value_loss_epoch_avg / epochs_done
            final_policy_loss = policy_loss_epoch_avg / epochs_done

            logger.info(
                f"Learn Step Summary ({epochs_done} epochs): Avg Losses: "
                f"Total={final_total_loss:.4f}, Value={final_value_loss:.4f}, Policy={final_policy_loss:.4f}"
            )

            return final_total_loss, final_value_loss, final_policy_loss
        else:
            logger.warning("No epochs completed in learn step.")
            return None  # Indicate no learning happened

    def _get_save_path(self) -> Path:
        """Constructs the save file path for the network weights."""
        env_type_name = type(self.env).__name__
        filename = f"alphazero_net_{env_type_name}.pth"
        return DATA_DIR / filename

    def save(self) -> None:
        """Save the neural network weights."""
        if not self.network:
            logger.warning("Cannot save: Network not initialized.")
            return
        filepath = self._get_save_path()
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.network.state_dict(), filepath)
            logger.info(f"AlphaZero network saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving AlphaZero network to {filepath}: {e}")

    def load(self) -> bool:
        """Load the neural network weights."""
        if not self.network:
            logger.warning("Cannot load: Network not initialized.")
            return False
        filepath = self._get_save_path()
        try:
            if filepath.exists():
                # Load state dict using the agent's device
                map_location = self.device
                self.network.load_state_dict(
                    torch.load(filepath, map_location=map_location)
                )
                # Ensure model is on the correct device after loading (should be redundant if map_location works)
                self.network.to(self.device)
                self.network.eval()  # Set to evaluation mode after loading
                logger.info(
                    f"AlphaZero network loaded from {filepath} to {self.device}"
                )
                return True
            else:
                logger.info(f"Network weights file not found: {filepath}")
                return False
        except Exception as e:
            logger.error(f"Error loading AlphaZero network from {filepath}: {e}")
            return False

    def reset(self) -> None:
        """Reset agent state (e.g., MCTS tree)."""
        # Only reset MCTS root, internal episode history is no longer used by agent.
        self.mcts.reset_root()
