from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random
import copy

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from loguru import logger

from agents.mcts_agent import BaseMCTSAgent
from environments.base import BaseEnvironment, StateType, ActionType
from algorithms.mcts import (
    DummyAlphaZeroNet,
    MCTSNode,
    UCB1Selection,
    ExpansionStrategy,
    EvaluationStrategy,
    StandardBackpropagation,
    Edge,
    BackpropagationStrategy,
    SelectionStrategy,
)
from models.networks import AutoGraphNet
from core.config import AlphaZeroConfig, DATA_DIR, TrainingConfig


class ReplayBufferDataset(Dataset):
    def __init__(self, buffer: deque, state_model: nn.Module):
        self.buffer_list = list(buffer)
        self.state_model = state_model

    def __len__(self):
        return len(self.buffer_list)

    def __getitem__(self, idx):
        state_dict, policy_target, value_target = self.buffer_list[idx]
        self.state_model.env.set_state(state_dict)
        src, src_key_padding_mask = self.state_model.create_input_tensors_from_state()

        return (
            src.squeeze(0),  # Remove batch dim
            src_key_padding_mask.squeeze(0),  # Remove batch dim
            torch.tensor(policy_target, dtype=torch.float32),
            torch.tensor([value_target], dtype=torch.float32),
        )


@dataclass
class EpisodeResult:
    """Holds the results of a finished self-play episode."""

    buffer_experiences: List[Tuple[StateType, np.ndarray, float]]
    logged_history: List[Tuple[StateType, ActionType, np.ndarray, float]]


@dataclass
class EpochMetrics:
    """Metrics for a single training/validation epoch."""

    loss: float
    policy_loss: float
    value_loss: float
    acc: float
    mse: float


@dataclass
class BestEpochMetrics:
    """Metrics from the best validation epoch during training."""

    train: EpochMetrics
    val: EpochMetrics


class AlphaZeroEvaluation(EvaluationStrategy):
    def __init__(self, network: nn.Module):
        self.network = network

    def evaluate(self, node: "MCTSNode", env: BaseEnvironment) -> float:
        if env.state.done:
            return env.state.rewards.get(env.state.players.current_index, 0.0)

        _, value = get_policy_value(network=self.network, node=node)

        return float(value)


class AlphaZeroExpansion(ExpansionStrategy):
    def __init__(self, network: nn.Module):
        self.network = network

    def expand(self, node: "MCTSNode", env: BaseEnvironment) -> None:
        if node.is_expanded or env.state.done:
            return

        policy_np, _ = get_policy_value(network=self.network, node=node)

        legal_actions = env.get_legal_actions()
        for action in legal_actions:
            action_idx = env.map_action_to_policy_index(action)
            if action_idx is not None:
                prior = policy_np[action_idx]
                action_key = tuple(action) if isinstance(action, list) else action
                node.edges[action_key] = Edge(prior=prior)
        node.is_expanded = True


class AlphaZeroAgent(BaseMCTSAgent):
    """Agent implementing the AlphaZero algorithm."""

    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
        network: nn.Module,
        optimizer,
        env: BaseEnvironment,
        config: AlphaZeroConfig,
        training_config: TrainingConfig,
    ):
        super().__init__(
            num_simulations=config.num_simulations,
            selection_strategy=selection_strategy,
            expansion_strategy=expansion_strategy,
            evaluation_strategy=evaluation_strategy,
            backpropagation_strategy=backpropagation_strategy,
        )
        self.network = network
        self.optimizer = optimizer
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.network:
            self.network.to(self.device)

        self.config = config
        self.training_config = training_config

        # Learning
        val_buffer_size = config.replay_buffer_size // 5
        train_buffer_size = config.replay_buffer_size - val_buffer_size
        # TODO now, this won't handle the run being continued later I think
        self.train_replay_buffer = deque(maxlen=train_buffer_size)
        self.val_replay_buffer = deque(maxlen=val_buffer_size)

    def act(self, env: BaseEnvironment, train: bool = False) -> ActionType:
        self.search(env=env, train=train)

        temperature = self.config.temperature if train else 0.0

        assert self.root is not None
        if not self.root.edges:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                logger.warning("No legal actions from a non-expanded root. Cannot act.")
                return None
            raise RuntimeError(
                f"Search finished but root has no edges. Legal actions: {legal_actions}"
            )

        action_visits: Dict[ActionType, int] = {
            action: edge.num_visits for action, edge in self.root.edges.items()
        }
        actions = list(action_visits.keys())
        visits = np.array(
            [action_visits[action] for action in actions], dtype=np.float64
        )

        if temperature == 0:
            if len(visits) == 0:
                raise RuntimeError("Cannot select action, no visits recorded.")
            best_action_index = np.argmax(visits)
            chosen_action = actions[best_action_index]
        else:
            visit_probs = visits / np.sum(visits)
            chosen_action_index = np.random.choice(len(actions), p=visit_probs)
            chosen_action = actions[chosen_action_index]

        return chosen_action

    def get_policy_target(self) -> np.ndarray:
        """
        Returns the policy target vector based on the visit counts from the last search.
        This should be called after `act()` to get the data for training.
        """
        if not self.root:
            raise RuntimeError(
                "Must run `act()` to perform a search before getting a policy target."
            )

        policy_vector = np.zeros(self.env.num_action_types, dtype=np.float32)
        if not self.root.edges:
            return policy_vector  # Return zeros if no actions were possible/explored

        action_visits: Dict[ActionType, int] = {
            action: edge.num_visits for action, edge in self.root.edges.items()
        }
        visits = np.array(list(action_visits.values()), dtype=np.float64)
        visit_sum = np.sum(visits)

        if visit_sum > 0:
            for action, visit_count in action_visits.items():
                action_idx = self.env.map_action_to_policy_index(action)
                if action_idx is not None:
                    policy_vector[action_idx] = visit_count / visit_sum
        return policy_vector

    def _expand_leaf(self, leaf_node: MCTSNode, leaf_env: BaseEnvironment, train: bool):
        if not leaf_node.is_expanded and not leaf_env.state.done:
            self.expansion_strategy.expand(leaf_node, leaf_env)

            if leaf_node == self.root and train and self.config.dirichlet_epsilon > 0:
                self._apply_dirichlet_noise(self.root)

    def _apply_dirichlet_noise(self, node: MCTSNode):
        if not node.edges:
            return
        actions = list(node.edges.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            node.edges[action].prior = (
                node.edges[action].prior * (1 - eps) + noise[i] * eps
            )

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
            player_at_step = state_at_step["players"].current_index

            if player_at_step == 0:
                value_target = final_outcome
            elif player_at_step == 1:
                value_target = -final_outcome  # Flip outcome for opponent
            else:
                assert False

            # The state dictionary is stored as-is. `env.set_state()` will reconstruct it.
            buffer_state = state_at_step.copy()
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
        """Adds a list of experiences to the replay buffer, splitting it between train and val sets."""
        random.shuffle(experiences)
        for exp in experiences:
            # 20% chance to be added to validation set
            if random.random() < 0.2:
                self.val_replay_buffer.append(exp)
            else:
                self.train_replay_buffer.append(exp)

    def _calculate_loss(
        self, policy_logits, value_preds, policy_targets, value_targets
    ):
        """Calculates the combined loss for AlphaZero and performance metrics."""

        # --- Add Temporary Debug Logging (using WARNING level) ---
        # Log ~10% of batches, showing first 8 samples
        try:
            targets_np = value_targets.flatten().cpu().detach().numpy()[:8]
            preds_np = value_preds.flatten().cpu().detach().numpy()[:8]
            logger.warning(f"[Loss Debug] Value Targets: {targets_np}")
            logger.warning(f"[Loss Debug] Value Preds  : {preds_np}")
            # Check if any target is +1.0 and corresponding pred is <= 0
            positive_targets_mask = targets_np > 0.99
            if np.any(positive_targets_mask):
                negative_problematic_preds = preds_np[positive_targets_mask]
                if np.any(negative_problematic_preds <= 0.3):
                    logger.warning(
                        f"[Loss Debug] Problematic Negative Preds (Target=1.0, Pred<=0): {negative_problematic_preds[negative_problematic_preds <= 0]}"
                    )

            negative_targets_mask = targets_np < 0.99
            if np.any(negative_targets_mask):
                positive_problematic_preds = preds_np[negative_targets_mask]
                if np.any(positive_problematic_preds >= 0.3):
                    logger.warning(
                        f"[Loss Debug] Problematic Positive Preds (Target=1.0, Pred<=0): {positive_problematic_preds[positive_problematic_preds <= 0]}"
                    )

        except Exception as e:
            # Added shape info to error log
            logger.error(
                f"Error during loss debug logging: {e}, Target shape: {value_targets.shape}, Pred shape: {value_preds.shape}"
            )
        # --- End Temporary Debug Logging ---

        # Value loss: Mean Squared Error
        value_loss = F.mse_loss(value_preds, value_targets)

        # Policy loss: Cross-Entropy between predicted policy logits and MCTS policy target
        policy_loss = F.cross_entropy(policy_logits, policy_targets)

        # Combine losses using configured weight for value loss
        total_loss = (self.config.value_loss_weight * value_loss) + policy_loss

        # --- Calculate Metrics ---
        value_mse = value_loss.item()
        # Policy accuracy: compare argmax of predicted policy with argmax of MCTS policy
        _, predicted_policy_indices = torch.max(policy_logits, 1)
        _, target_policy_indices = torch.max(policy_targets, 1)
        policy_acc = (
            (predicted_policy_indices == target_policy_indices).float().sum().item()
        )

        return total_loss, value_loss, policy_loss, policy_acc, value_mse

    def _train_epoch(
        self, train_loader: DataLoader, epoch: int, max_epochs: int
    ) -> EpochMetrics:
        """Runs one epoch of training and returns metrics."""
        self.network.train()
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0
        total_policy_acc, total_value_mse = 0.0, 0.0
        train_batches = 0

        train_iterator = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{max_epochs} (Train)",
                leave=False,
            )
            if not self.config.debug_mode
            else train_loader
        )
        for batch_data in train_iterator:
            (
                src_batch,
                mask_batch,
                policy_targets_batch,
                value_targets_batch,
            ) = batch_data
            src_batch = src_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)
            policy_targets_batch = policy_targets_batch.to(self.device)
            value_targets_batch = value_targets_batch.to(self.device)

            self.optimizer.zero_grad()
            policy_logits, value_preds = self.network(src_batch, mask_batch)
            (
                batch_loss,
                value_loss,
                policy_loss,
                policy_acc,
                value_mse,
            ) = self._calculate_loss(
                policy_logits,
                value_preds,
                policy_targets_batch,
                value_targets_batch,
            )
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            total_policy_acc += policy_acc
            total_value_mse += value_mse
            train_batches += 1

        return EpochMetrics(
            loss=total_loss / train_batches,
            policy_loss=total_policy_loss / train_batches,
            value_loss=total_value_loss / train_batches,
            acc=total_policy_acc / len(train_loader.dataset),
            mse=total_value_mse / train_batches,
        )

    def _validate_epoch(self, val_loader: DataLoader) -> Optional[EpochMetrics]:
        """Runs one epoch of validation and returns metrics."""
        self.network.eval()
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0
        total_policy_acc, total_value_mse = 0.0, 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                (
                    src_batch,
                    mask_batch,
                    policy_targets_batch,
                    value_targets_batch,
                ) = batch_data
                src_batch = src_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                policy_targets_batch = policy_targets_batch.to(self.device)
                value_targets_batch = value_targets_batch.to(self.device)
                policy_logits, value_preds = self.network(src_batch, mask_batch)
                (
                    batch_loss,
                    value_loss,
                    policy_loss,
                    policy_acc,
                    value_mse,
                ) = self._calculate_loss(
                    policy_logits,
                    value_preds,
                    policy_targets_batch,
                    value_targets_batch,
                )
                total_loss += batch_loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                total_policy_acc += policy_acc
                total_value_mse += value_mse
                val_batches += 1

        if val_batches == 0:
            return None

        return EpochMetrics(
            loss=total_loss / val_batches,
            policy_loss=total_policy_loss / val_batches,
            value_loss=total_value_loss / val_batches,
            acc=total_policy_acc / len(val_loader.dataset),
            mse=total_value_mse / val_batches,
        )

    def learn(self) -> Optional[BestEpochMetrics]:
        """
        Update the neural network by training for multiple epochs over the replay buffer,
        using early stopping based on a validation set.
        Returns a dictionary of losses and metrics from the best epoch.
        """
        try:
            train_loader, val_loader = self._get_train_val_loaders()
        except ValueError as e:
            logger.warning(e)
            return None

        max_epochs = 100
        early_stopping_patience = 10
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_state = None
        best_epoch_metrics: Optional[BestEpochMetrics] = None

        for epoch in range(max_epochs):
            train_metrics = self._train_epoch(train_loader, epoch, max_epochs)
            val_metrics = self._validate_epoch(val_loader)

            if val_metrics is None:
                continue

            logger.info(
                f"Epoch {epoch+1}/{max_epochs}: "
                f"Train Loss={train_metrics.loss:.4f}, Val Loss={val_metrics.loss:.4f} | "
                f"Train Acc={train_metrics.acc:.4f}, Val Acc={val_metrics.acc:.4f}"
            )

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(self.network.state_dict())
                best_epoch_metrics = BestEpochMetrics(
                    train=train_metrics, val=val_metrics
                )
                logger.info(
                    f"  New best validation loss: {best_val_loss:.4f}. Saving model state."
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        if best_model_state:
            logger.info(
                f"Restoring best model from epoch with validation loss: {best_val_loss:.4f}"
            )
            self.network.load_state_dict(best_model_state)

        self.network.eval()

        if best_epoch_metrics:
            logger.info(
                f"Learn Step Summary (best epoch): "
                f"Total Loss={best_epoch_metrics.train.loss:.4f}, "
                f"Value Loss={best_epoch_metrics.train.value_loss:.4f}, "
                f"Policy Loss={best_epoch_metrics.train.policy_loss:.4f}"
            )
            return best_epoch_metrics
        else:
            logger.warning("No learning occurred or no improvement found.")
            return None

    def _get_train_val_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if not self.network or not self.optimizer:
            raise ValueError("Cannot learn: Network or optimizer not initialized.")
        if len(self.train_replay_buffer) < self.config.training_batch_size:
            raise ValueError(
                f"Skipping learn step: Not enough training data for one batch. "
                f"Have {len(self.train_replay_buffer)}, need {self.config.training_batch_size}."
            )
        if not self.val_replay_buffer:
            raise ValueError("Skipping learn step: Validation buffer is empty.")

        train_dataset = ReplayBufferDataset(
            self.train_replay_buffer, self.network.state_model
        )
        val_dataset = ReplayBufferDataset(
            self.val_replay_buffer, self.network.state_model
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        logger.info(
            f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples."
        )
        return train_loader, val_loader

    def _get_save_path(self) -> Path:
        """Constructs the save file path for the network weights."""
        env_type_name = type(self.env).__name__
        filename = f"alphazero_net_{env_type_name}.pth"
        return DATA_DIR / filename

    def _get_optimizer_save_path(self) -> Path:
        """Constructs the save file path for the optimizer state."""
        env_type_name = type(self.env).__name__
        filename = f"alphazero_optimizer_{env_type_name}.pth"
        return DATA_DIR / filename

    def save(self) -> None:
        """Save the neural network weights and optimizer state."""
        if not self.network or not self.optimizer:
            logger.warning("Cannot save: Network or optimizer not initialized.")
            return
        filepath = self._get_save_path()
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.network.state_dict(), filepath)
            logger.info(f"AlphaZero network saved to {filepath}")

            # Save optimizer state
            optimizer_filepath = self._get_optimizer_save_path()
            torch.save(self.optimizer.state_dict(), optimizer_filepath)
            logger.info(f"AlphaZero optimizer state saved to {optimizer_filepath}")

        except Exception as e:
            logger.error(f"Error saving AlphaZero network or optimizer: {e}")

    def load(self) -> bool:
        """Load the neural network weights and optimizer state."""
        if not self.network or not self.optimizer:
            logger.warning("Cannot load: Network or optimizer not initialized.")
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
                self.network.eval()
                logger.info(
                    f"AlphaZero network loaded from {filepath} to {self.device}"
                )

                # Load optimizer state
                optimizer_filepath = self._get_optimizer_save_path()
                if optimizer_filepath.exists():
                    try:
                        self.optimizer.load_state_dict(
                            torch.load(optimizer_filepath, map_location=map_location)
                        )
                        logger.info(
                            f"AlphaZero optimizer state loaded from {optimizer_filepath}"
                        )
                    except Exception as opt_e:
                        logger.error(
                            f"Error loading AlphaZero optimizer state from {optimizer_filepath}: {opt_e}"
                        )
                        # Decide if failure to load optimizer should prevent network load (currently doesn't)
                else:
                    logger.info(
                        f"Optimizer state file not found: {optimizer_filepath}. Optimizer not loaded."
                    )

                return True  # Network loaded successfully, even if optimizer didn't
            else:
                logger.info(f"Network weights file not found: {filepath}")
                return False
        except Exception as net_e:
            logger.error(f"Error loading AlphaZero network from {filepath}: {net_e}")
            return False

    def reset(self) -> None:
        """Reset agent state (e.g., MCTS tree)."""
        self.root = None


def make_pure_az(
    env: BaseEnvironment,
    config: AlphaZeroConfig,
    training_config: TrainingConfig,
    should_use_network: bool,
):
    if should_use_network:
        network = AutoGraphNet(
            env=env,
            state_model_params=config.state_model_params,
            policy_model_params=config.policy_model_params,
        )
        optimizer = optim.AdamW(network.parameters(), lr=training_config.learning_rate)
    else:
        network = DummyAlphaZeroNet(env)
        optimizer = None

    return AlphaZeroAgent(
        selection_strategy=UCB1Selection(exploration_constant=config.cpuct),
        expansion_strategy=AlphaZeroExpansion(network=network),
        evaluation_strategy=AlphaZeroEvaluation(network=network),
        backpropagation_strategy=StandardBackpropagation(),
        network=network,
        optimizer=optimizer,
        env=env,
        config=config,
        training_config=training_config,
    )


def get_policy_value(network: nn.Module, node: "MCTSNode"):
    key = node.state_with_key.key
    cached_result = network.cache.get(key)

    if cached_result:
        policy_np, value = cached_result
    else:
        network.eval()
        with torch.no_grad():
            policy_np, value = network.predict(node.state_with_key)
        network.cache[key] = (policy_np, value)
    return policy_np, value
