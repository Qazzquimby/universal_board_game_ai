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
from environments.base import BaseEnvironment, StateType, ActionType, DataFrame
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
from experiments.architectures.shared import INFERENCE_DEVICE, TRAINING_DEVICE
from core.config import (
    AlphaZeroConfig,
    DATA_DIR,
    TrainingConfig,
)
from models.networks import AlphaZeroNet


def az_collate_fn(batch):
    """
    Custom collate function to handle batches of experiences, where states are
    dictionaries of DataFrames.
    """
    state_dicts, policy_targets, value_targets, legal_actions_batch = zip(*batch)

    batched_state = {}
    if state_dicts:
        # Get table names from the first sample, assuming all samples have the same tables.
        table_names = state_dicts[0].keys()
        for table_name in table_names:
            # For each table, gather all the DataFrames from the batch, adding a batch index.
            all_dfs_for_table = []
            for i, state_dict in enumerate(state_dicts):
                original_df = state_dict.get(table_name)
                # Skip if a state is missing this table or the table is empty.
                if original_df is None or original_df.is_empty():
                    continue

                new_data = [row + [i] for row in original_df._data]
                new_columns = original_df.columns + ["batch_idx"]
                new_df = DataFrame(data=new_data, columns=new_columns)
                all_dfs_for_table.append(new_df)

            if all_dfs_for_table:
                # Concatenate the list of DataFrames into a single DataFrame.
                concatenated_df = all_dfs_for_table[0].clone()
                for df in all_dfs_for_table[1:]:
                    concatenated_df = concatenated_df.concat(df)
                batched_state[table_name] = concatenated_df

    # Pad the policy targets to have the same length.
    policy_targets = nn.utils.rnn.pad_sequence(
        list(policy_targets), batch_first=True, padding_value=0.0
    )
    value_targets = torch.stack(list(value_targets), 0)

    return batched_state, policy_targets, value_targets, legal_actions_batch


class ReplayBufferDataset(Dataset):
    def __init__(self, buffer: deque):
        self.buffer_list = list(buffer)

    def __len__(self):
        return len(self.buffer_list)

    def __getitem__(self, idx):
        state_dict, policy_target, value_target, legal_actions = self.buffer_list[idx]

        return (
            state_dict,
            torch.tensor(policy_target, dtype=torch.float32),
            torch.tensor([value_target], dtype=torch.float32),
            legal_actions,
        )


@dataclass
class EpisodeResult:
    """Holds the results of a finished self-play episode."""

    buffer_experiences: List[Tuple[StateType, np.ndarray, float, List[ActionType]]]
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
        if env.is_done:
            return env.get_reward_for_player(player=env.get_current_player())

        _, value = get_policy_value(network=self.network, node=node, env=env)

        return float(value)


class AlphaZeroExpansion(ExpansionStrategy):
    def __init__(self, network: nn.Module):
        self.network = network

    def expand(self, node: "MCTSNode", env: BaseEnvironment) -> None:
        if node.is_expanded or env.is_done:
            return

        policy_dict, _ = get_policy_value(network=self.network, node=node, env=env)

        for action, prior in policy_dict.items():
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
        self.device = INFERENCE_DEVICE
        if self.network:
            self.network.to(self.device)
            self.network.eval()
            if not hasattr(self.network, "cache"):
                self.network.cache = {}

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
            self.search(env=env, train=train)  # for debugging
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
            # More numerically stable temperature scaling
            log_visits = np.log(visits + 1e-10)
            scaled_log_visits = log_visits / temperature
            scaled_log_visits -= np.max(scaled_log_visits)
            exp_scaled_log_visits = np.exp(scaled_log_visits)
            visit_probs = exp_scaled_log_visits / np.sum(exp_scaled_log_visits)

            chosen_action_index = np.random.choice(len(actions), p=visit_probs)
            chosen_action = actions[chosen_action_index]

        return chosen_action

    def get_policy_target(self, legal_actions: List[ActionType]) -> np.ndarray:
        """
        Returns the policy target vector based on the visit counts from the last search,
        ordered according to the provided legal_actions list.
        """
        if not self.root:
            raise RuntimeError(
                "Must run `act()` to perform a search before getting a policy target."
            )

        if not self.root.edges:
            # If there are no edges, it implies no actions were explored, possibly because
            # it's a terminal state. The policy target should be uniform or zero,
            # but it must match the length of legal_actions.
            if not legal_actions:
                return np.array([], dtype=np.float32)
            # This case might indicate an issue, but we return a uniform distribution
            # to avoid crashing.
            return np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)

        action_visits: Dict[ActionType, int] = {
            action: edge.num_visits for action, edge in self.root.edges.items()
        }
        total_visits = sum(action_visits.values())

        if total_visits == 0:
            return np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)

        # Create the policy target vector, ensuring the order matches legal_actions.
        policy_target = np.zeros(len(legal_actions), dtype=np.float32)
        for i, action in enumerate(legal_actions):
            action_key = tuple(action) if isinstance(action, list) else action
            visit_count = action_visits.get(action_key, 0)
            policy_target[i] = visit_count / total_visits

        # Normalize again to be safe, although it should sum to 1.
        if np.sum(policy_target) > 0:
            policy_target /= np.sum(policy_target)

        return policy_target

    def _expand_leaf(self, leaf_node: MCTSNode, leaf_env: BaseEnvironment, train: bool):
        if not leaf_node.is_expanded and not leaf_env.is_done:
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
            game_history: A list of tuples (state, action, policy_target, legal_actions) for the episode.
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
            legal_actions_df = state_at_step.get("legal_actions")
            if legal_actions_df is None or legal_actions_df.is_empty():
                continue

            # Reconstruct the list of legal actions from the DataFrame
            legal_actions = [row[0] for row in legal_actions_df.rows()]

            player_at_step = state_at_step["game"]["current_player"][0]

            if player_at_step == 0:
                value_target = final_outcome
            elif player_at_step == 1:
                value_target = -final_outcome  # Flip outcome for opponent
            else:
                assert False

            buffer_state = state_at_step.copy()
            buffer_experiences.append(
                (buffer_state, policy_target, value_target, legal_actions)
            )

            logged_history.append(
                (state_at_step, action_taken, policy_target, value_target)
            )

        return EpisodeResult(
            buffer_experiences=buffer_experiences, logged_history=logged_history
        )

    def add_experiences_to_buffer(
        self, experiences: List[Tuple[StateType, np.ndarray, float, List[ActionType]]]
    ):
        """Adds a list of experiences to the replay buffer, splitting it between train and val sets."""
        random.shuffle(experiences)
        for exp in experiences:
            # 20% chance to be added to validation set
            if random.random() < 0.2:
                self.val_replay_buffer.append(exp)
            else:
                self.train_replay_buffer.append(exp)

    def _convert_state_df_to_tensors(
        self, state_df: Dict[str, DataFrame]
    ) -> Dict[str, torch.Tensor]:
        """Converts a dictionary of batched DataFrames to a dictionary of tensors."""
        network_spec = self.env.get_network_spec()
        tensors = {}

        game_df = state_df.get("game")
        game_context_map = {}
        if game_df:
            # Create a map from batch_idx to a single-row DataFrame for that game's state
            game_cols = game_df.columns
            for row in game_df._data:
                batch_idx = row[game_df._col_to_idx["batch_idx"]]
                game_context_map[batch_idx] = DataFrame(data=[row], columns=game_cols)

        for table_name, table_df in state_df.items():
            for col_name in table_df.columns:
                key = f"{table_name}_{col_name}"
                raw_values = table_df[col_name]

                # batch_idx is already a tensor, just move to device
                if col_name == "batch_idx":
                    tensors[key] = torch.tensor(
                        raw_values, dtype=torch.long, device=self.device
                    )
                    continue

                transform = network_spec.get("transforms", {}).get(col_name)
                if transform and game_context_map:
                    batch_indices = table_df["batch_idx"]
                    transformed_values = []
                    for val, batch_idx in zip(raw_values, batch_indices):
                        # The state passed to transform only contains the 'game' table.
                        # This is a simplification that works for connect4-style transforms.
                        game_state_for_transform = {
                            "game": game_context_map.get(batch_idx)
                        }
                        if game_state_for_transform["game"]:
                            transformed_values.append(
                                transform(val, game_state_for_transform)
                            )
                        else:
                            transformed_values.append(val)  # Fallback
                else:
                    transformed_values = raw_values

                cardinality = network_spec.get("cardinalities", {}).get(col_name)
                if cardinality is not None:
                    final_values = [
                        v if v is not None else cardinality for v in transformed_values
                    ]
                else:
                    final_values = transformed_values  # Assumes values are numerical

                tensors[key] = torch.tensor(
                    final_values, dtype=torch.long, device=self.device
                )

        return tensors

    def _calculate_loss(
        self, policy_logits, value_preds, policy_targets, value_targets
    ):
        value_loss = F.mse_loss(value_preds, value_targets.squeeze(-1))

        # Policy loss for padded sequences
        log_probs = F.log_softmax(policy_logits, dim=1)
        safe_log_probs = torch.where(log_probs == -torch.inf, 0.0, log_probs)
        policy_loss_per_item = -torch.sum(policy_targets * safe_log_probs, dim=1)
        policy_loss = policy_loss_per_item.mean()

        total_loss = (self.config.value_loss_weight * value_loss) + policy_loss

        value_mse = value_loss.item()

        # Accuracy calculation for padded sequences
        predicted_indices = torch.argmax(policy_logits, dim=1)
        target_indices = torch.argmax(policy_targets, dim=1)

        # Only calculate accuracy for samples that have legal actions
        has_legal_actions = torch.any(policy_logits != -torch.inf, dim=1)
        num_valid_samples = has_legal_actions.sum().item()

        if num_valid_samples > 0:
            policy_acc = (
                (predicted_indices == target_indices)[has_legal_actions].sum().item()
            )
        else:
            policy_acc = 0

        return total_loss, value_loss, policy_loss, policy_acc, value_mse

    def _train_epoch(
        self, train_loader: DataLoader, epoch: int, max_epochs: int
    ) -> EpochMetrics:
        """Runs one epoch of training and returns metrics."""
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
                state_df_batch,
                policy_targets_batch,
                value_targets_batch,
                legal_actions_batch,
            ) = batch_data

            state_tensor_batch = self._convert_state_df_to_tensors(state_df_batch)
            policy_targets_batch = policy_targets_batch.to(self.device)
            value_targets_batch = value_targets_batch.to(self.device)

            self.optimizer.zero_grad()
            policy_logits, value_preds = self.network(
                state_tensor_batch, legal_actions=legal_actions_batch
            )
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
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0
        total_policy_acc, total_value_mse = 0.0, 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                (
                    state_df_batch,
                    policy_targets_batch,
                    value_targets_batch,
                    legal_actions_batch,
                ) = batch_data
                state_tensor_batch = self._convert_state_df_to_tensors(state_df_batch)
                policy_targets_batch = policy_targets_batch.to(self.device)
                value_targets_batch = value_targets_batch.to(self.device)
                policy_logits, value_preds = self.network(
                    state_tensor_batch, legal_actions=legal_actions_batch
                )
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

    def _set_device_and_mode(self, training: bool):
        """Sets the device and mode (train/eval) for the network and optimizer."""
        if training:
            self.device = TRAINING_DEVICE
            self.network.train()
        else:
            self.device = INFERENCE_DEVICE
            self.network.eval()

        self.network.to(self.device)

        if training and self.optimizer:
            # Move optimizer state to the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

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

        self._set_device_and_mode(training=True)
        for epoch in range(max_epochs):
            self.network.train()
            train_metrics = self._train_epoch(train_loader, epoch, max_epochs)

            self.network.eval()
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

        self._set_device_and_mode(training=False)
        logger.info(f"Network set to {self.device} and eval mode.")

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

        train_dataset = ReplayBufferDataset(self.train_replay_buffer)
        val_dataset = ReplayBufferDataset(self.val_replay_buffer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=az_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=az_collate_fn,
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

    def save(self, filepath: Optional[Path] = None) -> None:
        """Save the neural network weights and optimizer state."""
        if not self.network or not self.optimizer:
            logger.warning("Cannot save: Network or optimizer not initialized.")
            return

        if filepath is None:
            filepath = self._get_save_path()
            optimizer_filepath = self._get_optimizer_save_path()
        else:
            optimizer_filepath = filepath.with_name(
                f"{filepath.stem.replace('_net', '_optimizer')}{filepath.suffix}"
            )

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.network.state_dict(), filepath)
            logger.info(f"AlphaZero network saved to {filepath}")

            # Save optimizer state
            torch.save(self.optimizer.state_dict(), optimizer_filepath)
            logger.info(f"AlphaZero optimizer state saved to {optimizer_filepath}")

        except Exception as e:
            logger.error(f"Error saving AlphaZero network or optimizer: {e}")

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Load the neural network weights and optimizer state."""
        if not self.network or not self.optimizer:
            logger.warning("Cannot load: Network or optimizer not initialized.")
            return False

        if filepath is None:
            filepath = self._get_save_path()
            optimizer_filepath = self._get_optimizer_save_path()
        else:
            optimizer_filepath = filepath.with_name(
                f"{filepath.stem.replace('_net', '_optimizer')}{filepath.suffix}"
            )

        try:
            if filepath.exists():
                map_location = self.device
                self.network.load_state_dict(
                    torch.load(filepath, map_location=map_location)
                )
                self.network.to(self.device)
                self.network.eval()
                logger.info(
                    f"AlphaZero network loaded from {filepath} to {self.device}"
                )

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
                else:
                    logger.info(
                        f"Optimizer state file not found: {optimizer_filepath}. Optimizer not loaded."
                    )
                return True
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
        smp = config.state_model_params
        network = AlphaZeroNet(
            env=env,
            embedding_dim=smp.get("embedding_dim", 64),
            num_heads=smp.get("num_heads", 4),
            num_encoder_layers=smp.get("num_encoder_layers", 2),
            dropout=smp.get("dropout", 0.1),
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


def get_policy_value(network: nn.Module, node: "MCTSNode", env: BaseEnvironment):
    key = node.state_with_key.key
    cached_result = network.cache.get(key)

    if cached_result:
        policy_dict, value = cached_result
    else:
        network.eval()
        with torch.no_grad():
            legal_actions = env.get_legal_actions()
            policy_dict, value = network.predict(node.state_with_key, legal_actions)
        network.cache[key] = (policy_dict, value)
    return policy_dict, value
