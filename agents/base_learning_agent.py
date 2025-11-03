import abc
import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from loguru import logger

from agents.mcts_agent import BaseMCTSAgent
from environments.base import BaseEnvironment, StateType, ActionType, DataFrame
from algorithms.mcts import (
    SelectionStrategy,
    ExpansionStrategy,
    EvaluationStrategy,
    BackpropagationStrategy,
)
from core.config import (
    DATA_DIR,
    TrainingConfig,
    SomethingZeroConfig,
    TRAINING_DEVICE,
    INFERENCE_DEVICE,
)


@dataclass
class BaseCollation:
    state_tokens: torch.Tensor
    state_padding_mask: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    action_tokens: torch.Tensor
    action_batch_indices: torch.Tensor


def _get_batched_state(state_dicts: List[Dict[str, DataFrame]]) -> Dict:
    batched_state = {}
    if not state_dicts:
        return batched_state

    # Get table names from the first sample, assuming all samples have the same tables.
    table_names = state_dicts[0].keys()
    for table_name in table_names:
        all_rows = []
        final_columns = None
        for i, state_dict in enumerate(state_dicts):
            original_df = state_dict.get(table_name)
            # Skip if a state is missing this table or the table is empty.
            if original_df is None or original_df.is_empty():
                continue

            if final_columns is None:
                final_columns = original_df.columns + ["batch_idx"]

            for row in original_df._data:
                all_rows.append(row + [i])

        if all_rows:
            batched_state[table_name] = DataFrame(data=all_rows, columns=final_columns)
    return batched_state


def get_tokenizing_collate_fn(network: nn.Module) -> callable:
    """
    Returns a collate function that tokenizes game states and legal actions
    using the provided network.
    """

    def collate_fn(
        batch: Tuple[Dict, torch.Tensor, torch.Tensor, List[ActionType]]
    ) -> BaseCollation:
        state_dicts, policy_targets, value_targets, legal_actions_batch = zip(*batch)
        batch_size = len(state_dicts)

        # Tokenize states
        batched_state = _get_batched_state(state_dicts=state_dicts)
        state_tokens, state_padding_mask = network.tokenize_state_batch(
            batched_state, batch_size=batch_size
        )

        # Tokenize actions
        flat_legal_actions = []
        batch_indices_for_policy = []
        for i, actions in enumerate(legal_actions_batch):
            if actions:
                flat_legal_actions.extend(actions)
                batch_indices_for_policy.extend([i] * len(actions))

        action_tokens = network.tokenize_actions(flat_legal_actions)
        action_batch_indices = torch.tensor(batch_indices_for_policy, dtype=torch.long)

        # Pad targets
        policy_targets = nn.utils.rnn.pad_sequence(
            list(policy_targets), batch_first=True, padding_value=0.0
        )
        value_targets = torch.stack(list(value_targets), 0)

        return BaseCollation(
            state_tokens=state_tokens,
            state_padding_mask=state_padding_mask,
            policy_targets=policy_targets,
            value_targets=value_targets,
            action_tokens=action_tokens,
            action_batch_indices=action_batch_indices,
        )

    return collate_fn


@dataclass
class EpisodeResult:
    """Holds the results of a finished self-play episode."""

    buffer_experiences: List[Any]
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


@dataclass
class LossStatistics:
    batch_loss: torch.Tensor
    value_loss: torch.Tensor
    policy_loss: torch.Tensor
    policy_acc: float
    value_mse: float


@dataclass
class GameHistoryStep:
    state: StateType
    action_index: int
    policy: np.ndarray
    legal_actions: List[ActionType]  # Only used by muzero for training


class BaseLearningAgent(BaseMCTSAgent, abc.ABC):
    """Base agent for MCTS-based learning agents like AlphaZero and MuZero."""

    model_type: str = NotImplemented

    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
        network: nn.Module,
        optimizer,
        env: BaseEnvironment,
        config: SomethingZeroConfig,
        training_config: TrainingConfig,
        model_name: str,
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
        self.model_name = model_name
        self.name = model_name.capitalize()
        if self.network:
            self.network.to(self.device)
            self.network.eval()
            if not hasattr(self.network, "cache"):
                self.network.cache = {}

        self.config = config
        self.training_config = training_config

        self.iteration_to_start_training_at: int = 0

        val_buffer_size = config.replay_buffer_size // 5
        train_buffer_size = config.replay_buffer_size - val_buffer_size
        self.train_replay_buffer = deque(maxlen=train_buffer_size)
        self.val_replay_buffer = deque(maxlen=val_buffer_size)

        self.loaded = False
        self.printed_not_loaded_warning = False

    def print_not_loaded_warning(self):
        if not self.loaded and not self.printed_not_loaded_warning:
            print("WARN: Network weights not loaded")
            self.printed_not_loaded_warning = True

    def act(self, env: BaseEnvironment, train: bool = False) -> ActionType:
        self.print_not_loaded_warning()
        self.search(env=env, train=train)

        temperature = self.config.temperature if train else 0.0
        policy_result = self.get_policy_from_visits(temperature)
        return policy_result.chosen_action

    def process_finished_episode(
        self,
        game_history: List[GameHistoryStep],
        final_outcome: float,
    ) -> EpisodeResult:
        """
        Processes the history of a completed episode to generate training data.
        Assigns the final outcome to all steps and prepares data for buffer and logging.
        """
        if not game_history:
            logger.warning("process_finished_episode called with empty history.")
            return EpisodeResult(buffer_experiences=[], logged_history=[])

        value_targets = []
        for game_history_step in game_history:
            player = game_history_step.state["game"]["current_player"][0]
            value = final_outcome if player == 0 else -final_outcome
            value_targets.append(value)

        logged_history = []
        for i, game_history_step in enumerate(game_history):
            logged_history.append(
                (
                    game_history_step.state,
                    game_history_step.action_index,
                    game_history_step.policy,
                    value_targets[i],
                )
            )

        buffer_experiences = self._create_buffer_experiences(
            game_history, value_targets
        )
        return EpisodeResult(
            buffer_experiences=buffer_experiences, logged_history=logged_history
        )

    def get_policy_target(self, legal_actions: List[ActionType]) -> np.ndarray:
        """
        Returns the policy target vector based on the visit counts from the last search,
        ordered according to the provided legal_actions list.
        """
        if not self.root:
            raise RuntimeError(
                "Must run `act()` to perform a search before getting a policy target."
            )

        assert self.root.edges

        action_visits: Dict[int, int] = {
            action_index: edge.num_visits
            for action_index, edge in self.root.edges.items()
        }
        total_visits = sum(action_visits.values())

        if total_visits == 0:
            return np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)

        # Create the policy target vector, ensuring the order matches legal_actions.
        policy_target = np.zeros(len(legal_actions), dtype=np.float32)
        for action_index, action in enumerate(legal_actions):
            visit_count = action_visits.get(action_index, 0)
            policy_target[action_index] = visit_count / total_visits

        # Normalize again to be safe, although it should sum to 1.
        if np.sum(policy_target) > 0:
            policy_target /= np.sum(policy_target)

        return policy_target

    @abc.abstractmethod
    def _create_buffer_experiences(
        self,
        game_history: List[GameHistoryStep],
        value_targets: List[float],
    ) -> List[Any]:
        """Creates experiences for the replay buffer."""
        pass

    def add_experiences_to_buffer(self, experiences: List[Any]):
        """Adds experiences to the replay buffer, splitting between train and val."""
        random.shuffle(experiences)
        for exp in experiences:
            if random.random() < 0.2:
                self.val_replay_buffer.append(exp)
            else:
                self.train_replay_buffer.append(exp)

    @abc.abstractmethod
    def _calculate_loss(
        self, policy_logits, value_preds, policy_targets, value_targets
    ) -> LossStatistics:
        pass

    @abc.abstractmethod
    def _get_dataset(self, buffer: deque) -> Dataset:
        """Creates a dataset from a replay buffer."""
        pass

    @abc.abstractmethod
    def _get_collate_fn(self) -> callable:
        """Returns the collate function for the DataLoader."""
        pass

    @abc.abstractmethod
    def _process_game_log_data(self, game_data: List[Dict]) -> List[Any]:
        """Processes data from a single game log file into a list of experiences."""
        pass

    def load_game_logs(self, env_name: str, buffer_limit: int):
        """
        Loads existing game logs from LOG_DIR into the agent's train and validation
        replay buffers.
        """
        loaded_games = 0
        log_dir = DATA_DIR / env_name / "game_logs"
        if not log_dir.exists():
            logger.info(
                f"Log directory '{log_dir}' not found. Starting with empty buffers."
            )
            return

        logger.info(f"Scanning {log_dir} for existing '{env_name}' game logs...")
        log_files = sorted(log_dir.glob("**/*.json"), reverse=True)

        if not log_files:
            logger.info("No existing game logs found for this environment.")
            return

        all_experiences = []
        for filepath in tqdm(log_files, desc="Scanning Logs"):
            if len(all_experiences) >= buffer_limit:
                break

            with open(filepath, "r") as f:
                try:
                    game_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {filepath}. Skipping.")
                    continue

            loaded_games += 1
            if not game_data:
                continue

            experiences_from_game = self._process_game_log_data(game_data)
            if experiences_from_game:
                all_experiences.extend(experiences_from_game)

        self.add_experiences_to_buffer(all_experiences)
        loaded_steps = len(self.train_replay_buffer) + len(self.val_replay_buffer)

        logger.info(
            f"Loaded {loaded_steps} experience objects from {loaded_games} games into replay buffers. "
            f"Train: {len(self.train_replay_buffer)}, Val: {len(self.val_replay_buffer)}"
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        is_training: bool,
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> Optional[EpochMetrics]:
        """Runs a single epoch of training or validation."""
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0
        total_policy_acc, total_value_mse = 0.0, 0.0
        num_batches = 0

        iterator = loader
        if is_training and not self.config.debug_mode:
            desc = f"Epoch {epoch + 1}/{max_epochs} (Train)"
            iterator = tqdm(loader, desc=desc, leave=False)

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for batch_data in iterator:
                batch_data: BaseCollation
                policy_targets_batch = batch_data.policy_targets.to(self.device)
                value_targets_batch = batch_data.value_targets.to(self.device)

                if is_training:
                    self.optimizer.zero_grad()

                policy_logits, value_preds = self.network(
                    state_tokens=batch_data.state_tokens.to(self.device),
                    state_padding_mask=batch_data.state_padding_mask.to(self.device),
                    action_tokens=batch_data.action_tokens.to(self.device),
                    action_batch_indices=batch_data.action_batch_indices.to(
                        self.device
                    ),
                )
                loss_statistics = self._calculate_loss(
                    policy_logits,
                    value_preds,
                    policy_targets_batch,
                    value_targets_batch,
                )

                if is_training:
                    loss_statistics.batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                total_loss += loss_statistics.batch_loss.item()
                total_value_loss += loss_statistics.value_loss.item()
                total_policy_loss += loss_statistics.policy_loss.item()
                total_policy_acc += loss_statistics.policy_acc
                total_value_mse += loss_statistics.value_mse
                num_batches += 1

        if num_batches == 0:
            return None

        return EpochMetrics(
            loss=total_loss / num_batches,
            policy_loss=total_policy_loss / num_batches,
            value_loss=total_value_loss / num_batches,
            acc=total_policy_acc / len(loader.dataset),
            mse=total_value_mse / num_batches,
        )

    def _train_epoch(
        self, train_loader: DataLoader, epoch: int, max_epochs: int
    ) -> EpochMetrics:
        """Runs one epoch of training and returns metrics."""
        # _get_train_val_loaders ensures train_loader is not empty, so _run_epoch won't return None.
        return self._run_epoch(
            train_loader, is_training=True, epoch=epoch, max_epochs=max_epochs
        )

    def _validate_epoch(self, val_loader: DataLoader) -> Optional[EpochMetrics]:
        """Runs one epoch of validation and returns metrics."""
        return self._run_epoch(val_loader, is_training=False)

    def _set_device_and_mode(self, training: bool):
        """Sets the device and mode (train/eval) for the network."""
        self.device = TRAINING_DEVICE if training else INFERENCE_DEVICE
        self.network.train() if training else self.network.eval()
        self.network.to(self.device)
        if training and self.optimizer:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

    def train_network(
        self, epoch_callback: Optional[Callable[[int, float], None]] = None
    ) -> Optional[BestEpochMetrics]:
        """
        Trains the network with early stopping.
        An optional callback can be provided to report metrics after each epoch,
        which can also be used to prune trials in hyperparameter optimization.
        """
        try:
            train_loader, val_loader = self._get_train_val_loaders()
        except ValueError as e:
            logger.warning(e)
            return None

        max_epochs = 100
        patience = 10
        best_val_loss, epochs_no_improve = float("inf"), 0
        best_model_state, best_metrics = None, None

        self._set_device_and_mode(training=True)
        for epoch in range(max_epochs):
            self.network.train()
            train_metrics = self._train_epoch(train_loader, epoch, max_epochs)
            self.network.eval()
            val_metrics = self._validate_epoch(val_loader)
            if val_metrics is None:
                continue

            logger.info(
                f"Epoch {epoch+1}/{max_epochs}:\n"
                f"Train Loss=\t{train_metrics}\n"
                f"Val Loss=\t{val_metrics}"
            )

            if epoch_callback:
                try:
                    epoch_callback(epoch, val_metrics.loss)
                except Exception:  # Catches optuna.exceptions.TrialPruned
                    raise  # Re-raise to stop training

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(self.network.state_dict())
                best_metrics = BestEpochMetrics(train=train_metrics, val=val_metrics)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs.")
                    break

        if best_model_state:
            self.network.load_state_dict(best_model_state)
        self._set_device_and_mode(training=False)
        self.network.cache = {}
        return best_metrics

    def _get_model_dir(self) -> Path:
        env_name = type(self.env).__name__.lower()
        return DATA_DIR / env_name / "models"

    def get_latest_model_iter_num(self) -> Optional[int]:
        if not self.model_type:
            raise ValueError("model_type not set on agent")
        model_dir = self._get_model_dir()
        latest_iter = -1
        if model_dir.exists():
            for f in model_dir.glob(f"{self.model_type}_iter_*_net.pth"):
                try:
                    iter_num_str = f.stem.split("_iter_")[1].split("_net")[0]
                    iter_num = int(iter_num_str)
                    if iter_num > latest_iter:
                        latest_iter = iter_num
                except (ValueError, IndexError):
                    continue
        if latest_iter == -1:
            return None
        return latest_iter

    def get_model_iter_path(self, iter_num: int, get_optimizer=False) -> Path:
        if not self.model_type:
            raise ValueError("model_type not set on agent")
        iter_num_string = str(iter_num).zfill(3)
        model_dir = self._get_model_dir()
        suffix = "optimizer.pth" if get_optimizer else "net.pth"
        return model_dir / f"{self.model_type}_iter_{iter_num_string}_{suffix}"

    def load_latest_version(self) -> bool:
        """Loads the latest version of the model and returns True if a checkpoint was loaded."""
        latest_iter_num = self.get_latest_model_iter_num()
        if latest_iter_num is None:
            logger.info(
                f"No existing models found for {self.model_type} for {type(self.env).__name__}. Starting fresh."
            )
            self.iteration_to_start_training_at = 0
            return False  # Nothing loaded

        latest_model_net_path = self.get_model_iter_path(iter_num=latest_iter_num)
        logger.info(
            f"Resuming from iteration {latest_iter_num + 1}. Loading {latest_model_net_path.name}"
        )
        self.iteration_to_start_training_at = latest_iter_num + 1
        return self.load(latest_model_net_path)

    def _get_self_play_version_path(self) -> Path:
        if not self.model_type:
            raise ValueError("model_type not set on agent")
        model_dir = self._get_model_dir()
        return model_dir / f"{self.model_type}_self_play_version.txt"

    def get_self_play_agent_iter(self) -> Optional[int]:
        version_path = self._get_self_play_version_path()
        if version_path.exists():
            with version_path.open("r") as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    return None
        return None

    def promote_to_self_play(self, iteration: int):
        version_path = self._get_self_play_version_path()
        version_path.parent.mkdir(parents=True, exist_ok=True)
        with version_path.open("w") as f:
            f.write(str(iteration))

    def _get_train_val_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates and returns training and validation data loaders."""
        if not self.network or not self.optimizer:
            raise ValueError("Network or optimizer not initialized.")
        # Temporarily disable buffer size checks for overfitting test.
        if not self.train_replay_buffer:
            raise ValueError("Training buffer is empty.")

        train_ds = self._get_dataset(self.train_replay_buffer)

        val_buffer = self.val_replay_buffer
        if not val_buffer:
            val_buffer = self.train_replay_buffer
        val_ds = self._get_dataset(val_buffer)
        collate_fn = self._get_collate_fn()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.training_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader

    def save(self, iteration: int):
        """Saves network and optimizer state for a given iteration."""
        if not self.network or not self.optimizer:
            return
        net_path = self.get_model_iter_path(iteration)
        opt_path = self.get_model_iter_path(iteration, get_optimizer=True)

        net_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), net_path)
        torch.save(self.optimizer.state_dict(), opt_path)

    def load(self, filepath: Path) -> bool:
        """Loads network and optimizer state from a file."""
        if not self.network or not self.optimizer:
            return False
        net_path = filepath
        opt_path = net_path.with_name(
            f"{net_path.stem.replace('_net', '_optimizer')}{net_path.suffix}"
        )
        if not net_path.exists():
            return False
        map_location = self.device
        self.network.load_state_dict(torch.load(net_path, map_location=map_location))
        if opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=map_location)
            )
        self.loaded = True
        return True
