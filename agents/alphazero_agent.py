from typing import List, Optional, Dict

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from loguru import logger

from agents.base_learning_agent import (
    BaseLearningAgent,
    EpochMetrics,
)
from environments.base import BaseEnvironment, ActionType
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
    MCTSNodeCache,
)
from core.config import (
    AlphaZeroConfig,
    TrainingConfig,
)
from experiments.architectures.shared import TRAINING_DEVICE, INFERENCE_DEVICE
from models.networks import AlphaZeroNet


# Seems reusable by muzero
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


class AlphaZeroAgent(BaseLearningAgent):
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
        model_name: str = "alphazero",
    ):
        super().__init__(
            selection_strategy=selection_strategy,
            expansion_strategy=expansion_strategy,
            evaluation_strategy=evaluation_strategy,
            backpropagation_strategy=backpropagation_strategy,
            network=network,
            optimizer=optimizer,
            env=env,
            config=config,
            training_config=training_config,
            model_name=model_name,
        )

    # todo share this. Search is what differs
    def act(self, env: BaseEnvironment, train: bool = False) -> ActionType:
        self.search(env=env, train=train)

        temperature = self.config.temperature if train else 0.0
        policy_result = self.get_policy_from_visits(temperature)
        return policy_result.chosen_action

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

    # todo determine how mucht his can be sahred with muzero agent
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

    # todo this is duplicated train epoch too much
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

    def reset_game(self) -> None:
        self.network.cache = {}
        self.node_cache = MCTSNodeCache()

    def reset_turn(self) -> None:
        """Reset agent state (e.g., MCTS tree)."""
        self.root = None


def make_pure_az(
    env: BaseEnvironment,
    config: AlphaZeroConfig,
    training_config: TrainingConfig,
    should_use_network: bool = True,
):
    if should_use_network:
        params = config.state_model_params
        network = AlphaZeroNet(
            env=env,
            embedding_dim=params.get("embedding_dim", 64),
            num_heads=params.get("num_heads", 4),
            num_encoder_layers=params.get("num_encoder_layers", 2),
            dropout=params.get("dropout", 0.1),
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


def get_policy_value(network: AlphaZeroNet, node: "MCTSNode", env: BaseEnvironment):
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
