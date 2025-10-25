from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque

import torch
from torch.utils.data import Dataset
from torch import optim, nn
import torch.nn.functional as F
import numpy as np

from agents.alphazero.alphazero_net import AlphaZeroNet
from agents.base_learning_agent import (
    BaseLearningAgent,
    get_tokenizing_collate_fn,
    LossStatistics,
    GameHistoryStep,
)
from environments.base import BaseEnvironment, ActionType, StateType, DataFrame
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


@dataclass
class AlphaZeroExperience:
    state: StateType
    policy_target: np.ndarray
    value_target: float
    legal_actions: List[ActionType]


class AlphaZeroDataset(Dataset):
    def __init__(self, buffer: deque):
        self.buffer_list = list(buffer)

    def __len__(self):
        return len(self.buffer_list)

    def __getitem__(self, idx):
        exp = self.buffer_list[idx]
        return (
            exp.state,
            torch.tensor(exp.policy_target, dtype=torch.float32),
            torch.tensor([exp.value_target], dtype=torch.float32),
            exp.legal_actions,
        )


class AlphaZeroAgent(BaseLearningAgent):
    """Agent implementing the AlphaZero algorithm."""

    model_type = "alphazero"
    config: AlphaZeroConfig

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

    def _create_buffer_experiences(
        self,
        game_history: List[GameHistoryStep],
        value_targets: List[float],
    ) -> List[AlphaZeroExperience]:
        """Creates AlphaZeroExperience objects for the replay buffer."""
        experiences = []
        for i, game_history in enumerate(game_history):
            legal_actions_df = game_history.state.get("legal_actions")
            if legal_actions_df is None or legal_actions_df.is_empty():
                continue

            legal_actions = [row[0] for row in legal_actions_df.rows()]
            transformed_state = self.network.apply_transforms(game_history.state)
            experiences.append(
                AlphaZeroExperience(
                    state=transformed_state,
                    policy_target=game_history.policy,
                    value_target=value_targets[i],
                    legal_actions=legal_actions,
                )
            )
        return experiences

    def _get_dataset(self, buffer: deque) -> Dataset:
        """Creates a dataset from a replay buffer."""
        return AlphaZeroDataset(buffer)

    def _get_collate_fn(self) -> callable:
        """Returns the collate function for the DataLoader."""
        return get_tokenizing_collate_fn(self.network)

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

    # todo determine how much this can be shared with muzero agent
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

        return LossStatistics(
            batch_loss=total_loss,
            value_loss=value_loss,
            policy_loss=policy_loss,
            policy_acc=policy_acc,
            value_mse=value_mse,
        )

    def _process_game_log_data(
        self, game_data: List[Dict]
    ) -> List["AlphaZeroExperience"]:
        """Processes data from a single game log file into a list of experiences."""
        experiences = []
        for step_data in game_data:
            state_json = step_data.get("state")
            policy_target_list = step_data.get("policy_target")
            value_target = step_data.get("value_target")

            if (
                state_json is not None
                and policy_target_list is not None
                and value_target is not None
            ):
                state = {
                    table_name: DataFrame(
                        data=table_data.get("_data"),
                        columns=table_data.get("columns"),
                    )
                    for table_name, table_data in state_json.items()
                }
                legal_actions_df = state.get("legal_actions")
                if legal_actions_df is None or legal_actions_df.is_empty():
                    continue

                legal_actions = [row[0] for row in legal_actions_df.rows()]
                policy_target = np.array(policy_target_list, dtype=np.float32)
                experiences.append(
                    AlphaZeroExperience(
                        state=state,
                        policy_target=policy_target,
                        value_target=value_target,
                        legal_actions=legal_actions,
                    )
                )
        return experiences

    def reset_game(self) -> None:
        self.network.cache = {}
        self.node_cache = MCTSNodeCache()

    def reset_turn(self) -> None:
        """Reset agent state (e.g., MCTS tree)."""
        self.root = None

    def add_experiences_to_buffer(self, experiences: List[AlphaZeroExperience]):
        """Adds experiences to the replay buffer, splitting between train and val."""
        super().add_experiences_to_buffer(experiences=experiences)


def make_pure_az(
    env: BaseEnvironment,
    config: AlphaZeroConfig,
    training_config: TrainingConfig,
    should_use_network: bool = True,
    network: AlphaZeroNet = None,
):
    if should_use_network:
        params = config.state_model_params
        if not network:
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
            policy_dict, value = network.predict_single(
                state_with_key=node.state_with_key, legal_actions=legal_actions
            )
        network.cache[key] = (policy_dict, value)
    return policy_dict, value
