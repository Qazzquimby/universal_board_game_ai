# Does not use Reward, only Policy and Value
# Needs to generate a variable length list of action tokens for each hidden state VAE.
# Hidden state VAE is a mean and std for generating specific hidden states for distribution

from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from agents.base_learning_agent import BaseLearningAgent, EpochMetrics
from agents.muzero.muzero_net import MuZeroNet
from algorithms.mcts import (
    SelectionStrategy,
    ExpansionStrategy,
    EvaluationStrategy,
    BackpropagationStrategy,
    UCB1Selection,
    StandardBackpropagation,
    MCTSNode,
    SearchPath,
    SelectionResult,
    Edge,
)
from environments.base import (
    BaseEnvironment,
    ActionType,
    StateWithKey,
    DataFrame,
)
from core.config import MuZeroConfig, TrainingConfig

DUMMY_STATE_WITH_KEY = StateWithKey.from_state(
    {"game": DataFrame(data=[[0]], columns=["dummy"])}
)


class MuZeroExpansion(ExpansionStrategy):
    def __init__(self, network: nn.Module):
        self.network = network

    def expand(self, node: "MCTSNode", env: BaseEnvironment) -> None:
        if node.is_expanded:
            return

        # NOTE: env.get_legal_actions() is only correct for the root node.
        # For nodes deeper in the search tree, the environment state is not
        # updated, and this could lead to incorrect legal move sets.
        # A full MuZero implementation would require a dynamics model that
        # also predicts legal moves or a way to derive them from hidden state.
        legal_actions = env.get_legal_actions()
        policy_dict, _ = self.network.predict(node.hidden_state, legal_actions)

        for action, prior in policy_dict.items():
            action_key = tuple(action) if isinstance(action, list) else action
            node.edges[action_key] = Edge(prior=prior)
        node.is_expanded = True


class MuZeroEvaluation(EvaluationStrategy):
    def __init__(self, network: nn.Module):
        self.network = network

    def evaluate(self, node: "MCTSNode", env: BaseEnvironment) -> float:
        if node.hidden_state is None:
            if env.is_done:
                return env.get_reward_for_player(player=env.get_current_player())
            return 0.0

        _, value = self.network.prediction(node.hidden_state)
        return float(value)


class MuZeroSelection(UCB1Selection):
    def __init__(self, exploration_constant: float, network: nn.Module):
        super().__init__(exploration_constant)
        self.network = network

    def select(
        self,
        node: MCTSNode,
        sim_env: BaseEnvironment,
        cache: "MCTSNodeCache",
        remaining_sims: int,
        contender_actions: Optional[set],
    ) -> SelectionResult:
        path = SearchPath(initial_node=node)
        current_node: MCTSNode = node

        while current_node.is_expanded and current_node.edges:
            best_score = -float("inf")
            best_action: Optional[ActionType] = None

            edges_to_consider = current_node.edges
            if current_node is node and contender_actions is not None:
                edges_to_consider = {
                    action: edge
                    for action, edge in current_node.edges.items()
                    if action in contender_actions
                }

            for action, edge in edges_to_consider.items():
                score = self._score_edge(
                    edge=edge, parent_node_num_visits=current_node.num_visits
                )
                if score > best_score:
                    best_score = score
                    best_action = action

            assert best_action is not None

            edge = current_node.edges[best_action]
            if edge.child_node:
                current_node = edge.child_node
                path.add(current_node, best_action)
            else:
                next_node = MCTSNode(state_with_key=DUMMY_STATE_WITH_KEY)
                (
                    next_hidden_state,
                    reward,
                ) = self.network.dynamics(current_node.hidden_state, best_action)
                next_node.hidden_state = next_hidden_state
                next_node.reward = reward
                next_node.player_idx = 1 - current_node.player_idx
                edge.child_node = next_node
                path.add(next_node, best_action)
                return SelectionResult(path=path, leaf_env=sim_env)

        return SelectionResult(path=path, leaf_env=sim_env)


class MuZeroAgent(BaseLearningAgent):
    """Agent implementing the MuZero algorithm."""

    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        expansion_strategy: ExpansionStrategy,
        evaluation_strategy: EvaluationStrategy,
        backpropagation_strategy: BackpropagationStrategy,
        network: nn.Module,
        optimizer,
        env: BaseEnvironment,
        config: MuZeroConfig,
        training_config: TrainingConfig,
        model_name: str = "muzero",
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

    def search(self, env: BaseEnvironment, train: bool = False):
        if self.root is None:
            state_with_key = env.get_state_with_key()
            self.root = MCTSNode(state_with_key=state_with_key)
            self.root.player_idx = env.get_current_player()
            self.root.hidden_state = self.network.representation(state_with_key)

        for i in range(self.num_simulations):
            sim_env = env.copy()
            selection_result = self.selection_strategy.select(
                self.root, sim_env, self.node_cache, self.num_simulations - i, None
            )

            leaf_node = selection_result.leaf_node
            leaf_env = selection_result.leaf_env

            self._expand_leaf(leaf_node, leaf_env, train)
            value = self.evaluation_strategy.evaluate(leaf_node, leaf_env)
            self.backpropagation_strategy.backpropagate(
                selection_result.path, {0: value, 1: -value}
            )

    def _expand_leaf(self, leaf_node: MCTSNode, leaf_env: BaseEnvironment, train: bool):
        if not leaf_node.is_expanded and not leaf_env.is_done:
            self.expansion_strategy.expand(leaf_node, leaf_env)

    def act(self, env: BaseEnvironment, train: bool = False) -> ActionType:
        self.search(env=env, train=train)
        temperature = self.config.temperature if train else 0.0
        policy_result = self.get_policy_from_visits(temperature)
        return policy_result.chosen_action

    def _calculate_loss(
        self, policy_logits, value_preds, policy_targets, value_targets
    ):
        value_loss = F.mse_loss(value_preds, value_targets.squeeze(-1))
        log_probs = F.log_softmax(policy_logits, dim=1)
        safe_log_probs = torch.where(log_probs == -torch.inf, 0.0, log_probs)
        policy_loss_per_item = -torch.sum(policy_targets * safe_log_probs, dim=1)
        policy_loss = policy_loss_per_item.mean()

        total_loss = (self.config.value_loss_weight * value_loss) + policy_loss
        value_mse = value_loss.item()
        predicted_indices = torch.argmax(policy_logits, dim=1)
        target_indices = torch.argmax(policy_targets, dim=1)
        has_legal_actions = torch.any(policy_logits != -torch.inf, dim=1)
        num_valid_samples = has_legal_actions.sum().item()

        policy_acc = (
            ((predicted_indices == target_indices)[has_legal_actions].sum().item())
            if num_valid_samples > 0
            else 0
        )
        return total_loss, value_loss, policy_loss, policy_acc, value_mse

    def _train_epoch(
        self, train_loader: DataLoader, epoch: int, max_epochs: int
    ) -> "EpochMetrics":
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


def make_pure_muzero(
    env: BaseEnvironment,
    config: MuZeroConfig,
    training_config: TrainingConfig,
):
    params = config.state_model_params
    network = MuZeroNet(
        env=env,
        embedding_dim=params.get("embedding_dim", 64),
        num_heads=params.get("num_heads", 4),
        num_encoder_layers=params.get("num_encoder_layers", 2),
        dropout=params.get("dropout", 0.1),
    )
    optimizer = optim.AdamW(network.parameters(), lr=training_config.learning_rate)

    return MuZeroAgent(
        selection_strategy=MuZeroSelection(
            exploration_constant=config.cpuct, network=network
        ),
        expansion_strategy=MuZeroExpansion(network=network),
        evaluation_strategy=MuZeroEvaluation(network=network),
        backpropagation_strategy=StandardBackpropagation(),
        network=network,
        optimizer=optimizer,
        env=env,
        config=config,
        training_config=training_config,
    )
