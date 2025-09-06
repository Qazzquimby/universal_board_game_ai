# don't delete.
# Does not use Reward, only Policy and Value
# Hidden state is a Vae that can be sampled from. Need progressive widening.
# Legal actions for hidden states come from a network prediction. Variable length list of encoded action tokens.

# In muzero the root node and inner nodes are handled differently.
# The root node uses the actual state and legal actions
# The inner nodes derive both the hidden state and the encoded action tokens.
# Transitioning from a hidden state tensor to the next hidden state vae is done with the encoded action, not the ActionType action.

# Note that the root node also needs to use Vae and sampling for hidden info,
# but the root node has a known set of legal actions

from typing import Optional, List, Tuple, Dict, Union
from collections import defaultdict, deque
from dataclasses import dataclass

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from agents.base_learning_agent import (
    BaseLearningAgent,
    az_collate_fn,
)
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
    MCTSNodeCache,
)
from environments.base import (
    BaseEnvironment,
    ActionType,
    StateWithKey,
    DataFrame,
    StateType,
)
from core.config import MuZeroConfig, TrainingConfig


@dataclass
class MuZeroExperience:
    """Holds a trajectory of experience for MuZero training."""

    # todo, rather than same sized lists better to have trajectory step class?

    initial_state: StateType
    actions: List[ActionType]
    policy_targets: List[np.ndarray]
    value_targets: List[float]


class MuZeroDataset(Dataset):
    def __init__(self, buffer: List[MuZeroExperience]):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        exp = self.buffer[idx]
        policy_targets = [
            torch.tensor(p, dtype=torch.float32) for p in exp.policy_targets
        ]
        value_targets = torch.tensor(exp.value_targets, dtype=torch.float32)
        return exp.initial_state, exp.actions, policy_targets, value_targets


def muzero_collate_fn(batch):
    """Collates a batch of MuZero experiences."""
    initial_states, action_seqs, policy_target_seqs, value_target_seqs = zip(*batch)

    batched_state, _, _, _ = az_collate_fn([(s, [], 0.0, []) for s in initial_states])

    max_action_len = max((len(seq) for seq in action_seqs), default=0)
    actions_batch = torch.zeros((len(action_seqs), max_action_len), dtype=torch.long)
    for i, seq in enumerate(action_seqs):
        if seq:
            actions_batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    all_policies = [p for p_seq in policy_target_seqs for p in p_seq]
    if all_policies:
        padded_policies = torch.nn.utils.rnn.pad_sequence(
            all_policies, batch_first=True, padding_value=0.0
        )
        num_unroll_steps = len(policy_target_seqs[0])
        policy_targets_batch = padded_policies.view(len(batch), num_unroll_steps, -1)
    else:
        policy_targets_batch = torch.empty(len(batch), 0, 0)

    value_targets_batch = torch.stack(list(value_target_seqs), 0)

    # The 'legal_actions' part of the tuple in the base training loop will be our actions_batch.
    return batched_state, policy_targets_batch, value_targets_batch, actions_batch


DUMMY_STATE_WITH_KEY = StateWithKey.from_state(
    {"game": DataFrame(data=[[0]], columns=["dummy"])}
)


def _calculate_child_limit(num_visits: int) -> int:
    """A simple formula for progressive widening. More visits allow more children."""
    # This formula allows for a new child for roughly every 5 visits.
    return 1 + num_visits // 5


class MuZeroEdge(Edge):
    def __init__(self, prior: float):
        super().__init__(prior)
        # A single edge can lead to multiple outcomes (child nodes) due to stochastic dynamics.
        self.child_nodes: List["MuZeroNode"] = []


class MuZeroNode(MCTSNode):
    """Represents a node in the MCTS tree for MuZero."""

    def __init__(
        self,
        player_idx: int,
        hidden_state: torch.Tensor,
        state_with_key: Optional[StateWithKey] = None,
    ):
        # Use a dummy or provided state_with_key for MCTSNode compatibility
        super().__init__(state_with_key or DUMMY_STATE_WITH_KEY)
        self.edges: Dict[Union[ActionType, int], MuZeroEdge]
        self.hidden_state = hidden_state
        self.player_idx = player_idx
        self.action_tokens: Optional[List[torch.Tensor]] = None


class MuZeroRootNode(MCTSNode):
    """A special root node for MuZero that holds samples of hidden states."""

    def __init__(self, player_idx: int, state_with_key: StateWithKey):
        super().__init__(state_with_key)
        self.edges: Dict[Union[ActionType, int], MuZeroEdge]
        self.player_idx = player_idx
        self.child_samples: List[MuZeroNode] = []


class MuZeroExpansion(ExpansionStrategy):
    def __init__(self, network: nn.Module):
        self.network = network

    def expand(self, node: "MuZeroNode", env: BaseEnvironment) -> None:
        if node.is_expanded:
            return

        legal_actions = None
        # If root node, use real environment actions
        if node.state_with_key is not DUMMY_STATE_WITH_KEY:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                node.is_expanded = True
                return
            node.action_tokens = [
                self.network._action_to_token(a) for a in legal_actions
            ]
        else:  # If internal node, generate actions from hidden state
            node.action_tokens = self.network.get_actions_for_hidden_state(
                node.hidden_state
            )
            if not node.action_tokens:
                node.is_expanded = True
                return

        policy_dict, _ = self.network.get_policy_and_value(
            node.hidden_state, node.action_tokens
        )

        for action_idx, prior in policy_dict.items():
            # For root node, key edges by ActionType
            if legal_actions:
                action = legal_actions[action_idx]
                action_key = tuple(action) if isinstance(action, list) else action
            # For internal nodes, key edges by action index
            else:
                action_key = action_idx
            node.edges[action_key] = MuZeroEdge(prior=prior)
        node.is_expanded = True


class MuZeroEvaluation(EvaluationStrategy):
    def __init__(self, network: nn.Module):
        self.network = network

    def evaluate(self, node: "MuZeroNode", env: BaseEnvironment) -> float:
        if node.hidden_state is None:
            if env.is_done:
                return env.get_reward_for_player(player=env.get_current_player())
            return 0.0

        _, value = self.network.get_policy_and_value(node.hidden_state, [])
        return float(value)


class MuZeroSelection(UCB1Selection):
    def __init__(self, exploration_constant: float, network: nn.Module):
        super().__init__(exploration_constant)
        self.network = network

    def _select_action_from_edges(
        self, node: MCTSNode, contender_actions: Optional[set]
    ) -> ActionType:
        """Selects the best action from a node's edges based on UCB score."""
        best_score = -float("inf")
        best_action: Optional[ActionType] = None

        edges_to_consider = node.edges
        # At the root, we might have a restricted set of actions to consider.
        if contender_actions is not None:
            edges_to_consider = {
                action: edge
                for action, edge in node.edges.items()
                if action in contender_actions
            }

        for action, edge in edges_to_consider.items():
            score = self._score_edge(edge=edge, parent_node_num_visits=node.num_visits)
            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None
        return best_action

    def _traverse_or_expand_edge(
        self,
        current_node: "MuZeroNode",
        action: ActionType,
    ) -> Tuple[MuZeroNode, bool]:
        """
        Handles progressive widening for a selected edge.

        Either creates a new child node by sampling the dynamics model (terminating
        the selection phase for this simulation) or traverses to an existing child
        selected by a UCB-like formula.

        Returns:
            A tuple containing:
            - The next MCTSNode in the path.
            - A boolean indicating if the selection phase was terminated (True if a
              new node was created).
        """
        edge: MuZeroEdge = current_node.edges[action]

        child_limit = _calculate_child_limit(edge.num_visits)

        if len(edge.child_nodes) < child_limit:
            # Widen the edge by creating a new child node from a new dynamics sample.
            is_root = current_node.state_with_key is not DUMMY_STATE_WITH_KEY
            if is_root:
                action_token = self.network._action_to_token(action).unsqueeze(0)
            else:
                action_token = current_node.action_tokens[action]

            next_hidden_state_vae = self.network.get_next_hidden_state_vae(
                current_node.hidden_state, action_token
            )
            next_hidden_state = next_hidden_state_vae.take_sample()

            next_player_idx = 1 - current_node.player_idx
            next_node = MuZeroNode(
                player_idx=next_player_idx,
                hidden_state=next_hidden_state,
            )
            edge.child_nodes.append(next_node)
            return next_node, True  # Terminate selection
        else:
            # Select from existing children using a UCB-like formula.
            best_child = None
            best_score = -float("inf")
            for child in edge.child_nodes:
                # Negate Q-value as child's value is from opponent's perspective.
                q_value = (
                    -child.total_value / child.num_visits
                    if child.num_visits > 0
                    else 0.0
                )
                exploration_term = self.exploration_constant * (
                    (edge.num_visits) ** 0.5 / (1 + child.num_visits)
                )
                score = q_value + exploration_term
                if score > best_score:
                    best_score = score
                    best_child = child

            return best_child, False  # Continue selection

    def select(
        self,
        node: MuZeroNode,
        sim_env: BaseEnvironment,
        cache: MCTSNodeCache,
        remaining_sims: int,
        contender_actions: Optional[set],
    ) -> SelectionResult:
        path = SearchPath(initial_node=node)
        current_node: MuZeroNode = node

        while current_node.is_expanded and current_node.edges:
            # Only apply contender_actions at the root of the search.
            contenders = contender_actions if current_node is node else None
            best_action = self._select_action_from_edges(current_node, contenders)

            next_node, terminated = self._traverse_or_expand_edge(
                current_node, best_action
            )

            path.add(next_node, best_action)
            if terminated:
                return SelectionResult(path=path, leaf_env=sim_env)

            current_node = next_node

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
        self.root: Optional["MuZeroRootNode"] = None
        # MuZero requires its own replay buffer to store MuZeroExperience objects.
        val_buffer_size = config.replay_buffer_size // 5
        train_buffer_size = config.replay_buffer_size - val_buffer_size
        self.train_replay_buffer = deque(maxlen=train_buffer_size)
        self.val_replay_buffer = deque(maxlen=val_buffer_size)

    def search(self, env: BaseEnvironment, train: bool = False):
        if self.root is None:
            state_with_key = env.get_state_with_key()
            # The root node will act as a container for different hidden state samples.
            self.root = MuZeroRootNode(
                player_idx=env.get_current_player(), state_with_key=state_with_key
            )

        for i in range(self.num_simulations):
            # Progressive widening at the root.
            child_limit = _calculate_child_limit(self.root.num_visits)
            if len(self.root.child_samples) < child_limit:
                hidden_state_vae = self.network.get_hidden_state_vae(
                    self.root.state_with_key.state
                )
                hidden_state = hidden_state_vae.take_sample()
                new_sample_node = MuZeroNode(
                    player_idx=self.root.player_idx,
                    hidden_state=hidden_state,
                    state_with_key=self.root.state_with_key,
                )
                self.root.child_samples.append(new_sample_node)

            # Select a root sample for this simulation using UCB.
            best_sample_node = self._select_root_sample()

            sim_env = env.copy()
            selection_result = self.selection_strategy.select(
                best_sample_node,
                sim_env,
                self.node_cache,
                self.num_simulations - i,
                None,
            )

            leaf_node = selection_result.leaf_node
            leaf_env = selection_result.leaf_env

            self._expand_leaf(leaf_node, leaf_env, train)
            value = self.evaluation_strategy.evaluate(leaf_node, leaf_env)

            # Backpropagate through the path, including the main root.
            selection_result.path.nodes.insert(0, self.root)
            self.backpropagation_strategy.backpropagate(
                selection_result.path, {0: value, 1: -value}
            )

        # After simulations, aggregate edges from samples to the root for policy selection.
        self._aggregate_root_edges()

    def _select_root_sample(self) -> MuZeroNode:
        """Selects a root sample using a UCB formula."""
        exploration_constant = self.selection_strategy.exploration_constant
        best_sample = None
        best_score = -float("inf")

        for sample_node in self.root.child_samples:
            q_value = (
                sample_node.total_value / sample_node.num_visits
                if sample_node.num_visits > 0
                else 0.0
            )
            exploration_term = exploration_constant * (
                (self.root.num_visits) ** 0.5 / (1 + sample_node.num_visits)
            )
            score = q_value + exploration_term
            if score > best_score:
                best_score = score
                best_sample = sample_node

        return best_sample

    def _aggregate_root_edges(self):
        """Aggregates edges from all root samples into the main root node."""
        self.root.edges = {}
        aggregated_edges = defaultdict(lambda: MuZeroEdge(prior=0.0))

        for sample_node in self.root.child_samples:
            for action, edge in sample_node.edges.items():
                aggregated_edges[action].num_visits += edge.num_visits
                aggregated_edges[action].total_value += edge.total_value

        # We don't set priors correctly here as they are not used after search.
        self.root.edges = dict(aggregated_edges)

    def _expand_leaf(
        self, leaf_node: "MuZeroNode", leaf_env: BaseEnvironment, train: bool
    ):
        if not leaf_node.is_expanded and not leaf_env.is_done:
            self.expansion_strategy.expand(leaf_node, leaf_env)

    def act(self, env: BaseEnvironment, train: bool = False) -> ActionType:
        self.search(env=env, train=train)
        temperature = self.config.temperature if train else 0.0
        policy_result = self.get_policy_from_visits(temperature)
        return policy_result.chosen_action

    def process_finished_episode(
        self,
        game_history: List[Tuple[StateType, ActionType, np.ndarray]],
        final_outcome: float,
    ):
        # todo this looks wrong to me
        # Why are we discounting value? We are tracking end value, not immediate reward.

        experiences = []
        num_unroll_steps = self.config.num_unroll_steps
        for i in range(len(game_history)):
            initial_state_info = game_history[i]
            initial_state = initial_state_info[0]

            actions = []
            policy_targets = [initial_state_info[2]]
            value_targets = []

            # Calculate value targets with discounting
            for j in range(i, len(game_history)):
                player = game_history[j][0]["game"]["current_player"][0]
                outcome = final_outcome if player == 0 else -final_outcome
                value_targets.append(outcome * (self.config.discount_factor ** (j - i)))

            for j in range(i, min(i + num_unroll_steps, len(game_history))):
                actions.append(game_history[j][1])
                # Policy target for the next state
                if j + 1 < len(game_history):
                    policy_targets.append(game_history[j + 1][2])

            experiences.append(
                MuZeroExperience(
                    initial_state=initial_state,
                    actions=actions,
                    policy_targets=policy_targets,
                    value_targets=value_targets,
                )
            )
        self.add_experiences_to_buffer(experiences)

    def add_experiences_to_buffer(self, experiences: List[MuZeroExperience]):
        """Adds MuZero experiences to the replay buffer."""
        super().add_experiences_to_buffer(
            experiences=experiences
        )

    # TODO: The base agent's train_network() and _run_epoch() need to be adapted
    # to use this custom data loader and handle the MuZero batch format.
    # This will likely require overriding them in this class.
    def _get_train_val_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates and returns training and validation data loaders for MuZero."""
        train_ds = MuZeroDataset(list(self.train_replay_buffer))
        val_ds = MuZeroDataset(list(self.val_replay_buffer))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            collate_fn=muzero_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.training_batch_size,
            shuffle=False,
            collate_fn=muzero_collate_fn,
        )
        return train_loader, val_loader

    def _calculate_loss(
        self, policy_logits, value_preds, policy_targets, value_targets
    ):
        """Calculates the MuZero loss over an unrolled trajectory."""
        total_policy_loss = 0
        total_value_loss = 0
        num_steps = policy_logits.shape[1]

        for i in range(num_steps):
            step_policy_logits = policy_logits[:, i, :]
            step_value_preds = value_preds[:, i]
            step_policy_targets = policy_targets[:, i, :]
            step_value_targets = value_targets[:, i]

            # Value loss (MSE)
            value_loss = F.mse_loss(step_value_preds, step_value_targets)

            # Policy loss (Cross-Entropy)
            log_probs = F.log_softmax(step_policy_logits, dim=1)
            policy_loss = -torch.sum(step_policy_targets * log_probs, dim=1).mean()

            total_value_loss += value_loss
            total_policy_loss += policy_loss
            # TODO: Add reward prediction loss and VAE KL-divergence loss if applicable.

        total_loss = total_value_loss + total_policy_loss

        # TODO: Accuracy and MSE metrics need to be re-evaluated for sequential data.
        # Placeholder metrics for now.
        policy_acc = 0.0
        value_mse = (total_value_loss / num_steps).item()

        return (
            total_loss,
            total_value_loss / num_steps,
            total_policy_loss / num_steps,
            policy_acc,
            value_mse,
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
