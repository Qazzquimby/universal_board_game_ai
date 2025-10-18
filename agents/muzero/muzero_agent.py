# don't delete.
# Does not use Reward, only Policy and Value
# Hidden state is a Vae that can be sampled from. Need progressive widening.
# Legal actions for hidden states come from a network prediction. Variable length list of encoded action tokens.
#
# In muzero the root node and inner nodes are handled differently.
# The root node uses the actual state and legal actions
# The inner nodes derive both the hidden state and the encoded action tokens.
# Transitioning from a hidden state tensor to the next hidden state vae is done with the encoded action, not the ActionType action.
#
# Note that the root node also needs to use Vae and sampling for hidden info,
# but the root node has a known set of legal actions
import math
import random
from typing import Optional, List, Tuple, Dict, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from geomloss import SamplesLoss
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from agents.base_learning_agent import (
    BaseLearningAgent,
    _get_batched_state,
    GameHistoryStep,
)
from agents.muzero.muzero_net import MuZeroNet, MuZeroNetworkOutput, vae_take_sample
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
class MuZeroTrainingStep:
    """A single step of experience for MuZero training."""

    state: StateType
    policy_target: np.ndarray
    value_target: float
    legal_actions: List[ActionType]
    action: Optional[ActionType] = None
    # Action taken from this step's state. None for the last state in unroll.


@dataclass
class MuZeroExperience:
    """Holds a trajectory of experience for MuZero training."""

    steps: List[MuZeroTrainingStep]


class MuZeroDataset(Dataset):
    def __init__(self, buffer: List[MuZeroExperience]):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        exp = self.buffer[idx]
        states = [step.state for step in exp.steps]
        actions = [step.action for step in exp.steps if step.action is not None]
        policy_targets = [
            torch.tensor(step.policy_target, dtype=torch.float32) for step in exp.steps
        ]
        value_targets = torch.tensor(
            [step.value_target for step in exp.steps], dtype=torch.float32
        )
        legal_actions_per_step = [step.legal_actions for step in exp.steps]
        return (
            states,
            actions,
            policy_targets,
            value_targets,
            legal_actions_per_step,
        )


@dataclass
class MuZeroCollation:
    batched_state: Dict[str, DataFrame]
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    action_batch: torch.Tensor  # tensor rather than ActionType because encoded
    candidate_actions: List[List[List[ActionType]]]
    target_states_batch: List[Dict[str, DataFrame]]


def muzero_collate_fn(batch):
    """Collates a batch of MuZero experiences."""
    (
        states_seqs,
        action_seqs,
        policy_target_seqs,
        value_target_seqs,
        candidate_actions_seqs,
    ) = zip(*batch)

    initial_states = [seq[0] for seq in states_seqs]
    batched_state = _get_batched_state(state_dicts=initial_states)

    target_states_batch = []
    if states_seqs:
        # Longest sequence of states
        max_len = max(len(s) for s in states_seqs)
        # We need states for steps 1...N for hidden state consistency loss
        for i in range(1, max_len):
            states_for_step = [
                seq[i] if i < len(seq) else seq[-1] for seq in states_seqs
            ]
            target_states_batch.append(_get_batched_state(state_dicts=states_for_step))

    max_action_len = max((len(seq) for seq in action_seqs), default=0)
    actions_batch = torch.zeros((len(action_seqs), max_action_len), dtype=torch.long)
    for i, seq in enumerate(action_seqs):
        if seq:
            actions_batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    # policy_target_seqs is a tuple of lists of tensors (policy vectors).
    # First, pad each policy vector in each sequence to the max policy vector length in the batch.
    all_policies = [p for p_seq in policy_target_seqs for p in p_seq]
    if all_policies:
        padded_policy_vectors = torch.nn.utils.rnn.pad_sequence(
            all_policies, batch_first=True, padding_value=0.0
        )

        # Reconstruct the sequences of padded policy vectors.
        seq_lengths = [len(s) for s in policy_target_seqs]
        padded_policy_seqs = []
        current_idx = 0
        for length in seq_lengths:
            padded_policy_seqs.append(
                padded_policy_vectors[current_idx : current_idx + length]
            )
            current_idx += length

        # Pad the sequences of policy vectors to the max sequence length.
        policy_targets_batch = torch.nn.utils.rnn.pad_sequence(
            padded_policy_seqs, batch_first=True, padding_value=0.0
        )
    else:
        # Handle case with no policies (e.g., batch of terminal states).
        policy_targets_batch = torch.empty(len(batch), 0, 0)

    value_targets_batch = torch.nn.utils.rnn.pad_sequence(
        list(value_target_seqs), batch_first=True, padding_value=0.0
    )

    batch_size = policy_targets_batch.shape[0]
    assert (
        value_targets_batch.shape[0]
        == actions_batch.shape[0]
        == len(candidate_actions_seqs)
        == batch_size
    )

    future_unroll_length = len(target_states_batch)
    assert actions_batch.shape[1] == future_unroll_length
    assert (
        policy_targets_batch.shape[1]
        == value_targets_batch.shape[1]
        == future_unroll_length + 1
    )

    return MuZeroCollation(
        batched_state=batched_state,
        policy_targets=policy_targets_batch,
        value_targets=value_targets_batch,
        action_batch=actions_batch,
        candidate_actions=list(candidate_actions_seqs),
        target_states_batch=target_states_batch,
    )


def _calculate_child_limit(num_visits: int) -> int:
    """A simple formula for progressive widening. More visits allow more children."""
    return 1
    # todo disable later

    if num_visits <= 10:
        return 1
    return math.floor(math.log2(num_visits / 10)) + 2


class MuZeroEdge(Edge):
    def __init__(self, prior: float):
        super().__init__(prior)
        # A single edge can lead to multiple outcomes (child nodes) due to stochastic dynamics.
        self.child_nodes: List["MuZeroNode"] = []


class MuZeroNode(MCTSNode):
    """Represents a node in the MCTS tree for MuZero."""

    edges: Dict[Union[ActionType, int], MuZeroEdge]

    def __init__(
        self,
        player_idx: int,
        hidden_state: torch.Tensor,
        state_with_key: Optional[StateWithKey] = None,
    ):
        super().__init__(state_with_key=state_with_key)
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
        if node.state_with_key is not None:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                node.is_expanded = True
                return
            node.action_tokens = [
                self.network._action_to_token(a) for a in legal_actions
            ]
        else:  # If internal node, generate actions from hidden state
            node.action_tokens = self.network.get_actions_for_hidden_state(
                node.hidden_state.unsqueeze(0)
            )[0]
            if not node.action_tokens:
                node.is_expanded = True
                return

        policy_dict = self.network.get_policy(
            hidden_state=node.hidden_state, legal_action_tokens=node.action_tokens
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

        value = self.network.get_value(hidden_state=node.hidden_state)
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
            is_root = current_node.state_with_key is not None
            if is_root:
                action_token = self.network._action_to_token(action).unsqueeze(0)
            else:
                action_token = current_node.action_tokens[action]

            (
                next_hidden_state_mu,
                next_hidden_state_log_var,
            ) = self.network.get_next_hidden_state_vae(
                current_node.hidden_state, action_token
            )
            next_hidden_state = vae_take_sample(
                next_hidden_state_mu, next_hidden_state_log_var
            )

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


@dataclass
class MuZeroLossStatistics:
    batch_loss: torch.Tensor

    total_value_loss: torch.Tensor
    value_losses_per_step: torch.Tensor

    total_policy_loss: torch.Tensor
    policy_losses_per_step: torch.Tensor

    total_hidden_state_loss: torch.Tensor
    hidden_state_losses_per_step: torch.Tensor

    total_action_pred_loss: torch.Tensor
    action_pred_losses_per_step: torch.Tensor


@dataclass
class MuZeroEpochMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    hidden_state_loss: float
    action_pred_loss: float
    policy_loss_by_step: List[float]
    value_loss_by_step: List[float]
    hidden_state_loss_by_step: List[float]
    action_pred_loss_by_step: List[float]

    def __str__(self):
        return (
            f"Total: {self.loss:.3f} "
            f"- Policy: {self.policy_loss:.3f} "
            f"- Value: {self.value_loss:.3f} "
            f"- State: {self.hidden_state_loss:.3f} "
            f"- Action: {self.action_pred_loss:.3f} "
        )


class MuZeroAgent(BaseLearningAgent):
    """Agent implementing the MuZero algorithm."""

    model_type = "muzero"
    config: MuZeroConfig

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

    def search(self, env: BaseEnvironment, train: bool = False):
        # Skip cache when setting root since muzero will never get cache hits
        self.root = MuZeroRootNode(
            player_idx=env.get_current_player(), state_with_key=env.get_state_with_key()
        )

        for i in range(self.num_simulations):
            # Progressive widening at the root.
            child_limit = _calculate_child_limit(self.root.num_visits)
            if len(self.root.child_samples) < child_limit:
                (
                    hidden_state_mu,
                    hidden_state_log_var,
                ) = self.network.get_hidden_state_vae(self.root.state_with_key.state)
                hidden_state = vae_take_sample(hidden_state_mu, hidden_state_log_var)
                new_sample_node = MuZeroNode(
                    player_idx=self.root.player_idx,
                    hidden_state=hidden_state,
                    state_with_key=self.root.state_with_key,
                )
                self.root.child_samples.append(new_sample_node)

            root_sample = random.choice(self.root.child_samples)

            sim_env = env.copy()
            selection_result = self.selection_strategy.select(
                root_sample,
                sim_env,
                self.node_cache,
                self.num_simulations - i,
                None,
            )

            leaf_node = selection_result.leaf_node
            leaf_env = selection_result.leaf_env

            self._expand_leaf(leaf_node, leaf_env, train)
            value = self.evaluation_strategy.evaluate(leaf_node, leaf_env)
            # todo check path that root is included

            self.backpropagation_strategy.backpropagate(
                selection_result.path, {0: value, 1: -value}
            )

        # After simulations, aggregate edges from samples to the root for policy selection.
        self._aggregate_root_edges()

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

    def _create_buffer_experiences(
        self,
        game_history: List[GameHistoryStep],
        value_targets: List[float],
    ) -> List[MuZeroExperience]:
        """Creates MuZeroExperience objects for the replay buffer."""
        muzero_experiences = []
        num_unroll_steps = self.config.num_unroll_steps

        for turn_index in range(len(game_history)):
            steps = []
            for unroll_index in range(num_unroll_steps + 1):
                step_idx = turn_index + unroll_index
                if step_idx >= len(game_history):
                    break

                game_step = game_history[step_idx]
                transformed_state = self.network._apply_transforms(game_step.state)
                value_target = value_targets[step_idx]

                is_last_step = (unroll_index == num_unroll_steps) or (
                    step_idx + 1 >= len(game_history)
                )
                if is_last_step:
                    action_for_step = None
                else:
                    action_for_step = game_step.action

                steps.append(
                    MuZeroTrainingStep(
                        state=transformed_state,
                        policy_target=game_step.policy,
                        value_target=value_target,
                        legal_actions=game_step.legal_actions,
                        action=action_for_step,
                    )
                )

            if steps:
                muzero_experiences.append(MuZeroExperience(steps=steps))
        return muzero_experiences

    def _get_dataset(self, buffer: deque) -> Dataset:
        """Creates a dataset from a replay buffer for MuZero."""
        return MuZeroDataset(list(buffer))

    def _get_collate_fn(self) -> callable:
        """Returns the collate function for the DataLoader for MuZero."""
        return muzero_collate_fn

    def _process_game_log_data(
        self, game_data: List[Dict[str, DataFrame]]
    ) -> List["MuZeroExperience"]:
        """Processes data from a single game log file into a list of experiences."""
        if not game_data:
            return []

        game_history: List[GameHistoryStep] = []
        value_targets: List[float] = []
        for step_data in game_data:
            state_json = step_data.get("state")
            action = step_data.get("action")
            policy_target_list = step_data.get("policy_target")
            value_target = step_data.get("value_target")

            if (
                state_json is not None
                # and action is not None # I believe this was filtering out terminal states
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
                policy_target = np.array(policy_target_list, dtype=np.float32)

                legal_actions_df = state.get("legal_actions")
                legal_actions = []
                if legal_actions_df and not legal_actions_df.is_empty():
                    action_id_idx = legal_actions_df._col_to_idx["action_id"]
                    legal_actions = [
                        row[action_id_idx] for row in legal_actions_df._data
                    ]
                game_history_step = GameHistoryStep(
                    state=state,
                    action=action,
                    policy=policy_target,
                    legal_actions=legal_actions,
                )

                game_history.append(game_history_step)
                value_targets.append(value_target)

        if game_history:
            return self._create_buffer_experiences(game_history, value_targets)
        return []

    def _run_epoch(
        self,
        loader: DataLoader,
        is_training: bool,
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> Optional[MuZeroEpochMetrics]:
        """Runs a single epoch of training or validation for MuZero."""
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0
        total_hidden_state_loss, total_action_pred_loss = 0.0, 0.0
        policy_loss_by_step = defaultdict(float)
        value_loss_by_step = defaultdict(float)
        hidden_state_loss_by_step = defaultdict(float)
        action_pred_loss_by_step = defaultdict(float)
        num_batches = 0

        iterator = loader
        if is_training and not self.config.debug_mode:
            desc = f"Epoch {epoch + 1}/{max_epochs} (Train)"
            iterator = tqdm(loader, desc=desc, leave=False)

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for batch_data in iterator:
                batch_data: MuZeroCollation
                policy_targets_batch = batch_data.policy_targets.to(self.device)
                value_targets_batch = batch_data.value_targets.to(self.device)
                action_batch = batch_data.action_batch.to(self.device)

                if is_training:
                    self.optimizer.zero_grad()

                network_output: MuZeroNetworkOutput = self.network(
                    initial_state=batch_data.batched_state,
                    action_history=action_batch,
                    legal_actions=batch_data.candidate_actions,
                    unrolled_state=batch_data.target_states_batch,
                )
                loss_statistics: MuZeroLossStatistics = self._calculate_loss(
                    network_output=network_output,
                    policy_targets=policy_targets_batch,
                    value_targets=value_targets_batch,
                    candidate_actions=batch_data.candidate_actions,
                )

                if is_training:
                    loss_statistics.batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                total_loss += loss_statistics.batch_loss.item()
                total_value_loss += loss_statistics.total_value_loss.item()
                total_policy_loss += loss_statistics.total_policy_loss.item()
                total_hidden_state_loss += (
                    loss_statistics.total_hidden_state_loss.item()
                )
                total_action_pred_loss += loss_statistics.total_action_pred_loss.item()
                for i, v in enumerate(loss_statistics.policy_losses_per_step):
                    policy_loss_by_step[i] += v.item()
                for i, v in enumerate(loss_statistics.value_losses_per_step):
                    value_loss_by_step[i] += v.item()
                for i, v in enumerate(loss_statistics.hidden_state_losses_per_step):
                    hidden_state_loss_by_step[i] += v.item()
                for i, v in enumerate(loss_statistics.action_pred_losses_per_step):
                    action_pred_loss_by_step[i] += v.item()
                num_batches += 1

        if num_batches == 0:
            return None

        policy_loss_list = [
            v / num_batches for _, v in sorted(policy_loss_by_step.items())
        ]
        value_loss_list = [
            v / num_batches for _, v in sorted(value_loss_by_step.items())
        ]
        hidden_state_loss_list = [
            v / num_batches for _, v in sorted(hidden_state_loss_by_step.items())
        ]
        action_pred_loss_list = [
            v / num_batches for _, v in sorted(action_pred_loss_by_step.items())
        ]
        return MuZeroEpochMetrics(
            loss=total_loss / num_batches,
            policy_loss=total_policy_loss / num_batches,
            value_loss=total_value_loss / num_batches,
            hidden_state_loss=total_hidden_state_loss / num_batches,
            action_pred_loss=total_action_pred_loss / num_batches,
            policy_loss_by_step=policy_loss_list,
            value_loss_by_step=value_loss_list,
            hidden_state_loss_by_step=hidden_state_loss_list,
            action_pred_loss_by_step=action_pred_loss_list,
        )

    def _calculate_action_prediction_loss(
        self,
        pred_actions: torch.Tensor,
        pred_actions_mask: torch.Tensor,
        target_actions: torch.Tensor,
        target_actions_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
        step_losses = []
        device = self.network.get_device()

        if pred_actions.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        batch_size, num_unroll_steps, _, _ = pred_actions.shape

        for step_index in range(num_unroll_steps):
            batch_step_losses = []
            for batch_index in range(batch_size):
                pred_set = pred_actions[batch_index, step_index][
                    pred_actions_mask[batch_index, step_index]
                ]
                target_set = target_actions[batch_index, step_index][
                    target_actions_mask[batch_index, step_index]
                ]

                if target_set.shape[0] == 0:
                    # no actions exist, means state is terminal, and we wouldn't ask it to generate actions anyway
                    continue

                # geomloss' SamplesLoss crashes if one of the sets is empty.
                _pred_set = pred_set
                if _pred_set.shape[0] == 0:
                    _pred_set = torch.zeros(1, pred_actions.shape[-1], device=device)

                loss = loss_fn(_pred_set, target_set)
                batch_step_losses.append(loss)

            if batch_step_losses:
                step_losses.append(torch.stack(batch_step_losses).mean())

        if not step_losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(step_losses)

    def _calculate_loss(
        self,
        network_output: MuZeroNetworkOutput,
        policy_targets: torch.Tensor,
        value_targets: torch.Tensor,
        candidate_actions: List[List[List[ActionType]]],
    ):
        """Calculates the MuZero loss over an unrolled trajectory."""

        policy_losses = self._calculate_policy_loss(
            pred_policies=network_output.pred_policies, policy_targets=policy_targets
        )
        scaled_policy_losses = scale_loss_by_step(policy_losses)
        total_policy_loss = torch.sum(scaled_policy_losses)

        value_losses = self._calculate_value_loss(
            pred_values=network_output.pred_values, value_targets=value_targets
        )
        scaled_value_loss = scale_loss_by_step(value_losses)
        total_value_loss = torch.sum(scaled_value_loss)

        hidden_state_losses = self._calculate_hidden_state_consistency_loss(
            network_output=network_output
        )
        scaled_hidden_state_loss = scale_loss_by_step(hidden_state_losses)
        total_hidden_state_loss = torch.sum(scaled_hidden_state_loss)

        action_pred_losses = self._calculate_action_prediction_loss(
            pred_actions=network_output.pred_actions,
            pred_actions_mask=network_output.pred_actions_mask,
            target_actions=network_output.candidate_action_tokens,
            target_actions_mask=network_output.candidate_action_tokens_mask,
        )
        scaled_action_pred_loss = scale_loss_by_step(action_pred_losses)
        total_action_pred_loss = torch.sum(scaled_action_pred_loss)

        total_loss = (
            total_value_loss
            + total_policy_loss
            + total_hidden_state_loss
            + total_action_pred_loss
        )

        return MuZeroLossStatistics(
            batch_loss=total_loss,
            #
            total_value_loss=total_value_loss,
            value_losses_per_step=value_losses,
            #
            total_policy_loss=total_policy_loss,
            policy_losses_per_step=policy_losses,
            #
            total_hidden_state_loss=total_hidden_state_loss,
            hidden_state_losses_per_step=hidden_state_losses,
            #
            total_action_pred_loss=total_action_pred_loss,
            action_pred_losses_per_step=action_pred_losses,
        )

    def _calculate_value_loss(self, pred_values, value_targets) -> torch.Tensor:
        num_steps = pred_values.shape[1]
        assert value_targets.shape[1] == num_steps

        value_losses_per_step = []

        for i in range(num_steps):
            step_value_preds = pred_values[:, i]
            step_value_targets = value_targets[:, i]

            value_loss = F.mse_loss(step_value_preds, step_value_targets)
            value_losses_per_step.append(value_loss)
        value_losses_tensor = torch.stack(value_losses_per_step)

        return value_losses_tensor

    def _calculate_policy_loss(self, pred_policies, policy_targets) -> torch.Tensor:
        num_steps = pred_policies.shape[1]
        assert policy_targets.shape[1] == num_steps

        policy_losses_per_step = []
        for i in range(num_steps):
            # Policy loss (Cross-Entropy)
            step_policy_logits = pred_policies[:, i, :]
            step_policy_targets = policy_targets[:, i, :]
            log_probs = F.log_softmax(step_policy_logits, dim=1)
            difference = step_policy_targets * log_probs
            difference = torch.nan_to_num(difference, nan=0.0)
            policy_loss = -torch.sum(difference, dim=1).mean()
            policy_losses_per_step.append(policy_loss)
        policy_losses_tensor = torch.stack(policy_losses_per_step)
        return policy_losses_tensor

    def _calculate_hidden_state_consistency_loss(self, network_output):
        if not network_output.pred_dynamics_mu.numel():
            return torch.tensor(0.0, network_output.pred_policies.device)

        loss = wasserstein_distance_loss(
            mu1=network_output.pred_dynamics_mu,
            logvar1=network_output.pred_dynamics_log_var,
            mu2=network_output.target_representation_mu,
            logvar2=network_output.target_representation_log_var,
        )
        return loss


def scale_loss_by_step(loss: torch.Tensor, discount: float = 0.8):
    scales = discount ** torch.arange(
        0, loss.size(0), dtype=loss.dtype, device=loss.device
    )
    scaled = loss * scales
    return scaled


def wasserstein_distance_loss(
    mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor
):
    # W^2(p, q) = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2
    mean_diff_squared = torch.sum((mu1 - mu2).pow(2), dim=2)
    sigma1 = torch.exp(0.5 * logvar1)
    sigma2 = torch.exp(0.5 * logvar2)
    std_diff_squared = torch.sum((sigma1 - sigma2).pow(2), dim=2)
    distance = mean_diff_squared + std_diff_squared
    return torch.mean(distance, dim=0)


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
