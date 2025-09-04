from torch import nn, optim
from torch.utils.data import DataLoader

from agents.base_learning_agent import BaseLearningAgent, EpochMetrics
from agents.muzero.muzero_net import MuZeroNet
from algorithms.mcts import (
    SelectionStrategy,
    ExpansionStrategy,
    EvaluationStrategy,
    BackpropagationStrategy,
    UCB1Selection,
    StandardBackpropagation,
)
from environments.base import BaseEnvironment, ActionType
from core.config import AlphaZeroConfig, TrainingConfig, MuZeroConfig


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
        # TODO: MuZero will need its own MCTS strategies that use the network's
        # dynamics and prediction functions. These will need to be passed to super().__init__.

    def act(self, env: BaseEnvironment, train: bool = False) -> ActionType:
        # TODO: Implement act using MuZero-style MCTS search.
        # The search will be different from AlphaZero's as it operates on hidden states.
        raise NotImplementedError("MuZero act method not implemented.")

    def _train_epoch(
        self, train_loader: DataLoader, epoch: int, max_epochs: int
    ) -> "EpochMetrics":
        # TODO: Implement the training epoch for MuZero, which involves unrolling
        # the dynamics model and calculating losses over the unrolled trajectory.
        raise NotImplementedError("MuZero _train_epoch method not implemented.")

    def _calculate_loss(self, *args, **kwargs):
        # TODO: Implement MuZero's loss function, which includes policy, value,
        # and potentially reward and state consistency losses over multiple steps.
        raise NotImplementedError("MuZero _calculate_loss method not implemented.")


def make_pure_muzero(
    env: BaseEnvironment,
    config: MuZeroConfig,
    training_config: TrainingConfig,
):
    # TODO: Create and return a MuZeroAgent, similar to make_pure_az.
    # This will involve creating a MuZeroNet and MuZero-specific MCTS strategies.

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
        selection_strategy=UCB1Selection(exploration_constant=config.cpuct),
        expansion_strategy=MuZeroExpansion(network=network),
        evaluation_strategy=MuZeroEvaluation(network=network),
        backpropagation_strategy=StandardBackpropagation(),
        network=network,
        optimizer=optimizer,
        env=env,
        config=config,
        training_config=training_config,
    )
