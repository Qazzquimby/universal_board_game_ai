from pathlib import Path

from torch import optim

from agents.alphazero_agent import (
    AlphaZeroAgent,
    AlphaZeroExpansion,
    AlphaZeroEvaluation,
)
from algorithms.mcts import DummyAlphaZeroNet, UCB1Selection, StandardBackpropagation
from environments.base import BaseEnvironment
from core.config import AlphaZeroConfig, TrainingConfig, DATA_DIR
from models.networks import AlphaZeroNet


class MuZeroAgent(AlphaZeroAgent):
    def _get_save_path(self) -> Path:
        env_type_name = type(self.env).__name__
        filename = f"muzero_net_{env_type_name}.pth"
        return DATA_DIR / filename

    def _get_optimizer_save_path(self) -> Path:
        env_type_name = type(self.env).__name__
        filename = f"muzero_optimizer_{env_type_name}.pth"
        return DATA_DIR / filename


def make_pure_muzero(
    env: BaseEnvironment,
    config: AlphaZeroConfig,
    training_config: TrainingConfig,
    should_use_network: bool,
) -> MuZeroAgent:
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

    return MuZeroAgent(
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
