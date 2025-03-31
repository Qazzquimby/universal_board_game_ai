from dataclasses import dataclass, field
from typing import List

# --- Environment Configuration ---
@dataclass
class EnvConfig:
    name: str = "FourInARow" # Default environment
    board_size: int = 4 # Specific to FourInARow
    num_players: int = 2
    max_steps: int = field(init=False) # Calculated post-init
    nim_piles: List[int] = field(default_factory=lambda: [3, 5, 7]) # Specific to Nim

    def __post_init__(self):
        # Calculate max_steps based on board_size if applicable
        self.max_steps = self.board_size * self.board_size + 1


# --- Agent-Specific Configurations ---
@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0 # Initial exploration
    exploration_decay: float = 0.999
    min_exploration: float = 0.01


@dataclass
class MCTSConfig:
    exploration_constant: float = 1.41
    discount_factor: float = 1.0 # Discount within the search tree
    num_simulations_short: int = 50
    num_simulations_long: int = 200


@dataclass
class AlphaZeroConfig:
    num_simulations: int = 100 # MCTS simulations per move
    cpuct: float = 1.0 # Exploration constant in PUCT formula
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_layer_size: int = 128 # Size for the MLP hidden layers
    num_hidden_layers: int = 2 # Number of hidden layers in the MLP
    replay_buffer_size: int = 10000
    batch_size: int = 64


# --- Training Configuration ---
@dataclass
class TrainingConfig:
    # Specific to Q-learning training loop in factories.py
    num_episodes: int = 5000
    plot_window: int = 200

    # Specific to AlphaZero training loop in train_alphazero.py
    num_iterations: int = 100
    num_episodes_per_iteration: int = 25


# --- Evaluation Configuration ---
@dataclass
class EvaluationConfig:
    num_games: int = 50
    elo_k_factor: int = 64
    elo_iterations: int = 100
    elo_baseline_agent: str = "Random"
    elo_baseline_rating: float = 1000.0


# --- Main Application Configuration ---
@dataclass
class AppConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    alpha_zero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Flag to indicate if running in smoke test mode (can be set by test runner)
    smoke_test: bool = False

# Example usage:
# config = AppConfig()
# config.env.name = "Nim"
# print(config.env)
# print(config.q_learning.learning_rate)
