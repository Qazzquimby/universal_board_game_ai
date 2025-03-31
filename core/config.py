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

    def __post_init__(self):
        # Allow overrides via environment variables for smoke tests
        import os
        self.learning_rate = float(os.getenv("SMOKE_TEST_QL_LR", self.learning_rate))
        self.discount_factor = float(os.getenv("SMOKE_TEST_QL_DISCOUNT", self.discount_factor))
        self.exploration_rate = float(os.getenv("SMOKE_TEST_QL_EXPL_START", self.exploration_rate))
        self.exploration_decay = float(os.getenv("SMOKE_TEST_QL_EXPL_DECAY", self.exploration_decay))
        self.min_exploration = float(os.getenv("SMOKE_TEST_QL_EXPL_MIN", self.min_exploration))


@dataclass
class MCTSConfig:
    exploration_constant: float = 1.41
    discount_factor: float = 1.0 # Discount within the search tree
    num_simulations_short: int = 50
    num_simulations_long: int = 200

    def __post_init__(self):
        # Allow overrides via environment variables for smoke tests
        import os
        self.exploration_constant = float(os.getenv("SMOKE_TEST_MCTS_C", self.exploration_constant))
        self.num_simulations_short = int(os.getenv("SMOKE_TEST_MCTS_SIMS_SHORT", self.num_simulations_short))
        self.num_simulations_long = int(os.getenv("SMOKE_TEST_MCTS_SIMS_LONG", self.num_simulations_long))


@dataclass
class AlphaZeroConfig:
    num_simulations: int = 100 # MCTS simulations per move
    cpuct: float = 1.0 # Exploration constant in PUCT formula
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_layer_size: int = 128 # Size for the MLP hidden layers
    num_hidden_layers: int = 2 # Number of hidden layers in the MLP
    replay_buffer_size: int = 10000 # Max size of the replay buffer
    batch_size: int = 64 # Batch size for training

    def __post_init__(self):
        # Allow overrides via environment variables for smoke tests
        import os
        self.num_simulations = int(os.getenv("SMOKE_TEST_AZ_SIMS", self.num_simulations))
        self.cpuct = float(os.getenv("SMOKE_TEST_AZ_CPUCT", self.cpuct))
        self.learning_rate = float(os.getenv("SMOKE_TEST_AZ_LR", self.learning_rate))
        self.replay_buffer_size = int(os.getenv("SMOKE_TEST_AZ_BUFFER", self.replay_buffer_size))
        self.batch_size = int(os.getenv("SMOKE_TEST_AZ_BATCH", self.batch_size))


# --- Training Configuration ---
@dataclass
class TrainingConfig:
    # Specific to Q-learning training loop in factories.py
    num_episodes: int = 5000
    plot_window: int = 200

    # Specific to AlphaZero training loop in train_alphazero.py
    num_iterations: int = 100 # Number of self-play + learn cycles
    num_episodes_per_iteration: int = 25 # Self-play games per cycle

    def __post_init__(self):
        # Allow overrides via environment variables for smoke tests
        import os
        self.num_episodes = int(os.getenv("SMOKE_TEST_QL_EPISODES", self.num_episodes))
        self.num_iterations = int(os.getenv("SMOKE_TEST_AZ_ITERATIONS", self.num_iterations))
        self.num_episodes_per_iteration = int(os.getenv("SMOKE_TEST_AZ_EPISODES_PER_ITER", self.num_episodes_per_iteration))


# --- Evaluation Configuration ---
@dataclass
class EvaluationConfig:
    num_games: int = 50
    elo_k_factor: int = 64
    elo_iterations: int = 100
    elo_baseline_agent: str = "Random"
    elo_baseline_rating: float = 1000.0

    def __post_init__(self):
        # Allow overrides via environment variables for smoke tests
        import os
        self.num_games = int(os.getenv("SMOKE_TEST_EVAL_GAMES", self.num_games))
        self.elo_iterations = int(os.getenv("SMOKE_TEST_EVAL_ELO_ITER", self.elo_iterations))


# --- Main Application Configuration ---
@dataclass
class AppConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    alpha_zero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig) # Add AlphaZero config
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

# Example usage:
# config = AppConfig()
# config.env.name = "Nim"
# print(config.env)
# print(config.q_learning.learning_rate)
