from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# --- Environment Configuration ---
@dataclass
class EnvConfig:
    name: str = "connect4"  # Default environment
    width: int = 7
    height: int = 6
    num_players: int = 2
    max_steps: int = field(init=False)  # Calculated post-init
    nim_piles: List[int] = field(default_factory=lambda: [3, 5, 7])  # Specific to Nim

    def __post_init__(self):
        # Calculate max_steps based on board_size if applicable
        self.max_steps = self.width * self.height + 1


# --- Agent-Specific Configurations ---
@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0  # Initial exploration
    exploration_decay: float = 0.999
    min_exploration: float = 0.01


@dataclass
class MCTSConfig:
    exploration_constant: float = 1.41
    discount_factor: float = 1.0  # Discount within the search tree
    num_simulations: int = 100  # Default simulations for the benchmark MCTS


@dataclass
class AlphaZeroConfig:
    num_simulations: int = 400  # MCTS simulations per move
    cpuct: float = 1.0  # Exploration constant in PUCT formula
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_layer_size: int = 128  # Size for the MLP hidden layers
    num_hidden_layers: int = 2  # Number of hidden layers in the MLP
    replay_buffer_size: int = 10000
    batch_size: int = 64
    # Weight for value loss (default 1.0, try increasing)
    value_loss_weight: float = 2.0
    # Number of game steps before reducing temperature
    temperature_decay_steps: int = 30
    # Parallel Self-Play & Batching
    num_parallel_games: int = 32  # Number of games to run in parallel during self-play
    inference_batch_size: int = 32  # Max batch size for network inference
    # --- Dirichlet Noise for Exploration during Self-Play ---
    dirichlet_alpha: float = 0.3  # Shape parameter for noise (typical value 0.3)
    dirichlet_epsilon: float = 0.25  # Weight of noise vs. priors (typical value 0.25)
    # --- Learning Rate ---
    lr_scheduler_step_size: int = 100  # Decay LR every N training iterations
    lr_scheduler_gamma: float = 0.9  # Multiplicative factor for LR decay
    debug_mode: bool = False
    should_use_network: bool = True


@dataclass
class MuZeroConfig:
    # Inherit/share some params with AlphaZero? Or keep separate? Let's keep separate for now.
    num_simulations: int = 50  # Reduced default for MuZero as it's more complex per sim
    cpuct: float = 1.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_state_size: int = 128  # Size of the latent state in dynamics/prediction
    replay_buffer_size: int = 10000
    batch_size: int = 32  # Smaller batch size might be needed due to unrolling
    num_unroll_steps: int = 5  # Number of game steps to simulate in dynamics (k)
    td_steps: int = 10  # Number of steps for n-step return calculation
    value_loss_weight: float = 0.25  # Weight for value loss component
    reward_loss_weight: float = 1.0  # Weight for reward loss component (often 1.0)
    policy_loss_weight: float = 1.0  # Weight for policy loss component (often 1.0)
    debug_mode: bool = False


# --- Training Configuration ---
@dataclass
class TrainingConfig:
    # Specific to Q-learning training loop in factories.py
    num_episodes: int = 5000
    plot_window: int = 200

    # Specific to AlphaZero/MuZero training loops
    num_iterations: int = 1000  # Total training iterations
    num_games_per_iteration: int = 32
    # Number of epochs (passes over replay buffer) per learning phase
    num_epochs_per_iteration: int = 4  # Increased epochs
    # How often (in iterations) to run sanity checks (0=only at end, 1=every iteration)
    sanity_check_frequency: int = 5  # Run every 5 iterations
    # MCTS Profiling configuration
    enable_mcts_profiling: bool = True
    # How often (in iterations) to report MCTS profiling stats (0=only at end)
    mcts_profiling_report_frequency: int = 10


# --- Evaluation Configuration ---
@dataclass
class EvaluationConfig:
    num_games: int = 50  # Number of games per matchup
    # Elo parameters removed


# --- Main Application Configuration ---
@dataclass
class AppConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    alpha_zero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
    muzero: MuZeroConfig = field(default_factory=MuZeroConfig)  # Add MuZero config
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Flag to indicate if running in smoke test mode (can be set by test runner)
    smoke_test: bool = False


PROJECT_ROOT = Path(__file__).resolve().parents
for parent in PROJECT_ROOT:
    if (parent / ".git").exists():
        PROJECT_ROOT = parent
        break

DATA_DIR = PROJECT_ROOT / "data"

# Example usage:
# config = AppConfig()
# config.env.name = "Nim"
# print(config.env)
# print(config.q_learning.learning_rate)
