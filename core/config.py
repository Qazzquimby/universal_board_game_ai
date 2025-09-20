from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any

import torch
import wandb

GAMES_PER_TRAINING_LOOP = 0  # todo
MCTS_SIMULATIONS = 400

TRAINING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_DEVICE = "cpu"

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


@dataclass
class MCTSConfig:
    exploration_constant: float = 1.41
    discount_factor: float = 1.0  # Discount within the search tree
    num_simulations: int = MCTS_SIMULATIONS


TRAINING_BATCH_SIZE = 1

REPLAY_BUFFER_SIZE = 1


@dataclass
class SomethingZeroConfig:
    num_simulations: int = MCTS_SIMULATIONS  # MCTS simulations per move
    cpuct: float = 1.0  # Exploration constant in PUCT formula
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    value_loss_weight: float = 0.5

    training_batch_size: int = TRAINING_BATCH_SIZE
    replay_buffer_size: int = REPLAY_BUFFER_SIZE

    temperature: float = 0.1

    debug_mode: bool = False


@dataclass
class AlphaZeroConfig(SomethingZeroConfig):
    # Parallel Self-Play & Batching
    num_self_play_workers: int = 2
    inference_batch_size: int = 32  # Max batch size for network inference
    # --- Dirichlet Noise for Exploration during Self-Play ---
    dirichlet_alpha: float = 0.3  # Shape parameter for noise (typical value 0.3)
    dirichlet_epsilon: float = 0.25  # Weight of noise vs. priors (typical value 0.25)

    state_model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "dropout": 0.1,
        }
    )
    policy_model_params: Dict[str, Any] = field(
        default_factory=lambda: {"embedding_dim": 64}
    )

    should_use_network: bool = True


@dataclass
class MuZeroConfig(SomethingZeroConfig):
    num_unroll_steps: int = 5  # 1  # Number of game steps to simulate in dynamics (k)
    td_steps: int = 10  # Number of steps for n-step return calculation
    policy_loss_weight: float = 1.0  # Weight for policy loss component (often 1.0)
    discount_factor: float = 0.99
    state_model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "dropout": 0.1,
        }
    )


# --- Training Configuration ---
@dataclass
class TrainingConfig:
    # Specific to AlphaZero/MuZero training loops
    num_iterations: int = 1000  # Total training iterations
    num_games_per_iteration: int = GAMES_PER_TRAINING_LOOP  # 128 * 2
    # # Number of epochs (passes over replay buffer) per learning phase
    # num_epochs_per_iteration: int = 4
    # How often (in iterations) to run sanity checks (0=only at end, 1=every iteration)
    sanity_check_frequency: int = 0
    # How often (in iterations) to save agent checkpoints (0=only at end)
    save_checkpoint_frequency: int = 2
    # MCTS Profiling configuration
    enable_mcts_profiling: bool = True
    learning_rate: float = 0.001


# --- WandB Configuration ---
@dataclass
class WandBConfig:
    enabled: bool = True
    project_name: str = "board_game_ai"
    entity: str = ""  # Your WandB username or team name (optional)
    run_name: str = (
        "add discount"  # Optional: Set a specific run name, otherwise auto-generated
    )
    log_freq: int = 1  # Log metrics every N iterations
    log_config: bool = True  # Log the entire AppConfig to WandB


# --- Evaluation Configuration ---
@dataclass
class EvaluationConfig:
    full_eval_num_games: int = 40
    run_periodic_evaluation: bool = True
    periodic_eval_frequency: int = 10
    periodic_eval_num_games: int = 40


# --- Main Application Configuration ---
@dataclass
class AppConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    alphazero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
    muzero: MuZeroConfig = field(default_factory=MuZeroConfig)  # Add MuZero config
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    # Flag to indicate if running in smoke test mode (can be set by test runner)
    smoke_test: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary, handling nested dataclasses."""
        return asdict(self)

    def init_wandb(self):
        if not self.wandb.enabled:
            return

        wandb.login(key=WANDB_KEY)
        wandb.init(
            project=self.wandb.project_name,
            entity=self.wandb.entity or None,
            name=self.wandb.run_name or None,
            config=self.to_dict(),
        )


PROJECT_ROOT = Path(__file__).resolve().parents
for parent in PROJECT_ROOT:
    if (parent / ".git").exists():
        PROJECT_ROOT = parent
        break

DATA_DIR = PROJECT_ROOT / "data"

WANDB_KEY = (PROJECT_ROOT / "wandb_key").read_text()

# Example usage:
# config = AppConfig()
# config.env.name = "Nim"
# print(config.env)
# print(config.q_learning.learning_rate)
