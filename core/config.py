from pathlib import Path
from typing import Dict, Any

import torch
import wandb
from pydantic import BaseModel, Field

GAMES_PER_TRAINING_LOOP = 1500
MCTS_SIMULATIONS = 400

TRAINING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_DEVICE = "cpu"

# --- Environment Configuration ---
class EnvConfig(BaseModel):
    name: str = "connect4"


class MCTSConfig(BaseModel):
    exploration_constant: float = 1.41
    discount_factor: float = 1.0  # Discount within the search tree
    num_simulations: int = MCTS_SIMULATIONS


TRAINING_BATCH_SIZE = 256

REPLAY_BUFFER_SIZE = 10_000  # 1_000_000


class SomethingZeroConfig(BaseModel):
    num_simulations: int = MCTS_SIMULATIONS  # MCTS simulations per move
    cpuct: float = 1.0  # Exploration constant in PUCT formula
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    value_loss_weight: float = 0.5

    training_batch_size: int = TRAINING_BATCH_SIZE
    replay_buffer_size: int = REPLAY_BUFFER_SIZE

    temperature: float = 0.1

    debug_mode: bool = False


class AlphaZeroConfig(SomethingZeroConfig):
    # Parallel Self-Play & Batching
    num_self_play_workers: int = 2
    inference_batch_size: int = 32  # Max batch size for network inference
    # --- Dirichlet Noise for Exploration during Self-Play ---
    dirichlet_alpha: float = 0.3  # Shape parameter for noise (typical value 0.3)
    dirichlet_epsilon: float = 0.25  # Weight of noise vs. priors (typical value 0.25)

    state_model_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "dropout": 0.1,
        }
    )
    policy_model_params: Dict[str, Any] = Field(
        default_factory=lambda: {"embedding_dim": 64}
    )

    should_use_network: bool = True


class MuZeroConfig(SomethingZeroConfig):
    num_unroll_steps: int = 5  # 1  # Number of game steps to simulate in dynamics (k)
    td_steps: int = 10  # Number of steps for n-step return calculation
    policy_loss_weight: float = 1.0  # Weight for policy loss component (often 1.0)
    discount_factor: float = 0.99
    state_model_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "embedding_dim": 64,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "dropout": 0,  # 0.1, # wasn't converging at 0.1
        }
    )


# --- Training Configuration ---
class TrainingConfig(BaseModel):
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
class WandBConfig(BaseModel):
    enabled: bool = True
    project_name: str = "board_game_ai"
    entity: str = ""  # Your WandB username or team name (optional)
    run_name: str = "muzero_overfit"
    log_freq: int = 1  # Log metrics every N iterations
    log_config: bool = True  # Log the entire AppConfig to WandB


# --- Evaluation Configuration ---
class EvaluationConfig(BaseModel):
    full_eval_num_games: int = 40
    run_periodic_evaluation: bool = True
    periodic_eval_frequency: int = 10
    periodic_eval_num_games: int = 40


# --- Main Application Configuration ---
class AppConfig(BaseModel):
    env: EnvConfig = Field(default_factory=EnvConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    alphazero: AlphaZeroConfig = Field(default_factory=AlphaZeroConfig)
    muzero: MuZeroConfig = Field(default_factory=MuZeroConfig)  # Add MuZero config
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    # Flag to indicate if running in smoke test mode (can be set by test runner)
    smoke_test: bool = False

    def init_wandb(self):
        if not self.wandb.enabled or not WANDB_KEY:
            return

        wandb.login(key=WANDB_KEY)
        wandb.init(
            project=self.wandb.project_name,
            entity=self.wandb.entity or None,
            name=self.wandb.run_name or None,
            config=dict(self),
        )


parents = Path(__file__).resolve().parents
PROJECT_ROOT = None
for parent in parents:
    if (parent / ".git").exists():
        PROJECT_ROOT = parent
        break


DATA_DIR = PROJECT_ROOT / "data"

try:
    WANDB_KEY = (PROJECT_ROOT / "wandb_key").read_text()
except FileNotFoundError:
    WANDB_KEY = None

# Example usage:
# config = AppConfig()
# config.env.name = "Nim"
# print(config.env)
# print(config.q_learning.learning_rate)
