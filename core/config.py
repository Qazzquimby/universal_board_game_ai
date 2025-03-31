from dataclasses import dataclass


# Forward declare Agent if needed, or import later if circular dependency is resolved
# from core.agent_interface import Agent


@dataclass
class MainConfig:
    """Configuration settings for the main script."""

    # Environment settings
    board_size: int = 4
    num_players: int = 2
    env_max_steps: int = (
        board_size * board_size + 1
    )  # Max steps slightly more than board size

    # Training settings
    num_episodes_train: int = 5000
    plot_window: int = 200
    ql_save_file: str = "q_agent_{board_size}x{board_size}.pkl"

    # Testing settings
    num_games_test: int = 100
    elo_k_factor: int = 32
    elo_iterations: int = 100
    elo_baseline_agent: str = "Random"
    elo_baseline_rating: float = 1000.0

    # Agent specific configurations can be added here or in separate dataclasses
    mcts_simulations_short: int = 50
    mcts_simulations_long: int = 200

    # Agent registry (mapping names to classes - requires agent classes to be imported)
    # This might be better placed elsewhere if config needs to be loaded before agent classes
    # agent_registry: Dict[str, Type[Agent]] = field(default_factory=dict)

    def __post_init__(self):
        # Dynamically create filename if needed
        self.ql_save_file = self.ql_save_file.format(board_size=self.board_size)


# Example of creating a config instance
# config = MainConfig()
# print(config.ql_save_file)
