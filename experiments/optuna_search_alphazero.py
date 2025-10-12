import sys

import optuna
from loguru import logger

from agents.alphazero.alphazero_agent import make_pure_az
from agents.alphazero.alphazero_net import AlphaZeroNet
from core.config import AppConfig
from factories import get_environment


def objective(trial: optuna.Trial):
    """
    Objective function for Optuna to minimize.
    Trains an AlphaZero agent with a given set of hyperparameters and returns the final loss.
    """
    config = AppConfig()
    config.wandb.enabled = False  # Disable wandb for hyperparameter search

    # --- Hyperparameters to Tune ---
    config.alphazero.learning_rate = trial.suggest_loguniform(
        "learning_rate", 1e-5, 1e-2
    )
    config.alphazero.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    config.alphazero.value_loss_weight = trial.suggest_uniform(
        "value_loss_weight", 0.1, 1.0
    )

    # --- State Model Hyperparameters ---
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    if embedding_dim % num_heads != 0:
        # Prune trial if num_heads is not a divisor of embedding_dim
        raise optuna.exceptions.TrialPruned()

    config.alphazero.state_model_params["embedding_dim"] = embedding_dim
    config.alphazero.state_model_params["num_heads"] = num_heads
    config.alphazero.state_model_params["num_encoder_layers"] = trial.suggest_int(
        "num_encoder_layers", 1, 4
    )
    config.alphazero.state_model_params["dropout"] = trial.suggest_uniform(
        "dropout", 0.0, 0.5
    )

    env = get_environment(config.env)

    # --- Create Agent with Tuned Hyperparameters ---
    net = AlphaZeroNet(
        env=env,
        embedding_dim=config.alphazero.state_model_params["embedding_dim"],
        num_heads=config.alphazero.state_model_params["num_heads"],
        num_encoder_layers=config.alphazero.state_model_params["num_encoder_layers"],
        dropout=config.alphazero.state_model_params["dropout"],
    )

    agent = make_pure_az(
        env=env,
        config=config.alphazero,
        training_config=config.training,
        network=net,
    )

    agent.load_game_logs(config.env.name, agent.config.replay_buffer_size)

    # --- Training Loop ---
    num_epochs = 10  # Train for a fixed number of epochs
    final_loss = float("inf")

    logger.info(f"Starting trial {trial.number} with params: {trial.params}")

    for epoch in range(num_epochs):
        metrics = agent.train_network()
        if not metrics:
            logger.warning(f"Trial {trial.number}: training failed, returning inf")
            return float("inf")  # Agent didn't train, e.g. buffer too small

        final_loss = metrics.val.loss
        logger.info(
            f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs} - Loss: {final_loss:.4f}"
        )

        trial.report(final_loss, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"Trial {trial.number} finished with loss: {final_loss:.4f}")
    return final_loss


if __name__ == "__main__":
    config = AppConfig()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    journal_path = "./optuna_journal_storage.log"
    study = optuna.create_study(
        study_name=f"alphazero_tuning_{config.env.name}",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        # storage=f"sqlite:///data/{config.env.name}/optuna_study.db",
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                journal_path,
                lock_obj=optuna.storages.journal.JournalFileOpenLock(journal_path),
            )
        ),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

    logger.info(f"Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(
        f"  Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}"
    )
    logger.info(
        f"  Number of complete trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}"
    )

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
