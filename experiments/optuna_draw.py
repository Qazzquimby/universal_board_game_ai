import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_timeline,
)

from core.config import AppConfig

if __name__ == "__main__":
    config = AppConfig()

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
    # todo reuse

    print(study.best_params)
    opt_history = plot_optimization_history(study)
    param_importance = plot_param_importances(study)
    contour = plot_contour(study)
    timeline = plot_timeline(study)
    print("done")

    # plot_optimization_history(): Shows the objective value over trials.
# plot_intermediate_values(): Visualizes intermediate objective values for trials with pruning.
# plot_param_importances(): Displays the importance of each hyperparameter.
# plot_slice(): Shows the relationship between a hyperparameter and the objective value.
# plot_contour()
