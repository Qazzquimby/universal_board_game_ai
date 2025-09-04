import time
from typing import Optional

import wandb
from loguru import logger

from agents.alphazero.alphazero_agent import AlphaZeroAgent, BestEpochMetrics
from core.config import AppConfig


class TrainingReporter:
    def __init__(self, config: AppConfig, agent: AlphaZeroAgent, start_time: float):
        self.config = config
        self.agent = agent
        self.start_time = start_time
        # if self.config.wandb.enabled:
        #     wandb.login(key=WANDB_KEY)
        #     wandb.init(
        #         project=self.config.wandb.project_name,
        #         entity=self.config.wandb.entity or None,
        #         name=self.config.wandb.run_name or None,
        #         config=self.config.to_dict(),
        #     )

    def log_iteration_start(self, iteration: int):
        logger.info(
            f"\n--- Iteration {iteration + 1}/{self.config.training.num_iterations} ---"
        )

    def log_iteration_end(self, iteration: int, metrics: Optional[BestEpochMetrics]):
        train_buffer_size = len(self.agent.train_replay_buffer)
        val_buffer_size = len(self.agent.val_replay_buffer)
        total_buffer_size = train_buffer_size + val_buffer_size
        logger.info(
            f"Iteration {iteration + 1} complete. "
            f"Buffers: Train={train_buffer_size}/{self.agent.train_replay_buffer.maxlen}, "
            f"Val={val_buffer_size}/{self.agent.val_replay_buffer.maxlen} "
            f"({total_buffer_size}/{self.config.alphazero.replay_buffer_size})"
        )
        if metrics:
            logger.info(
                f"  Latest Losses: Train Total={metrics.train.loss:.4f}, Val Total={metrics.val.loss:.4f}"
            )
            logger.info(
                f"  Latest Accs:   Train Policy={metrics.train.acc:.4f}, Val Policy={metrics.val.acc:.4f}"
            )
        else:
            logger.info("Learning Time: Skipped (buffer too small)")

        if self.config.wandb.enabled:
            log_data = {
                "iteration": iteration + 1,
                "buffer_size_total": total_buffer_size,
                "buffer_size_train": train_buffer_size,
                "buffer_size_val": val_buffer_size,
                "wall_clock_time_s": time.time() - self.start_time,
            }
            if metrics:
                wandb_metrics = {
                    f"learn/train_{k}": v for k, v in metrics.train.__dict__.items()
                }
                wandb_metrics.update(
                    {f"learn/val_{k}": v for k, v in metrics.val.__dict__.items()}
                )
                log_data.update(wandb_metrics)
            try:
                wandb.log(log_data)
            except Exception as e:
                logger.warning(f"Failed to log metrics to WandB: {e}")

    def log_evaluation_results(
        self, eval_results: dict, benchmark_agent_name: str, iteration: int
    ):
        if self.config.wandb.enabled:
            wandb_eval_log = {
                f"eval_vs_{benchmark_agent_name}/win_rate": eval_results.get(
                    "AlphaZero_win_rate", 0.0
                ),
                f"eval_vs_{benchmark_agent_name}/loss_rate": eval_results.get(
                    f"{benchmark_agent_name}_win_rate", 0.0
                ),
                f"eval_vs_{benchmark_agent_name}/draw_rate": eval_results.get(
                    "draw_rate", 0.0
                ),
                "iteration": iteration + 1,
            }
            try:
                wandb.log(wandb_eval_log)
            except Exception as e:
                print("Couldn't log to wandb")
                pass
            logger.info(f"Logged periodic evaluation results to WandB.")

    def finish(self):
        if self.config.wandb.enabled and wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished.")
