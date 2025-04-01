### MUZERO

# import copy
# import importlib
# import json
# import math
# import pathlib
# import pickle
# import sys
# import time
#
# import nevergrad
# import numpy
# import ray
# import torch
# from torch.utils.tensorboard import SummaryWriter
#
# import diagnose_model
# import models
# import replay_buffer
# import self_play
# import shared_storage
# import trainer
#
#
# class MuZero:
#     """
#     Main class to manage MuZero.
#
#     Args:
#         game_name (str): Name of the game module, it should match the name of a .py file
#         in the "./games" directory.
#
#         config (dict, MuZeroConfig, optional): Override the default config of the game.
#
#         split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.
#
#     Example:
#         >>> muzero = MuZero("cartpole")
#         >>> muzero.train()
#         >>> muzero.test(render=True)
#     """
#
#     def __init__(self, game_name, config=None, split_resources_in=1):
#         # Load the game and the config from the module with the game name
#         try:
#             game_module = importlib.import_module("games." + game_name)
#             self.Game = game_module.Game
#             self.config = game_module.MuZeroConfig()
#         except ModuleNotFoundError as err:
#             print(
#                 f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
#             )
#             raise err
#
#         # Overwrite the config
#         if config:
#             if type(config) is dict:
#                 for param, value in config.items():
#                     if hasattr(self.config, param):
#                         setattr(self.config, param, value)
#                     else:
#                         raise AttributeError(
#                             f"{game_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
#                         )
#             else:
#                 self.config = config
#
#         # Fix random generator seed
#         numpy.random.seed(self.config.seed)
#         torch.manual_seed(self.config.seed)
#
#         # Manage GPUs
#         if self.config.max_num_gpus == 0 and (
#             self.config.selfplay_on_gpu
#             or self.config.train_on_gpu
#             or self.config.reanalyse_on_gpu
#         ):
#             raise ValueError(
#                 "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
#             )
#         if (
#             self.config.selfplay_on_gpu
#             or self.config.train_on_gpu
#             or self.config.reanalyse_on_gpu
#         ):
#             total_gpus = (
#                 self.config.max_num_gpus
#                 if self.config.max_num_gpus is not None
#                 else torch.cuda.device_count()
#             )
#         else:
#             total_gpus = 0
#         self.num_gpus = total_gpus / split_resources_in
#         if 1 < self.num_gpus:
#             self.num_gpus = math.floor(self.num_gpus)
#
#         ray.init(num_gpus=total_gpus, ignore_reinit_error=True)
#
#         # Checkpoint and replay buffer used to initialize workers
#         self.checkpoint = {
#             "weights": None,
#             "optimizer_state": None,
#             "total_reward": 0,
#             "muzero_reward": 0,
#             "opponent_reward": 0,
#             "episode_length": 0,
#             "mean_value": 0,
#             "training_step": 0,
#             "lr": 0,
#             "total_loss": 0,
#             "value_loss": 0,
#             "reward_loss": 0,
#             "policy_loss": 0,
#             "num_played_games": 0,
#             "num_played_steps": 0,
#             "num_reanalysed_games": 0,
#             "terminate": False,
#         }
#         self.replay_buffer = {}
#
#         cpu_actor = CPUActor.remote()
#         cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
#         self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))
#
#         # Workers
#         self.self_play_workers = None
#         self.test_worker = None
#         self.training_worker = None
#         self.reanalyse_worker = None
#         self.replay_buffer_worker = None
#         self.shared_storage_worker = None
#
#     def train(self, log_in_tensorboard=True):
#         """
#         Spawn ray workers and launch the training.
#
#         Args:
#             log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
#         """
#         if log_in_tensorboard or self.config.save_model:
#             self.config.results_path.mkdir(parents=True, exist_ok=True)
#
#         # Manage GPUs
#         if 0 < self.num_gpus:
#             num_gpus_per_worker = self.num_gpus / (
#                 self.config.train_on_gpu
#                 + self.config.num_workers * self.config.selfplay_on_gpu
#                 + log_in_tensorboard * self.config.selfplay_on_gpu
#                 + self.config.use_last_model_value * self.config.reanalyse_on_gpu
#             )
#             if 1 < num_gpus_per_worker:
#                 num_gpus_per_worker = math.floor(num_gpus_per_worker)
#         else:
#             num_gpus_per_worker = 0
#
#         # Initialize workers
#         self.training_worker = trainer.Trainer.options(
#             num_cpus=0,
#             num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
#         ).remote(self.checkpoint, self.config)
#
#         self.shared_storage_worker = shared_storage.SharedStorage.remote(
#             self.checkpoint,
#             self.config,
#         )
#         self.shared_storage_worker.set_info.remote("terminate", False)
#
#         self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
#             self.checkpoint, self.replay_buffer, self.config
#         )
#
#         if self.config.use_last_model_value:
#             self.reanalyse_worker = replay_buffer.Reanalyse.options(
#                 num_cpus=0,
#                 num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
#             ).remote(self.checkpoint, self.config)
#
#         self.self_play_workers = [
#             self_play.SelfPlay.options(
#                 num_cpus=0,
#                 num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
#             ).remote(
#                 self.checkpoint,
#                 self.Game,
#                 self.config,
#                 self.config.seed + seed,
#             )
#             for seed in range(self.config.num_workers)
#         ]
#
#         # Launch workers
#         [
#             self_play_worker.continuous_self_play.remote(
#                 self.shared_storage_worker, self.replay_buffer_worker
#             )
#             for self_play_worker in self.self_play_workers
#         ]
#         self.training_worker.continuous_update_weights.remote(
#             self.replay_buffer_worker, self.shared_storage_worker
#         )
#         if self.config.use_last_model_value:
#             self.reanalyse_worker.reanalyse.remote(
#                 self.replay_buffer_worker, self.shared_storage_worker
#             )
#
#         if log_in_tensorboard:
#             self.logging_loop(
#                 num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
#             )
#
#     def logging_loop(self, num_gpus):
#         """
#         Keep track of the training performance.
#         """
#         # Launch the test worker to get performance metrics
#         self.test_worker = self_play.SelfPlay.options(
#             num_cpus=0,
#             num_gpus=num_gpus,
#         ).remote(
#             self.checkpoint,
#             self.Game,
#             self.config,
#             self.config.seed + self.config.num_workers,
#         )
#         self.test_worker.continuous_self_play.remote(
#             self.shared_storage_worker, None, True
#         )
#
#         # Write everything in TensorBoard
#         writer = SummaryWriter(self.config.results_path)
#
#         print(
#             "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
#         )
#
#         # Save hyperparameters to TensorBoard
#         hp_table = [
#             f"| {key} | {value} |" for key, value in self.config.__dict__.items()
#         ]
#         writer.add_text(
#             "Hyperparameters",
#             "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
#         )
#         # Save model representation
#         writer.add_text(
#             "Model summary",
#             self.summary,
#         )
#         # Loop for updating the training performance
#         counter = 0
#         keys = [
#             "total_reward",
#             "muzero_reward",
#             "opponent_reward",
#             "episode_length",
#             "mean_value",
#             "training_step",
#             "lr",
#             "total_loss",
#             "value_loss",
#             "reward_loss",
#             "policy_loss",
#             "num_played_games",
#             "num_played_steps",
#             "num_reanalysed_games",
#         ]
#         info = ray.get(self.shared_storage_worker.get_info.remote(keys))
#         try:
#             while info["training_step"] < self.config.training_steps:
#                 info = ray.get(self.shared_storage_worker.get_info.remote(keys))
#                 writer.add_scalar(
#                     "1.Total_reward/1.Total_reward",
#                     info["total_reward"],
#                     counter,
#                 )
#                 writer.add_scalar(
#                     "1.Total_reward/2.Mean_value",
#                     info["mean_value"],
#                     counter,
#                 )
#                 writer.add_scalar(
#                     "1.Total_reward/3.Episode_length",
#                     info["episode_length"],
#                     counter,
#                 )
#                 writer.add_scalar(
#                     "1.Total_reward/4.MuZero_reward",
#                     info["muzero_reward"],
#                     counter,
#                 )
#                 writer.add_scalar(
#                     "1.Total_reward/5.Opponent_reward",
#                     info["opponent_reward"],
#                     counter,
#                 )
#                 writer.add_scalar(
#                     "2.Workers/1.Self_played_games",
#                     info["num_played_games"],
#                     counter,
#                 )
#                 writer.add_scalar(
#                     "2.Workers/2.Training_steps", info["training_step"], counter
#                 )
#                 writer.add_scalar(
#                     "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
#                 )
#                 writer.add_scalar(
#                     "2.Workers/4.Reanalysed_games",
#                     info["num_reanalysed_games"],
#                     counter,
#                 )
#                 writer.add_scalar(
#                     "2.Workers/5.Training_steps_per_self_played_step_ratio",
#                     info["training_step"] / max(1, info["num_played_steps"]),
#                     counter,
#                 )
#                 writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
#                 writer.add_scalar(
#                     "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
#                 )
#                 writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
#                 writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
#                 writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
#                 print(
#                     f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
#                     end="\r",
#                 )
#                 counter += 1
#                 time.sleep(0.5)
#         except KeyboardInterrupt:
#             pass
#
#         self.terminate_workers()
#
#         if self.config.save_model:
#             # Persist replay buffer to disk
#             path = self.config.results_path / "replay_buffer.pkl"
#             print(f"\n\nPersisting replay buffer games to disk at {path}")
#             pickle.dump(
#                 {
#                     "buffer": self.replay_buffer,
#                     "num_played_games": self.checkpoint["num_played_games"],
#                     "num_played_steps": self.checkpoint["num_played_steps"],
#                     "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
#                 },
#                 open(path, "wb"),
#             )
#
#     def terminate_workers(self):
#         """
#         Softly terminate the running tasks and garbage collect the workers.
#         """
#         if self.shared_storage_worker:
#             self.shared_storage_worker.set_info.remote("terminate", True)
#             self.checkpoint = ray.get(
#                 self.shared_storage_worker.get_checkpoint.remote()
#             )
#         if self.replay_buffer_worker:
#             self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())
#
#         print("\nShutting down workers...")
#
#         self.self_play_workers = None
#         self.test_worker = None
#         self.training_worker = None
#         self.reanalyse_worker = None
#         self.replay_buffer_worker = None
#         self.shared_storage_worker = None
#
#     def test(
#         self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
#     ):
#         """
#         Test the model in a dedicated thread.
#
#         Args:
#             render (bool): To display or not the environment. Defaults to True.
#
#             opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
#             for a random agent, None will use the opponent in the config. Defaults to None.
#
#             muzero_player (int): Player number of MuZero in case of multiplayer
#             games, None let MuZero play all players turn by turn, None will use muzero_player in
#             the config. Defaults to None.
#
#             num_tests (int): Number of games to average. Defaults to 1.
#
#             num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
#         """
#         opponent = opponent if opponent else self.config.opponent
#         muzero_player = muzero_player if muzero_player else self.config.muzero_player
#         self_play_worker = self_play.SelfPlay.options(
#             num_cpus=0,
#             num_gpus=num_gpus,
#         ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
#         results = []
#         for i in range(num_tests):
#             print(f"Testing {i+1}/{num_tests}")
#             results.append(
#                 ray.get(
#                     self_play_worker.play_game.remote(
#                         0,
#                         0,
#                         render,
#                         opponent,
#                         muzero_player,
#                     )
#                 )
#             )
#         self_play_worker.close_game.remote()
#
#         if len(self.config.players) == 1:
#             result = numpy.mean([sum(history.reward_history) for history in results])
#         else:
#             result = numpy.mean(
#                 [
#                     sum(
#                         reward
#                         for i, reward in enumerate(history.reward_history)
#                         if history.to_play_history[i - 1] == muzero_player
#                     )
#                     for history in results
#                 ]
#             )
#         return result
#
#     def load_model(self, checkpoint_path=None, replay_buffer_path=None):
#         """
#         Load a model and/or a saved replay buffer.
#
#         Args:
#             checkpoint_path (str): Path to model.checkpoint or model.weights.
#
#             replay_buffer_path (str): Path to replay_buffer.pkl
#         """
#         # Load checkpoint
#         if checkpoint_path:
#             checkpoint_path = pathlib.Path(checkpoint_path)
#             self.checkpoint = torch.load(checkpoint_path)
#             print(f"\nUsing checkpoint from {checkpoint_path}")
#
#         # Load replay buffer
#         if replay_buffer_path:
#             replay_buffer_path = pathlib.Path(replay_buffer_path)
#             with open(replay_buffer_path, "rb") as f:
#                 replay_buffer_infos = pickle.load(f)
#             self.replay_buffer = replay_buffer_infos["buffer"]
#             self.checkpoint["num_played_steps"] = replay_buffer_infos[
#                 "num_played_steps"
#             ]
#             self.checkpoint["num_played_games"] = replay_buffer_infos[
#                 "num_played_games"
#             ]
#             self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
#                 "num_reanalysed_games"
#             ]
#
#             print(f"\nInitializing replay buffer with {replay_buffer_path}")
#         else:
#             print(f"Using empty buffer.")
#             self.replay_buffer = {}
#             self.checkpoint["training_step"] = 0
#             self.checkpoint["num_played_steps"] = 0
#             self.checkpoint["num_played_games"] = 0
#             self.checkpoint["num_reanalysed_games"] = 0
#
#     def diagnose_model(self, horizon):
#         """
#         Play a game only with the learned model then play the same trajectory in the real
#         environment and display information.
#
#         Args:
#             horizon (int): Number of timesteps for which we collect information.
#         """
#         game = self.Game(self.config.seed)
#         obs = game.reset()
#         dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
#         dm.compare_virtual_with_real_trajectories(obs, game, horizon)
#         input("Press enter to close all plots")
#         dm.close_all()
#
#
# @ray.remote(num_cpus=0, num_gpus=0)
# class CPUActor:
#     # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
#     def __init__(self):
#         pass
#
#     def get_initial_weights(self, config):
#         model = models.MuZeroNetwork(config)
#         weigths = model.get_weights()
#         summary = str(model).replace("\n", " \n\n")
#         return weigths, summary
#
#
# def hyperparameter_search(
#     game_name, parametrization, budget, parallel_experiments, num_tests
# ):
#     """
#     Search for hyperparameters by launching parallel experiments.
#
#     Args:
#         game_name (str): Name of the game module, it should match the name of a .py file
#         in the "./games" directory.
#
#         parametrization : Nevergrad parametrization, please refer to nevergrad documentation.
#
#         budget (int): Number of experiments to launch in total.
#
#         parallel_experiments (int): Number of experiments to launch in parallel.
#
#         num_tests (int): Number of games to average for evaluating an experiment.
#     """
#     optimizer = nevergrad.optimizers.OnePlusOne(
#         parametrization=parametrization, budget=budget
#     )
#
#     running_experiments = []
#     best_training = None
#     try:
#         # Launch initial experiments
#         for i in range(parallel_experiments):
#             if 0 < budget:
#                 param = optimizer.ask()
#                 print(f"Launching new experiment: {param.value}")
#                 muzero = MuZero(game_name, param.value, parallel_experiments)
#                 muzero.param = param
#                 muzero.train(False)
#                 running_experiments.append(muzero)
#                 budget -= 1
#
#         while 0 < budget or any(running_experiments):
#             for i, experiment in enumerate(running_experiments):
#                 if experiment and experiment.config.training_steps <= ray.get(
#                     experiment.shared_storage_worker.get_info.remote("training_step")
#                 ):
#                     experiment.terminate_workers()
#                     result = experiment.test(False, num_tests=num_tests)
#                     if not best_training or best_training["result"] < result:
#                         best_training = {
#                             "result": result,
#                             "config": experiment.config,
#                             "checkpoint": experiment.checkpoint,
#                         }
#                     print(f"Parameters: {experiment.param.value}")
#                     print(f"Result: {result}")
#                     optimizer.tell(experiment.param, -result)
#
#                     if 0 < budget:
#                         param = optimizer.ask()
#                         print(f"Launching new experiment: {param.value}")
#                         muzero = MuZero(game_name, param.value, parallel_experiments)
#                         muzero.param = param
#                         muzero.train(False)
#                         running_experiments[i] = muzero
#                         budget -= 1
#                     else:
#                         running_experiments[i] = None
#
#     except KeyboardInterrupt:
#         for experiment in running_experiments:
#             if isinstance(experiment, MuZero):
#                 experiment.terminate_workers()
#
#     recommendation = optimizer.provide_recommendation()
#     print("Best hyperparameters:")
#     print(recommendation.value)
#     if best_training:
#         # Save best training weights (but it's not the recommended weights)
#         best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
#         torch.save(
#             best_training["checkpoint"],
#             best_training["config"].results_path / "model.checkpoint",
#         )
#         # Save the recommended hyperparameters
#         text_file = open(
#             best_training["config"].results_path / "best_parameters.txt",
#             "w",
#         )
#         text_file.write(str(recommendation.value))
#         text_file.close()
#     return recommendation.value
#
#
# def load_model_menu(muzero, game_name):
#     # Configure running options
#     options = ["Specify paths manually"] + sorted(
#         (pathlib.Path("results") / game_name).glob("*/")
#     )
#     options.reverse()
#     print()
#     for i in range(len(options)):
#         print(f"{i}. {options[i]}")
#
#     choice = input("Enter a number to choose a model to load: ")
#     valid_inputs = [str(i) for i in range(len(options))]
#     while choice not in valid_inputs:
#         choice = input("Invalid input, enter a number listed above: ")
#     choice = int(choice)
#
#     if choice == (len(options) - 1):
#         # manual path option
#         checkpoint_path = input(
#             "Enter a path to the model.checkpoint, or ENTER if none: "
#         )
#         while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
#             checkpoint_path = input("Invalid checkpoint path. Try again: ")
#         replay_buffer_path = input(
#             "Enter a path to the replay_buffer.pkl, or ENTER if none: "
#         )
#         while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
#             replay_buffer_path = input("Invalid replay buffer path. Try again: ")
#     else:
#         checkpoint_path = options[choice] / "model.checkpoint"
#         replay_buffer_path = options[choice] / "replay_buffer.pkl"
#
#     muzero.load_model(
#         checkpoint_path=checkpoint_path,
#         replay_buffer_path=replay_buffer_path,
#     )
#
#
# if __name__ == "__main__":
#     if len(sys.argv) == 2:
#         # Train directly with: python muzero.py cartpole
#         muzero = MuZero(sys.argv[1])
#         muzero.train()
#     elif len(sys.argv) == 3:
#         # Train directly with: python muzero.py cartpole '{"lr_init": 0.01}'
#         config = json.loads(sys.argv[2])
#         muzero = MuZero(sys.argv[1], config)
#         muzero.train()
#     else:
#         print("\nWelcome to MuZero! Here's a list of games:")
#         # Let user pick a game
#         games = [
#             filename.stem
#             for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
#             if filename.name != "abstract_game.py"
#         ]
#         for i in range(len(games)):
#             print(f"{i}. {games[i]}")
#         choice = input("Enter a number to choose the game: ")
#         valid_inputs = [str(i) for i in range(len(games))]
#         while choice not in valid_inputs:
#             choice = input("Invalid input, enter a number listed above: ")
#
#         # Initialize MuZero
#         choice = int(choice)
#         game_name = games[choice]
#         muzero = MuZero(game_name)
#
#         while True:
#             # Configure running options
#             options = [
#                 "Train",
#                 "Load pretrained model",
#                 "Diagnose model",
#                 "Render some self play games",
#                 "Play against MuZero",
#                 "Test the game manually",
#                 "Hyperparameter search",
#                 "Exit",
#             ]
#             print()
#             for i in range(len(options)):
#                 print(f"{i}. {options[i]}")
#
#             choice = input("Enter a number to choose an action: ")
#             valid_inputs = [str(i) for i in range(len(options))]
#             while choice not in valid_inputs:
#                 choice = input("Invalid input, enter a number listed above: ")
#             choice = int(choice)
#             if choice == 0:
#                 muzero.train()
#             elif choice == 1:
#                 load_model_menu(muzero, game_name)
#             elif choice == 2:
#                 muzero.diagnose_model(30)
#             elif choice == 3:
#                 muzero.test(render=True, opponent="self", muzero_player=None)
#             elif choice == 4:
#                 muzero.test(render=True, opponent="human", muzero_player=0)
#             elif choice == 5:
#                 env = muzero.Game()
#                 env.reset()
#                 env.render()
#
#                 done = False
#                 while not done:
#                     action = env.human_to_action()
#                     observation, reward, done = env.step(action)
#                     print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
#                     env.render()
#             elif choice == 6:
#                 # Define here the parameters to tune
#                 # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
#                 muzero.terminate_workers()
#                 del muzero
#                 budget = 20
#                 parallel_experiments = 2
#                 lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
#                 discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
#                 parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
#                 best_hyperparameters = hyperparameter_search(
#                     game_name, parametrization, budget, parallel_experiments, 20
#                 )
#                 muzero = MuZero(game_name, best_hyperparameters)
#             else:
#                 break
#             print("\nDone")
#
#     ray.shutdown()

### MODELS

# import math
# from abc import ABC, abstractmethod
#
# import torch
#
#
# class MuZeroNetwork:
#     def __new__(cls, config):
#         if config.network == "fullyconnected":
#             return MuZeroFullyConnectedNetwork(
#                 config.observation_shape,
#                 config.stacked_observations,
#                 len(config.action_space),
#                 config.encoding_size,
#                 config.fc_reward_layers,
#                 config.fc_value_layers,
#                 config.fc_policy_layers,
#                 config.fc_representation_layers,
#                 config.fc_dynamics_layers,
#                 config.support_size,
#             )
#         elif config.network == "resnet":
#             return MuZeroResidualNetwork(
#                 config.observation_shape,
#                 config.stacked_observations,
#                 len(config.action_space),
#                 config.blocks,
#                 config.channels,
#                 config.reduced_channels_reward,
#                 config.reduced_channels_value,
#                 config.reduced_channels_policy,
#                 config.resnet_fc_reward_layers,
#                 config.resnet_fc_value_layers,
#                 config.resnet_fc_policy_layers,
#                 config.support_size,
#                 config.downsample,
#             )
#         else:
#             raise NotImplementedError(
#                 'The network parameter should be "fullyconnected" or "resnet".'
#             )
#
#
# def dict_to_cpu(dictionary):
#     cpu_dict = {}
#     for key, value in dictionary.items():
#         if isinstance(value, torch.Tensor):
#             cpu_dict[key] = value.cpu()
#         elif isinstance(value, dict):
#             cpu_dict[key] = dict_to_cpu(value)
#         else:
#             cpu_dict[key] = value
#     return cpu_dict
#
#
# class AbstractNetwork(ABC, torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass
#
#     @abstractmethod
#     def initial_inference(self, observation):
#         pass
#
#     @abstractmethod
#     def recurrent_inference(self, encoded_state, action):
#         pass
#
#     def get_weights(self):
#         return dict_to_cpu(self.state_dict())
#
#     def set_weights(self, weights):
#         self.load_state_dict(weights)
#
#
# ##################################
# ######## Fully Connected #########
#
#
# class MuZeroFullyConnectedNetwork(AbstractNetwork):
#     def __init__(
#         self,
#         observation_shape,
#         stacked_observations,
#         action_space_size,
#         encoding_size,
#         fc_reward_layers,
#         fc_value_layers,
#         fc_policy_layers,
#         fc_representation_layers,
#         fc_dynamics_layers,
#         support_size,
#     ):
#         super().__init__()
#         self.action_space_size = action_space_size
#         self.full_support_size = 2 * support_size + 1
#
#         self.representation_network = torch.nn.DataParallel(
#             mlp(
#                 observation_shape[0]
#                 * observation_shape[1]
#                 * observation_shape[2]
#                 * (stacked_observations + 1)
#                 + stacked_observations * observation_shape[1] * observation_shape[2],
#                 fc_representation_layers,
#                 encoding_size,
#             )
#         )
#
#         self.dynamics_encoded_state_network = torch.nn.DataParallel(
#             mlp(
#                 encoding_size + self.action_space_size,
#                 fc_dynamics_layers,
#                 encoding_size,
#             )
#         )
#         self.dynamics_reward_network = torch.nn.DataParallel(
#             mlp(encoding_size, fc_reward_layers, self.full_support_size)
#         )
#
#         self.prediction_policy_network = torch.nn.DataParallel(
#             mlp(encoding_size, fc_policy_layers, self.action_space_size)
#         )
#         self.prediction_value_network = torch.nn.DataParallel(
#             mlp(encoding_size, fc_value_layers, self.full_support_size)
#         )
#
#     def prediction(self, encoded_state):
#         policy_logits = self.prediction_policy_network(encoded_state)
#         value = self.prediction_value_network(encoded_state)
#         return policy_logits, value
#
#     def representation(self, observation):
#         encoded_state = self.representation_network(
#             observation.view(observation.shape[0], -1)
#         )
#         # Scale encoded state between [0, 1] (See appendix paper Training)
#         min_encoded_state = encoded_state.min(1, keepdim=True)[0]
#         max_encoded_state = encoded_state.max(1, keepdim=True)[0]
#         scale_encoded_state = max_encoded_state - min_encoded_state
#         scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
#         encoded_state_normalized = (
#             encoded_state - min_encoded_state
#         ) / scale_encoded_state
#         return encoded_state_normalized
#
#     def dynamics(self, encoded_state, action):
#         # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
#         action_one_hot = (
#             torch.zeros((action.shape[0], self.action_space_size))
#             .to(action.device)
#             .float()
#         )
#         action_one_hot.scatter_(1, action.long(), 1.0)
#         x = torch.cat((encoded_state, action_one_hot), dim=1)
#
#         next_encoded_state = self.dynamics_encoded_state_network(x)
#
#         reward = self.dynamics_reward_network(next_encoded_state)
#
#         # Scale encoded state between [0, 1] (See paper appendix Training)
#         min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
#         max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
#         scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
#         scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
#         next_encoded_state_normalized = (
#             next_encoded_state - min_next_encoded_state
#         ) / scale_next_encoded_state
#
#         return next_encoded_state_normalized, reward
#
#     def initial_inference(self, observation):
#         encoded_state = self.representation(observation)
#         policy_logits, value = self.prediction(encoded_state)
#         # reward equal to 0 for consistency
#         reward = torch.log(
#             (
#                 torch.zeros(1, self.full_support_size)
#                 .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
#                 .repeat(len(observation), 1)
#                 .to(observation.device)
#             )
#         )
#
#         return (
#             value,
#             reward,
#             policy_logits,
#             encoded_state,
#         )
#
#     def recurrent_inference(self, encoded_state, action):
#         next_encoded_state, reward = self.dynamics(encoded_state, action)
#         policy_logits, value = self.prediction(next_encoded_state)
#         return value, reward, policy_logits, next_encoded_state
#
#
# ###### End Fully Connected #######
# ##################################
#
#
# ##################################
# ############# ResNet #############
#
#
# def conv3x3(in_channels, out_channels, stride=1):
#     return torch.nn.Conv2d(
#         in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
#     )
#
#
# # Residual block
# class ResidualBlock(torch.nn.Module):
#     def __init__(self, num_channels, stride=1):
#         super().__init__()
#         self.conv1 = conv3x3(num_channels, num_channels, stride)
#         self.bn1 = torch.nn.BatchNorm2d(num_channels)
#         self.conv2 = conv3x3(num_channels, num_channels)
#         self.bn2 = torch.nn.BatchNorm2d(num_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = torch.nn.functional.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += x
#         out = torch.nn.functional.relu(out)
#         return out
#
#
# # Downsample observations before representation network (See paper appendix Network Architecture)
# class DownSample(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(
#             in_channels,
#             out_channels // 2,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias=False,
#         )
#         self.resblocks1 = torch.nn.ModuleList(
#             [ResidualBlock(out_channels // 2) for _ in range(2)]
#         )
#         self.conv2 = torch.nn.Conv2d(
#             out_channels // 2,
#             out_channels,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias=False,
#         )
#         self.resblocks2 = torch.nn.ModuleList(
#             [ResidualBlock(out_channels) for _ in range(3)]
#         )
#         self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         self.resblocks3 = torch.nn.ModuleList(
#             [ResidualBlock(out_channels) for _ in range(3)]
#         )
#         self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         for block in self.resblocks1:
#             x = block(x)
#         x = self.conv2(x)
#         for block in self.resblocks2:
#             x = block(x)
#         x = self.pooling1(x)
#         for block in self.resblocks3:
#             x = block(x)
#         x = self.pooling2(x)
#         return x
#
#
# class DownsampleCNN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, h_w):
#         super().__init__()
#         mid_channels = (in_channels + out_channels) // 2
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv2d(
#                 in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
#             ),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=3, stride=2),
#             torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         return x
#
#
# class RepresentationNetwork(torch.nn.Module):
#     def __init__(
#         self,
#         observation_shape,
#         stacked_observations,
#         num_blocks,
#         num_channels,
#         downsample,
#     ):
#         super().__init__()
#         self.downsample = downsample
#         if self.downsample:
#             if self.downsample == "resnet":
#                 self.downsample_net = DownSample(
#                     observation_shape[0] * (stacked_observations + 1)
#                     + stacked_observations,
#                     num_channels,
#                 )
#             elif self.downsample == "CNN":
#                 self.downsample_net = DownsampleCNN(
#                     observation_shape[0] * (stacked_observations + 1)
#                     + stacked_observations,
#                     num_channels,
#                     (
#                         math.ceil(observation_shape[1] / 16),
#                         math.ceil(observation_shape[2] / 16),
#                     ),
#                 )
#             else:
#                 raise NotImplementedError('downsample should be "resnet" or "CNN".')
#         self.conv = conv3x3(
#             observation_shape[0] * (stacked_observations + 1) + stacked_observations,
#             num_channels,
#         )
#         self.bn = torch.nn.BatchNorm2d(num_channels)
#         self.resblocks = torch.nn.ModuleList(
#             [ResidualBlock(num_channels) for _ in range(num_blocks)]
#         )
#
#     def forward(self, x):
#         if self.downsample:
#             x = self.downsample_net(x)
#         else:
#             x = self.conv(x)
#             x = self.bn(x)
#             x = torch.nn.functional.relu(x)
#
#         for block in self.resblocks:
#             x = block(x)
#         return x
#
#
# class DynamicsNetwork(torch.nn.Module):
#     def __init__(
#         self,
#         num_blocks,
#         num_channels,
#         reduced_channels_reward,
#         fc_reward_layers,
#         full_support_size,
#         block_output_size_reward,
#     ):
#         super().__init__()
#         self.conv = conv3x3(num_channels, num_channels - 1)
#         self.bn = torch.nn.BatchNorm2d(num_channels - 1)
#         self.resblocks = torch.nn.ModuleList(
#             [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
#         )
#
#         self.conv1x1_reward = torch.nn.Conv2d(
#             num_channels - 1, reduced_channels_reward, 1
#         )
#         self.block_output_size_reward = block_output_size_reward
#         self.fc = mlp(
#             self.block_output_size_reward,
#             fc_reward_layers,
#             full_support_size,
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = torch.nn.functional.relu(x)
#         for block in self.resblocks:
#             x = block(x)
#         state = x
#         x = self.conv1x1_reward(x)
#         x = x.view(-1, self.block_output_size_reward)
#         reward = self.fc(x)
#         return state, reward
#
#
# class PredictionNetwork(torch.nn.Module):
#     def __init__(
#         self,
#         action_space_size,
#         num_blocks,
#         num_channels,
#         reduced_channels_value,
#         reduced_channels_policy,
#         fc_value_layers,
#         fc_policy_layers,
#         full_support_size,
#         block_output_size_value,
#         block_output_size_policy,
#     ):
#         super().__init__()
#         self.resblocks = torch.nn.ModuleList(
#             [ResidualBlock(num_channels) for _ in range(num_blocks)]
#         )
#
#         self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
#         self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
#         self.block_output_size_value = block_output_size_value
#         self.block_output_size_policy = block_output_size_policy
#         self.fc_value = mlp(
#             self.block_output_size_value, fc_value_layers, full_support_size
#         )
#         self.fc_policy = mlp(
#             self.block_output_size_policy,
#             fc_policy_layers,
#             action_space_size,
#         )
#
#     def forward(self, x):
#         for block in self.resblocks:
#             x = block(x)
#         value = self.conv1x1_value(x)
#         policy = self.conv1x1_policy(x)
#         value = value.view(-1, self.block_output_size_value)
#         policy = policy.view(-1, self.block_output_size_policy)
#         value = self.fc_value(value)
#         policy = self.fc_policy(policy)
#         return policy, value
#
#
# class MuZeroResidualNetwork(AbstractNetwork):
#     def __init__(
#         self,
#         observation_shape,
#         stacked_observations,
#         action_space_size,
#         num_blocks,
#         num_channels,
#         reduced_channels_reward,
#         reduced_channels_value,
#         reduced_channels_policy,
#         fc_reward_layers,
#         fc_value_layers,
#         fc_policy_layers,
#         support_size,
#         downsample,
#     ):
#         super().__init__()
#         self.action_space_size = action_space_size
#         self.full_support_size = 2 * support_size + 1
#         block_output_size_reward = (
#             (
#                 reduced_channels_reward
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
#         )
#
#         block_output_size_value = (
#             (
#                 reduced_channels_value
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_value * observation_shape[1] * observation_shape[2])
#         )
#
#         block_output_size_policy = (
#             (
#                 reduced_channels_policy
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
#         )
#
#         self.representation_network = torch.nn.DataParallel(
#             RepresentationNetwork(
#                 observation_shape,
#                 stacked_observations,
#                 num_blocks,
#                 num_channels,
#                 downsample,
#             )
#         )
#
#         self.dynamics_network = torch.nn.DataParallel(
#             DynamicsNetwork(
#                 num_blocks,
#                 num_channels + 1,
#                 reduced_channels_reward,
#                 fc_reward_layers,
#                 self.full_support_size,
#                 block_output_size_reward,
#             )
#         )
#
#         self.prediction_network = torch.nn.DataParallel(
#             PredictionNetwork(
#                 action_space_size,
#                 num_blocks,
#                 num_channels,
#                 reduced_channels_value,
#                 reduced_channels_policy,
#                 fc_value_layers,
#                 fc_policy_layers,
#                 self.full_support_size,
#                 block_output_size_value,
#                 block_output_size_policy,
#             )
#         )
#
#     def prediction(self, encoded_state):
#         policy, value = self.prediction_network(encoded_state)
#         return policy, value
#
#     def representation(self, observation):
#         encoded_state = self.representation_network(observation)
#
#         # Scale encoded state between [0, 1] (See appendix paper Training)
#         min_encoded_state = (
#             encoded_state.view(
#                 -1,
#                 encoded_state.shape[1],
#                 encoded_state.shape[2] * encoded_state.shape[3],
#             )
#             .min(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         max_encoded_state = (
#             encoded_state.view(
#                 -1,
#                 encoded_state.shape[1],
#                 encoded_state.shape[2] * encoded_state.shape[3],
#             )
#             .max(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         scale_encoded_state = max_encoded_state - min_encoded_state
#         scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
#         encoded_state_normalized = (
#             encoded_state - min_encoded_state
#         ) / scale_encoded_state
#         return encoded_state_normalized
#
#     def dynamics(self, encoded_state, action):
#         # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
#         action_one_hot = (
#             torch.ones(
#                 (
#                     encoded_state.shape[0],
#                     1,
#                     encoded_state.shape[2],
#                     encoded_state.shape[3],
#                 )
#             )
#             .to(action.device)
#             .float()
#         )
#         action_one_hot = (
#             action[:, :, None, None] * action_one_hot / self.action_space_size
#         )
#         x = torch.cat((encoded_state, action_one_hot), dim=1)
#         next_encoded_state, reward = self.dynamics_network(x)
#
#         # Scale encoded state between [0, 1] (See paper appendix Training)
#         min_next_encoded_state = (
#             next_encoded_state.view(
#                 -1,
#                 next_encoded_state.shape[1],
#                 next_encoded_state.shape[2] * next_encoded_state.shape[3],
#             )
#             .min(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         max_next_encoded_state = (
#             next_encoded_state.view(
#                 -1,
#                 next_encoded_state.shape[1],
#                 next_encoded_state.shape[2] * next_encoded_state.shape[3],
#             )
#             .max(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
#         scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
#         next_encoded_state_normalized = (
#             next_encoded_state - min_next_encoded_state
#         ) / scale_next_encoded_state
#         return next_encoded_state_normalized, reward
#
#     def initial_inference(self, observation):
#         encoded_state = self.representation(observation)
#         policy_logits, value = self.prediction(encoded_state)
#         # reward equal to 0 for consistency
#         reward = torch.log(
#             (
#                 torch.zeros(1, self.full_support_size)
#                 .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
#                 .repeat(len(observation), 1)
#                 .to(observation.device)
#             )
#         )
#         return (
#             value,
#             reward,
#             policy_logits,
#             encoded_state,
#         )
#
#     def recurrent_inference(self, encoded_state, action):
#         next_encoded_state, reward = self.dynamics(encoded_state, action)
#         policy_logits, value = self.prediction(next_encoded_state)
#         return value, reward, policy_logits, next_encoded_state
#
#
# ########### End ResNet ###########
# ##################################
#
#
# def mlp(
#     input_size,
#     layer_sizes,
#     output_size,
#     output_activation=torch.nn.Identity,
#     activation=torch.nn.ELU,
# ):
#     sizes = [input_size] + layer_sizes + [output_size]
#     layers = []
#     for i in range(len(sizes) - 1):
#         act = activation if i < len(sizes) - 2 else output_activation
#         layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
#     return torch.nn.Sequential(*layers)
#
#
# def support_to_scalar(logits, support_size):
#     """
#     Transform a categorical representation to a scalar
#     See paper appendix Network Architecture
#     """
#     # Decode to a scalar
#     probabilities = torch.softmax(logits, dim=1)
#     support = (
#         torch.tensor([x for x in range(-support_size, support_size + 1)])
#         .expand(probabilities.shape)
#         .float()
#         .to(device=probabilities.device)
#     )
#     x = torch.sum(support * probabilities, dim=1, keepdim=True)
#
#     # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
#     x = torch.sign(x) * (
#         ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
#         ** 2
#         - 1
#     )
#     return x
#
#
# def scalar_to_support(x, support_size):
#     """
#     Transform a scalar to a categorical representation with (2 * support_size + 1) categories
#     See paper appendix Network Architecture
#     """
#     # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
#     x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x
#
#     # Encode on a vector
#     x = torch.clamp(x, -support_size, support_size)
#     floor = x.floor()
#     prob = x - floor
#     logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
#     logits.scatter_(
#         2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
#     )
#     indexes = floor + support_size + 1
#     prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
#     indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
#     logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
#     return logits
