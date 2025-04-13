from typing import Tuple, List, Dict, Optional

import numpy as np
import ray
from loguru import logger

from core.config import AlphaZeroConfig
from core.serialization import save_game_log
from environments.base import BaseEnvironment, StateType, ActionType
from algorithms.mcts import AlphaZeroMCTS, get_state_key
from factories import get_environment


@ray.remote
class SelfPlayWorkerActor:
    """
    Ray actor managing self-play for a subset of games and interacting
    with a central InferenceActor.
    """

    def __init__(
        self,
        actor_id: int,
        env_config,
        agent_config: AlphaZeroConfig,
        inference_actor_handle: ray.actor.ActorHandle,
        num_internal_parallel_games: int,
        iteration: int,
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.agent_config = agent_config
        self.inference_actor = inference_actor_handle
        self.num_internal_parallel_games = num_internal_parallel_games
        self.iteration = iteration
        self.env_name = env_config.name

        self.env_factory = lambda: get_environment(self.env_config)
        self.inference_batch_size = agent_config.inference_batch_size
        self.network_cache = {}

        # Game state tracking for games managed by this worker
        self.all_experiences_collected: List[Tuple[StateType, np.ndarray, float]] = []
        self.num_finished_games_total = 0
        self.num_steps_taken = 0
        self.envs: List[BaseEnvironment] = []
        self.observations: List[StateType] = []
        self.histories: Dict[int, List[Tuple[StateType, ActionType, np.ndarray]]] = {}
        self.states_before_action: Dict[int, StateType] = {}
        self.mcts_instances: Dict[int, AlphaZeroMCTS] = {}
        self.pending_requests: Dict[
            int, StateType
        ] = {}  # game_idx (local) -> state_dict

        # Initialize internal game states
        self._initialize_internal_games()

    def _initialize_internal_games(self):
        """Sets up the environments and tracking for the worker's internal games."""
        self.envs = [
            self.env_factory() for _ in range(self.num_internal_parallel_games)
        ]
        self.observations = [env.reset() for env in self.envs]
        self.histories = {i: [] for i in range(self.num_internal_parallel_games)}
        self.states_before_action = {}
        self.mcts_instances = {}
        self.pending_requests = {}
        # Reset experience list for the new collection task
        self.all_experiences_collected = []

    def collect_n_games(
        self, num_games_to_collect_this_call: int
    ) -> List[Tuple[StateType, np.ndarray, float]]:
        """Runs the self-play loop until N games are completed by this worker."""
        logger.debug(
            f"Actor {self.actor_id}: Starting collection of {num_games_to_collect_this_call} games."
        )
        self._initialize_internal_games()  # Reset state for this collection task
        num_finished_this_call = 0

        while num_finished_this_call < num_games_to_collect_this_call:
            finished_before_iter = self.num_finished_games_total
            self._run_one_iter()
            finished_after_iter = self.num_finished_games_total
            num_finished_this_call += finished_after_iter - finished_before_iter

        logger.debug(
            f"Actor {self.actor_id}: Finished collecting {num_finished_this_call} games "
            f"(Total finished by worker: {self.num_finished_games_total})."
        )
        return self.all_experiences_collected

    def _run_one_iter(self):
        self._start_needed_searches()
        self._repeatedly_handle_pending_requests()
        self._handle_completed_searches()

    def _start_needed_searches(self):
        for game_idx in range(self.num_internal_parallel_games):
            # Check if game is already finished *within this collection task*
            # This needs refinement - how to know if a game slot is truly done?
            # Let's assume we just keep resetting finished games within the worker.
            if self.envs[game_idx].is_game_over() or game_idx in self.pending_requests:
                # If game over, it should have been reset in _handle_finished_game
                # If still pending, wait.
                continue

            if game_idx not in self.mcts_instances:
                self.states_before_action[game_idx] = self.observations[game_idx].copy()
                mcts = AlphaZeroMCTS(
                    env=self.envs[game_idx],
                    config=self.agent_config,
                    network_cache=self.network_cache,
                )
                mcts.env = self.envs[game_idx].copy()
                mcts.env.set_state(self.observations[game_idx])
                mcts.prepare_for_next_search(train=True)
                self.mcts_instances[game_idx] = mcts
                self._get_mcts_network_request(game_idx=game_idx, response=None)
            else:
                # Continue existing game
                self.states_before_action[game_idx] = self.observations[game_idx].copy()
                mcts = self.mcts_instances[game_idx]
                mcts.prepare_for_next_search(train=True)
                self._get_mcts_network_request(game_idx=game_idx, response=None)

    def _get_mcts_network_request(
        self, game_idx: int, response: Optional[Tuple[np.ndarray, float]] = None
    ):
        if game_idx not in self.mcts_instances:
            # Game might have finished and been deleted between cycles
            return
        mcts: AlphaZeroMCTS = self.mcts_instances[game_idx]
        request_state = mcts.get_network_request(previous_response=response)
        if request_state is not None:
            self.pending_requests[game_idx] = request_state

    def _repeatedly_handle_pending_requests(self):
        # This worker sends requests to the central InferenceActor
        while len(self.pending_requests) >= self.inference_batch_size // 2:
            batch_size = min(self.inference_batch_size, len(self.pending_requests))
            # Use local game indices
            local_game_indices = list(self.pending_requests.keys())[:batch_size]
            batch_states = [self.pending_requests[idx] for idx in local_game_indices]

            # Call remote inference actor
            inference_task_ref = self.inference_actor.predict_batch.remote(batch_states)
            try:
                policy_list, value_list = ray.get(inference_task_ref)
            except ray.exceptions.RayTaskError as e:
                logger.error(f"Actor {self.actor_id}: Inference task failed: {e}")
                # Clear failed requests for this worker
                for idx in local_game_indices:
                    if idx in self.pending_requests:
                        del self.pending_requests[idx]
                break  # Stop processing batches this iteration

            if len(policy_list) != len(local_game_indices):
                logger.error(f"Actor {self.actor_id}: Inference result size mismatch!")
                for idx in local_game_indices:
                    if idx in self.pending_requests:
                        del self.pending_requests[idx]
                continue

            for i, game_idx in enumerate(local_game_indices):
                if game_idx not in self.pending_requests:
                    continue  # Skip if cleared by error

                evaluated_state = self.pending_requests[game_idx]
                state_key = get_state_key(evaluated_state)
                response = (policy_list[i], value_list[i])

                # Use local cache
                self.network_cache[state_key] = response
                del self.pending_requests[game_idx]
                # Feed result back to local MCTS instance
                self._get_mcts_network_request(game_idx=game_idx, response=response)

    def _handle_completed_searches(self):
        # Handles games where MCTS search (within this worker) is complete
        completed_local_indices = []
        for game_idx, mcts in list(self.mcts_instances.items()):
            if game_idx not in self.pending_requests:
                completed_local_indices.append(game_idx)

        for game_idx in completed_local_indices:
            if game_idx not in self.mcts_instances:
                continue  # Might have been deleted by finish
            mcts = self.mcts_instances[game_idx]
            action, action_visits = mcts.get_result()
            policy = self._calculate_policy(action_visits, self.envs[game_idx])

            current_legal_actions = self.envs[game_idx].get_legal_actions()
            if action not in current_legal_actions:
                logger.error(
                    f"Actor {self.actor_id} Game {game_idx}: Illegal action {action} chosen! Legal: {current_legal_actions}"
                )
                done = True
                next_obs = self.observations[game_idx]
            else:
                next_obs, reward, done = self.envs[game_idx].step(action)
                self.num_steps_taken += 1

            if policy is None:
                logger.warning(
                    f"Actor {self.actor_id} Game {game_idx}: Could not calculate policy."
                )
            elif game_idx in self.states_before_action and action is not None:
                self.histories[game_idx].append(
                    (self.states_before_action[game_idx], action, policy)
                )

            self.observations[game_idx] = next_obs

            if game_idx in self.states_before_action:
                del self.states_before_action[game_idx]

            if done:
                # Pass the total finished count offset for unique game logging index
                self._handle_finished_game(game_idx, self.num_finished_games_total)
                self.num_finished_games_total += 1  # Increment worker's total count
            elif action is not None:
                mcts.advance_root(action)
                mcts.env = self.envs[game_idx].copy()
            else:  # Illegal action case
                if game_idx in self.mcts_instances:
                    del self.mcts_instances[game_idx]  # Reset MCTS

    def _handle_finished_game(self, game_idx, finished_game_offset):
        # Processes a finished game and adds experiences to the worker's list
        winner = self.envs[game_idx].get_winning_player()
        final_outcome = 1.0 if winner == 0 else -1.0 if winner == 1 else 0.0
        valid_history = [
            (s, a, p) for s, a, p in self.histories[game_idx] if p is not None
        ]

        if valid_history:
            num_steps = len(valid_history)
            logged_history_for_saving = []
            for i, (state, action, policy) in enumerate(valid_history):
                value_target = final_outcome  # Simplistic assignment
                # Add to the list that will be returned by this actor
                self.all_experiences_collected.append((state, policy, value_target))
                logged_history_for_saving.append((state, action, policy, value_target))

            # Logging from actors can be tricky, consider using Ray logging
            save_game_log(
                logged_history=logged_history_for_saving,
                iteration=self.iteration,
                # Use offset for a potentially unique index across all workers
                game_index=finished_game_offset + 1,
                env_name=self.env_name,
            )
        else:
            logger.warning(
                f"Actor {self.actor_id} Game {game_idx}: Finished with no valid history."
            )

        # Reset environment and history for this game index *within the worker*
        self.observations[game_idx] = self.envs[game_idx].reset()
        self.histories[game_idx] = []
        if game_idx in self.mcts_instances:
            del self.mcts_instances[game_idx]
        if game_idx in self.states_before_action:
            del self.states_before_action[game_idx]

    def _calculate_policy(
        self, action_visits: Dict[ActionType, int], env: BaseEnvironment
    ) -> np.ndarray:
        # This logic remains the same
        policy_size = env.policy_vector_size
        policy_target = np.zeros(policy_size, dtype=np.float32)
        total_visits = sum(action_visits.values())
        if total_visits <= 0:  # Should be caught by MCTS assertion, but double-check
            logger.error(
                f"Actor {self.actor_id}: Total visits zero in _calculate_policy."
            )
            # Return uniform as fallback
            policy_target.fill(1.0 / policy_size if policy_size > 0 else 1.0)
            return policy_target

        for action, visits in action_visits.items():
            action_key = tuple(action) if isinstance(action, list) else action
            action_idx = env.map_action_to_policy_index(action_key)
            if action_idx is not None and 0 <= action_idx < policy_size:
                policy_target[action_idx] = visits / total_visits
            else:
                logger.warning(
                    f"Actor {self.actor_id}: Action {action_key} could not be mapped."
                )

        current_sum = policy_target.sum()
        if abs(current_sum - 1.0) > 1e-5:
            if current_sum > 1e-6:
                policy_target /= current_sum
            elif policy_size > 0:
                policy_target.fill(1.0 / policy_size)
            else:
                logger.error(
                    f"Actor {self.actor_id}: Cannot normalize zero-size policy."
                )

        return policy_target
