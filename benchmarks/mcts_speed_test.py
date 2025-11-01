import time
from typing import Dict

from agents.mcts_agent import BaseMCTSAgent
from environments.base import BaseEnvironment, ActionType
from environments.connect4.connect4 import Connect4 as OldConnect4
from environments_new.connect4 import Connect4 as NewConnect4


class GenericBaseMCTSAgent(BaseMCTSAgent):
    """
    An MCTS agent that can work with both old and new environment styles by
    overriding the simulation loop to dynamically determine the number of players.
    """

    def _run_simulation(self, env: BaseEnvironment, train: bool):
        """
        Runs a single simulation from selection to backpropagation.
        This implementation is adapted from the original BaseMCTSAgent to be
        compatible with multiple environment types.
        """
        sim_env = env.copy()

        # 1. Selection
        selection_result = self.selection_strategy.select(
            node=self.root, sim_env=sim_env, cache=self.node_cache
        )
        path = selection_result.path
        leaf_node = selection_result.leaf_node
        leaf_env = selection_result.leaf_env

        # 2. Expansion
        self._expand_leaf(leaf_node, leaf_env, train)

        # 3. Evaluation
        player_at_leaf = leaf_env.get_current_player()
        value = float(self.evaluation_strategy.evaluate(leaf_node, leaf_env))

        # --- Start Generic Adaptation ---
        num_players = 0
        if hasattr(env, "num_players"):  # New env style
            num_players = env.num_players
        elif hasattr(env, "state") and hasattr(env.state, "players"):  # Old env style
            num_players = len(env.state.players)
        else:
            raise ValueError("Could not determine number of players for environment.")

        player_to_value: Dict[int, float] = {}
        for player in range(num_players):
            if player == player_at_leaf:
                player_to_value[player] = value
            else:
                # Assuming zero-sum game for other players
                player_to_value[player] = -value
        # --- End Generic Adaptation ---

        # 4. Backpropagation
        self.backpropagation_strategy.backpropagate(
            path=path, player_to_value=player_to_value
        )

    def act(self, env: BaseEnvironment, train: bool = False) -> ActionType:
        """Act is implemented here to make the class concrete, but search is what we benchmark."""
        self.search(env, train)
        # A simple policy for acting, not used in benchmark itself
        return max(self.root.edges.items(), key=lambda item: item[1].num_visits)[0]


def run_benchmark(agent_class, env_class, num_simulations, test_name):
    """Generic function to run a benchmark test."""
    print(f"--- Running Benchmark: {test_name} ---")
    env = env_class()

    # We can't use `make_pure_mcts` because it returns the original agent.
    # We must instantiate our generic agent with the same strategies.
    from algorithms.mcts import (
        UCB1Selection,
        UniformExpansion,
        RandomRolloutEvaluation,
        StandardBackpropagation,
    )

    agent = agent_class(
        num_simulations=num_simulations,
        selection_strategy=UCB1Selection(exploration_constant=1.41),
        expansion_strategy=UniformExpansion(),
        evaluation_strategy=RandomRolloutEvaluation(
            max_rollout_depth=100, discount_factor=0.98
        ),
        backpropagation_strategy=StandardBackpropagation(),
    )

    start_time = time.perf_counter()
    agent.search(env, train=False)
    end_time = time.perf_counter()

    duration = end_time - start_time
    sims_per_second = num_simulations / duration
    print(f"Total time for {num_simulations} sims: {duration:.4f} seconds")
    print(f"Simulations per second: {sims_per_second:.2f}")
    print("-" * (25 + len(test_name)))
    print()


if __name__ == "__main__":
    sim_count = 1000

    # Benchmark the old environment
    run_benchmark(
        agent_class=GenericBaseMCTSAgent,
        env_class=OldConnect4,
        num_simulations=sim_count,
        test_name="Old Connect4 (Pydantic State)",
    )

    # Benchmark the new environment
    run_benchmark(
        agent_class=GenericBaseMCTSAgent,
        env_class=NewConnect4,
        num_simulations=sim_count,
        test_name="New Connect4 (Polars DataFrame State)",
    )
