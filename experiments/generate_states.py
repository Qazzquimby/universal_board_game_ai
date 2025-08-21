# Using a structure similar to eval, run two instances of mcts many times.
# The intention is to collect game states for model training, similar to alphazero training but off of regular mcts.

# states can be represented by a string of actions taken to get to that state
# eg, R playing in column 0, then Y playing in column 4 = "04"
# each state must record the next action that was taken from that state, and the the final game result
# 0 if player 0 wins, 1 if player 1 wins.

# Rather than starting games at an empty board, have players take 5 random moves each (which should not be recorded).

# save the states as a json file to /data/connect_4_states/pure_mcts

import json
import random
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from agents.mcts_agent import MCTSAgent
from algorithms.mcts import (
    UCB1Selection,
    UniformExpansion,
    RandomRolloutEvaluation,
    StandardBackpropagation,
)
from environments.base import BaseEnvironment
from environments.connect4 import Connect4

# --- Constants ---
TARGET_NUM_STATES = 15_000

NUM_SIMULATIONS_PER_MOVE = 800
NUM_INITIAL_RANDOM_MOVES = 10
OUTPUT_DIR = Path("../data/connect_4_states/pure_mcts")
OUTPUT_FILE = OUTPUT_DIR / "mcts_generated_states.json"


def _play_and_collect_one_game(
    env: BaseEnvironment, agent0: MCTSAgent, agent1: MCTSAgent
) -> List[Dict[str, Any]]:
    """
    Plays a single game, collecting state, action, and final result data.
    Starts with a number of random moves before MCTS agents take over.
    """
    env.reset()
    agent0.reset_turn()
    agent1.reset_turn()

    action_history = []

    # 1. Play initial random moves
    for _ in range(NUM_INITIAL_RANDOM_MOVES):
        if env.get_winning_player() is not None or not env.get_legal_actions():
            # Game ended prematurely during random moves, discard this game
            return []

        action = random.choice(env.get_legal_actions())
        env.step(action)
        action_history.append(str(action))

    # 2. Play with MCTS and collect data points
    game_data_points = []
    done = False
    while not env.done:
        current_action_history_str = "".join(action_history)
        current_player = env.get_current_player()

        agent = agent0 if current_player == 0 else agent1
        action = agent.act(env)

        if action is None:  # No legal moves or game already ended
            break

        # Record state (as action history) and the action taken from it
        game_data_points.append(
            {
                "action_history": current_action_history_str,
                "next_action": action,
            }
        )

        result = env.step(action)
        action_history.append(str(action))

    # 3. Add final game result to all data points from this game
    winner = env.get_winning_player()  # Can be 0, 1, or None for a draw

    for data_point in game_data_points:
        data_point["winner"] = winner

    return game_data_points


def main():
    """
    Main function to generate and save game states from MCTS self-play.
    """
    print("--- Starting MCTS State Generation ---")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_collected_states = []
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                all_collected_states = json.load(f)
            print(
                f"Loaded {len(all_collected_states)} existing states from {OUTPUT_FILE}."
            )
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Could not load existing states from {OUTPUT_FILE}, starting fresh. Error: {e}"
            )
            all_collected_states = []

    env = Connect4()
    selection_strategy = UCB1Selection(exploration_constant=1.41)
    expansion_strategy = UniformExpansion()
    evaluation_strategy = RandomRolloutEvaluation(max_rollout_depth=100)
    backpropagation_strategy = StandardBackpropagation()

    agent_config = {
        "num_simulations": NUM_SIMULATIONS_PER_MOVE,
        "selection_strategy": selection_strategy,
        "expansion_strategy": expansion_strategy,
        "evaluation_strategy": evaluation_strategy,
        "backpropagation_strategy": backpropagation_strategy,
    }

    agent0 = MCTSAgent(**agent_config)
    agent1 = MCTSAgent(**agent_config)

    num_states_to_generate = TARGET_NUM_STATES - len(all_collected_states)
    if num_states_to_generate <= 0:
        print(
            f"Already have {len(all_collected_states)} states, which meets or exceeds the target of {TARGET_NUM_STATES}."
        )
        print("--- State Generation Complete ---")
        return

    print(
        f"Targeting {TARGET_NUM_STATES} total states. Need to generate {num_states_to_generate} more."
    )

    with tqdm(
        total=TARGET_NUM_STATES,
        initial=len(all_collected_states),
        desc="Generating states",
    ) as pbar:
        while len(all_collected_states) < TARGET_NUM_STATES:
            game_states = _play_and_collect_one_game(env, agent0, agent1)
            if game_states:
                all_collected_states.extend(game_states)
                pbar.n = len(all_collected_states)
                pbar.refresh()

    # Save collected data to a JSON file
    print(
        f"\nSaving {len(all_collected_states)} total state entries to {OUTPUT_FILE}..."
    )
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_collected_states, f, indent=2)

    print("--- State Generation Complete ---")


if __name__ == "__main__":
    main()
