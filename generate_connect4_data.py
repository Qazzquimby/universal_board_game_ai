# nvm api is out.
# https://github.com/PascalPons/connect4


# (old) use https://connect4.gamesolver.org/solve?pos=475 api
# {"pos":"475","score":[-18,-2,-2,-18,-18,-2,-18]}
import json
import random
from pathlib import Path
from typing import List

import requests
from pydantic import BaseModel

# generate board positions. 15k probably good.
# How to generate positions?
# Would prefer not to get duplicates or mirrors
# Would prefer realistic positions, thatd arise from skilled play
# want to minimize api calls

# From action code, normalize symmetry, get policy (use cache), save to file
# randomly select move from policy, get new action code, loop


# Send to the api and record board code to option scores. Jsonl

# convert to normalized policy and value (1 if best option positive, 0 if all 0, else -1
# Flip all since symmetrical

# train various ml models

mid_column = 4


class Policy(BaseModel):
    pos: str
    score: List[int]
    visits: int = 0


class Connect4Cache:
    def __init__(self):
        self.data_path = Path("data/connect4_policies/connect4_policies.json")
        self._cache = self._load_cache()

    def get_policy(self, action_code: str) -> Policy:
        action_code = normalize_action_code(action_code)
        policy_match = self._cache.get(action_code)
        if policy_match:
            policy_match.visits += 1
            return policy_match

        policy = self._call_api_for_policy(action_code)
        return policy

    def _call_api_for_policy(self, action_code: str):
        pos = "".join(str(column) for column in action_code)
        if not pos:
            pos = "0"
        url = f"https://connect4.gamesolver.org/solve?pos={pos}"
        response = requests.get(url)
        api_response = Policy(**response.json())
        self._update_cache(action_code, api_response)
        return api_response

    def _load_cache(self):
        if self.data_path.exists():
            with self.data_path.open("r") as f:
                return json.load(f)
        else:
            return {}

    def _update_cache(self, action_code, policy: Policy):
        self._cache[action_code] = policy
        self.save()

    def save(self):
        with self.data_path.open("w") as f:
            json.dump(self._cache, f)


def normalize_action_code(action_code: str):
    # If first number(?) greater than half, inverse all?
    if action_code and int(action_code[0]) > 4:
        inverted_action_code = str([8 - int(column) for column in action_code])
        return inverted_action_code
    else:
        return action_code


IMPOSSIBLE_MOVE_FLAG = 100


def main():
    cache = Connect4Cache()
    rows_to_fetch = 3  # eventually 15_000

    action_code = ""
    for i in range(rows_to_fetch):
        policy: Policy = cache.get_policy(action_code)
        raw_policy = {
            i + 1: p for i, p in enumerate(policy.score) if p != IMPOSSIBLE_MOVE_FLAG
        }
        sum_policy = sum(raw_policy.values())
        normalized_policy = {
            action_num: p / sum_policy for action_num, p in raw_policy.values()
        }
        action_nums = list(normalized_policy.keys())
        action_values = [normalized_policy[action_num] for action_num in action_nums]
        chosen_action = random.choices(action_nums, weights=action_values, k=1)[0]
        action_code = action_code + str(chosen_action)
    cache.save()


if __name__ == "__main__":
    main()
