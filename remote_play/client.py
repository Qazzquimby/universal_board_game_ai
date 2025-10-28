from __future__ import annotations
import asyncio
import json
from typing import List, Tuple

from agents.base_learning_agent import GameHistoryStep


class RemotePlayClient:
    def __init__(self):
        self.servers = []
        try:
            with open("servers.json", "r") as f:
                self.servers = json.load(f)
        except FileNotFoundError:
            print(
                "servers.json not found. Remote play client will not be able to connect."
            )
        self.ips = [s["ip"] for s in self.servers]

    async def run_self_play_games(
        self, model_path: str, num_games: int
    ) -> List[Tuple[List[GameHistoryStep], float]]:
        # This is a placeholder for the actual implementation that would
        # distribute game execution across remote servers.
        print(
            f"Requesting {num_games} games from {len(self.ips)} servers with model {model_path}"
        )

        # In a real implementation, we would use aiohttp to make requests to each server,
        # pass the model (or path to it), and collect game logs.
        # e.g. tasks = [self.run_games_on_server(ip, games_per_server, model_path) for ip in self.ips]
        # results = await asyncio.gather(*tasks)
        # return [item for sublist in results for item in sublist] # flatten list of lists
        return []
