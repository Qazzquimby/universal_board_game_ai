from __future__ import annotations

import asyncio
import json
from typing import List, Tuple
from pathlib import Path

import aiohttp
import yaml
import numpy as np
from loguru import logger

from agents.base_learning_agent import GameHistoryStep
from core.config import AppConfig
from environments.base import DataFrame


class RemotePlayClient:
    def __init__(self):
        self.servers = []
        try:
            with open("servers.json", "r") as f:
                self.servers = json.load(f)
        except FileNotFoundError:
            logger.warning(
                "servers.json not found. Remote play client will not be able to connect."
            )
        self.ips = [s["ip"] for s in self.servers]

    async def _upload_model_to_server(
        self, session: aiohttp.ClientSession, ip: str, model_path: str
    ) -> str:
        url = f"http://{ip}:8000/upload-model/"
        data = aiohttp.FormData()
        data.add_field("file", open(model_path, "rb"), filename=Path(model_path).name)
        async with session.post(url, data=data) as response:
            response.raise_for_status()
            return (await response.json())["filename"]

    def _deserialize_game_results(
        self, raw_results: List[Tuple[list, float]]
    ) -> List[Tuple[List[GameHistoryStep], float]]:
        deserialized_results = []
        for raw_history, final_outcome in raw_results:
            history = []
            for step_dict in raw_history:
                state_dict = step_dict["state"]
                state = {}
                for k, df_dict in state_dict.items():
                    df = DataFrame(data=df_dict["_data"], columns=df_dict["columns"])
                    state[k] = df

                step = GameHistoryStep(
                    state=state,
                    action=step_dict["action"],
                    policy=np.array(step_dict["policy"], dtype=np.float32),
                    legal_actions=step_dict["legal_actions"],
                )
                history.append(step)
            deserialized_results.append((history, final_outcome))
        return deserialized_results

    async def _run_games_on_server(
        self,
        session: aiohttp.ClientSession,
        ip: str,
        model_filename: str,
        num_games: int,
        config: AppConfig,
        model_type: str,
    ) -> List[Tuple[List[GameHistoryStep], float]]:
        url = f"http://{ip}:8000/run-self-play/"
        payload = {
            "model_filename": model_filename,
            "num_games": num_games,
            "config_yaml": yaml.dump(config.dict()),
            "model_type": model_type,
        }
        try:
            async with session.post(url, json=payload, timeout=3600) as response:
                response.raise_for_status()
                raw_results = await response.json()
                return self._deserialize_game_results(raw_results)
        except aiohttp.ClientError as e:
            logger.error(f"Error running games on server {ip}: {e}")
            return []

    async def run_self_play_games(
        self, model_path: str, num_games: int, config: AppConfig, model_type: str
    ) -> List[Tuple[List[GameHistoryStep], float]]:
        if not self.ips:
            return []

        num_servers = len(self.ips)
        games_per_server = [num_games // num_servers] * num_servers
        for i in range(num_games % num_servers):
            games_per_server[i] += 1

        async with aiohttp.ClientSession() as session:
            upload_tasks = [
                self._upload_model_to_server(session, ip, model_path) for ip in self.ips
            ]
            model_filenames = await asyncio.gather(*upload_tasks)

            game_tasks = []
            for ip, model_filename, games_count in zip(
                self.ips, model_filenames, games_per_server
            ):
                if games_count > 0:
                    game_tasks.append(
                        self._run_games_on_server(
                            session,
                            ip,
                            model_filename,
                            games_count,
                            config,
                            model_type,
                        )
                    )

            results_from_all_servers = await asyncio.gather(*game_tasks)

        return [
            item for sublist in results_from_all_servers for item in sublist
        ]  # flatten list
