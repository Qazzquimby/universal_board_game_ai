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

    def _deserialize_game_result(
        self, raw_result: Tuple[list, float]
    ) -> Tuple[List[GameHistoryStep], float]:
        raw_history, final_outcome = raw_result
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
        return history, final_outcome

    async def _run_game_on_server(
        self,
        session: aiohttp.ClientSession,
        ip: str,
        model_filename: str,
        config: AppConfig,
        model_type: str,
    ) -> Tuple[List[GameHistoryStep], float]:
        url = f"http://{ip}:8000/run-self-play/"
        payload = {
            "model_filename": model_filename,
            "config_yaml": yaml.dump(config.dict()),
            "model_type": model_type,
        }
        try:
            async with session.post(url, json=payload, timeout=3600) as response:
                response.raise_for_status()
                raw_result = await response.json()
                return self._deserialize_game_result(raw_result)
        except aiohttp.ClientError as e:
            logger.error(f"Error running game on server {ip}: {e}")
            raise

    async def run_self_play_games(
        self, model_path: str, num_games: int, config: AppConfig, model_type: str
    ):
        if not self.ips:
            return

        async with aiohttp.ClientSession() as session:
            upload_tasks = [
                self._upload_model_to_server(session, ip, model_path) for ip in self.ips
            ]
            model_filenames = await asyncio.gather(*upload_tasks)
            model_filenames_map = {
                ip: filename for ip, filename in zip(self.ips, model_filenames)
            }

            # todo We'd like to make another request after it finishes its first, not dump all requests at the start.
            tasks = []
            for i in range(num_games):
                ip = self.ips[i % len(self.ips)]
                task = asyncio.create_task(
                    self._run_game_on_server(
                        session, ip, model_filenames_map[ip], config, model_type
                    )
                )
                tasks.append(task)

            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    yield result
                except Exception as e:
                    logger.error(f"A game task failed: {e}")
