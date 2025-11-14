from __future__ import annotations

import asyncio
import json
from typing import List, Tuple
from pathlib import Path

import aiohttp
import numpy as np
from loguru import logger
from pydantic import BaseModel

from agents.base_learning_agent import GameHistoryStep
from core.config import AppConfig
from environments.base import DataFrame


class SelfPlayRequest(BaseModel):
    filename: str
    config_json: str
    type: str


class RunGameRequest(BaseModel):
    agent_id: str


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
        async with session.post(url, data=data, timeout=360) as response:
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
                action_index=step_dict["action_index"],
                policy=np.array(step_dict["policy"], dtype=np.float32),
                legal_actions=step_dict["legal_actions"],
            )
            history.append(step)
        return history, final_outcome

    async def _setup_agent_on_server(
        self,
        session: aiohttp.ClientSession,
        ip: str,
        model_filename: str,
        config: AppConfig,
        model_type: str,
    ) -> str:
        url = f"http://{ip}:8000/setup-agent/"
        config_json = config.model_dump_json()
        payload = SelfPlayRequest(
            filename=model_filename,
            config_json=config_json,
            type=model_type,
        ).model_dump()
        try:
            async with session.post(url, json=payload, timeout=3600) as response:
                response.raise_for_status()
                return (await response.json())["agent_id"]
        except aiohttp.ClientError as e:
            logger.error(f"Error setting up agent on server {ip}: {e}")
            raise

    async def _run_game_on_server(
        self,
        session: aiohttp.ClientSession,
        ip: str,
        agent_id: str,
    ) -> Tuple[List[GameHistoryStep], float]:
        url = f"http://{ip}:8000/run-self-play/"
        payload = RunGameRequest(agent_id=agent_id).model_dump()
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

            setup_tasks = [
                self._setup_agent_on_server(
                    session, ip, model_filenames_map[ip], config, model_type
                )
                for ip in self.ips
            ]
            agent_ids = await asyncio.gather(*setup_tasks)
            agent_ids_map = {ip: agent_id for ip, agent_id in zip(self.ips, agent_ids)}

            game_queue = list(range(num_games))

            # Start initial tasks (one per server)
            active_tasks = {}
            for ip in self.ips:
                if game_queue:
                    game_queue.pop(0)
                    task = asyncio.create_task(
                        self._run_game_on_server(session, ip, agent_ids_map[ip])
                    )
                    active_tasks[task] = ip

            # Process completed tasks and start new ones
            while active_tasks:
                done, pending = await asyncio.wait(
                    active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    ip = active_tasks.pop(task)
                    try:
                        result = await task
                        yield result
                    except Exception as e:
                        logger.error(f"A game task failed: {e}")

                    # Start a new game on the now-free server
                    if game_queue:
                        game_queue.pop(0)
                        new_task = asyncio.create_task(
                            self._run_game_on_server(session, ip, agent_ids_map[ip])
                        )
                        active_tasks[new_task] = ip
