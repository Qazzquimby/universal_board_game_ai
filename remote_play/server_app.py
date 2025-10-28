import os
import uuid
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger

from agents.base_learning_agent import GameHistoryStep
from core.config import AppConfig
from factories import get_environment, _create_learning_agent
from models.training import _run_one_self_play_game

app = FastAPI()

MODEL_DIR = Path("uploaded_models")
MODEL_DIR.mkdir(exist_ok=True)


class SelfPlayRequest(BaseModel):
    model_filename: str
    num_games: int
    config_yaml: str
    model_type: str


def serialize_game_step(step: GameHistoryStep) -> Dict[str, Any]:
    """Manually serialize a GameHistoryStep to be JSON-compatible."""
    return {
        "state": {k: vars(v) for k, v in step.state.items()},
        "action": step.action,
        "policy": step.policy.tolist(),
        "legal_actions": step.legal_actions,
    }


@app.post("/upload-model/")
async def upload_model(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}_{file.filename}"
    path = MODEL_DIR / filename
    try:
        with open(path, "wb") as buffer:
            buffer.write(await file.read())
        return {"filename": filename}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@app.post("/run-self-play/")
async def run_self_play_endpoint(
    request: SelfPlayRequest,
) -> List[Tuple[List[Dict[str, Any]], float]]:
    config = AppConfig.parse_obj(yaml.safe_load(request.config_yaml))
    env = get_environment(config.env)

    # Note: MCTSAgent is not a learning agent and is handled differently
    agent = _create_learning_agent(request.model_type, env, config)
    model_path = MODEL_DIR / request.model_filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    agent.load(model_path)
    if agent.network:
        agent.network.eval()

    results = []
    for _ in range(request.num_games):
        game_history, final_outcome = _run_one_self_play_game(env, agent)
        serializable_history = [serialize_game_step(step) for step in game_history]
        results.append((serializable_history, final_outcome))

    return results
