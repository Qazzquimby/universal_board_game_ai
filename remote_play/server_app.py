import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger

from agents.base_learning_agent import GameHistoryStep
from core.config import AppConfig
from factories import get_environment, create_learning_agent
from models.training import _run_one_self_play_game
from remote_play.client import SelfPlayRequest

app = FastAPI()

MODEL_DIR = Path("uploaded_models")
MODEL_DIR.mkdir(exist_ok=True)


def serialize_game_step(step: GameHistoryStep) -> Dict[str, Any]:
    """Manually serialize a GameHistoryStep to be JSON-compatible."""
    return {
        "state": {k: vars(v) for k, v in step.state.items()},
        "action_index": step.action_index,
        "policy": step.policy.tolist(),
        "legal_actions": step.legal_actions,
    }


@app.get("/")
async def health():
    return "OK"


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
) -> Tuple[List[Dict[str, Any]], float]:
    config = AppConfig.model_validate(request.config_json)
    env = get_environment(config.env)

    # Note: MCTSAgent is not a learning agent and is handled differently
    agent = create_learning_agent(request.type, env, config)
    model_path = MODEL_DIR / request.filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    agent.load(model_path)
    if agent.network:
        agent.network.eval()

    game_history, final_outcome = _run_one_self_play_game(env, agent)
    serializable_history = [serialize_game_step(step) for step in game_history]
    return serializable_history, final_outcome
