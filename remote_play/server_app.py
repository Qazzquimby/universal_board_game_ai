import json
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any
from urllib.request import Request

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette import status
from starlette.responses import JSONResponse

from agents.base_learning_agent import GameHistoryStep
from core.config import AppConfig
from factories import get_environment, create_learning_agent
from models.training import _run_one_self_play_game
from remote_play.client import SelfPlayRequest, RunGameRequest

app = FastAPI()

MODEL_DIR = Path("uploaded_models")
MODEL_DIR.mkdir(exist_ok=True)

CACHED_AGENTS: Dict[str, Tuple[Any, Any]] = {}


def serialize_game_step(step: GameHistoryStep) -> Dict[str, Any]:
    """Manually serialize a GameHistoryStep to be JSON-compatible."""
    return {
        "state": {k: vars(v) for k, v in step.state.items()},
        "action_index": step.action_index,
        "policy": step.policy.tolist(),
        "legal_actions": step.legal_actions,
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # todo, this actually catches all exceptions, not just 422
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    print(f"ERROR: {request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.get("/")
async def health():
    return "OK"

    # @app.post("/upload-model/")
    # def upload_model(file: UploadFile = File(...)):
    #     filename = f"{uuid.uuid4()}_{file.filename}"
    #     path = MODEL_DIR / filename
    #     try:
    #         with open(path, "wb") as buffer:
    #             shutil.copyfileobj(file.file, buffer)
    #         return {"filename": filename}
    #     except Exception as e:
    #         logger.error(f"Error uploading file: {e}")
    #         raise HTTPException(status_code=500, detail="File upload failed")


import anyio


@app.post("/upload-model/")
async def upload_model(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}_{file.filename}"
    path = MODEL_DIR / filename

    contents = await file.read()

    def write_file():
        with open(path, "wb") as buffer:
            buffer.write(contents)

    await anyio.to_thread.run_sync(write_file)

    return {"filename": filename}


@app.post("/setup-agent/")
def setup_agent(request: SelfPlayRequest):
    config = AppConfig.model_validate(json.loads(request.config_json))
    env = get_environment(config.env)

    # Note: MCTSAgent is not a learning agent and is handled differently
    agent = create_learning_agent(request.type, env, config)
    model_path = MODEL_DIR / request.filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    agent.load(model_path)
    if agent.network:
        agent.network.eval()

    agent_id = str(uuid.uuid4())
    CACHED_AGENTS[agent_id] = (agent, env)

    return {"agent_id": agent_id}


@app.post("/run-self-play/")
def run_self_play_endpoint(
    request: RunGameRequest,
) -> Tuple[List[Dict[str, Any]], float]:
    if request.agent_id not in CACHED_AGENTS:
        raise HTTPException(
            status_code=404, detail="Agent not found. Please set it up first."
        )

    agent, env = CACHED_AGENTS[request.agent_id]
    game_history, final_outcome = _run_one_self_play_game(env, agent)
    serializable_history = [serialize_game_step(step) for step in game_history]
    return serializable_history, final_outcome
