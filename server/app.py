import os
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import ALLOWED_ORIGINS, WS_TOKEN_SECRET, WS_TOKEN_TTL_SECONDS
from server.auth import create_ws_token, verify_ws_token
from server.session_worker import SessionWorker
from server.storage import Storage


class StartSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)


app = FastAPI(title="Aura Remote Server", version="1.0.0")
storage = Storage()
workers: Dict[str, SessionWorker] = {}

if ALLOWED_ORIGINS.strip() == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [item.strip() for item in ALLOWED_ORIGINS.split(",") if item.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="web app not found")
    return FileResponse(index_path)


@app.post("/api/v1/session/start")
async def start_session(payload: StartSessionRequest) -> dict:
    session_id = uuid.uuid4().hex
    worker = SessionWorker(session_id=session_id, user_id=payload.user_id, storage=storage)
    workers[session_id] = worker
    storage.create_session(session_id, payload.user_id, status="created")

    token = create_ws_token(
        secret=WS_TOKEN_SECRET,
        session_id=session_id,
        user_id=payload.user_id,
        ttl_seconds=WS_TOKEN_TTL_SECONDS,
    )
    ws_url = f"/api/v1/session/{session_id}/stream?token={token}"
    return {"session_id": session_id, "ws_url": ws_url}


@app.get("/api/v1/session/{session_id}/history")
async def get_history(session_id: str) -> dict:
    data = storage.get_history(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="session not found")
    return data


@app.websocket("/api/v1/session/{session_id}/stream")
async def session_stream(websocket: WebSocket, session_id: str, token: str):
    worker = workers.get(session_id)
    if not worker:
        await websocket.close(code=4404, reason="session_not_found")
        return

    claims = verify_ws_token(token, secret=WS_TOKEN_SECRET)
    if not claims or claims.get("sid") != session_id or claims.get("uid") != worker.user_id:
        await websocket.close(code=4401, reason="invalid_token")
        return

    await websocket.accept()
    await worker.attach(websocket)

    try:
        while worker.running:
            message = await websocket.receive_json()
            await worker.handle_client_message(message)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await worker._send_error("stream_failed", str(exc))
    finally:
        await worker.close("websocket_disconnected")
        workers.pop(session_id, None)
