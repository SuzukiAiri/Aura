import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List

from config import CONVERSATIONS_PATH, SQLITE_PATH


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class Storage:
    def __init__(self, sqlite_path: str = SQLITE_PATH):
        self.sqlite_path = sqlite_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.sqlite_path)

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_session(self, session_id: str, user_id: str, status: str = "idle") -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, user_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, status, now, now),
            )

    def set_session_status(self, session_id: str, status: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (status, now, session_id),
            )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def append_event(self, session_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        now = _utc_now()
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events(session_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, event_type, payload_json, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def get_history(self, session_id: str) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            session = conn.execute(
                """
                SELECT session_id, user_id, status, created_at, updated_at
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if not session:
                return {}

            messages = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

            events = conn.execute(
                """
                SELECT event_type, payload_json, created_at
                FROM events
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return {
            "session": {
                "session_id": session[0],
                "user_id": session[1],
                "status": session[2],
                "created_at": session[3],
                "updated_at": session[4],
            },
            "messages": [
                {"role": role, "content": content, "created_at": created_at}
                for role, content, created_at in messages
            ],
            "events": [
                {
                    "event_type": event_type,
                    "payload": json.loads(payload_json),
                    "created_at": created_at,
                }
                for event_type, payload_json, created_at in events
            ],
        }

    def append_conversation_json(self, conversation_history: List[Dict[str, str]]) -> None:
        if not conversation_history:
            return

        conversations: List[Dict[str, Any]] = []
        if os.path.exists(CONVERSATIONS_PATH):
            try:
                with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as fh:
                    raw = fh.read().strip()
                    if raw:
                        conversations = json.loads(raw)
            except Exception:
                conversations = []

        conversations.append(
            {
                "time": datetime.now().replace(microsecond=0).isoformat(),
                "messages": conversation_history,
            }
        )

        with open(CONVERSATIONS_PATH, "w", encoding="utf-8") as fh:
            json.dump(conversations, fh, ensure_ascii=False, indent=4)
