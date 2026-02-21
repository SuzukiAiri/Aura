import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = Path(os.getenv("AURA_DATA_DIR", str(BASE_DIR))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

AWS_FACE_KEY_ID = os.getenv("AWS_FACE_KEY_ID", "")
AWS_FACE_ACCESS_KEY = os.getenv("AWS_FACE_ACCESS_KEY", "")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-west-2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DAILY_LOG_PATH = str(DATA_DIR / os.getenv("AURA_DAILY_LOG_FILE", "daily_log.json"))
USER_PROFILE_PATH = str(DATA_DIR / os.getenv("AURA_USER_PROFILE_FILE", "user_profile.json"))
EVENT_STREAM_PATH = str(DATA_DIR / os.getenv("AURA_EVENT_STREAM_FILE", "event_stream.json"))
CONVERSATIONS_PATH = str(DATA_DIR / os.getenv("AURA_CONVERSATIONS_FILE", "conversations.json"))
MEMCELLS_PATH = str(DATA_DIR / os.getenv("AURA_MEMCELLS_FILE", "memcells.json"))
SQLITE_PATH = str(DATA_DIR / os.getenv("AURA_SQLITE_FILE", "aura.db"))

WHITELIST_DIR = os.getenv("AURA_WHITELIST_DIR", str((BASE_DIR / "whitelist").resolve()))
YOLO_DIR = os.getenv("AURA_YOLO_DIR", str((BASE_DIR / "yolo").resolve()))

LLM_INPUT_MODE = os.getenv("LLM_INPUT_MODE", "audio")
LLM_OUTPUT_MODE = os.getenv("LLM_OUTPUT_MODE", "audio")

WS_TOKEN_SECRET = os.getenv("AURA_WS_TOKEN_SECRET", "change-me-in-production")
WS_TOKEN_TTL_SECONDS = int(os.getenv("AURA_WS_TOKEN_TTL_SECONDS", "1800"))
ALLOWED_ORIGINS = os.getenv("AURA_ALLOWED_ORIGINS", "*")
