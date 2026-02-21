import asyncio
import os
import queue
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import edge_tts
import numpy as np
import soundfile as sf
import whisper
from openai import OpenAI

from call_llm_api import call_llm_api
from call_rekognition_api import call_rekognition_api
from config import OPENAI_API_KEY
from decide_should_speak import decide_should_speak
import event_stream
from server.media_pipeline import (
    FrameThrottle,
    decode_jpeg_base64,
    decode_pcm16_base64,
    encode_pcm16_base64,
    float32_to_pcm16,
    pcm16_to_float32,
    resample_float32,
)
from server.storage import Storage

_WHISPER_MODEL: Optional[Any] = None
_WHISPER_LOCK = threading.Lock()


def get_whisper_model() -> Any:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            model_name = os.getenv("AURA_WHISPER_MODEL", "base")
            _WHISPER_MODEL = whisper.load_model(model_name)
    return _WHISPER_MODEL


class SessionWorker:
    def __init__(self, session_id: str, user_id: str, storage: Storage):
        self.session_id = session_id
        self.user_id = user_id
        self.storage = storage
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.websocket = None
        self._send_lock: Optional[asyncio.Lock] = None

        self.state = "idle"
        self.running = False
        self.closed = False

        self.conversation_history: List[Dict[str, str]] = []
        self.audio_buffers: List[np.ndarray] = []
        self.audio_sample_rate = 16000

        self.frame_queue: queue.Queue = queue.Queue(maxsize=120)
        self._event_stream_thread_started = False
        self.frame_ingest_throttle = FrameThrottle(fps=float(os.getenv("AURA_FRAME_INGEST_FPS", "2.0")))
        self.frame_detect_throttle = FrameThrottle(fps=float(os.getenv("AURA_FRAME_DETECT_FPS", "1.0")))

        self.conversation_active = False
        self.last_greet_time = 0.0
        self.last_rekognition = None
        self.face_detection_cooldown = float(os.getenv("AURA_FACE_DETECTION_COOLDOWN", "2.0"))
        self.session_end_prompt = os.getenv("AURA_SESSION_END_PROMPT", "如果有需要，可以再呼唤我。")

    async def attach(self, websocket: Any) -> None:
        self.websocket = websocket
        self._send_lock = asyncio.Lock()
        self.running = True
        self.storage.set_session_status(self.session_id, "connected")
        self._ensure_event_stream_thread()
        await self._set_state("idle", "session_connected")

    async def close(self, reason: str = "session_closed") -> None:
        if self.closed:
            return
        self.closed = True
        self.running = False
        self.storage.set_session_status(self.session_id, "closed")
        if self.conversation_history:
            self.storage.append_conversation_json(self.conversation_history)
        await self._safe_send({"type": "status", "state": "closed", "detail": reason})

    async def handle_client_message(self, message: Dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "audio_chunk":
            await self._handle_audio_chunk(message)
            return
        if kind == "video_frame":
            await self._handle_video_frame(message)
            return
        if kind == "control":
            await self._handle_control(message)
            return
        await self._send_error("bad_message", f"unknown message type: {kind}")

    async def _handle_audio_chunk(self, message: Dict[str, Any]) -> None:
        if "pcm16_base64" not in message:
            await self._send_error("bad_audio_chunk", "missing pcm16_base64")
            return
        try:
            chunk = decode_pcm16_base64(message["pcm16_base64"])
            if chunk.size == 0:
                return
            self.audio_sample_rate = int(message.get("sample_rate", self.audio_sample_rate))
            self.audio_buffers.append(chunk)
            if self.state == "idle":
                await self._set_state("listening", "receiving_audio")
        except Exception as exc:
            await self._send_error("audio_decode_failed", str(exc))

    async def _handle_video_frame(self, message: Dict[str, Any]) -> None:
        if not self.frame_ingest_throttle.allow():
            return
        frame = decode_jpeg_base64(message.get("jpeg_base64", ""))
        if frame is None:
            await self._send_error("bad_video_frame", "invalid jpeg payload")
            return

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        if not self.frame_detect_throttle.allow():
            return

        now = time.time()
        if now - self.last_greet_time < self.face_detection_cooldown:
            return

        image_bytes = self._image_to_jpeg_bytes(frame)
        if not image_bytes:
            return

        rekognition = await asyncio.to_thread(call_rekognition_api, image_bytes)
        if not rekognition:
            return

        prev_rekognition = self.last_rekognition
        self.last_rekognition = rekognition
        self.storage.append_event(
            self.session_id,
            "emotion_update",
            {"faces": rekognition.get("faces", [])},
        )
        await self._safe_send({"type": "emotion_update", "faces": rekognition.get("faces", [])})

        should_speak = decide_should_speak(rekognition, self.last_greet_time, prev_rekognition)
        if should_speak and not self.conversation_active:
            await self._auto_greet(rekognition)

    async def _handle_control(self, message: Dict[str, Any]) -> None:
        action = message.get("action")
        if action == "end_turn":
            await self._process_end_turn()
            return
        if action == "stop":
            await self.close("client_stop")
            return
        await self._send_error("bad_control", f"unknown control action: {action}")

    async def _auto_greet(self, rekognition: Dict[str, Any]) -> None:
        self.conversation_active = True
        await self._set_state("thinking", "face_triggered")
        reply = await asyncio.to_thread(call_llm_api, self.client, rekognition, self.conversation_history)
        await self._append_assistant_reply(reply)
        self.last_greet_time = time.time()
        self.conversation_active = True

    async def _process_end_turn(self) -> None:
        if not self.audio_buffers:
            if self.conversation_active:
                await self._append_assistant_reply(self.session_end_prompt)
                self.conversation_active = False
                await self._set_state("idle", "conversation_ended_by_silence")
            else:
                await self._set_state("listening", "no_audio_buffered")
            return

        await self._set_state("thinking", "transcribing_audio")
        audio = np.concatenate(self.audio_buffers, axis=0)
        self.audio_buffers = []

        user_text = await asyncio.to_thread(self._transcribe_audio, audio, self.audio_sample_rate)
        if not user_text:
            await self._set_state("listening", "empty_transcription")
            return

        self._append_message("user", user_text)
        await self._safe_send({"type": "assistant_text_delta", "text": f"[用户] {user_text}"})

        reply = await asyncio.to_thread(call_llm_api, self.client, None, self.conversation_history)
        await self._append_assistant_reply(reply)
        self.last_greet_time = time.time()
        self.conversation_active = True

    async def _append_assistant_reply(self, reply: str) -> None:
        self._append_message("assistant", reply)
        await self._safe_send({"type": "assistant_text_delta", "text": reply})
        await self._speak_reply(reply)
        await self._set_state("listening", "assistant_done")

    def _append_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        self.storage.append_message(self.session_id, role, content)

    def _transcribe_audio(self, pcm16: np.ndarray, sample_rate: int) -> str:
        if pcm16.size == 0:
            return ""
        model = get_whisper_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            sf.write(path, pcm16_to_float32(pcm16), sample_rate, subtype="PCM_16")
            result = model.transcribe(path, language="zh")
            text = result.get("text", "").strip()
            return text
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def _speak_reply(self, text: str) -> None:
        await self._set_state("speaking", "tts_streaming")
        voice = os.getenv("AURA_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
        target_rate = int(os.getenv("AURA_OUTPUT_SAMPLE_RATE", "24000"))
        chunk_ms = int(os.getenv("AURA_OUTPUT_CHUNK_MS", "20"))
        chunk_samples = max(1, int(target_rate * chunk_ms / 1000))

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(mp3_path)
            wav, sr = sf.read(mp3_path, dtype="float32")
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass

        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = resample_float32(np.asarray(wav, dtype=np.float32), sr, target_rate)
        pcm16 = float32_to_pcm16(wav)

        for i in range(0, pcm16.size, chunk_samples):
            chunk = pcm16[i : i + chunk_samples]
            await self._safe_send(
                {
                    "type": "assistant_audio_chunk",
                    "sample_rate": target_rate,
                    "pcm16_base64": encode_pcm16_base64(chunk),
                }
            )

    async def _set_state(self, state: str, detail: str) -> None:
        if state != self.state:
            self.state = state
            self.storage.set_session_status(self.session_id, state)
        await self._safe_send({"type": "status", "state": state, "detail": detail})

    async def _send_error(self, code: str, message: str) -> None:
        self.storage.append_event(self.session_id, "error", {"code": code, "message": message})
        await self._safe_send({"type": "error", "code": code, "message": message})

    async def _safe_send(self, payload: Dict[str, Any]) -> None:
        if not self.websocket or not self._send_lock:
            return
        try:
            async with self._send_lock:
                await self.websocket.send_json(payload)
        except Exception:
            self.running = False

    def _ensure_event_stream_thread(self) -> None:
        if self._event_stream_thread_started:
            return
        self._event_stream_thread_started = True
        thread = threading.Thread(
            target=event_stream.start_event_stream_thread,
            args=(self.frame_queue,),
            daemon=True,
        )
        thread.start()

    @staticmethod
    def _image_to_jpeg_bytes(img_bgr: np.ndarray) -> Optional[bytes]:
        ok, buffer = cv2.imencode(".jpg", img_bgr)
        return buffer.tobytes() if ok else None
