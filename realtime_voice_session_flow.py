import asyncio
import base64
import json
import os
import threading
import time
import uuid
import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

class RealtimeVoiceSession:
    """
    Realtime API Speech-to-Speech session (WebSocket).

    ✅ 单一模式（不分首轮/之后轮）
      - server_vad: create_response=True, interrupt_response=True
      - mic uplink always enabled
      - 模型自动分轮 + 自动回复
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-realtime",
        voice: str = "marin",
        sample_rate: int = 24000,
        chunk_ms: int = 20,
        idle_timeout_seconds: float = 10.0,
        enable_input_transcription: bool = True,
        language: str = "zh",
        debug_print_events: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.idle_timeout_seconds = idle_timeout_seconds
        self.enable_input_transcription = enable_input_transcription
        self.language = language
        self.debug_print_events = debug_print_events

        # Conversation
        self.conversation_history: List[Dict[str, Any]] = []
        self._history_lock = threading.Lock()

        self._connected = threading.Event()
        self._stop_flag = threading.Event()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws = None

        # Audio
        self._out_stream: Optional[sd.OutputStream] = None
        self._in_stream: Optional[sd.InputStream] = None
        self.mic_enabled = True  # ✅ 单一模式：一直开
        self._audio_send_q: Optional[asyncio.Queue] = None

        # Response tracking
        self._assistant_speaking = False
        self._last_activity_ts = time.time()

        # Streaming assistant transcript (from response.output_audio_transcript.delta)
        self._assistant_text_deltas: Dict[str, List[str]] = {}

        # Callback
        self.on_session_end: Optional[Callable[[], None]] = None

    # ---------------- Public ----------------

    def start(self, instructions: str):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, args=(instructions,), daemon=True)
        self._thread.start()
        self._connected.wait(timeout=15)

    def stop(self):
        self._stop_flag.set()
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._close(), self._loop)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def send_text_and_speak_now(self, user_text: str, max_output_tokens: int = 1200):
        """
        主动插入一条“用户文本”，并强制让模型立刻回一段语音。
        适合：人脸触发首句 / 你想让模型马上说一句
        """
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._manual_speak_flow(user_text=user_text, max_output_tokens=max_output_tokens),
            self._loop,
        )

    # ---------------- Main loop ----------------

    def _run_loop(self, instructions: str):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main(instructions))

    async def _main(self, instructions: str):
        self._audio_send_q = asyncio.Queue()

        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        print("[RT] connecting", url)

        connect_kwargs = {}
        sig = inspect.signature(websockets.connect)
        if "additional_headers" in sig.parameters:
            connect_kwargs["additional_headers"] = headers
        else:
            connect_kwargs["extra_headers"] = headers

        async with websockets.connect(url, **connect_kwargs) as ws:
            self._ws = ws
            print("[RT] connected")

            self._start_audio_devices()

            # ✅ 单一模式：上来就 realtime server_vad（自动分轮+自动回复+可打断）
            await self._session_update_single_mode(instructions)
            print("[RT] session.update(single_mode) sent")

            self._connected.set()
            print("[RT] connected event set")

            recv_task = asyncio.create_task(self._recv_loop())
            send_task = asyncio.create_task(self._send_audio_loop())
            idle_task = asyncio.create_task(self._idle_watchdog())

            done, pending = await asyncio.wait(
                [recv_task, send_task, idle_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

        self._stop_audio_devices()

    async def _close(self):
        try:
            if self._ws:
                await self._ws.close()
        finally:
            self._stop_audio_devices()

    # ---------------- Session update (single mode) ----------------

    async def _session_update_single_mode(self, instructions: str):
        session = {
            "type": "realtime",
            "model": self.model,
            "output_modalities": ["audio"],  # Realtime 单次只支持 audio 或 text
            "instructions": instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.sample_rate},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 600,
                        "create_response": True,
                        "interrupt_response": False,
                    },
                    "noise_reduction": {"type": "near_field"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": self.sample_rate},
                    "voice": self.voice,
                },
            },
        }

        if self.enable_input_transcription:
            session["audio"]["input"]["transcription"] = {
                "model": "gpt-4o-transcribe",
                "language": self.language,
            }

        await self._send_event({"type": "session.update", "session": session})

    # ---------------- Manual speak (optional) ----------------

    async def _manual_speak_flow(self, user_text: str, max_output_tokens: int):
        """
        手动触发一轮回复：创建用户消息 + response.create
        不关闭麦克风（单一模式保持自然）
        但如果你希望“这句更稳不被噪声打断”，可以在外部暂时把 mic_enabled=False。
        """
        await self._send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            }
        })
        await self._send_event({
            "type": "response.create",
            "response": {"max_output_tokens": max_output_tokens}
        })

    # ---------------- Audio devices ----------------

    def _start_audio_devices(self):
        self._out_stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16", blocksize=0
        )
        self._out_stream.start()

        frames_per_chunk = int(self.sample_rate * self.chunk_ms / 1000)

        def in_callback(indata, frames, time_info, status):
            if self._stop_flag.is_set():
                return
            if not self._loop or not self._audio_send_q:
                return
            pcm = indata.reshape(-1).tobytes()
            self._loop.call_soon_threadsafe(self._audio_send_q.put_nowait, pcm)

        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frames_per_chunk,
            callback=in_callback,
        )
        self._in_stream.start()

    def _stop_audio_devices(self):
        try:
            if self._in_stream:
                self._in_stream.stop()
                self._in_stream.close()
        except Exception:
            pass
        try:
            if self._out_stream:
                self._out_stream.stop()
                self._out_stream.close()
        except Exception:
            pass

    # ---------------- WS loops ----------------

    async def _send_audio_loop(self):
        assert self._audio_send_q is not None
        while not self._stop_flag.is_set():
            pcm = await self._audio_send_q.get()

            if self._assistant_speaking:
                await asyncio.sleep(0.005)
                continue

            b64 = base64.b64encode(pcm).decode("ascii")
            await self._send_event({"type": "input_audio_buffer.append", "audio": b64})
            self._last_activity_ts = time.time()

    async def _recv_loop(self):
        async for msg in self._ws:
            try:
                evt = json.loads(msg)
            except Exception:
                continue

            etype = evt.get("type", "")
            self._last_activity_ts = time.time()

            if self.debug_print_events:
                print("[RT EVT]", etype)

            if etype == "error":
                code = ((evt.get("error") or {}).get("code") or "")
                if code == "response_cancel_not_active":
                    # 时机问题，直接忽略
                    continue
                print("Realtime error full:", evt)
                continue

            # Barge-in：用户说话时，若助手正在说，取消并清空播放缓冲
            if etype == "input_audio_buffer.speech_started":
                if self._assistant_speaking:
                    await self._send_event({"type": "response.cancel"})
                continue

            # Audio delta（两种名字都兼容）
            if etype in ("response.output_audio.delta", "response.audio.delta"):
                delta = evt.get("delta") or evt.get("audio")
                response_id = evt.get("response_id") or evt.get("id") or "unknown"
                if delta:
                    pcm_bytes = base64.b64decode(delta)
                    self._play_pcm16(pcm_bytes)
                continue

            # Assistant streaming transcript delta (字幕流)
            if etype == "response.output_audio_transcript.delta":
                response_id = evt.get("response_id") or evt.get("id") or "unknown"
                delta_text = (evt.get("delta") or evt.get("text") or "")
                if delta_text:
                    self._assistant_text_deltas.setdefault(response_id, []).append(delta_text)
                continue

            # User transcription -> history
            if etype == "conversation.item.input_audio_transcription.delta":
                response_id = evt.get("response_id") or evt.get("id") or "unknown"
                delta_text = (evt.get("delta") or evt.get("text") or "")
                print(delta_text)

            if etype == "conversation.item.input_audio_transcription.completed":
                transcript = (evt.get("transcript") or evt.get("text") or "").strip()
                if transcript:
                    with self._history_lock:
                        self.conversation_history.append({"role": "user", "content": transcript})
                continue

            if etype == "conversation.item.input_audio_transcription.failed":
                with self._history_lock:
                    self.conversation_history.append({"role": "user", "content": "[语音转写失败]"})
                continue

            # Response done -> finalize assistant turn（保存音频 + 文本）
            if etype == "response.done":

                # 规范：response id 在 evt["response"]["id"]
                resp = evt.get("response") or {}
                response_id = resp.get("id") or evt.get("response_id") or evt.get("id") or "unknown"

                # 从 _assistant_text_deltas 拿文本
                transcript_text = ''.join(self._assistant_text_deltas.pop(response_id, []))

                with self._history_lock:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": transcript_text
                    })
                continue

    # ---------------- Playback / Save ----------------

    def _play_pcm16(self, pcm_bytes: bytes):
        if not self._out_stream:
            return
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        if not audio.size:
            return

        # 再真正写入声卡
        self._out_stream.write(audio)

        # ✅ 关键：让 self._assistant_speaking 跟随扬声器时间线
        self._assistant_speaking = True

    async def _send_event(self, event: dict):
        if not self._ws:
            return
        await self._ws.send(json.dumps(event, ensure_ascii=False))

    # ---------------- Idle watchdog ----------------

    async def _idle_watchdog(self):
        while not self._stop_flag.is_set():
            await asyncio.sleep(0.25)

            if (time.time() - self._last_activity_ts) > self.idle_timeout_seconds and not self._assistant_speaking:
                if self.on_session_end:
                    try:
                        self.on_session_end()
                    except Exception:
                        pass
                break
            