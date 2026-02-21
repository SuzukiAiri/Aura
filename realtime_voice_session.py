import asyncio
import base64
import json
import threading
import time
import inspect
import collections
from typing import Callable, List, Optional, Dict, Any

import numpy as np
import sounddevice as sd
import websockets

class RealtimeVoiceSession:
    """
    Realtime API Speech-to-Speech session (WebSocket).

    ✅ 关键改动：
    - 建立播放缓冲区（deque + OutputStream callback），防止流式 delta 播放卡顿
    - 用“缓冲是否耗尽 + 尾巴时间 + 是否仍在接收音频”来准确维护 assistant_speaking
    - 提供 playback_drained_event：等价于“播完”事件（OutputStream 场景不能用 sd.wait）
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
        self.mic_enabled = True
        self._audio_send_q: Optional[asyncio.Queue] = None

        # Speaking state (driven by playback buffer timeline)
        self._assistant_speaking = False
        self._speaking_lock = threading.Lock()

        self._last_activity_ts = time.monotonic()

        # Streaming assistant transcript
        self._assistant_text_deltas: Dict[str, List[str]] = {}

        # Callback
        self.on_session_end: Optional[Callable[[], None]] = None

        # ---------------- Playback buffer (anti-stutter) ----------------
        self._play_lock = threading.Lock()
        self._play_deque = collections.deque()  # elements are np.int16 arrays
        self._play_frames = 0                   # total samples in buffer (mono)
        self._play_max_frames = int(self.sample_rate * 2.0)  # cap buffer to 2 seconds

        # “播完”事件：缓冲排空 + 尾巴过去后置位
        self.playback_drained_event = threading.Event()
        self.playback_drained_event.set()  # initial drained

        # recv-side hint: whether server may still send more audio deltas
        self._receiving_audio = False
        self._last_audio_pop_mono = 0.0

        # tail window to absorb device/driver scheduling jitter
        self._spk_play_until_mono = 0.0   # 预计“声卡队列播放到”的时间线（monotonic）
        self._spk_tail_ms = 1000.0        # 建议外放 120~200ms；耳机 30~80ms

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
        self._audio_send_q = asyncio.Queue(maxsize=10)

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
            "output_modalities": ["audio"],
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
        frames_per_chunk = int(self.sample_rate * self.chunk_ms / 1000)

        def out_callback(outdata, frames, time_info, status):
            # outdata expects shape (frames, channels), dtype=int16
            need = frames
            out = np.zeros(need, dtype=np.int16)

            popped_any = False # ★ NEW：是否真的 pop 了音频

            with self._play_lock:
                while need > 0 and self._play_deque:
                    cur = self._play_deque[0]
                    take = min(need, cur.size)

                    start = frames - need
                    out[start:start + take] = cur[:take]

                    if take == cur.size:
                        self._play_deque.popleft()
                    else:
                        self._play_deque[0] = cur[take:]

                    self._play_frames -= take
                    need -= take

                    popped_any = True # ★ 发生真实消费

                if self._play_frames < 0:
                    self._play_frames = 0

            outdata[:] = out.reshape(-1, 1)
            if popped_any:
                now2 = time.monotonic()
                self._last_audio_pop_mono = now2

                # 本次 callback 实际输出的“音频样本数” = frames - need
                frames_filled = frames - need
                dur = frames_filled / float(self.sample_rate)

                # 推进“预计播放到”的时间线（核心）
                if self._spk_play_until_mono < now2:
                    self._spk_play_until_mono = now2
                self._spk_play_until_mono += dur
                self._last_activity_ts = self._spk_play_until_mono

            # speaking/drained 状态：以“输出回调时间线”为准
            now = time.monotonic()
            empty = self._buffer_is_empty()

            if not empty:
                self._set_assistant_speaking(True)
                self.playback_drained_event.clear()
                return

            # empty: 留 tail 窗口，避免刚好抖动导致过早关闭
            # 计算 output latency（有的设备返回 float，有的返回(in,out)）
            out_lat = 0.2
            # try:
            #     lat = getattr(self._out_stream, "latency", 0.0)
            #     out_lat = float(lat[1]) if isinstance(lat, (tuple, list)) else float(lat)
            #     print("outlat", out_lat)
            # except Exception:
            #     out_lat = 0.0

            tail = self._spk_tail_ms / 1000.0
            gate_until = self._spk_play_until_mono + out_lat + tail

            if (not self._receiving_audio) and (now >= gate_until):
                self._set_assistant_speaking(False)
                self.playback_drained_event.set()
            else:
                self._set_assistant_speaking(True)
                self.playback_drained_event.clear()

        self._out_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frames_per_chunk,
            callback=out_callback,
        )
        self._out_stream.start()

        def in_callback(indata, frames, time_info, status):
            if self._stop_flag.is_set():
                return
            if not self._loop or not self._audio_send_q:
                return
            pcm = indata.reshape(-1).tobytes()

            def try_put():
                try:
                    self._audio_send_q.put_nowait(pcm)
                except asyncio.QueueFull:
                    pass # ✅ 直接丢帧，不缓存

            self._loop.call_soon_threadsafe(try_put)

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

            # ✅ 防回声：只要“扬声器仍在播放时间线内”，就不上传麦克风
            if self._get_assistant_speaking():
                await asyncio.sleep(0.01)  # 节流
                continue

            if not self.mic_enabled:
                await asyncio.sleep(0.01)
                continue

            b64 = base64.b64encode(pcm).decode("ascii")
            await self._send_event({"type": "input_audio_buffer.append", "audio": b64})

    async def _recv_loop(self):
        async for msg in self._ws:
            try:
                evt = json.loads(msg)
            except Exception:
                continue

            etype = evt.get("type", "")
            self._last_activity_ts = time.monotonic()

            if self.debug_print_events:
                print("[RT EVT]", etype)

            if etype == "error":
                code = ((evt.get("error") or {}).get("code") or "")
                if code == "response_cancel_not_active":
                    continue
                print("Realtime error full:", evt)
                continue

            if etype == "input_audio_buffer.speech_started":
                # 你如果要支持 barge-in（插嘴），这里可以 cancel
                if self._get_assistant_speaking():
                    await self._send_event({"type": "response.cancel"})
                continue

            if etype in ("response.output_audio.delta", "response.audio.delta"):
                delta = evt.get("delta") or evt.get("audio")
                if delta:
                    pcm_bytes = base64.b64decode(delta)
                    self._play_pcm16(pcm_bytes)
                continue

            if etype == "response.output_audio_transcript.delta":
                response_id = evt.get("response_id") or evt.get("id") or "unknown"
                delta_text = (evt.get("delta") or evt.get("text") or "")
                if delta_text:
                    self._assistant_text_deltas.setdefault(response_id, []).append(delta_text)
                continue

            if etype == "conversation.item.input_audio_transcription.delta":
                delta_text = (evt.get("delta") or evt.get("text") or "")
                # if delta_text:
                #     print("input", delta_text)
                continue

            if etype == "conversation.item.input_audio_transcription.completed":
                transcript = (evt.get("transcript") or evt.get("text") or "").strip()
                if transcript:
                    print("input", transcript)
                    with self._history_lock:
                        self.conversation_history.append({"role": "user", "content": transcript})
                continue

            if etype == "conversation.item.input_audio_transcription.failed":
                with self._history_lock:
                    self.conversation_history.append({"role": "user", "content": "[语音转写失败]"})
                continue

            # ✅ 服务器明确“音频输出结束”（但声卡可能还有缓冲）
            if etype in ("response.output_audio.done", "response.audio.done"):
                self._receiving_audio = False
                # speaking 的关闭交给 out_callback：等缓冲排空+尾巴过去再关
                continue

            if etype == "response.done":
                self._receiving_audio = False

                resp = evt.get("response") or {}
                response_id = resp.get("id") or evt.get("response_id") or evt.get("id") or "unknown"
                transcript_text = "".join(self._assistant_text_deltas.pop(response_id, []))

                with self._history_lock:
                    self.conversation_history.append({"role": "assistant", "content": transcript_text})
                continue

    # ---------------- Playback buffer push ----------------

    def _play_pcm16(self, pcm_bytes: bytes):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        if not audio.size:
            return

        self._receiving_audio = True

        # 只要有新音频推入，就认为“正在说”，drained 取消
        self._set_assistant_speaking(True)
        self.playback_drained_event.clear()

        with self._play_lock:
            self._play_deque.append(audio.copy())
            self._play_frames += audio.size

    def _buffer_is_empty(self) -> bool:
        with self._play_lock:
            return self._play_frames <= 0

    # ---------------- Speaking helpers ----------------

    def _set_assistant_speaking(self, v: bool):
        with self._speaking_lock:
            self._assistant_speaking = v

    def _get_assistant_speaking(self) -> bool:
        with self._speaking_lock:
            return self._assistant_speaking

    # ---------------- WS send ----------------

    async def _send_event(self, event: dict):
        if not self._ws:
            return
        await self._ws.send(json.dumps(event, ensure_ascii=False))

    # ---------------- Idle watchdog ----------------

    async def _idle_watchdog(self):
        while not self._stop_flag.is_set():
            await asyncio.sleep(0.25)

            # 空闲判断：只有在“没有说话”时才允许触发 session end
            if (time.monotonic() - self._last_activity_ts) > self.idle_timeout_seconds and not self._get_assistant_speaking():
                if self.on_session_end:
                    try:
                        self.on_session_end()
                    except Exception:
                        pass
                break