import base64
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def decode_pcm16_base64(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload)
    return np.frombuffer(raw, dtype=np.int16).copy()


def encode_pcm16_base64(samples: np.ndarray) -> str:
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16)
    return base64.b64encode(samples.tobytes()).decode("ascii")


def decode_jpeg_base64(payload: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def resample_float32(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    dst_size = max(1, int(round(audio.size * dst_rate / src_rate)))
    dst_x = np.linspace(0.0, 1.0, num=dst_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def pcm16_to_float32(samples: np.ndarray) -> np.ndarray:
    return np.clip(samples.astype(np.float32) / 32768.0, -1.0, 1.0)


def float32_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


@dataclass
class FrameThrottle:
    fps: float
    _last_ts: float = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if self.fps <= 0:
            return True
        period = 1.0 / self.fps
        if now - self._last_ts < period:
            return False
        self._last_ts = now
        return True
