import os
import cv2
import time
import threading
import queue
import tempfile

import numpy as np
import psutil
import tracemalloc
import comtypes.client

import sounddevice as sd
import soundfile as sf
import whisper

from openai import OpenAI

from config import OPENAI_API_KEY, CONVERSATIONS_PATH
from decide_should_speak import decide_should_speak
from call_llm_api import call_llm_api
from call_rekognition_api import call_rekognition_api

import event_stream

import json
from datetime import datetime

import edge_tts
import asyncio

# =========================
# OpenAI
# =========================
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# =========================
# ★ NEW：会话参数
# =========================
SESSION_END_PROMPT = "如果有需要，可以再呼唤我。"

# =========================
# 全局状态
# =========================
tts_queue = queue.Queue()
frame_queue = queue.Queue()

conversation_active = False
tts_speaking_event = threading.Event()

FACE_DETECTION_COOLDOWN = 2.0
processing_lock = threading.Lock()

last_greet_time = 0.0
last_rekognition = None
last_face_seen = 0.0

conversations_path = CONVERSATIONS_PATH

# =========================
# 性能监控
# =========================
tracemalloc.start()
process = psutil.Process(os.getpid())

# =========================
# 工具函数
# =========================
def image_to_jpeg_bytes(img_bgr):
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    return buffer.tobytes() if is_success else None


async def tts(text):
    communicate = edge_tts.Communicate(
        text=text,
        voice="zh-CN-XiaoxiaoNeural"
    )
    await communicate.save("temp.mp3")

    data, samplerate = sf.read("temp.mp3", dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def tts_worker():
    while True:
        text = tts_queue.get()
        try:
            if text:
                # ★ NEW：标记正在说话
                tts_speaking_event.set()

                # speaker = comtypes.client.CreateObject("SAPI.SpVoice")

                # speaker.Speak(text)

                asyncio.run(tts(text))

        except Exception as e:
            print("TTS 出错：", e)

        finally:
            # ★ NEW：说话结束
            tts_speaking_event.clear()
            tts_queue.task_done()

# ★ NEW：加载本地 Whisper 模型
whisper_model = whisper.load_model("base") # 你可以根据需求选择不同的模型（如 base, small, medium, large）

def local_stt(wav_path):
    """
    使用 Whisper 本地模型进行语音转文本
    """
    result = whisper_model.transcribe(wav_path, language="zh")
    return result["text"].strip()

# =========================
# ★ MOD：修改 listen_from_microphone 逻辑
# =========================
def listen_from_microphone(
    least_duration=5.0, # 最小说话时长
    silence_duration=1.5, # ★ NEW：连续静音多久算结束
    volume_threshold=0.1 # ★ NEW：音量阈值（可调）
):
    """
    实时监听麦克风：
    - 用户开始说话后开始录音
    - 连续 silence_duration 秒无声音 → 停止并 STT
    """

    # ★ NEW：TTS 播放期间不监听
    while tts_speaking_event.is_set():
        time.sleep(0.05)

    fs = 16000
    channels = 1
    chunk_duration = 1 # ★ NEW：100ms 一个 chunk
    chunk_size = int(fs * chunk_duration)

    audio_buffer = [] # ★ NEW：存放有效音频
    last_voice_time = None
    start_time = time.time()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name

    try:
        with sd.InputStream(
            samplerate=fs,
            channels=channels,
            blocksize=chunk_size,
            dtype="float32"
        ) as stream:
            
            last_voice_time = time.time()

            while True:
                chunk, _ = stream.read(chunk_size)
                chunk = chunk.reshape(-1)

                volume = np.max(np.abs(chunk))

                if volume > volume_threshold:
                    # ★ NEW：检测到说话
                    print("检测到说话", volume)
                    speaking = True
                    last_voice_time = time.time()
                    audio_buffer.append(chunk)
                else:
                    current_time = time.time()
                    if current_time - start_time > least_duration and current_time - last_voice_time > silence_duration:
                        break # ★ NEW：确认用户说完了

        # ★ NEW：如果从未说话
        if not audio_buffer:
            return None

        audio_data = np.concatenate(audio_buffer, axis=0)
        sf.write(filename, audio_data, fs)

        user_text = local_stt(filename)
        print("user_text:", user_text)

        return user_text.strip() if user_text else None

    except Exception as e:
        print("STT 失败：", e)
        return None
    

def save_conversation_to_json(conversation_history, conversations_path):
    """
    将一次完整会话追加保存到 conversations.json
    文件不存在则创建，存在则追加
    """
    if not conversation_history:
        return

    conversations = []

    # 读取已有文件
    if os.path.exists(conversations_path):
        try:
            with open(conversations_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    conversations = json.loads(content)
        except Exception:
            conversations = []

    # 追加本次会话（加时间戳，方便后续总结）
    conversations.append({
        "time": datetime.now().replace(microsecond=0).isoformat(),
        "messages": conversation_history
    })

    # 写回文件
    with open(conversations_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

    print("会话已保存到 conversations.json")


# =========================
# ★ MOD：人脸 → 会话（加入自动结束）
# =========================
def handle_face_detected(frame_bgr):
    global conversation_active, last_greet_time, last_rekognition, last_face_seen

    if processing_lock.locked():
        return

    def worker(img):
        global conversation_active, last_greet_time, last_rekognition, last_face_seen

        with processing_lock:
            last_face_seen = time.time()

            image_bytes = image_to_jpeg_bytes(img)
            if not image_bytes:
                return
            
            rekognition = call_rekognition_api(image_bytes)
            if not rekognition:
                return  

            should = decide_should_speak(rekognition, last_greet_time, last_rekognition)
            if not should:
                last_rekognition = rekognition
                conversation_active = False
                return
            
            conversation_active = True
            last_rekognition = rekognition

            print("会话开始")
            # =========================
            # 会话上下文
            # =========================
            conversation_history = []

            # ---------- 第一轮 ----------
            reply = call_llm_api(client, rekognition, conversation_history)
            conversation_history.append(
                {"role": "assistant", "content": reply}
            )
            tts_speaking_event.set()
            tts_queue.put(reply)
            print("reply_text: ", reply)

            last_greet_time = time.time()

            # ---------- 多轮 ----------
            while conversation_active:
                user_text = listen_from_microphone()

                # ★ NEW：沉默超时 → 结束会话
                if user_text is None:
                    tts_speaking_event.set()
                    tts_queue.put(SESSION_END_PROMPT)

                    last_greet_time = time.time()
                    break

                conversation_history.append(
                    {"role": "user", "content": user_text}
                )

                reply = call_llm_api(client, None, conversation_history)
                conversation_history.append(
                    {"role": "assistant", "content": reply}
                )

                tts_speaking_event.set()
                tts_queue.put(reply)

                last_greet_time = time.time()
                

            conversation_active = False
            print("会话结束")

            # 保存会话到 JSON
            save_conversation_to_json(conversation_history, conversations_path)

    threading.Thread(
        target=worker,
        args=(frame_bgr.copy(),),
        daemon=True
    ).start()

# =========================
# 主循环（基本不变）
# =========================
def main():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(
        target=event_stream.start_event_stream_thread,
        args=(frame_queue,),
        daemon=True
    ).start()

    print("按 q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(80, 80)
        )

        now = time.time()
        if len(faces) > 0 and now - last_face_seen > FACE_DETECTION_COOLDOWN:
            handle_face_detected(frame)

        # 放入事件流提取队列
        frame_queue.put(frame)

        # 在画面上画出人脸框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)

        # 显示情绪/状态提示（如果有 last_rekognition）
        if last_rekognition is not None:
            y_offset = 30  # 第一行起始位置

            for face in last_rekognition['faces']:
                emotions = face['emotions']
                if not emotions:
                    top_emotion = "N/A"
                else:
                    top_emotion = max(emotions.items(), key=lambda kv: kv[1])[0]

                cv2.putText(
                    frame,
                    f"id: {face['matched_person_id']}, last emotion: {top_emotion}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                y_offset += 30  # 每张脸单独占一行

        cv2.imshow("Face Greeter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
