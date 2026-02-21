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

from config import OPENAI_API_KEY, CONVERSATIONS_PATH, DAILY_LOG_PATH, USER_PROFILE_PATH, EVENT_STREAM_PATH, MEMCELLS_PATH
from decide_should_speak import decide_should_speak
from call_llm_api import call_llm_api, _read_json_as_text
from call_rekognition_api import call_rekognition_api

import event_stream

import json
from datetime import datetime

from realtime_voice_session import RealtimeVoiceSession

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
frame_queue = queue.Queue()

conversation_active = False

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
            print("会话开始 (Realtime)")

            # ---- 读取背景（你原来的 JSON）----
            user_profile_text = _read_json_as_text(USER_PROFILE_PATH, "用户画像暂无。")
            daily_log_text = _read_json_as_text(DAILY_LOG_PATH, "暂无近期记录。")
            event_stream_text = _read_json_as_text(EVENT_STREAM_PATH, "暂无当日画面记录。")
            memcells_text = _read_json_as_text(MEMCELLS_PATH, "暂无原子记忆记录。")

            # ---- 组装首轮信息（当作 user text）----
            # （Realtime 模型会直接产出语音，无需本地 TTS）
            lines = []
            for face in rekognition["faces"]:
                item_str = ", ".join([f"{k}:{v:.2f}" for k, v in face["emotions"].items()])
                lines.append(f"id: {face['matched_person_id']}, emotions: {{{item_str}}}")
            emotion_str = "\n".join(lines) if lines else "未检测到情绪"

            first_turn_text = (
                f"当前检测到的情绪信息：\n{emotion_str}\n\n"
                f"今天镜头前出现人物的画面总结：\n{event_stream_text}\n\n"
                "请用1-2句不超过50字，温暖自然地开场，并可顺势开启一个轻话题。"
            )

            # ---- 建立 Realtime session ----
            instructions = (
                "你是一个温柔、体贴、善于倾听的智能管家。\n"
                "你正在与用户进行自然的多轮语音对话。\n"
                "请结合上下文自然回应，不要重复寒暄。\n\n"
                "【用户画像】\n"
                f"{user_profile_text}\n\n"
                "【日志记录】\n"
                f"{daily_log_text}\n\n"
                "【原子记忆】\n"
                f"{memcells_text}\n\n"
                "规则：\n"
                "语气口语化、有分寸；不总结对话；不提及系统、模型、检测；使用中文；不要包含序号。"
            )

            session = RealtimeVoiceSession(
                api_key=OPENAI_API_KEY,
                model="gpt-realtime",
                voice="marin",
                debug_print_events=False,
                enable_input_transcription=True,
            )

            session.start(instructions=instructions)

            # 首轮稳：发首轮信息并让它说出来
            session.send_text_and_speak_now(first_turn_text, max_output_tokens=3000)

            while session.is_alive():
                time.sleep(1)

            # 保存（建议 snapshot）
            history_snapshot = list(session.conversation_history)
            save_conversation_to_json(history_snapshot, conversations_path)

            conversation_active = False
            print("会话结束 (Realtime)")
            last_greet_time = time.time()
            

    threading.Thread(target=worker, args=(frame_bgr.copy(),), daemon=True).start()

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
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)

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