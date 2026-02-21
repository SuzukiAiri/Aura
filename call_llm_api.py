import json
import os

from config import DAILY_LOG_PATH, USER_PROFILE_PATH, EVENT_STREAM_PATH, MEMCELLS_PATH

def _read_json_as_text(file_path, default_text):
    """
    安全读取 JSON 文件并转为可读文本，文件不存在或异常时返回默认文本
    """
    if not os.path.exists(file_path):
        return default_text

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return default_text
    

def call_llm_api(client, rekognition, conversation_history):
    """
    ★ NEW
    多轮对话版本 LLM 调用
    emotion_list：仅在会话第一轮传入
    conversation_history：[
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    """

    # ===============================
    # 1. 情绪描述（仅首轮有效）
    # ===============================
    if rekognition:
        lines = []
        for face in rekognition['faces']:
            item_str = ", ".join([f"{k}:{v:.2f}" for k, v in face['emotions'].items()])
            lines.append(f"id: {face['matched_person_id']}, emotions: {{{item_str}}}")
        emotion_str = "\n".join(lines)
        # print(emotion_str)
    else:
        emotion_str = "（本轮不再重复提供情绪信息）"

    # ===============================
    # 2. 用户背景
    # ===============================
    user_profile_text = _read_json_as_text(
        USER_PROFILE_PATH, "用户画像暂无。"
    )
    daily_log_text = _read_json_as_text(
        DAILY_LOG_PATH, "暂无近期记录。"
    )
    event_stream_text = _read_json_as_text(
        EVENT_STREAM_PATH, "暂无当日画面记录。"
    )
    memcells_text = _read_json_as_text(
        MEMCELLS_PATH, "暂无原子记忆记录。"
    )

    # ===============================
    # 3. System Prompt
    # ===============================
    system_prompt = (
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
        "1. 每次回复以回答用户问题或提出建议为主\n"
        "2. 语气口语化、有分寸\n"
        "3. 不总结对话\n"
        "4. 不提及系统、模型、检测\n"
        "5. 使用中文"
        "6. 不要包含序号"
    )

    # ===============================
    # 4. Messages（关键）
    # ===============================
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"当前检测到的情绪信息：\n{emotion_str}\n\n"
                f"今天镜头前出现人物的画面的总结：\n{event_stream_text}"
            )
        }
    ]

    # ★ NEW：加入历史对话
    messages.extend(conversation_history)

    # ===============================
    # 5. 调用 LLM
    # ===============================
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
            max_output_tokens=200,
            temperature=0.7,
        )
        return resp.output_text.strip()
    except Exception as e:
        print("调用 LLM 失败：", e)
        return "我在听，你可以慢慢说。"
    