import json
import os
import base64
import numpy as np

import re

from config import DAILY_LOG_PATH, USER_PROFILE_PATH, EVENT_STREAM_PATH, LLM_INPUT_MODE, LLM_OUTPUT_MODE

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
    

def extract_between(text, start_tag, end_tag):
    """
    从 text 中提取 start_tag 和 end_tag 之间的内容
    找不到则返回 None
    """
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    match = re.search(pattern, text, re.S)
    return match.group(1).strip() if match else None
    

def call_llm_api(
    client,
    rekognition,
    conversation_history,
    input_audio_pcm16=None
):
    # ===============================
    # 1. Emotion（仅首轮）
    # ===============================
    if rekognition:
        lines = []
        for face in rekognition["faces"]:
            emo = ", ".join([f"{k}:{v:.2f}" for k, v in face["emotions"].items()])
            lines.append(f"id:{face['matched_person_id']} {{{emo}}}")
        emotion_str = "\n".join(lines)
    else:
        emotion_str = "（无新的情绪信息）"

    # ===============================
    # 2. Context
    # ===============================
    user_profile_text = _read_json_as_text(USER_PROFILE_PATH, "用户画像暂无。")
    daily_log_text = _read_json_as_text(DAILY_LOG_PATH, "暂无记录。")
    event_stream_text = _read_json_as_text(EVENT_STREAM_PATH, "暂无画面记录。")

    # ===============================
    # 3. System Prompt
    # ===============================
    system_prompt = (
        "你是一个温柔、体贴、善于倾听的智能管家。\n"
        "请自然对话，不重复寒暄，不提及系统。\n"
        "使用中文，口语化。\n"
        "当用户通过语音输入时，请你遵循以下规则：\n"
        "1. 你必须准确识别用户刚才说的话\n"
        "2. 在你的回复中，先输出一段：\n"
        "<user_transcript>这里是你听到的用户原话</user_transcript>\n"
        "3. 然后再给出正常的口语化回复\n"
        "4. <user_transcript> 内容不要润色，不要补全\n"
        "5. 不要向用户解释这个标签的存在\n"
        "没有语音输入时则忽略上述五条规则。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"情绪信息：\n{emotion_str}\n\n"
                f"画面总结：\n{event_stream_text}\n\n"
                f"用户画像：\n{user_profile_text}\n\n"
                f"近期日常：\n{daily_log_text}"
            )
        }
    ]

    messages.extend(conversation_history)

    # ===============================
    # ★ AUDIO LLM CALL
    # ===============================
    result = call_gpt4o_mini_audio(
        client,
        messages=messages,
        input_audio_pcm16=input_audio_pcm16 if LLM_INPUT_MODE == "audio" else None,
        output_mode=LLM_OUTPUT_MODE
    )

    return result
    

def pcm16_to_base64(audio_pcm16: np.ndarray):
    return base64.b64encode(audio_pcm16.tobytes()).decode("utf-8")
    

def call_gpt4o_mini_audio(
    client,
    messages,
    input_audio_pcm16=None,
    input_samplerate=16000,
    output_mode="text"  # "text" | "audio"
):
    """
    messages: 标准 messages（system + history）
    input_audio_pcm16: np.int16 mono
    """

    input_payload = messages

    if input_audio_pcm16 is not None:
        input_payload.append({
            "role": "user",
            "content": [{
                "type": "input_audio",
                "audio": pcm16_to_base64(input_audio_pcm16),
                "format": "pcm16",
                "sample_rate": input_samplerate
            }]
        })

    # resp = client.responses.create(
    #     model="gpt-4o",
    #     input=input_payload,
    #     # modalities=["text", "audio"] if output_mode == "audio" else ["text"],
    #     # audio={"voice": "alloy"} if output_mode == "audio" else None,
    #     temperature=0.7,
    #     max_output_tokens=300,
    # )

    # ===== 解析输出 =====
    # output_text = resp.output_text.strip() if resp.output_text else ""

    # output_audio = None
    # transcript = output_text

    # for item in resp.output:
    #     for content in item["content"]:
    #         if content["type"] == "output_audio":
    #             audio_b64 = content["audio"]
    #             audio_pcm16 = np.frombuffer(
    #                 base64.b64decode(audio_b64),
    #                 dtype=np.int16
    #             )
    #             output_audio = audio_pcm16
    #             transcript = content.get("transcript", transcript)
    #             user_transcript = extract_between(transcript, "<user_transcript>", "</user_transcript>")
    #             assistant_text = transcript.replace(f"<user_transcript>{user_transcript}</user_transcript>", "").strip()

    resp = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=input_payload,
        modalities=["text", "audio"] if output_mode == "audio" else ["text"],
        audio={"voice": "marin", "format": "pcm16"} if output_mode == "audio" else None,
        temperature=0.7,
        max_tokens=500,
    )

    choice = resp.choices[0]
    message = choice.message

    output_audio = None
    transcript = None
    output_text = None
    user_transcript = None

    # audio output
    if message.audio:
        audio_b64 = message.audio.data
        output_audio = np.frombuffer(
            base64.b64decode(audio_b64),
            dtype=np.int16
        )
        transcript = message.audio.transcript

    # text output（兜底）
    if message.content:
        output_text = message.content

    # 解析 <user_transcript>
    if transcript:
        user_transcript = extract_between(
            transcript,
            "<user_transcript>",
            "</user_transcript>"
        )

        if user_transcript:
            output_text = transcript.replace(
                f"<user_transcript>{user_transcript}</user_transcript>",
                ""
            ).strip()

    return {
        "input_text": user_transcript,
        "output_text": output_text,
        "output_audio": output_audio
    }