import re
import cv2
import time
import json
import numpy as np
from datetime import datetime
from config import (
    GEMINI_API_KEY,
    DAILY_LOG_PATH,
    USER_PROFILE_PATH,
    EVENT_STREAM_PATH,
    CONVERSATIONS_PATH,
    MEMCELLS_PATH,
    YOLO_DIR,
    DATA_DIR,
)

import os
from google import genai

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        client = None
        print("Gemini client init failed, event summary disabled:", e)
else:
    client = None
    print("Gemini API key missing, event summary disabled.")

# 事件流存储
event_stream = []

# 存储标识及容器
video_writing = False
out_mp4 = None
video_start_time = None
captured_video_path = str(DATA_DIR / "captured_video.mp4")
event_stream_path = EVENT_STREAM_PATH
daily_log_path = DAILY_LOG_PATH
user_profile_path = USER_PROFILE_PATH
conversations_path = CONVERSATIONS_PATH
memcells_path = MEMCELLS_PATH
current_date = datetime.now().date()

# 隔帧检测，加快处理时间
detect_cnt = 0
detect_per_frames = 20

# Load YOLOv3 model.
try:
    net = cv2.dnn.readNet(
        os.path.join(YOLO_DIR, "yolov3.weights"),
        os.path.join(YOLO_DIR, "yolov3.cfg"),
    )
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    net = None
    output_layers = []
    print("YOLO model load failed, event stream detector disabled:", e)


def save_daily_log(date, summary):
    logs = []

    if os.path.exists(daily_log_path):
        try:
            with open(daily_log_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
        except Exception:
            logs = []

    logs.append({
        "date": date.isoformat(),
        "summary": summary
    })

    with open(daily_log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print("已写入 daily_log.json")


def extract_json_from_text(text: str):
    """
    从模型输出中提取 JSON 对象（支持 ```json ... ``` 代码块、或纯 JSON）。
    解析失败返回 None。
    """
    if not text:
        return None

    t = text.strip()

    # 1) 优先处理 ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2) 尝试直接解析纯 JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # 3) 兜底：尝试截取第一个 { 到最后一个 }（有时模型会在 JSON 前后加说明）
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = t[l:r+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


memcells_path = MEMCELLS_PATH

def load_memcells():
    if not os.path.exists(memcells_path):
        return []
    try:
        with open(memcells_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_memcells(memcells):
    with open(memcells_path, "w", encoding="utf-8") as f:
        json.dump(memcells, f, ensure_ascii=False, indent=2)

def _next_memcell_id(existing: list) -> str:
    mx = 0
    for m in existing:
        if isinstance(m, dict):
            mid = str(m.get("id", ""))
            if mid.startswith("m_"):
                try:
                    mx = max(mx, int(mid.split("_", 1)[1]))
                except Exception:
                    pass
    return f"m_{mx+1:06d}"

def merge_memcells(existing: list, extracted: list, today_iso: str):
    """
    extracted: 由模型输出的 memcell patch 列表，每个元素结构：
      - match_id: 可选，命中旧 memcell 的 id
      - type: 必填
      - content: 当 match_id 为空时必填
      - confidence/sources/tags: 可选
    """
    if not isinstance(existing, list):
        existing = []
    if not isinstance(extracted, list):
        extracted = []

    # 建索引：按 id
    id_index = {}
    merged = []

    for m in existing:
        if not isinstance(m, dict):
            continue
        if "id" not in m or not str(m["id"]).strip():
            m["id"] = _next_memcell_id(existing)  # 极少发生：老数据没 id 的兜底
        m.setdefault("first_seen", today_iso)
        m.setdefault("last_seen", m.get("first_seen", today_iso))
        m.setdefault("count", 1)
        m.setdefault("sources", [])
        m.setdefault("confidence", 0.5)
        merged.append(m)
        id_index[str(m["id"])] = m

    for n in extracted:
        if not isinstance(n, dict):
            continue

        match_id = str(n.get("match_id", "")).strip()
        n_type = str(n.get("type", "")).strip()
        n_content = str(n.get("content", "")).strip()

        if not n_type:
            continue

        if match_id and match_id in id_index:
            m = id_index[match_id]
            m["last_seen"] = today_iso
            m["count"] = int(m.get("count", 1)) + 1

            # sources 合并去重
            src = n.get("sources", [])
            if isinstance(src, list):
                ms = m.get("sources", [])
                if not isinstance(ms, list):
                    ms = []
                m["sources"] = sorted(set([str(x) for x in (ms + src) if str(x).strip()]))

            # tags 合并去重
            if isinstance(n.get("tags"), list):
                mt = m.get("tags", [])
                if not isinstance(mt, list):
                    mt = []
                m["tags"] = sorted(set([str(x) for x in (mt + n["tags"]) if str(x).strip()]))

            # confidence 轻微上调（封顶 0.95）
            try:
                cur = float(m.get("confidence", 0.5))
                inc = float(n.get("confidence", 0.55)) - 0.5
                m["confidence"] = max(0.05, min(0.95, cur + max(0.02, inc * 0.2)))
            except Exception:
                pass

        else:
            # 新增：必须有 content
            if not n_content:
                continue
            new_id = _next_memcell_id(merged)
            item = {
                "id": new_id,
                "type": n_type,
                "content": n_content,
                "first_seen": today_iso,
                "last_seen": today_iso,
                "count": 1,
                "confidence": float(n.get("confidence", 0.55)) if isinstance(n.get("confidence"), (int, float)) else 0.55,
                "sources": n.get("sources", []) if isinstance(n.get("sources"), list) else [],
            }
            if isinstance(n.get("tags"), list):
                item["tags"] = [str(x) for x in n["tags"] if str(x).strip()]
            merged.append(item)
            id_index[new_id] = item

    return merged

def extract_json_array_from_text(text: str):
    """
    从模型输出中提取 JSON 数组（支持 ```json [...] ``` 或纯 [...]）。
    解析失败返回 None。
    """
    if not text:
        return None
    t = text.strip()

    m = re.search(r"```(?:json)?\s*($$[\s\S]*?$$)\s*```", t, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, list) else None
    except Exception:
        pass

    l = t.find("[")
    r = t.rfind("]")
    if l != -1 and r != -1 and r > l:
        candidate = t[l:r+1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, list) else None
        except Exception:
            return None
    return None

def generate_memcells_for_day(date_obj, daily_summary_text, conversations, existing_memcells, max_catalog=120):
    if client is None:
        return []
    """
    用 Gemini 抽取 MemCell（原子记忆），并优先复用旧 memcell 的规范表述（通过 match_id）。
    """
    today_iso = date_obj.isoformat()

    # 目录裁剪：优先拿最近/更强的（count 高、last_seen 新）
    catalog = []
    for m in existing_memcells if isinstance(existing_memcells, list) else []:
        if isinstance(m, dict) and m.get("id") and m.get("type") and m.get("content"):
            catalog.append({
                "id": m["id"],
                "type": m["type"],
                "content": m["content"],
                "count": m.get("count", 1),
                "last_seen": m.get("last_seen", "")
            })
    catalog.sort(key=lambda x: (x.get("last_seen",""), x.get("count",1)), reverse=True)
    catalog = catalog[:max_catalog]

    prompt = f"""
    你是 EverMemOS 风格的“原子记忆抽取器（MemCell Extractor）”，并且你必须遵循“规范复用”策略：
    - 系统已有一份历史 MemCell 目录（Memory Catalog），每条都有稳定 id、type、content（content 是规范表述）
    - 你从今天的日记总结与对话中抽取长期记忆时：
    1) 如果语义与目录中某条相同/高度相似，必须输出 match_id = 该条 id，并且不要新写 content
    2) 只有当目录里找不到等价语义时，才新增一条（match_id 为空，提供新的 content）
    - 目标：最大化复用目录中的 content，让同一语义落到同一个 id 上

    【什么信息可以成为 MemCell】
    - 长期可复用：偏好、稳定习惯、长期目标、反复压力源、稳定人际/工作模式、持续健康趋势等
    - 一次性事件、偶发行为、短期情绪波动：不要写入长期记忆
    - 不确定就不要编造，宁缺毋滥

    【输出要求】
    - 只输出 JSON 数组（不要 markdown，不要解释）
    - 数组元素是对象，字段：
    - match_id: 字符串或空字符串（命中目录则给 id，否则为空）
    - type: 枚举之一：
        ["preference","habit","goal","stress_trigger","value","health","relationship","work_pattern","communication_style","emotion_baseline","environment_preference","other"]
    - content: 仅当 match_id 为空时提供；一句话中文、客观可复用（不要带日期/一次性细节）
    - confidence: 0.0~1.0（越像长期稳定事实越高）
    - sources: 字符串数组，可选值：["daily_summary","conversation"]
    - tags: 字符串数组（可选）

    【数量与去重】
    - 总数 3~12 条
    - 不要语义重复；同一语义只能输出一次（要么 match_id，要么新增 content）

    【Memory Catalog（历史规范记忆，优先复用）】
    {json.dumps(catalog, ensure_ascii=False)}

    【今日输入】
    日记总结：
    {daily_summary_text}

    对话记录：
    {json.dumps(conversations, ensure_ascii=False)}
    """

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    arr = extract_json_array_from_text(resp.text)
    if not isinstance(arr, list):
        return []
    return arr


def load_existing_profile():
    if not os.path.exists(user_profile_path):
        return {}
    try:
        with open(user_profile_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception:
        return {}


def generate_user_profile():
    if client is None:
        return
    if not os.path.exists(daily_log_path):
        return

    with open(daily_log_path, "r", encoding="utf-8") as f:
        daily_logs = json.load(f)

    old_profile = load_existing_profile()

    prompt = f"""
    你是一名长期用户记忆与画像建模专家，负责维护一个“可演化但高度稳定”的用户画像系统。
    该系统结合 PersonalTree（人格树）与 EverMemOS（长期记忆 OS）：
    - 核心人格、生活模式属于“树干”，极其稳定
    - 兴趣、偏好属于“树枝”，允许逐步演化
    - 新信息必须经过多次、跨天、重复证据才能写入长期记忆
    - 所有更新遵循：ADD / UPDATE / KEEP / DEPRECATE 的最小改动原则

    你将收到：
    1) 过去多天的生活记录 daily_log（短期记忆，有噪声）
    2) 上一次的用户画像 old_profile（长期稳定记忆）

    你的任务：
    在最大程度继承 old_profile 的前提下，
    基于 daily_log 中**重复出现的可验证证据**，
    生成“更新后的用户画像”（长期稳定画像）。

    ------------------------------------------------------------
    【非常重要：记忆更新原则】
    - 单次行为 ≠ 长期特征，不要更新画像
    - 情绪波动 ≠ 个性，不要更新 personality_traits
    - 偶然尝试 ≠ 兴趣，不要加入 hobbies
    - 若出现矛盾，新证据不足 → 保留 old_profile
    - 允许扩展字段，但必须是长期特征，而非短期事件

    ------------------------------------------------------------
    【字段要求】
    你必须输出以下字段，并可以在此基础上增加更多长期字段：

    必含字段（稳定画像树干/主枝）：
    - lifestyle（字符串）
    - work_pattern（字符串）
    - hobbies（字符串数组，去重 ≤ 8）
    - social_level（字符串）
    - personality_traits（字符串数组，去重 ≤ 8）

    新增必备（扩展长期记忆字段）：
    - health_pattern（字符串） # 作息与健康习惯趋势
    - emotion_baseline（字符串） # 长期情绪基线（如平稳/敏感/积极）
    - communication_style（字符串）  # 表达方式（如直接/含蓄/情绪化/逻辑强）
    - values_and_beliefs（字符串数组） # 价值观倾向（如家庭优先、自我提升）
    - stress_triggers（字符串数组 ≤ 6） # 重复出现的典型压力来源
    - goals_long_term（字符串数组 ≤ 6） # 长期目标（非短期任务）
    - environment_preferences（字符串数组）# 环境偏好（安静/整洁/自然光等）
    - energy_pattern（字符串） # 典型的日常能量节律（如上午效率高）

    允许扩展：
    - 你可以根据证据自行增加合理的长期画像字段，但必须稳定、持久。
    - 不能包含任何短期事件、一次性情绪、单日行为。

    ------------------------------------------------------------
    【输出格式要求】
    - 只输出一个干净的 JSON 对象
    - 不要 markdown、不加解释、不加额外文字

    ------------------------------------------------------------

    daily_log:
    {json.dumps(daily_logs, ensure_ascii=False)}

    old_profile:
    {json.dumps(old_profile, ensure_ascii=False)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    profile_obj = extract_json_from_text(response.text)

    # 如果仍失败：用最小可用结构兜底，避免写 raw_profile
    if not isinstance(profile_obj, dict):
        profile_obj = {
            "lifestyle": "",
            "work_pattern": "",
            "hobbies": [],
            "social_level": "",
            "personality_traits": []
        }

    # 最后再写入：顶层就是字段
    with open(user_profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_obj, f, ensure_ascii=False, indent=4)

    print("用户画像已更新")


# 生成当日总结
def generate_daily_summary(date, event_stream, conversations):
    if client is None:
        print("Gemini unavailable, skip daily summary.")
        return None
    if not (event_stream or conversations):
        print("当天无事件，跳过总结")
        return

    prompt = f"""
    你是长期记忆系统的“日记巩固器（Memory Consolidator）”，遵循 EverMemOS 原则：
    - 只写可复用的长期信息，忽略琐碎与重复
    - 区分：一次性事件 vs 可沉淀的规律/变化
    - 语言自然，不要提“事件流/JSON/模型/系统”等字样

    你会收到：
    - 事件记录（可能嘈杂、重复）
    - 对话记录（包含用户显式表达的偏好/情绪/目标/压力源等）

    请输出一段自然中文日记（只要一段话，不要分点），包含：
    - 今天真正重要的事情（对用户生活或目标有意义的）
    - 用户在对话中显式表达的重要信息（偏好、计划、困扰、决定）
    - 如果出现“持续性线索”（例如作息变化、反复出现的压力、反复提到的兴趣），请用更明确的措辞写出来
    - 不要写具体时间戳，不要逐条复述原始记录

    事件记录：
    {json.dumps(event_stream, ensure_ascii=False)}

    对话记录：
    {json.dumps(conversations, ensure_ascii=False)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    daily_summary = response.text.strip()
    print("当日总结：", daily_summary)

    save_daily_log(date, daily_summary)
    return daily_summary


def check_day_change():
    global current_date, event_stream

    now_date = datetime.now().date()
    if now_date != current_date:
        print("检测到跨天，开始生成当日总结...")

        with open(event_stream_path, "r", encoding="utf-8") as f:
            event_stream_file = json.load(f)

        with open(conversations_path, "r", encoding="utf-8") as f:
            conversations_file = json.load(f)

        # 更新日志（拿到当日总结文本）
        daily_summary_text = generate_daily_summary(current_date, event_stream_file, conversations_file)

        # 生成/合并 memcells（原子记忆）
        if daily_summary_text:
            existing_memcells = load_memcells()
            extracted = generate_memcells_for_day(
                current_date,
                daily_summary_text,
                conversations_file,
                existing_memcells
            )
            merged = merge_memcells(existing_memcells, extracted, current_date.isoformat())
            save_memcells(merged)
            print(f"memcells 已更新：本次输出 {len(extracted)} 条，库总量 {len(merged)} 条")


        # 更新用户画像
        generate_user_profile()

        # 清空当天事件流（内存）
        event_stream.clear()

        # 清空文件内容（磁盘）
        event_stream_file.clear()
        conversations_file.clear()

        with open(event_stream_path, "w", encoding="utf-8") as f:
            json.dump(event_stream_file, f, ensure_ascii=False, indent=2)

        with open(conversations_path, "w", encoding="utf-8") as f:
            json.dump(conversations_file, f, ensure_ascii=False, indent=2)

        current_date = now_date


def process_frame(frame):
    """处理每一帧：检测人物，并记录视频片段"""
    global video_start_time, out_mp4, video_writing, detect_cnt, detect_per_frames

    if net is None:
        return frame

    if video_writing is True:
        # 保存当前帧到视频文件
        out_mp4.write(frame)

    # 间隔帧不做检测
    if detect_cnt < detect_per_frames:
        detect_cnt += 1
        return(frame)
    
    # 抽取帧进行检测
    else:
        # 重置计数
        detect_cnt = 0

        # 获取图像的高度、宽度
        height, width, channels = frame.shape
        
        # 转换为YOLO输入格式
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        # 对每个检测框进行处理
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5: # 检测阈值
                    # 获取框的坐标
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 使用非极大值抑制来过滤重复的框
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            # 开始录制视频
            if video_writing is False:
                video_writing = True
                video_start_time = time.time()

                # MP4 输出
                out_mp4 = cv2.VideoWriter(
                    captured_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    20.0,
                    (width, height)
                )

                print("开始录制视频...")

                # 保存当前帧到视频文件
                out_mp4.write(frame)

        else:
            # 没有检测到人物时结束视频记录
            if video_writing is True:
                video_writing = False
                video_end_time = time.time()

                out_mp4.release() # 结束视频录制

                print(f"视频录制完毕，处理时长: {(video_end_time - video_start_time):.2f}秒")
                upload_video(captured_video_path, video_start_time, video_end_time) # 上传视频到Gemini
                video_start_time = None

    return frame


def upload_video(video_path, video_start_time, video_end_time):
    """将视频上传到Gemini并返回事件描述"""
    global event_stream

    if client is None:
        print("Gemini unavailable, skip upload_video.")
        return

    myfile = client.files.upload(file=video_path)
    file_id = myfile.name

    print("上传成功，文件ID:", file_id)
    print("初始状态:", myfile.state)

    # 等待文件处理完成
    while True:
        f = client.files.get(name=file_id)
        print("当前状态:", f.state)

        if f.state == "ACTIVE":
            break
        elif f.state == "FAILED":
            raise RuntimeError(f"文件处理失败: {f.error}")
        
        time.sleep(2)  # 每2秒查询一次状态

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                f,
                """
            你是 EverMemOS 风格的“事件语义提取器”。
            请用一句到两句中文描述视频中的主要事件，要求：
            - 只写“对用户有意义”的动作/互动/状态，不要写画面技术细节
            - 忽略绿色方框、文字叠加、检测框
            - 不要猜测身份、不要编造动机；不确定就用中性表述
            - 输出纯文本，不要加任何前缀或解释
            """
            ]
        )

        event = {
            "start_time": datetime.fromtimestamp(video_start_time).replace(microsecond=0).isoformat(),
            "end_time": datetime.fromtimestamp(video_end_time).replace(microsecond=0).isoformat(),
            "event_description": response.text
        }

        event_stream.append(event)
        save_event_to_json(event_stream)

        print(f"事件已保存：{event}")
    except Exception as e:
        print("调用 GEMINI 失败：", e)



def save_event_to_json(event_stream):
    """将事件流保存到JSON文件"""
    with open(event_stream_path, "w", encoding="utf-8") as f:
        json.dump(event_stream, f, ensure_ascii=False, indent=4)
    
    # 开始更新日志及用户画像，重置event_stream
    check_day_change()



def start_event_stream_thread(frame_queue):
    """事件流处理的线程"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            process_frame(frame)
            frame_queue.task_done()

if __name__ == "__main__":
    print("程序作为主程序运行")
    check_day_change()
