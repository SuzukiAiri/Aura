import time


MIN_SECONDS_BETWEEN_GREETS = 10  # 最小间隔，避免频繁打扰
EMOTION_CHANGE_THRESHOLD = 0.5  # 如果情绪置信度变化大于此值则认为有明显变化


def decide_should_speak(rekognition, last_greet_time, last_rekognition):
    """
    本地逻辑判断是否该生成并播报问候：
    - 若上次问候到现在超过 MIN_SECONDS_BETWEEN_GREETS（节流）
    - 若当前情绪与上一次显著不同（避免重复相同问候）
    - 若当前情绪表明可能需要关怀（例如 sadness, anger, neutral 可设置）
    返回 True/False
    """
    now = time.time()
    if now - last_greet_time < MIN_SECONDS_BETWEEN_GREETS:
        return False

    # 如果没有上次情绪记录，则可以问候
    if last_rekognition is None:
        return True

    # TODO: 其他判断逻辑，emotion_list : list[dict]
    return True

    # 否则不打扰
    return False