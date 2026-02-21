import os
import io
from typing import Optional, Dict, Any, List, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image

from config import AWS_FACE_ACCESS_KEY, AWS_FACE_KEY_ID, AWS_REGION_NAME, WHITELIST_DIR

def _load_whitelist_images(whitelist_dir: str) -> List[Tuple[str, str, bytes]]:
    """
    whitelist_dir/
      person_id1/ a.jpg b.jpg
      person_id2/ 1.png
    返回 [(person_id, image_path, image_bytes), ...]
    """
    if not os.path.isdir(whitelist_dir):
        raise FileNotFoundError(f"Whitelist dir not found: {whitelist_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items: List[Tuple[str, str, bytes]] = []

    for person_id in os.listdir(whitelist_dir):
        person_path = os.path.join(whitelist_dir, person_id)
        if not os.path.isdir(person_path):
            continue

        for fn in os.listdir(person_path):
            _, ext = os.path.splitext(fn.lower())
            if ext not in exts:
                continue
            img_path = os.path.join(person_path, fn)
            try:
                with open(img_path, "rb") as f:
                    items.append((person_id, img_path, f.read()))
            except Exception:
                continue

    return items

def _crop_face_from_bytes(image_bytes: bytes, bbox: Dict[str, float]) -> bytes:
    """
    bbox: Rekognition BoundingBox (0..1)
    裁剪后返回 JPEG bytes
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    left = int(max(0.0, bbox["Left"]) * w)
    top = int(max(0.0, bbox["Top"]) * h)
    right = int(min(1.0, bbox["Left"] + bbox["Width"]) * w)
    bottom = int(min(1.0, bbox["Top"] + bbox["Height"]) * h)

    if right <= left or bottom <= top:
        raise ValueError("Invalid bounding box (empty crop).")

    face_img = img.crop((left, top, right, bottom))
    out = io.BytesIO()
    face_img.save(out, format="JPEG", quality=95)
    return out.getvalue()

def call_rekognition_api(
    image_bytes: bytes,
    whitelist_dir: str = WHITELIST_DIR,
    region_name: str = AWS_REGION_NAME,
    aws_access_key_id: str = AWS_FACE_KEY_ID,
    aws_secret_access_key: str = AWS_FACE_ACCESS_KEY,
    aws_session_token: str = None,
    similarity_threshold: float = 90.0,
    max_faces: int = 10,
    return_emotion_confidence_as_ratio: bool = True,  # True: 0..1, False: 0..100
) -> Optional[Dict[str, Any]]:
    """
    单函数完成：
    1) DetectFaces(ALL) 获取 emotions + bounding boxes （一次网络请求）
    2) CompareFaces 对本地白名单进行鉴权（多次网络请求，无法避免）

    返回:
    {
      "is_authorized": bool,
      "faces": [
        {
          "face_index": int,
          "bounding_box": {...},
          "emotions": {"HAPPY":0.95, ...} 或 0..100,
          "is_authorized": bool,
          "matched_person_id": str|None,
          "matched_image_path": str|None,
          "similarity": float|None
        }, ...
      ],
      "best_match": {...}|None
    }
    未检测到人脸/出错返回 None
    """

    region = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        print("错误：必须指定 AWS 区域（region）。")
        return None

    try:
        whitelist_items = _load_whitelist_images(whitelist_dir)
        if not whitelist_items:
            print(f"白名单库为空：{whitelist_dir}")
            return None

        client_args = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            client_args.update(
                {"aws_access_key_id": aws_access_key_id, "aws_secret_access_key": aws_secret_access_key}
            )
        if aws_session_token:
            client_args["aws_session_token"] = aws_session_token

        rek = boto3.client("rekognition", **client_args)

        # ✅ 一次 DetectFaces 拿到 emotions + bbox（情绪识别合并在这里）
        detect_resp = rek.detect_faces(
            Image={"Bytes": image_bytes},
            Attributes=["ALL"],
        )
        faces = detect_resp.get("FaceDetails", [])
        if not faces:
            return None

        faces = faces[:max_faces]

        result_faces: List[Dict[str, Any]] = []
        best_match_overall = None
        best_similarity_overall = -1.0

        for i, face in enumerate(faces):
            bbox = face.get("BoundingBox")
            emotions_list = face.get("Emotions", []) or []

            # 规范化 emotions 输出
            emotions = {e["Type"]: float(e["Confidence"]) for e in emotions_list if "Type" in e and "Confidence" in e}
            if return_emotion_confidence_as_ratio:
                emotions = {k: (v / 100.0) for k, v in emotions.items()}
            emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))

            if not bbox:
                result_faces.append(
                    {
                        "face_index": i,
                        "bounding_box": None,
                        "emotions": emotions,
                        "is_authorized": False,
                        "matched_person_id": None,
                        "matched_image_path": None,
                        "similarity": None,
                        "error": "missing_bounding_box",
                    }
                )
                continue

            # 裁剪出该脸（后续 CompareFaces 的 Target 用裁剪脸，减少传输量）
            try:
                face_crop_bytes = _crop_face_from_bytes(image_bytes, bbox)
            except Exception as e:
                result_faces.append(
                    {
                        "face_index": i,
                        "bounding_box": bbox,
                        "emotions": emotions,
                        "is_authorized": False,
                        "matched_person_id": None,
                        "matched_image_path": None,
                        "similarity": None,
                        "error": f"crop_failed: {e}",
                    }
                )
                continue

            face_best_person = None
            face_best_path = None
            face_best_sim = -1.0

            # ❗白名单在本地 -> 只能逐个 compare_faces（多次网络请求，无法合并到 detect_faces）
            for person_id, img_path, ref_bytes in whitelist_items:
                try:
                    cmp_resp = rek.compare_faces(
                        SourceImage={"Bytes": ref_bytes},
                        TargetImage={"Bytes": face_crop_bytes},
                        SimilarityThreshold=similarity_threshold,
                    )
                    matches = cmp_resp.get("FaceMatches", [])
                    if not matches:
                        continue

                    sim = max(float(m.get("Similarity", 0.0)) for m in matches)
                    if sim > face_best_sim:
                        face_best_sim = sim
                        face_best_person = person_id
                        face_best_path = img_path

                except (BotoCoreError, ClientError):
                    continue

            face_is_auth = face_best_sim >= similarity_threshold
            face_result = {
                "face_index": i,
                "bounding_box": bbox,
                "emotions": emotions,
                "is_authorized": face_is_auth,
                "matched_person_id": face_best_person if face_is_auth else None,
                "matched_image_path": face_best_path if face_is_auth else None,
                "similarity": round(face_best_sim, 3) if face_best_sim >= 0 else None,
            }
            result_faces.append(face_result)

            if face_best_sim > best_similarity_overall:
                best_similarity_overall = face_best_sim
                best_match_overall = face_result

        is_authorized = any(f.get("is_authorized") for f in result_faces)

        return {
            "is_authorized": is_authorized,
            "faces": result_faces,
            "best_match": best_match_overall if best_similarity_overall >= 0 else None,
        }

    except (BotoCoreError, ClientError) as e:
        print("调用 AWS Rekognition 出错：", e)
        return None
    except Exception as e:
        print("未知错误：", e)
        return None