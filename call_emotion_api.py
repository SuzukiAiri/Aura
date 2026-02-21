import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import AWS_FACE_ACCESS_KEY, AWS_FACE_KEY_ID, AWS_REGION_NAME

# from deepface import DeepFace
import numpy as np
import cv2


def call_emotion_api_aws(
    image_bytes,
    region_name: str = AWS_REGION_NAME,
    aws_access_key_id: str = AWS_FACE_KEY_ID,
    aws_secret_access_key: str = AWS_FACE_ACCESS_KEY,
    aws_session_token: str = None,
    return_confidence_as_ratio: bool = True
):
    """
    使用 AWS Rekognition 进行情绪识别。
    - image_bytes: bytes 图片二进制
    - region_name: AWS 区域，如 'us-west-2'。若为 None，则使用环境或 AWS 配置中的默认值
    - aws_access_key_id / aws_secret_access_key / aws_session_token: 可选（若不传则使用环境或 IAM role）
    - return_confidence_as_ratio: 若 True 则返回 0..1 的置信度，否则返回 0..100（默认）
    返回示例（若检测到人脸）:
      {"HAPPY": 95.2, "CALM": 3.1, "SURPRISED": 1.7}
    若未检测到人脸或出错返回 None。
    """
    # 决定 region：优先参数，再环境变量，最后 boto3 默认（若都没有，报错）
    region = region_name or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')
    if not region:
        print("错误：必须指定 AWS 区域（region）。请传入 region_name 或设置 AWS_REGION / AWS_DEFAULT_REGION 环境变量，或运行 `aws configure`。")
        return None

    try:
        client_args = {'region_name': region}
        if aws_access_key_id and aws_secret_access_key:
            client_args.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })
        if aws_session_token:
            client_args['aws_session_token'] = aws_session_token

        client = boto3.client('rekognition', **client_args)

        response = client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )

        faces = response.get('FaceDetails', [])
        if not faces:
            # 没有人脸
            return None

        # 返回所有人脸的情绪
        all_face_emotions = []

        for face in faces:
            emotions = face.get('Emotions', [])
            if not emotions:
                return None

            # 转成 {emotion: confidence}
            emotion_dict = {e['Type']: e['Confidence'] for e in emotions}

            if return_confidence_as_ratio:
                # 把 0-100 转成 0-1
                emotion_dict = {k: (v / 100.0) for k, v in emotion_dict.items()}

            emotion_dict = dict(sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True))

            all_face_emotions.append(emotion_dict)

        return all_face_emotions

    except (BotoCoreError, ClientError) as e:
        print("调用 AWS Rekognition 出错：", e)
        return None
    except Exception as e:
        print("未知错误：", e)
        return None
    

# def call_emotion_api_deepface(image_bytes):
#     np_arr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     objs = DeepFace.analyze(
#     img_path=img, actions = ['emotion']
#     )
    
#     if isinstance(objs, dict):
#         objs = [objs] # 兼容 DeepFace 单脸/多脸返回格式

#     all_face_emotions = []

#     for obj in objs:
#         emotion_dict = obj.get("emotion", {})
#         if not emotion_dict:
#             continue

#         emotion_dict = {k: float(v) / 100.0 for k, v in emotion_dict.items()}

#         emotion_dict = dict(sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True))

#         all_face_emotions.append(emotion_dict)

#     return all_face_emotions if all_face_emotions else None
    