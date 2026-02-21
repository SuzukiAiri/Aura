# Aura

Aura 是一个“情绪识别 + 主动关怀 + 语音交互 + 长期记忆”的多模态 AI 系统。  
支持本地桌面运行，也支持迁移到 Ubuntu 云服务器进行 Web 远程访问。

---

## 1. 功能说明（全量）

### 1.1 核心链路

1. 浏览器或本地设备采集音视频  
2. AWS Rekognition 执行情绪识别与白名单匹配  
3. `decide_should_speak.py` 决策是否触发主动交流  
4. OpenAI 基于上下文生成回复  
5. Whisper 转写（STT）+ edge-tts 合成（TTS）  
6. YOLO + Gemini 生成事件流并沉淀长期记忆  

### 1.2 关键能力

- 人脸检测与白名单识别  
- 情绪识别与防打扰冷却机制  
- 多轮中文语音对话  
- 会话存档：`conversations.json`  
- 事件流：`event_stream.json`  
- 每日日志：`daily_log.json`  
- 用户画像：`user_profile.json`  
- 原子记忆：`memcells.json`  
- 云端 API + WebSocket 实时交互  
- SQLite 会话检索：`aura.db`  

### 1.3 运行入口

- `Aura.py`：本地经典版本  
- `Aura_realtime.py`：本地 Realtime 版本  
- `server/app.py`：云端 FastAPI + WebSocket 入口  

---

## 2. 目录结构

```text
Aura.py
Aura_realtime.py
Aura_audio.py
call_llm_api.py
call_rekognition_api.py
decide_should_speak.py
event_stream.py
config.py

server/
  app.py
  session_worker.py
  media_pipeline.py
  storage.py
  auth.py

web/
  index.html
  app.js
  styles.css

deploy/
  setup_server.sh
  Caddyfile
  aura.service
  tmux_hook.txt
  AGENTS_CLOUD_DEPLOY.md

Dockerfile
docker-compose.yml
.env.example
```

---

## 3. 环境要求

### 3.1 通用

- Python 3.10+（推荐 3.12）
- FFmpeg
- 能访问 OpenAI / AWS / Gemini

### 3.2 本地模式

- Windows 10/11
- 摄像头、麦克风、扬声器
- 建议使用纯英文路径（避免 OpenCV 路径问题）

### 3.3 云服务器模式

- Ubuntu 22.04 LTS
- 建议 4 vCPU / 8GB RAM 起步
- 生产推荐域名 + HTTPS/WSS

---

## 4. API Key 填写（.env）

### 4.1 创建 `.env`

```bash
cp .env.example .env
```

### 4.2 示例模板

```dotenv
DOMAIN=your-domain.example.com

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxx
AWS_FACE_KEY_ID=AKIAxxxxxxxxxxxx
AWS_FACE_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
AWS_REGION_NAME=us-west-2

AURA_WS_TOKEN_SECRET=replace-with-a-long-random-secret
AURA_WS_TOKEN_TTL_SECONDS=1800
AURA_ALLOWED_ORIGINS=https://your-domain.example.com

AURA_DATA_DIR=/app/data
AURA_WHITELIST_DIR=/app/whitelist
AURA_YOLO_DIR=/app/yolo

AURA_WHISPER_MODEL=base
AURA_TTS_VOICE=zh-CN-XiaoxiaoNeural
AURA_OUTPUT_SAMPLE_RATE=24000
AURA_OUTPUT_CHUNK_MS=20
AURA_FRAME_INGEST_FPS=2.0
AURA_FRAME_DETECT_FPS=1.0
AURA_FACE_DETECTION_COOLDOWN=2.0
AURA_SESSION_END_PROMPT=如果有需要，可以再呼唤我。
```

### 4.3 必填项

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `AWS_FACE_KEY_ID`
- `AWS_FACE_ACCESS_KEY`
- `AWS_REGION_NAME`
- `AURA_WS_TOKEN_SECRET`
- `AURA_ALLOWED_ORIGINS`

---

## 5. 本地运行

```bash
pip install -r requirements.txt
python Aura.py
```

---

## 6. 云端迁移部署（按新版 Agents.md）

新版 `Agents.md` 要求：在本地生成部署脚手架文件，由你手动推送并在 Ubuntu 服务器执行。  
核心文件如下：

1. `deploy/setup_server.sh`
2. `deploy/Caddyfile`
3. `deploy/aura.service`
4. `deploy/tmux_hook.txt`

详细执行手册见：`deploy/AGENTS_CLOUD_DEPLOY.md`

### 6.1 服务器执行步骤（简版）

1. 上传项目到服务器（例如 `/opt/aura`）
2. 执行初始化脚本：
   ```bash
   cd /opt/aura
   bash deploy/setup_server.sh
   ```
3. 配置 `.env`
4. 安装 systemd：
   - 复制 `deploy/aura.service` 到 `/etc/systemd/system/aura.service`
   - 替换 `<USER>`、`<PROJECT_DIR>`
   - 执行：
     ```bash
     sudo cat /etc/systemd/system/aura.service
     sudo systemctl daemon-reload
     sudo systemctl enable --now aura
     ```
5. 安装 Caddy：
   - 复制 `deploy/Caddyfile` 到 `/etc/caddy/Caddyfile`
   - 执行：
     ```bash
     sudo cat /etc/caddy/Caddyfile
     sudo systemctl reload caddy
     ```

---

## 7. API 与 WebSocket

### 7.1 REST

- `GET /healthz`
- `POST /api/v1/session/start`
- `GET /api/v1/session/{session_id}/history`

`POST /api/v1/session/start` 请求：

```json
{
  "user_id": "demo-user"
}
```

返回：

```json
{
  "session_id": "xxxx",
  "ws_url": "/api/v1/session/{session_id}/stream?token=..."
}
```

### 7.2 WebSocket

连接：

`WS /api/v1/session/{session_id}/stream?token=...`

客户端 -> 服务端：

- `audio_chunk`
- `video_frame`
- `control` (`end_turn` / `stop`)

服务端 -> 客户端：

- `assistant_audio_chunk`
- `assistant_text_delta`
- `emotion_update`
- `status`
- `error`

---

## 8. 逐条检查清单（迁移必看）

### 8.1 部署前检查

- [ ] 服务器系统为 Ubuntu 22.04
- [ ] 项目路径为纯英文（例如 `/opt/aura`）
- [ ] 已准备 OpenAI / AWS / Gemini Key
- [ ] `yolo/yolov3.cfg` 与 `yolo/yolov3.weights` 已上传
- [ ] `whitelist/` 已准备授权人脸样本

### 8.2 环境与依赖检查

- [ ] 已执行 `bash deploy/setup_server.sh`
- [ ] `venv` 已创建成功
- [ ] `ffmpeg -version` 可执行
- [ ] `python -c "import fastapi, cv2, whisper"` 成功

### 8.3 配置检查

- [ ] `.env` 已存在并填入真实 key
- [ ] `AURA_WS_TOKEN_SECRET` 已更换随机长串
- [ ] `AURA_ALLOWED_ORIGINS` 已设置成实际域名

### 8.4 systemd 检查

- [ ] 已复制并修改 `/etc/systemd/system/aura.service`
- [ ] 已执行 `sudo cat /etc/systemd/system/aura.service`
- [ ] 已执行 `sudo systemctl daemon-reload`
- [ ] `sudo systemctl status aura` 为 active

### 8.5 Caddy 检查

- [ ] 已复制并修改 `/etc/caddy/Caddyfile`
- [ ] 已执行 `sudo cat /etc/caddy/Caddyfile`
- [ ] `sudo systemctl status caddy` 正常

### 8.6 功能验收检查

- [ ] `GET /healthz` 返回 200
- [ ] `POST /api/v1/session/start` 返回 `session_id`
- [ ] 浏览器可授权摄像头/麦克风
- [ ] 可收到 `emotion_update`/`assistant_text_delta`
- [ ] 可播放 `assistant_audio_chunk`
- [ ] `conversations.json` 和 `aura.db` 有新增记录

### 8.7 稳定性检查

- [ ] 连续运行 30 分钟无崩溃
- [ ] 断网重连后可恢复会话
- [ ] 日志无持续报错刷屏

### 8.8 安全检查

- [ ] 仓库中无明文 API Key
- [ ] `.env` 未提交
- [ ] 生产不使用 `AURA_ALLOWED_ORIGINS=*`
- [ ] 生产环境启用 HTTPS/WSS

### 8.9 回滚检查

- [ ] 保留上一版本代码副本
- [ ] 保留上一版本 `.env` 备份
- [ ] 回滚命令已验证可执行

---

## 9. 运维命令

```bash
sudo systemctl restart aura
sudo systemctl status aura --no-pager
sudo journalctl -u aura -f

curl http://127.0.0.1:8000/healthz
```

---

## 10. 常见问题

1. OpenCV 报 `CascadeClassifier !empty()`  
- 常见于路径或模型文件问题，建议纯英文路径。

2. YOLO 加载失败  
- 检查 `AURA_YOLO_DIR`、`yolov3.cfg`、`yolov3.weights`。

3. 启动时报 key 缺失  
- 检查 `.env` 是否被 systemd 正确注入。

4. 浏览器无声音  
- 检查浏览器权限和自动播放策略。

---

## 11. 参考

- 迁移执行手册：`deploy/AGENTS_CLOUD_DEPLOY.md`
