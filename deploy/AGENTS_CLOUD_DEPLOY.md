# Aura 云端迁移执行手册（对齐最新版 Agents.md）

本手册严格对应 `Agents.md` 当前要求：

- 本地仅生成部署脚手架文件
- 不改业务逻辑代码
- 部署目标服务器：Ubuntu 22.04 LTS
- 使用 `venv`，禁止 `sudo pip install`

---

## 1. 本地需要准备的文件

请确认以下 4 个文件已在 `deploy/` 目录下：

1. `deploy/setup_server.sh`
2. `deploy/Caddyfile`
3. `deploy/aura.service`
4. `deploy/tmux_hook.txt`

---

## 2. 上传到云服务器

将整个 Aura 项目目录上传到服务器（示例路径：`/opt/aura`）。

---

## 3. 在服务器执行初始化脚本

```bash
cd /opt/aura
bash deploy/setup_server.sh
```

该脚本会完成：

- 系统依赖安装（Python venv、ffmpeg、tmux、curl、wget 等）
- Caddy 安装
- Python `venv` 创建与依赖安装
- `gunicorn` / `uvicorn` 安装

---

## 4. 填写环境变量

在服务器项目根目录创建/编辑 `.env`：

```bash
cd /opt/aura
cp .env.example .env
vim .env
```

至少填写：

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `AWS_FACE_KEY_ID`
- `AWS_FACE_ACCESS_KEY`
- `AWS_REGION_NAME`
- `AURA_WS_TOKEN_SECRET`
- `AURA_ALLOWED_ORIGINS`

---

## 5. 部署 Systemd 服务

1. 将模板复制到系统目录：

```bash
sudo cp /opt/aura/deploy/aura.service /etc/systemd/system/aura.service
```

2. 编辑并替换占位符：

- `<USER>` -> 你的 Linux 用户名
- `<PROJECT_DIR>` -> 例如 `/opt/aura`

```bash
sudo vim /etc/systemd/system/aura.service
```

3. 回显检查（必须）：

```bash
sudo cat /etc/systemd/system/aura.service
```

4. 重载并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aura
sudo systemctl status aura --no-pager
```

---

## 6. 部署 Caddy 反向代理

1. 复制模板：

```bash
sudo cp /opt/aura/deploy/Caddyfile /etc/caddy/Caddyfile
```

2. 按需修改（可将 `:80` 替换为域名）：

```bash
sudo vim /etc/caddy/Caddyfile
```

3. 回显检查（必须）：

```bash
sudo cat /etc/caddy/Caddyfile
```

4. 重载 Caddy：

```bash
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

---

## 7. （可选）启用登录自动进入 tmux 控制台

按 `deploy/tmux_hook.txt` 的说明，将代码追加到 `~/.bashrc` 并替换 `<PROJECT_DIR>`。

---

## 8. 验证

```bash
curl http://127.0.0.1:8000/healthz
```

预期返回：

```json
{"status":"ok"}
```

再测试创建会话：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/session/start \
  -H "Content-Type: application/json" \
  -d '{"user_id":"smoke-test"}'
```

---

## 9. 常见故障排查

- `aura.service` 启动失败：`journalctl -u aura -n 200 --no-pager`
- Caddy 无法转发：确认 `reverse_proxy 127.0.0.1:8000`
- WebSocket 失败：确认 `AURA_ALLOWED_ORIGINS` 与前端域名一致
- 模型文件缺失：确认 `yolo/yolov3.cfg` 与 `yolo/yolov3.weights` 存在

