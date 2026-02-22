# AGENTS.md
# Aura 系统本地配置与自动化部署脚手架生成规范
(Codex 本地文件修改手册)

## 1. 代理核心角色与操作边界约束
* **身份定位**: 你是一位资深的 DevOps 工程师。当前任务是在本地代码库中生成一系列部署脚本和配置文件，为后续手动推送到 Ubuntu 22.04 LTS 云服务器进行部署做好所有前置准备。
* **操作边界**:
  1. 绝对不要尝试运行系统级安装命令（如 `apt-get`）或试图连接远程服务器。
  2. 你的所有输出必须仅仅是创建或修改本地项目目录下的文本文件。
  3. 不得修改原有的业务逻辑代码（如 `Aura.py`, `server/app.py` 等），仅专注于环境配置。
* **目录规范**: 请在项目根目录下创建一个名为 `deploy/` 的文件夹，将所有生成的部署相关文件统一存放在此目录下。

## 2. 需要生成的配置文件与脚本清单

请依次在 `deploy/` 目录下创建以下 4 个文件：

### 文件 1：`deploy/setup_server.sh`
这是一个将在 Ubuntu 22.04 上执行的 Bash 脚本，用于自动化安装依赖和初始化 Python 环境。请写入以下内容（必须包含注释以指导用户）：
```bash
#!/bin/bash
# Aura 服务器自动化初始化脚本
# 请在云服务器的项目根目录下运行此脚本

echo "1. 更新系统并安装系统级依赖..."
sudo apt-get update && sudo apt-get install -y python3.10-venv python3-pip ffmpeg tmux curl wget software-properties-common apt-transport-https

echo "2. 安装 Caddy Server..."
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf '[https://dl.cloudsmith.io/public/caddy/stable/gpg.key](https://dl.cloudsmith.io/public/caddy/stable/gpg.key)' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf '[https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt](https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt)' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy

echo "3. 构建 Python 虚拟环境并安装项目依赖..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "uvicorn[standard]" gunicorn

echo "初始化完成！请手动配置 .env 文件并将 Systemd 和 Caddy 配置移至系统目录。"
```


### 文件 2：deploy/Caddyfile
用于安全转发 WebSocket 的反向代理配置模板。写入以下内容：

:80 {
    # 监听 80 端口（如果您有域名，请将 :80 替换为您的域名）
    # 将 HTTP 和 WebSocket 流量反向代理到本地 8000 端口
    reverse_proxy 127.0.0.1:8000
}


### 文件 3：deploy/aura.service
Systemd 守护进程配置文件模板。写入以下内容：

Ini, TOML
[Unit]
Description=Aura FastAPI Application Service
After=network.target

[Service]
# 请将 <USER> 替换为您的 Linux 用户名（非 root）
User=<USER>
# 请将 <PROJECT_DIR> 替换为您服务器上的 Aura 项目绝对路径
WorkingDirectory=<PROJECT_DIR>
EnvironmentFile=<PROJECT_DIR>/.env
ExecStart=<PROJECT_DIR>/venv/bin/gunicorn server.app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
Restart=always

[Install]
WantedBy=multi-user.target


### 文件 4：deploy/tmux_hook.txt
这是用于实现“登录即交互”的 SSH 挂载代码片段。写入以下内容，并在此文件顶部添加一段文字，提示用户手动将此代码段追加到服务器的 ~/.bashrc 文件中。

Bash
# Aura Auto-Attach Tmux Logic
# 请在添加到 .bashrc 前，将 <PROJECT_DIR> 替换为项目的实际绝对路径
if [[ $- =~ i ]]; then
    tmux new-session -A -s aura_console -d
    tmux send-keys -t aura_console:0 'journalctl -u aura.service -f' C-m
    tmux split-window -h -t aura_console:0
    tmux send-keys -t aura_console:0.1 'cd <PROJECT_DIR> && source venv/bin/activate' C-m
    exec tmux attach-session -t aura_console
fi