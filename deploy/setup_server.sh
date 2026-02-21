#!/bin/bash
# Aura 服务器自动化初始化脚本
# 请在云服务器的项目根目录下运行此脚本

set -euo pipefail

echo "1. 更新系统并安装系统级依赖..."
sudo apt-get update && sudo apt-get install -y python3.10-venv python3-pip ffmpeg tmux curl wget software-properties-common apt-transport-https

echo "2. 安装 Caddy Server..."
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
sudo apt-get update && sudo apt-get install -y caddy

echo "3. 构建 Python 虚拟环境并安装项目依赖..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "uvicorn[standard]" gunicorn

echo "初始化完成！请手动配置 .env 文件并将 Systemd 和 Caddy 配置移至系统目录。"
