#!/bin/bash
# ============================================================
# EmoEdit 一键启动脚本
# ============================================================

echo "=========================================="
echo "EmoEdit 一键启动"
echo "=========================================="

# 1. 关闭所有旧进程
echo "[1/4] 关闭旧进程..."
pkill -f streamlit 2>/dev/null
pkill -f "python.*app" 2>/dev/null
sleep 2

# 2. 安装缺失依赖
echo "[2/4] 检查依赖..."
pip install -q streamlit

# 3. 找一个可用端口
PORT=8502
for p in 8502 8503 8504 8505 8506; do
    if ! lsof -i:$p >/dev/null 2>&1; then
        PORT=$p
        break
    fi
done

# 4. 启动
echo "[3/4] 启动应用 (端口: $PORT)..."
echo ""
echo "=========================================="
echo "  启动成功后在 DSW '应用' 面板查看端口"
echo "=========================================="
echo ""

streamlit run /mnt/workspace/emotion-image-editor/app_streamlit.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true
