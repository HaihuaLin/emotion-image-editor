#!/bin/bash
# ============================================================
# Streamlit 版本启动脚本
# ============================================================

echo "=========================================="
echo "EmoEdit - Streamlit 版本启动"
echo "=========================================="

# 安装 Streamlit
pip install -q streamlit

# 启动应用
echo ""
echo "启动 Streamlit 应用..."
echo "访问地址: http://localhost:8501"
echo ""

streamlit run app_streamlit.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
