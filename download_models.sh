#!/bin/bash
# ============================================================
# 从魔搭下载所有模型
# 运行方式: bash download_models.sh
# ============================================================

set -e

echo "=========================================="
echo "从魔搭下载模型"
echo "=========================================="

# 创建目录
mkdir -p /mnt/workspace/models/modelscope

# 1. 下载 CLIP 模型
echo ""
echo "[1/4] 下载 CLIP 模型: AI-ModelScope/CLIP-GmP-ViT-L-14"
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/CLIP-GmP-ViT-L-14', cache_dir='/mnt/workspace/models/modelscope')
print('✓ CLIP 模型下载完成')
"

# 2. 下载 BLIP-2 模型
echo ""
echo "[2/4] 下载 BLIP-2 模型: goldslj/blip2-opt-2.7b"
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('goldslj/blip2-opt-2.7b', cache_dir='/mnt/workspace/models/modelscope')
print('✓ BLIP-2 模型下载完成')
"

# 3. 下载 Stable Diffusion 模型
echo ""
echo "[3/4] 下载 Stable Diffusion 模型: AI-ModelScope/stable-diffusion-v1.5-no-safetensor"
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/stable-diffusion-v1.5-no-safetensor', cache_dir='/mnt/workspace/models/modelscope')
print('✓ Stable Diffusion 模型下载完成')
"

# 4. 下载 ControlNet 模型
echo ""
echo "[4/4] 下载 ControlNet 模型: AI-ModelScope/sd-controlnet-canny"
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/sd-controlnet-canny', cache_dir='/mnt/workspace/models/modelscope')
print('✓ ControlNet 模型下载完成')
"

echo ""
echo "=========================================="
echo "✓ 所有模型下载完成！"
echo "=========================================="
echo ""
echo "模型已保存到: /mnt/workspace/models/modelscope/"
