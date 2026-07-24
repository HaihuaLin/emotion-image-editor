"""
魔搭（ModelScope）模型配置
用于从魔搭平台下载模型
"""

# ============================================================
# 魔搭模型ID配置
# ============================================================

MODELS = {
    # CLIP 模型 - 用于情感诊断和评分
    "clip": {
        "model_id": "AI-ModelScope/CLIP-GmP-ViT-L-14",
        "local_dir": "/mnt/user/models/modelscope/clip",
        "description": "CLIP 模型，用于零样本情感分类和生成质量评估",
    },

    # BLIP-2 模型 - 用于语义解析
    "blip2": {
        "model_id": "goldslj/blip2-opt-2.7b",
        "local_dir": "/mnt/user/models/modelscope/blip2",
        "description": "BLIP-2 模型，用于生成客观场景描述",
    },

    # Stable Diffusion 模型 - 用于图像生成
    "sd": {
        "model_id": "AI-ModelScope/stable-diffusion-v1.5-no-safetensor",
        "local_dir": "/mnt/user/models/modelscope/sd-v1.5",
        "description": "Stable Diffusion v1.5，用于图像生成",
    },

    # ControlNet 模型 - 用于结构控制
    "controlnet": {
        "model_id": "AI-ModelScope/sd-controlnet-canny",
        "local_dir": "/mnt/user/models/modelscope/controlnet",
        "description": "ControlNet 模型，用于保持图像结构",
    },
}

# ============================================================
# 下载脚本
# ============================================================

DOWNLOAD_SCRIPT = '''
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
mkdir -p /mnt/user/models/modelscope

# 1. 下载 CLIP 模型
echo ""
echo "[1/4] 下载 CLIP 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/CLIP-GmP-ViT-L-14', cache_dir='/mnt/user/models/modelscope')
print('✓ CLIP 模型下载完成')
"

# 2. 下载 BLIP-2 模型
echo ""
echo "[2/4] 下载 BLIP-2 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('goldslj/blip2-opt-2.7b', cache_dir='/mnt/user/models/modelscope')
print('✓ BLIP-2 模型下载完成')
"

# 3. 下载 Stable Diffusion 模型
echo ""
echo "[3/4] 下载 Stable Diffusion 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/stable-diffusion-v1.5-no-safetensor', cache_dir='/mnt/user/models/modelscope')
print('✓ Stable Diffusion 模型下载完成')
"

# 4. 下载 ControlNet 模型
echo ""
echo "[4/4] 下载 ControlNet 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/sd-controlnet-canny', cache_dir='/mnt/user/models/modelscope')
print('✓ ControlNet 模型下载完成')
"

echo ""
echo "=========================================="
echo "✓ 所有模型下载完成！"
echo "=========================================="
'''
